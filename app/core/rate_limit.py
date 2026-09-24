"""
认证接口限流
============
`/auth/login` 与 `/auth/register` 是可被**匿名**调用、且每次都要做昂贵计算的接口：
口令校验用 argon2id（单次约占 64MiB 内存、百毫秒级 CPU）。而且
`authenticate()` 在"用户不存在"时也会跑一次等价耗时的哈希（这是刻意用来抹平
响应时间差、防止用户名枚举的，见 auth_service 的注释）——也就是说攻击者
**不需要猜中任何东西**，只要持续打这两个接口就能把 CPU 打满。

所以这里做进程内的滑动窗口限流，两个维度**各自独立**计数：

- **按 IP**：统计全部尝试（不论成败）。挡住"换着用户名撞"和纯粹的 CPU 消耗；
- **按用户名**：只统计**失败**的尝试。挡住"盯着一个账号猜密码"，
  同时保证用户自己正常登录成功不会被计入而误伤自己。

关键点：限流必须在 `authenticate()` / `register()` **之前**执行。放在后面的话，
昂贵的口令哈希已经算完了，限流就成了纯粹的事后计数，挡不住 CPU 消耗。

为什么不引入 slowapi 之类的库：本项目一直坚持不引入非必要依赖（认证本身就是
只用标准库实现的），而且"只统计失败"这种定制规则用现成库反而更别扭。

计数默认放在**进程内存**里，因此单机单进程够用，但有两个已知局限：
- 多 worker / 多实例部署时各进程各算各的，实际放行量是配置值 × 进程数；
- 进程重启即清零。

配好 Redis 后计数改放 Redis（见 app/core/redis_client.py），上述两条随之消失，
并且多实例共享同一份计数。**拿不到 Redis 时自动退回进程内实现**——
所以 Redis 挂掉不会让登录变得不可用，只是限流精度退回单机水平。
"""
import threading
import time
import uuid
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from app.core import redis_client
from app.core.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# 最多追踪多少个 key。超过就先清掉已过期的空队列。
# 这是防"内存 DoS"的：攻击者用海量不同用户名/伪造来源，可以造出无限多的 key。
_MAX_KEYS = 10000


class SlidingWindowLimiter:
    """按 key 维护时间戳队列的滑动窗口计数器。进程内、线程安全。

    FastAPI 的同步端点跑在线程池里，所以内部必须加锁。
    """

    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = threading.Lock()

    # ---------- 内部：调用方必须已持有锁 ----------

    def _window(self, key: str, now: float, window: float) -> Deque[float]:
        """取出（必要时新建）该 key 的队列，并丢弃窗口外的旧时间戳。"""
        queue = self._hits.get(key)
        if queue is None:
            queue = deque()
            self._hits[key] = queue
        cutoff = now - window
        while queue and queue[0] <= cutoff:
            queue.popleft()
        return queue

    def _sweep_locked(self) -> None:
        """清理已过期的空队列（调用方必须已持有锁）。"""
        for key in [k for k, q in self._hits.items() if not q]:
            del self._hits[key]

    # ---------- 对外 ----------

    def check_and_record(self, key: str, limit: int, window: float) -> Tuple[bool, float]:
        """判断是否放行；放行则同时记一次。返回 (是否放行, 建议等待秒数)。"""
        if limit <= 0:  # 配置为 0 视为不限流
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > _MAX_KEYS:
                self._sweep_locked()
            queue = self._window(key, now, window)
            if len(queue) >= limit:
                return False, max(0.0, queue[0] + window - now)
            queue.append(now)
            return True, 0.0

    def peek(self, key: str, limit: int, window: float) -> Tuple[bool, float]:
        """只判断不记录。用于"失败才计入"的维度：先看是否已被锁，再决定要不要干活。"""
        if limit <= 0:
            return True, 0.0
        now = time.monotonic()
        with self._lock:
            queue = self._window(key, now, window)
            if len(queue) >= limit:
                return False, max(0.0, queue[0] + window - now)
            return True, 0.0

    def record(self, key: str, window: float) -> None:
        """只记录不判断（配合 peek 使用）。"""
        now = time.monotonic()
        with self._lock:
            self._window(key, now, window).append(now)

    def reset(self) -> None:
        """清空全部计数。仅供测试使用。"""
        with self._lock:
            self._hits.clear()


# Redis 侧的实现：与进程内实现语义一致（有序集合做滑动窗口），
# 但用 Lua 脚本把"清理 + 判断 + 记录"做成一个原子操作——
# 否则并发下会出现"读到的数量偏小、于是多放行几次"的丢更新问题。
#
# 返回值统一为 {是否放行, 最早一次的时间戳(毫秒，被拒时用于算 Retry-After)}
_LUA_CHECK_AND_RECORD = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', tonumber(ARGV[1]) - tonumber(ARGV[2]))
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then
    local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
    return {0, math.floor(tonumber(oldest[2]))}
end
redis.call('ZADD', KEYS[1], tonumber(ARGV[1]), ARGV[4])
redis.call('PEXPIRE', KEYS[1], math.ceil(tonumber(ARGV[2])))
return {1, 0}
"""

_LUA_PEEK = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', tonumber(ARGV[1]) - tonumber(ARGV[2]))
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[3]) then
    local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
    return {0, math.floor(tonumber(oldest[2]))}
end
return {1, 0}
"""

_LUA_RECORD = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', tonumber(ARGV[1]) - tonumber(ARGV[2]))
redis.call('ZADD', KEYS[1], tonumber(ARGV[1]), ARGV[3])
redis.call('PEXPIRE', KEYS[1], math.ceil(tonumber(ARGV[2])))
return {1, 0}
"""


class RedisSlidingWindow:
    """限流计数的 Redis 实现（滑动窗口 + Lua 保证原子性）。"""

    @staticmethod
    def _retry_after(oldest_ms: float, window_ms: float, now_ms: float) -> float:
        """把"最早一次的时间戳"换算成还需要等多少秒。"""
        return max(0.0, (oldest_ms + window_ms - now_ms) / 1000.0)

    def check_and_record(self, client, key: str, limit: int, window: float) -> Tuple[bool, float]:
        now_ms = time.time() * 1000
        window_ms = window * 1000
        allowed, oldest = client.eval(
            _LUA_CHECK_AND_RECORD, 1, key,
            now_ms, window_ms, limit, f"{now_ms:.3f}-{uuid.uuid4().hex[:8]}",
        )
        if allowed:
            return True, 0.0
        return False, self._retry_after(float(oldest), window_ms, now_ms)

    def peek(self, client, key: str, limit: int, window: float) -> Tuple[bool, float]:
        now_ms = time.time() * 1000
        window_ms = window * 1000
        allowed, oldest = client.eval(_LUA_PEEK, 1, key, now_ms, window_ms, limit)
        if allowed:
            return True, 0.0
        return False, self._retry_after(float(oldest), window_ms, now_ms)

    def record(self, client, key: str, window: float) -> None:
        now_ms = time.time() * 1000
        client.eval(
            _LUA_RECORD, 1, key, now_ms, window * 1000, f"{now_ms:.3f}-{uuid.uuid4().hex[:8]}"
        )


class RateLimiter:
    """
    限流门面：优先用 Redis，拿不到就退回进程内。

    这个"双后端"结构是有意为之：Redis 只是把精度从单机提升到多实例，
    而不是成为单点——Redis 挂了最多让限流退回单机水平，登录接口不会因此不可用。
    """

    def __init__(self) -> None:
        self.local = SlidingWindowLimiter()
        self.redis = RedisSlidingWindow()

    def _client(self) -> Optional[object]:
        return redis_client.get_redis()

    def _dispatch(self, method: str, key: str, *rest):
        """先试 Redis；出错就标记连接失效，并退回进程内实现。"""
        client = self._client()
        if client is not None:
            try:
                # 只有 Redis 侧需要加命名空间前缀（进程内的 key 是本进程私有的）
                return getattr(self.redis, method)(
                    client, redis_client.key("ratelimit", key), *rest
                )
            except Exception as e:
                redis_client.mark_down()
                logger.warning("Redis 限流操作失败，本次退回进程内实现: %s", e)
        return getattr(self.local, method)(key, *rest)

    def check_and_record(self, key: str, limit: int, window: float) -> Tuple[bool, float]:
        return self._dispatch("check_and_record", key, limit, window)

    def peek(self, key: str, limit: int, window: float) -> Tuple[bool, float]:
        return self._dispatch("peek", key, limit, window)

    def record(self, key: str, window: float) -> None:
        self._dispatch("record", key, window)

    def reset(self) -> None:
        """清空两边的计数。仅供测试使用。"""
        self.local.reset()
        client = self._client()
        if client is not None:
            try:
                # scan_iter 而不是 KEYS：KEYS 在大库上会阻塞 Redis
                keys = list(client.scan_iter(match=redis_client.key("ratelimit", "*"), count=500))
                if keys:
                    client.delete(*keys)
            except Exception as e:
                redis_client.mark_down()
                logger.warning("清理 Redis 限流计数失败: %s", e)

    def backend(self) -> str:
        """当前实际生效的后端（redis / in-process），便于排查与测试断言。"""
        return "redis" if self._client() is not None else "in-process"


# 模块级单例
auth_limiter = RateLimiter()


def client_ip(request) -> str:
    """
    取客户端 IP（用于限流 key）。

    默认只用 TCP 对端的地址。只有在 TRUST_PROXY_HEADERS=true 时才读
    `X-Forwarded-For` 的第一段——因为 XFF 是客户端**可以自己伪造**的请求头，
    只有"请求必然经过可信反向代理（且代理会覆写该头）"时才可信。
    否则攻击者随手编一个 XFF 就能绕过按 IP 的限流，比不限流还糟（会造成虚假的安全感）。
    """
    if settings.TRUST_PROXY_HEADERS:
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded:
            return forwarded
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


def allow_auth_attempt(request, username: str = "") -> Tuple[bool, float, str]:
    """
    认证入口的第一道闸。返回 (是否放行, 建议等待秒数, 被拒原因)。

    **必须在 authenticate() / register() 之前调用**，否则昂贵的口令哈希已经算完，限流失去意义。

    被拒原因取值：
      "ip"       —— 该来源的认证尝试过于频繁
      "username" —— 该账号连续失败次数过多（只统计失败）
    """
    if not settings.AUTH_RATE_LIMIT_ENABLED:
        return True, 0.0, ""

    ip_key = f"ip:{client_ip(request)}"
    allowed, retry_after = auth_limiter.check_and_record(
        ip_key, settings.AUTH_RATE_LIMIT_IP_PER_MINUTE, 60.0
    )
    if not allowed:
        logger.warning("[限流] 按来源拒绝认证尝试: ip=%s", client_ip(request))
        return False, retry_after, "ip"

    # 用户名维度用 peek（只看不记）：失败与否要等校验完才知道
    if username:
        username_key = f"user:{username.strip().lower()}"
        allowed, retry_after = auth_limiter.peek(
            username_key,
            settings.AUTH_RATE_LIMIT_USER_FAILURES,
            float(settings.AUTH_RATE_LIMIT_USER_WINDOW_SECONDS),
        )
        if not allowed:
            logger.warning("[限流] 按账号拒绝认证尝试: username=%s", username)
            return False, retry_after, "username"

    return True, 0.0, ""


def record_auth_failure(username: str) -> None:
    """认证失败后调用：只累加到"按用户名"那一维。"""
    if not settings.AUTH_RATE_LIMIT_ENABLED or not username:
        return
    auth_limiter.record(
        f"user:{username.strip().lower()}",
        float(settings.AUTH_RATE_LIMIT_USER_WINDOW_SECONDS),
    )

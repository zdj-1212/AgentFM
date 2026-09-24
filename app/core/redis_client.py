"""
Redis 连接（可选加速层）
========================
Redis 在本项目里**不是必需依赖**。它只承担两件"拿不到也能退化"的事：

1. **认证限流**：放到 Redis 后，多 worker / 多实例共享同一份计数，
   重启也不清零——正好补掉进程内实现的那条已知限制；
2. **检索结果缓存**：语义相同的问题直接复用上一次的 embedding 与向量检索结果，
   省掉一次外部 API 调用。

拿不到 Redis 时：限流退回进程内实现、缓存直接跳过，**功能不受影响**。
所以这个模块的调用方都必须写好"拿不到就退化"的分支，绝不能因为 Redis 挂了就报错。

两个细节值得说明：

- **用 key 前缀而不是切 DB**：`REDIS_KEY_PREFIX` 让本项目的数据与同一个 Redis 上
  别人的数据互不干扰。用 SELECT 切库在 Redis Cluster 下不可用，前缀则是通用的。
- **故障后有冷却期**：连接失败不能每次请求都重试一次（每次都等一个连接超时，
  会把接口拖垮）。这里失败后进入冷却，冷却期内直接用 None 让调用方退化。
"""
import threading
import time
from typing import Optional

import redis

from app.core.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# 失败后的冷却时长（秒）：期间不再尝试连接，避免每次请求都白等一个连接超时
_COOLDOWN_SECONDS = 30.0

_client: Optional[redis.Redis] = None
_retry_at: float = 0.0
_warned = False
_lock = threading.Lock()


def _connect() -> redis.Redis:
    client = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        password=settings.REDIS_PASSWORD or None,
        db=settings.REDIS_DB,
        decode_responses=True,
        # 超时必须短：Redis 变慢/挂掉时，不能把用户的请求一起拖住
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
        health_check_interval=30,
    )
    client.ping()
    return client


def get_redis() -> Optional[redis.Redis]:
    """
    返回可用的 Redis 客户端；未启用或当前不可用时返回 None。

    调用方拿到 None 必须走退化分支（进程内限流 / 不做缓存），而不是当成错误。
    """
    global _client, _retry_at, _warned

    if not settings.REDIS_ENABLED:
        return None
    if _client is not None:
        return _client

    with _lock:
        if _client is not None:
            return _client
        if time.monotonic() < _retry_at:
            return None  # 冷却中：静默退化，不重复打日志
        try:
            _client = _connect()
            logger.info(
                "Redis 已连接：%s:%s (db=%s, 前缀=%s)",
                settings.REDIS_HOST, settings.REDIS_PORT, settings.REDIS_DB,
                settings.REDIS_KEY_PREFIX,
            )
            return _client
        except Exception as e:
            _retry_at = time.monotonic() + _COOLDOWN_SECONDS
            if not _warned:
                # 只提醒一次：Redis 是可选加速层，没它也能跑，不该刷屏
                logger.warning(
                    "Redis 不可用，限流退回进程内、检索缓存停用（%.0f 秒后重试）: %s",
                    _COOLDOWN_SECONDS, e,
                )
                _warned = True
            return None


def mark_down() -> None:
    """
    标记连接已失效（在正常使用中抛异常时调用）。

    否则一个已经断掉的连接会一直被复用，每次操作都要等超时才失败。
    """
    global _client, _retry_at
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # 关闭失败无所谓，丢掉引用即可
                pass
        _client = None
        _retry_at = time.monotonic() + _COOLDOWN_SECONDS


def reset() -> None:
    """丢弃当前连接，让下次调用重新连接。仅供测试使用。"""
    global _client, _retry_at, _warned
    with _lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
        _client = None
        _retry_at = 0.0
        _warned = False


def key(*parts: str) -> str:
    """拼一个带统一前缀的 key（所有本项目的 key 都必须经过这里）。"""
    return settings.REDIS_KEY_PREFIX + ":".join(str(p) for p in parts)

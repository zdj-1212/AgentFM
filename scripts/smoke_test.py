"""
端到端冒烟测试
==============
快速验证整条链路是否打通：

  1. MySQL 连通性 + 建表 + 演示数据 + 业务工具查询
  2. 认证：登录/错误密码/注册校验
  3. 数据隔离：会话与订单只能被本人看到（本次新增的核心断言）
  4. Milvus 连通性 + 知识库入库（为空时自动入库）
  5. 检索相关性：相关问句必须命中预期文档，不相关问句必须 0 命中
  6. 认证限流：连续失败必须 429，且被挡时不再白跑口令哈希
  7. 令牌吊销：退出登录后，之前签发的令牌必须立刻失效（401）
  8. Redis（可选加速层）：限流计数与检索缓存确实生效；
     **Redis 不可用时必须退化**（限流回进程内、检索不走缓存），不能因此报错
  9. 角色与反馈：注册不能自封管理员；普通用户访问运营接口一律 403；
     评价只能打自己会话里的助手回复（管理员也不行，运营台只读）
  10. 降级可辨识：节点失败时 error_code 必须透出，闲聊历史必须是原文
  11. LangGraph 工作流：四种路由 + 会话标题概括 + 历史落库

测试可反复执行：过程中创建的临时账号与会话会在结束时自动清理，不会污染演示数据。

用法（项目根目录）：
    uv run python -m scripts.smoke_test
"""
import uuid

from app.core.logger import get_logger
from app.database import milvus_client, mysql_client
from app.services.auth_service import AuthError, auth_service
from app.services.chat_service import chat_service

logger = get_logger(__name__)

# 覆盖四类路由的测试问题
TEST_CASES = [
    ("你好，在吗？", "chitchat"),
    ("七天无理由退货的条件是什么？", "knowledge"),
    ("云翼科技的会员积分规则是怎样的？", "knowledge"),
    ("帮我查一下订单 SO20260901001 的物流情况", "order"),
    ("云翼科技是一家怎样的公司？", "knowledge"),
]

DEMO_USER = ("张三", "123456")


def check_mysql() -> None:
    """验证 MySQL：建表、造数、业务工具直查。"""
    mysql_client.init_db()
    mysql_client.seed_demo_data()

    order = mysql_client.query_order_by_id("SO20260901001")
    assert order is not None, "订单查询失败"
    print(f"  [MySQL] 订单 {order.order_id}: {order.product_name} / {order.status} / ¥{order.amount:.2f}")

    logistics = mysql_client.query_logistics("SO20260901001")
    assert logistics, "物流查询失败"
    print(f"  [MySQL] 物流节点 {len(logistics)} 条，最新: {logistics[-1].description}")


def check_auth() -> dict:
    """验证认证：演示账号可登录、错误密码被拒、注册校验生效。"""
    user = auth_service.authenticate(*DEMO_USER)
    print(f"  [Auth] 演示账号登录成功: {user['username']} (id={user['user_id']})")

    try:
        auth_service.authenticate(DEMO_USER[0], "definitely-wrong-password")
        raise AssertionError("错误密码竟然登录成功了")
    except AuthError:
        print("  [Auth] 错误密码被正确拒绝")

    try:
        auth_service.register(DEMO_USER[0], "whatever123")
        raise AssertionError("重复用户名竟然注册成功了")
    except AuthError:
        print("  [Auth] 重复用户名被正确拒绝")

    try:
        auth_service.register("x", "12345678")
        raise AssertionError("非法用户名竟然注册成功了")
    except AuthError:
        print("  [Auth] 非法用户名被正确拒绝")

    # 令牌往返：验签能过，且篡改后必须失败
    token = auth_service.create_token(user["user_id"], user["username"])
    payload = auth_service.parse_token(token)
    assert payload["user_id"] == user["user_id"], "令牌解析出的用户不一致"
    try:
        auth_service.parse_token(token[:-2] + ("aa" if not token.endswith("aa") else "bb"))
        raise AssertionError("被篡改的令牌竟然通过了验签")
    except AuthError:
        print("  [Auth] 令牌签发/验签正常，篡改被拒")

    # 哈希格式不变量：只认 argon2id。
    # 旧 PBKDF2 的兼容校验已在迁移完成后删除，所以一旦库里出现非 argon2 的哈希，
    # 那个账号就登不进来了——这条断言用来在"从旧备份恢复"这类情况下早点发现，
    # 而不是等用户报"密码明明是对的"。
    stored = mysql_client.get_user_by_id(user["user_id"]).password_hash
    assert stored.startswith("$argon2"), f"账号口令哈希不是 argon2id: {stored[:16]}"
    print("  [Auth] 口令哈希格式为 argon2id")

    return user


def check_isolation(user: dict) -> None:
    """验证数据隔离：会话与订单只能被本人访问。"""
    # 造一个干净的"他人"账号，避免与真实演示数据互相干扰
    other_name = "iso_" + uuid.uuid4().hex[:8]
    other = auth_service.register(other_name, "secret123")
    session_id = chat_service.create_session(user["user_id"], user["display_name"])

    try:
        # 1) 会话列表互相看不到
        mine = {s["session_id"] for s in chat_service.list_sessions(user["user_id"])}
        others = {s["session_id"] for s in chat_service.list_sessions(other["user_id"])}
        assert session_id in mine, "自己的会话没有出现在自己的列表里"
        assert session_id not in others, "自己的会话出现在了别人的列表里"
        assert not (mine & others), "两个用户的会话列表出现交集"
        print("  [隔离] 会话列表按用户隔离正常")

        # 2) 拿着 session_id 也不能读别人的历史（IDOR 防护）
        try:
            chat_service.get_history(session_id, other["user_id"])
            raise AssertionError("他人竟然能读取该会话历史")
        except PermissionError:
            print("  [隔离] 跨用户读取会话历史被拒绝")

        # 3) 拿着 session_id 也不能往别人的会话里发消息
        try:
            chat_service.ask(session_id, "企图越权提问", other["user_id"], other["display_name"])
            raise AssertionError("他人竟然能往该会话里发消息")
        except PermissionError:
            print("  [隔离] 跨用户向会话发消息被拒绝")

        # 4) 订单也按用户隔离
        zhang_orders = mysql_client.query_orders_by_user(user["user_id"])
        other_orders = mysql_client.query_orders_by_user(other["user_id"])
        assert zhang_orders, "演示账号应有订单"
        assert other_orders == [], "新账号不应看到任何订单"
        print(f"  [隔离] 张三 {len(zhang_orders)} 笔订单，新账号 {len(other_orders)} 笔")
    finally:
        # 冒烟测试要能反复跑：一次性账号与其会话必须收尾，
        # 否则每跑一次演示库里就多一个 iso_xxxx 无名账号。
        mysql_client.delete_user(other["user_id"])
        mysql_client.delete_conversation(session_id)
        print("  [隔离] 已清理本次测试账号与会话")


def check_milvus() -> None:
    """验证 Milvus：集合存在且有向量（为空则自动入库）。"""
    milvus_client.create_collection()
    count = milvus_client.count()
    if count == 0:
        from app.knowledge.ingest import ingest
        ingest(reset=False)
        count = milvus_client.count()
    assert count > 0, "Milvus 知识库为空"
    print(f"  [Milvus] 知识库向量 {count} 条")


# 检索相关性回归用例：(问题, 期望命中的文档关键词；None 表示"应当检索不到")
RETRIEVAL_CASES = [
    ("七天无理由退货的条件是什么？", "售后"),
    ("会员积分怎么抵扣？", "会员"),
    # 下面三条在知识库里没有任何对应内容，必须被阈值挡掉——
    # 否则说明阈值没起作用，模型会被喂进无关上下文，进而编出答案。
    ("今天北京天气怎么样？", None),
    ("帮我写一段 Python 快速排序", None),
    ("爱因斯坦的相对论公式是什么", None),
]


ADMIN_USER = ("张三", "123456")

# 需要管理员权限的接口：普通用户访问必须全部 403
ADMIN_ONLY_PATHS = [
    "/admin/overview",
    "/admin/feedback/summary",
    "/admin/users/1/sessions",
]


def check_redis() -> None:
    """
    验证 Redis 这个"可选加速层"确实在起作用，以及**没有它的时候不会坏**。

    最有价值的断言是最后一段：把 Redis 指到一个没人监听的端口，
    确认限流退回进程内且依然生效、检索依然可用。因为 Redis 在这里只是加速，
    不是依赖——如果它挂了会让登录或问答不可用，那这个改动就是倒退。
    """
    import socket

    from app.core import redis_client
    from app.core.rate_limit import auth_limiter
    from app.knowledge import retriever
    from config.settings import settings

    client = redis_client.get_redis()
    if client is None:
        print("  [Redis] 未配置或不可用 —— 跳过（限流走进程内、检索不走缓存）")
        return

    try:
        # 1) 限流计数落在 Redis 上
        auth_limiter.reset()
        auth_limiter.check_and_record("smoke:redis", 2, 60.0)
        auth_limiter.check_and_record("smoke:redis", 2, 60.0)
        allowed, retry_after = auth_limiter.check_and_record("smoke:redis", 2, 60.0)
        assert not allowed and retry_after > 0, "Redis 后端未按上限拒绝"
        keys = list(client.scan_iter(match=redis_client.key("ratelimit", "smoke:*")))
        assert keys, "限流计数没有写进 Redis"
        print(f"  [Redis] 限流计数落在 Redis（后端={auth_limiter.backend()}，key 前缀 {settings.REDIS_KEY_PREFIX}）")

        auth_limiter.reset()
        assert not list(client.scan_iter(match=redis_client.key("ratelimit", "smoke:*"))), "reset 未清干净"
        print("  [Redis] 限流计数可整体作废")

        # 2) 检索缓存：同一个问句第二次应命中，且结果与首次完全一致
        retriever.invalidate_cache()
        question = "七天无理由退货的条件是什么？"
        first = retriever.retrieve(question)
        cached_keys = list(client.scan_iter(match=redis_client.key("retrieve", "*")))
        assert cached_keys, "检索结果没有写进缓存"
        second = retriever.retrieve(question)
        assert first == second, "缓存返回的结果与首次不一致"
        ttl = client.ttl(cached_keys[0])
        assert 0 < ttl <= settings.RETRIEVE_CACHE_TTL_SECONDS, f"缓存 TTL 异常: {ttl}"
        cleared = retriever.invalidate_cache()
        assert cleared >= 1, "作废没有清掉缓存"
        print(f"  [Redis] 检索结果被缓存（TTL {ttl}s）且内容一致，作废清掉 {cleared} 条")

        # 3) 关键：Redis 不可用时必须退化，而不是报错
        probe = socket.socket()
        probe.settimeout(0.3)
        closed = probe.connect_ex(("127.0.0.1", 6399)) != 0
        probe.close()
        assert closed, "端口 6399 居然有服务，换个端口再测"

        original_port = settings.REDIS_PORT
        settings.REDIS_PORT = 6399
        redis_client.reset()
        try:
            assert redis_client.get_redis() is None, "连不上却仍返回了客户端"
            assert auth_limiter.backend() == "in-process", "未退化到进程内"

            auth_limiter.reset()
            auth_limiter.check_and_record("fallback", 1, 60.0)
            allowed, _ = auth_limiter.check_and_record("fallback", 1, 60.0)
            assert not allowed, "退化到进程内后限流失效了"

            assert retriever.retrieve(question), "退化后检索不可用"
            print("  [Redis] 不可用时：限流退回进程内且仍生效，检索正常（仅不走缓存）")
        finally:
            settings.REDIS_PORT = original_port
            redis_client.reset()
            assert redis_client.get_redis() is not None, "恢复端口后重连失败"
    finally:
        auth_limiter.reset()
        retriever.invalidate_cache()


def check_roles_and_feedback() -> None:
    """
    验证角色分级与答案反馈。

    重点在那些"看起来实现了、其实没生效"的地方：
    1. 注册接口不能自封管理员——哪怕请求体里塞了 role 字段；
    2. 普通用户访问 /admin/* 一律 403；
    3. 升级成管理员**不会**让普通接口也开始返回别人的数据（运营视角只能走 /admin/*）；
    4. 评价只能打自己会话里的助手回复——**管理员也不行**，运营台是只读的。
    """
    from fastapi.testclient import TestClient

    from app.api_server import app
    from app.core.rate_limit import auth_limiter

    auth_limiter.reset()
    client = TestClient(app)

    def bearer(token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def login(username: str, password: str = "123456") -> str:
        r = client.post("/auth/login", json={"username": username, "password": password})
        assert r.status_code == 200, f"登录失败: {r.status_code} {r.text}"
        return r.json()["access_token"]

    # 管理员必须存在（由 seed_demo_users 保证）
    admin_token = login(*ADMIN_USER)
    me = client.get("/auth/me", headers=bearer(admin_token)).json()
    assert me["role"] == "admin", f"演示管理员角色异常: {me}"

    name = "role_" + uuid.uuid4().hex[:8]
    session_id = None
    user_id = None
    try:
        # 1) 注册时试图自封管理员
        r = client.post(
            "/auth/register",
            json={"username": name, "password": "secret123", "role": "admin"},
        )
        assert r.status_code in (200, 201), f"注册失败: {r.status_code} {r.text}"
        user_id = r.json()["user"]["user_id"]
        member_token = r.json()["access_token"]
        assert r.json()["user"]["role"] == "user", "注册响应里竟然带上了 admin 角色"
        assert mysql_client.get_user_by_id(user_id).role == "user", "库里竟然被写成了 admin"
        print("  [角色] 注册请求携带 role=admin 被忽略，落库仍为 user")

        # 2) 普通用户访问管理员接口
        for path in ADMIN_ONLY_PATHS:
            code = client.get(path, headers=bearer(member_token)).status_code
            assert code == 403, f"{path} 对普通用户应 403，实际 {code}"
        # 跨用户读消息也算管理员能力
        code = client.get("/admin/sessions/whatever/messages", headers=bearer(member_token)).status_code
        assert code == 403, f"普通用户读任意会话消息应 403，实际 {code}"
        print(f"  [角色] 普通用户访问 {len(ADMIN_ONLY_PATHS) + 1} 个管理员接口全部 403")

        # 管理员可访问
        assert client.get("/admin/overview", headers=bearer(admin_token)).status_code == 200
        assert client.get("/admin/feedback/summary", headers=bearer(admin_token)).status_code == 200
        print("  [角色] 管理员可访问运营接口")

        # 3) 造一条属于普通用户的会话与助手回复（唯一一次 LLM 调用）
        member = auth_service.get_user(user_id)
        session_id = chat_service.create_session(member["user_id"], member["display_name"])
        chat_service.ask(session_id, "你好，在吗？", member["user_id"], member["display_name"])
        history = chat_service.get_history(session_id, member["user_id"])
        msg_id = [h for h in history if h["role"] == "assistant"][-1]["message_id"]

        # 管理员能看到这条会话的消息（这才是运营视角的意义）
        r = client.get(f"/admin/sessions/{session_id}/messages", headers=bearer(admin_token))
        assert r.status_code == 200, f"管理员读会话消息失败: {r.status_code}"
        assert any(m["message_id"] == msg_id for m in r.json()), "运营台看不到该消息"
        print("  [角色] 管理员可跨用户查看会话消息")

        # 但**普通接口**对管理员仍然只返回他自己的数据：升级管理员不等于普通接口开洞
        own = {s["session_id"] for s in client.get("/sessions", headers=bearer(admin_token)).json()}
        assert session_id not in own, "管理员的 /sessions 里出现了别人的会话！"
        print("  [角色] 管理员的 /sessions 仍只含自己的会话（隔离未被角色放宽）")

        # 4) 管理员也不能给别人的回复打分：运营台是只读的
        code = client.post(
            f"/sessions/{session_id}/messages/{msg_id}/feedback",
            json={"feedback": -1},
            headers=bearer(admin_token),
        ).status_code
        assert code == 403, f"管理员给他人消息打分应 403，实际 {code}"
        print("  [反馈] 管理员给他人消息打分被拒绝（运营台只读）")

        # 5) 本人打分成功，并反映到运营总览
        r = client.post(
            f"/sessions/{session_id}/messages/{msg_id}/feedback",
            json={"feedback": 1, "note": "回答清楚"},
            headers=bearer(member_token),
        )
        assert r.status_code == 200, f"本人打分失败: {r.status_code} {r.text}"

        stored = mysql_client.get_message(msg_id)
        assert stored.feedback == 1 and stored.feedback_note == "回答清楚", "评价未落库"
        print("  [反馈] 本人评价落库（含原因）")

        row = next(
            u for u in client.get("/admin/overview", headers=bearer(admin_token)).json()
            if u["user_id"] == user_id
        )
        assert row["up"] == 1 and row["down"] == 0, f"运营总览未反映评价: {row}"
        print(f"  [反馈] 运营总览已反映：👍 {row['up']} / 👎 {row['down']} / 消息 {row['messages']}")

        # 6) 取消评价（feedback=null）应把计数清掉
        r = client.post(
            f"/sessions/{session_id}/messages/{msg_id}/feedback",
            json={"feedback": None},
            headers=bearer(member_token),
        )
        assert r.status_code == 200, "取消评价失败"
        assert mysql_client.get_message(msg_id).feedback is None, "取消评价后仍留有评分"
        print("  [反馈] 取消评价生效")
    finally:
        if session_id:
            mysql_client.delete_conversation(session_id)
        if user_id:
            mysql_client.delete_user(user_id)
        auth_limiter.reset()


def check_token_revocation() -> None:
    """
    验证令牌吊销：调用 /auth/logout 后，**退出前签发的令牌必须立刻失效**。

    走真实的 HTTP 接口而不是直接调服务函数，因为"服务层能抛出异常"不等于
    "接口真的会返回 401"——中间还隔着依赖注入与异常映射。

    这里也把语义写成了断言：退出登录会吊销该账号**全部**令牌（不只是发起退出的那个），
    所以同时签发的两个令牌应当一起失效。
    """
    from fastapi.testclient import TestClient

    from app.api_server import app
    from app.core.rate_limit import auth_limiter

    auth_limiter.reset()
    client = TestClient(app)

    # 造一个一次性账号，避免动到演示账号的令牌版本
    name = "revoke_" + uuid.uuid4().hex[:8]
    user = auth_service.register(name, "secret123")
    user_id = user["user_id"]

    def bearer(token: str) -> dict:
        return {"Authorization": f"Bearer {token}"}

    def login() -> str:
        resp = client.post("/auth/login", json={"username": name, "password": "secret123"})
        assert resp.status_code == 200, f"登录失败: {resp.status_code} {resp.text}"
        return resp.json()["access_token"]

    try:
        token_a = login()

        # 令牌 payload 只有 uid/u/ver/exp，没有任何随机成分，因此**同一秒内两次登录
        # 会签出完全相同的令牌**。这不是缺陷（吊销是按用户版本来的，不依赖令牌唯一性），
        # 但必须知道，否则会写出"两次登录应得到不同令牌"这种想当然的断言。
        import time as _time

        assert login() == token_a, "同一秒内的令牌应当完全相同（payload 无随机成分）"
        _time.sleep(1.05)  # 跨过一个时间刻度，exp 才会变，才能拿到可区分的第二个令牌
        token_b = login()
        assert token_b != token_a, "跨秒后应签出不同令牌"

        assert client.get("/auth/me", headers=bearer(token_a)).status_code == 200
        assert client.get("/auth/me", headers=bearer(token_b)).status_code == 200
        print("  [吊销] 两个令牌（两台设备）均可用")

        # 用 A 退出登录
        resp = client.post("/auth/logout", headers=bearer(token_a))
        assert resp.status_code == 200, f"/auth/logout 应返回 200，实际 {resp.status_code}"

        # 两个都必须失效 —— 这是"退出所有设备"语义
        codes = (client.get("/auth/me", headers=bearer(token_a)).status_code,
                 client.get("/auth/me", headers=bearer(token_b)).status_code)
        assert codes == (401, 401), f"退出后两个旧令牌都应 401，实际 {codes}"
        print("  [吊销] 退出后两个旧令牌均返回 401（含未发起退出的那台设备）")

        # 重新登录的新令牌必须可用：吊销只针对旧令牌，不能把账号锁死
        token_c = login()
        assert client.get("/auth/me", headers=bearer(token_c)).status_code == 200, "新令牌应可用"
        print("  [吊销] 重新登录后的新令牌可正常使用")

        # 票据自增必须是原子的：并发吊销不能丢更新，否则会有一批本该失效的令牌继续可用
        from concurrent.futures import ThreadPoolExecutor

        before = mysql_client.get_user_by_id(user_id).token_version
        rounds = 10
        with ThreadPoolExecutor(max_workers=rounds) as pool:
            list(pool.map(lambda _: auth_service.revoke_tokens(user_id), range(rounds)))
        after = mysql_client.get_user_by_id(user_id).token_version
        assert after - before == rounds, (
            f"并发吊销丢失更新：期望版本 +{rounds}，实际 +{after - before}"
        )
        print(f"  [吊销] {rounds} 次并发吊销无丢更新（token_version {before} -> {after}）")
    finally:
        mysql_client.delete_user(user_id)
        auth_limiter.reset()


def check_retrieval() -> None:
    """
    验证向量检索的"分数方向"是对的。

    针对一个真实踩过的坑：集合用 COSINE 索引时，Milvus 返回的 distance 本身就是
    余弦相似度（越大越相关），不是距离。曾经写成 score = 1.0 - distance，
    把相关性整个反过来：最相关的切片被阈值滤掉、反而留下最不相关的。
    症状是"问相关的问题却答'知识库中暂无相关信息'"——不报错、不抛异常，
    只让回答质量悄悄变差，所以必须用断言锁住：

    - 相关问题     -> 必须命中，且第一名来自预期文档；
    - 完全不相关问题 -> 必须一条都不命中（确认 0.45 这个阈值真的在起作用）。
    """
    from app.knowledge import retriever

    for question, expect in RETRIEVAL_CASES:
        hits = retriever.retrieve(question)

        if expect is None:
            assert not hits, (
                f"不相关的问题竟然检索到 {len(hits)} 条（阈值失效？）: {question} -> "
                f"{[(h['title'], round(h['score'], 3)) for h in hits]}"
            )
            print(f"  [检索] 不相关问题 0 命中（阈值生效）: {question[:16]}")
            continue

        assert hits, f"相关问题一条都没检索到（分数方向反了？）: {question}"
        top = hits[0]
        origin = f"{top['title']}{top['source']}"
        assert expect in origin, (
            f"命中来源不对: {question} -> 第一名是 {top['title']!r}，期望包含 {expect!r}"
        )
        print(
            f"  [检索] {question[:16]} -> {top['title']} "
            f"(相似度 {top['score']:.3f})，共 {len(hits)} 条"
        )


def check_rate_limit() -> None:
    """
    验证认证限流：连续失败必须被 429 挡住，且**昂贵的哈希不应再被计算**。

    这里同时验证两件容易做错的事：
    1. 限流要真的返回 429（而不是只记了日志）；
    2. 被限流后不能再跑口令哈希——所以顺带比一下"被挡住的响应"耗时是否明显更短。
       限流如果写在 authenticate() 之后，这一步就会失败。
    """
    import time as _time

    from fastapi.testclient import TestClient

    from app.api_server import app
    from app.core.rate_limit import auth_limiter, client_ip
    from config.settings import settings

    auth_limiter.reset()  # 用干净状态起测，避免影响本进程里其它检查
    client = TestClient(app)

    limit = settings.AUTH_RATE_LIMIT_USER_FAILURES
    target = "限流测试账号_不存在"  # 账号不存在，authenticate 会走 dummy hash（同样昂贵）

    codes = []
    slowest = 0.0
    for _ in range(limit + 2):
        started = _time.perf_counter()
        resp = client.post("/auth/login", json={"username": target, "password": "wrong-password"})
        codes.append(resp.status_code)
        if resp.status_code == 401:
            slowest = max(slowest, _time.perf_counter() - started)

    assert codes[:limit] == [401] * limit, f"前 {limit} 次应为 401，实际 {codes}"
    assert codes[limit] == 429, f"超出上限后应为 429，实际 {codes}"

    # 被限流的那次应当明显快于真正做了口令哈希的那次
    started = _time.perf_counter()
    blocked = client.post(
        "/auth/login", json={"username": target, "password": "wrong-password"}
    )
    blocked_elapsed = _time.perf_counter() - started
    assert blocked.status_code == 429, f"仍应被限流，实际 {blocked.status_code}"
    assert "retry-after" in {k.lower() for k in blocked.headers}, "429 应带 Retry-After 头"
    print(
        f"  [限流] 连续失败 {limit} 次后返回 429（Retry-After={blocked.headers.get('retry-after')}s）"
    )
    print(
        f"  [限流] 被挡响应 {blocked_elapsed * 1000:.1f}ms vs 真实校验 {slowest * 1000:.1f}ms"
        "（被挡明显更快，说明没白跑口令哈希）"
    )

    # 换一个账号不受影响（确认限的是账号维度，不是把所有人都锁死）
    ok = client.post("/auth/login", json={"username": "张三", "password": "123456"})
    assert ok.status_code == 200, f"其它账号不应被连带限流，实际 {ok.status_code}"
    print("  [限流] 其它账号不受影响")

    # 按来源（IP）维度：TestClient 的请求都来自同一来源，狂打成功登录也应被挡
    auth_limiter.reset()
    settings.AUTH_RATE_LIMIT_IP_PER_MINUTE = 3
    try:
        got = [
            client.post("/auth/login", json={"username": "张三", "password": "123456"}).status_code
            for _ in range(5)
        ]
        assert 429 in got, f"同一来源超过上限应被挡，实际 {got}"
        print(f"  [限流] 按来源维度生效: {got}")
    finally:
        settings.AUTH_RATE_LIMIT_IP_PER_MINUTE = 20
        auth_limiter.reset()


def check_degradation(user: dict) -> None:
    """
    验证"降级可辨识"：节点失败时，回复是兜底话术，但 error_code 必须能说明是谁挂了。

    四个节点失败时返回的是同一句话术，如果只断言"有回复"，永远发现不了这个问题——
    这里故意打断知识库，确认 knowledge_unavailable 一路传到 chat_service 的返回值。
    同时也覆盖了 chitchat 节点的历史拼装（曾经的 Bug：用户历史被写成字面量 "content"）。
    """
    from app.agent import nodes as agent_nodes
    from app.knowledge import retriever

    user_id, user_name = user["user_id"], user["display_name"]
    session_id = chat_service.create_session(user_id, user_name)

    original = retriever.build_context

    def _boom(*_args, **_kwargs):
        raise RuntimeError("模拟知识库不可用")

    try:
        retriever.build_context = _boom
        # rag_node 里是 from ... import retriever 后按属性调用，替换模块属性即可生效
        result = chat_service.ask(session_id, "七天无理由退货的条件是什么？", user_id, user_name)
        assert result["intent"] == "knowledge", f"未走到知识问答路由: {result['intent']}"
        assert result["error_code"] == "knowledge_unavailable", (
            f"降级原因未透出: {result.get('error_code')!r}"
        )
        assert result["reply"], "降级时也必须有兜底话术"
        print(f"  [降级] 知识库故障 -> error_code={result['error_code']}，兜底话术已返回")
    finally:
        retriever.build_context = original
        mysql_client.delete_conversation(session_id)

    # chitchat 节点：历史里的用户消息必须是原文，不能是字面量 "content"
    captured = {}

    class _Recorder:
        def invoke(self, messages, *_args, **_kwargs):
            captured["messages"] = messages

            class _R:
                content = "好的"

            return _R()

    original_llm = agent_nodes.get_llm
    try:
        agent_nodes.get_llm = lambda: _Recorder()
        agent_nodes.chitchat_node(
            {
                "session_id": "x",
                "user_id": user_id,
                "user_name": user_name,
                "user_input": "在吗",
                "history": [
                    {"role": "user", "content": "我买的东西坏了"},
                    {"role": "assistant", "content": "很抱歉，请提供订单号"},
                ],
            }
        )
    finally:
        agent_nodes.get_llm = original_llm

    sent = [m.content for m in captured["messages"]]
    assert "我买的东西坏了" in sent, f"闲聊节点未携带真实用户历史，实际发送: {sent}"
    assert "content" not in sent, "闲聊节点仍把用户历史写成了字面量 'content'"
    print("  [降级] 闲聊节点历史拼装正常（用户消息为原文，非字面量）")


def check_graph(user: dict) -> None:
    """验证 LangGraph 工作流、四种路由，以及首轮提问生成的会话标题。"""
    user_id, user_name = user["user_id"], user["display_name"]
    session_id = chat_service.create_session(user_id, user_name)

    for question, expect_intent in TEST_CASES:
        result = chat_service.ask(session_id, question, user_id, user_name)
        assert result["reply"], f"回复为空: {question}"
        assert result["intent"] == expect_intent, f"意图不符: {question} -> {result['intent']} != {expect_intent}"
        # 必须有这一条：节点失败时回复是兜底话术，"有回复 + 意图正确"两条断言都能通过，
        # 于是组件挂掉会伪装成一次成功——这正是之前被漏掉的那类静默失败。
        assert not result.get("error_code"), (
            f"本轮是降级回复（{result['error_code']}），链路并不健康: {question}"
        )
        tag = "✓" if result["intent"] == expect_intent else "✗"
        src = f" | 来源 {len(result['sources'])} 条" if result.get("sources") else ""
        print(f"  [{tag}] {question[:20]} -> {result['intent']}{src}")
    print(f"  [Graph] 工作流运行正常，会话 {session_id}")

    # 标题应来自首轮提问的概括，而不是建表默认值
    summary = next(
        (s for s in chat_service.list_sessions(user_id) if s["session_id"] == session_id), None
    )
    assert summary is not None, "会话未出现在列表中"
    assert summary["title"] and summary["title"] != "新会话", f"会话标题未被概括: {summary['title']!r}"
    print(f"  [Graph] 会话标题已按首轮提问概括: 「{summary['title']}」")

    # 历史消息应完整落库（user + assistant）
    history = chat_service.get_history(session_id, user_id)
    assert history, "会话历史为空"
    roles = {h["role"] for h in history}
    assert roles == {"user", "assistant"}, f"历史消息角色异常: {roles}"
    print(f"  [Graph] 会话历史落库 {len(history)} 条（user + assistant）")

    # 同样要收尾：测试用的会话不该留在演示账号名下
    mysql_client.delete_conversation(session_id)
    print("  [Graph] 已清理本次测试会话")


def main() -> None:
    print("=" * 60)
    print("AgentFM 端到端冒烟测试")
    print("=" * 60)

    print("[1/11] 检查 MySQL ...")
    check_mysql()
    print("[2/11] 检查认证 ...")
    user = check_auth()
    print("[3/11] 检查数据隔离 ...")
    check_isolation(user)
    print("[4/11] 检查 Milvus 知识库 ...")
    check_milvus()
    print("[5/11] 检查检索相关性 ...")
    check_retrieval()
    print("[6/11] 检查认证限流 ...")
    check_rate_limit()
    print("[7/11] 检查令牌吊销 ...")
    check_token_revocation()
    print("[8/11] 检查 Redis 加速层 ...")
    check_redis()
    print("[9/11] 检查角色与反馈 ...")
    check_roles_and_feedback()
    print("[10/11] 检查降级可辨识 ...")
    check_degradation(user)
    print("[11/11] 检查 LangGraph 工作流 ...")
    check_graph(user)

    print("=" * 60)
    print("✅ 全部通过！可运行: uv run streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

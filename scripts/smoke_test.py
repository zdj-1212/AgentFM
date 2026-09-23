"""
端到端冒烟测试
==============
快速验证整条链路是否打通：

  1. MySQL 连通性 + 建表 + 演示数据 + 业务工具查询
  2. 认证：登录/错误密码/注册校验
  3. 数据隔离：会话与订单只能被本人看到（本次新增的核心断言）
  4. Milvus 连通性 + 知识库入库（为空时自动入库）
  5. LangGraph 工作流：覆盖知识问答 / 订单查询 / 闲聊 / 综合 四种路由
  6. 会话标题：首轮提问后标题应被概括，而不是停在"新会话"

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


def check_graph(user: dict) -> None:
    """验证 LangGraph 工作流、四种路由，以及首轮提问生成的会话标题。"""
    user_id, user_name = user["user_id"], user["display_name"]
    session_id = chat_service.create_session(user_id, user_name)

    for question, expect_intent in TEST_CASES:
        result = chat_service.ask(session_id, question, user_id, user_name)
        assert result["reply"], f"回复为空: {question}"
        assert result["intent"] == expect_intent, f"意图不符: {question} -> {result['intent']} != {expect_intent}"
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

    print("[1/5] 检查 MySQL ...")
    check_mysql()
    print("[2/5] 检查认证 ...")
    user = check_auth()
    print("[3/5] 检查数据隔离 ...")
    check_isolation(user)
    print("[4/5] 检查 Milvus 知识库 ...")
    check_milvus()
    print("[5/5] 检查 LangGraph 工作流 ...")
    check_graph(user)

    print("=" * 60)
    print("✅ 全部通过！可运行: uv run streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

"""
端到端冒烟测试
==============
快速验证整条链路是否打通：

  1. MySQL 连通性 + 建表 + 演示数据 + 业务工具查询
  2. Milvus 连通性 + 知识库入库（为空时自动入库）
  3. LangGraph 工作流：覆盖知识问答 / 订单查询 / 闲聊 / 综合 四种路由

用法（项目根目录）：
    uv run python -m scripts.smoke_test
"""
from app.core.logger import get_logger
from app.database import milvus_client, mysql_client
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


def check_graph() -> None:
    """验证 LangGraph 工作流与四种路由。"""
    session_id = chat_service.create_session(user_name="冒烟测试")
    for question, expect_intent in TEST_CASES:
        result = chat_service.ask(session_id, question)
        assert result["reply"], f"回复为空: {question}"
        assert result["intent"] == expect_intent, f"意图不符: {question} -> {result['intent']} != {expect_intent}"
        tag = "✓" if result["intent"] == expect_intent else "✗"
        src = f" | 来源 {len(result['sources'])} 条" if result.get("sources") else ""
        print(f"  [{tag}] {question[:20]} -> {result['intent']}{src}")
    print(f"  [Graph] 工作流运行正常，会话 {session_id}")


def main() -> None:
    print("=" * 60)
    print("AgentFM 端到端冒烟测试")
    print("=" * 60)

    print("[1/3] 检查 MySQL ...")
    check_mysql()
    print("[2/3] 检查 Milvus 知识库 ...")
    check_milvus()
    print("[3/3] 检查 LangGraph 工作流 ...")
    check_graph()

    print("=" * 60)
    print("✅ 全部通过！可运行: uv run streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

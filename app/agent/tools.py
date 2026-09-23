"""
Agent 业务工具集
=================
LangChain @tool 装饰器把普通 Python 函数变成"大模型可调用的工具"：
- 函数的 docstring 就是给 LLM 看的工具说明（决定 LLM 何时调用）；
- 参数名与类型标注会被 LLM 用来生成调用参数。

本模块提供两类工具：
1. 业务查询工具（查 MySQL 订单/物流）—— 由 ORDER/GENERAL Agent 使用；
2. 知识库检索工具（查 Milvus）—— 由 GENERAL Agent 使用。

关于"当前用户是谁"（数据隔离的关键）：
订单类工具通过 ToolRuntime 注入的 UserContext 拿到登录用户 id，而**不把用户名作为工具参数**。
若把用户名做成参数，模型会从用户问句里猜（用户说"查张三的订单"就真去查张三的），
既不可靠也能被越权利用。标注了 ToolRuntime/InjectedState 的参数不会出现在
工具的 args schema 里，模型看不见、也伪造不了。
"""
from typing import Any,Optional

from langchain_core.tools import tool
from langchain.tools import ToolRuntime

from app.agent.state import UserContext
from app.database import mysql_client
from app.core.logger import get_logger

logger=get_logger(__name__)

@tool
def query_order_status(order_id: str, runtime: ToolRuntime[UserContext, Any]) -> str:
    """
        查询订单的当前状态、金额及最新物流轨迹。
        当用户提供订单号（形如 SO20260901001）询问"订单到哪了/发货没有/物流情况"时调用。
        参数 order_id: 订单号字符串。
    """
    order=mysql_client.query_order_by_id(order_id)

    # 归属校验：只能查自己的订单。订单号是可枚举的（SO2026...），
    # 不校验的话，用户随便报一个别人的订单号就能看到对方的订单与收货城市。
    user_id = _current_user_id(runtime)
    if order is None or (user_id is not None and order.user_id != user_id):
        return f"未查询到订单 {order_id}，请核对订单号是否正确。"

    lines=[
        f"订单号：{order.order_id}",
        f"商品：{order.product_name}",
        f"金额：{order.amount}",
        f"状态：{order.status}",
    ]
    #附带最新物流信息
    logistics=mysql_client.query_logistics(order_id)
    if logistics:
        latest=logistics[-1]
        lines.append(f"最新物流：{latest.node_time} {latest.description} ({latest.location})")
    else:
        lines.append("暂无物流记录（可能尚未发货）")
    return "\n".join(lines)

@tool
def query_orders_by_user(runtime: ToolRuntime[UserContext, Any]) -> str:
    """
        查询【当前登录用户本人】名下的全部订单列表，无需任何参数。
        当用户询问"我的订单/我买过什么/查一下我的订单"但未提供具体订单号时调用。
        不要向用户索要用户名：本工具自动按登录账号查询。
    """
    user_id = _current_user_id(runtime)
    if user_id is None:
        return "当前会话未登录，无法查询订单，请先登录。"

    orders = mysql_client.query_orders_by_user(user_id)
    if not orders:
        return "当前账号名下暂无订单记录。"

    lines=[f"您名下共有 {len(orders)} 笔订单："]
    for o in orders:
        lines.append(f"- {o.order_id} | {o.product_name} | ¥{o.amount} | {o.status}")
    return "\n".join(lines)

def _current_user_id(runtime: ToolRuntime[UserContext, Any]) -> Optional[int]:
    """从运行时上下文取出当前登录用户 id；取不到（如未登录调用）返回 None。"""
    ctx = getattr(runtime, "context", None)
    user_id = getattr(ctx, "user_id", None)
    return user_id or None

@tool
def search_knowledge(query: str)->str:
    """
        检索企业知识库，获取官方政策内容（售后/退换货/保修/物流政策/运费/会员/积分/优惠券/发票等）。
        当用户询问企业内部政策、规则、流程类问题时调用。
        参数 query: 检索关键词或完整问题。
    """
    # 延迟导入：Milvus 连接与 embedding 初始化较重，且离线环境需要在调用时才暴露问题
    from app.knowledge import retriever

    try:
        context, _hits = retriever.build_context(query)
    except Exception as e:
        logger.exception("知识库检索失败")
        return f"知识库检索暂时不可用，请稍后再试。（{e}）"

    if not context:
        return "知识库中未检索到与该问题相关的内容。"
    # 返回带 [1][2] 引用编号的片段，与 RAG 节点给模型的上下文格式一致
    return context

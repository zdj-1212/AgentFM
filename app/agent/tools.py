"""
Agent 业务工具集
=================
LangChain @tool 装饰器把普通 Python 函数变成"大模型可调用的工具"：
- 函数的 docstring 就是给 LLM 看的工具说明（决定 LLM 何时调用）；
- 参数名与类型标注会被 LLM 用来生成调用参数。

本模块提供两类工具：
1. 业务查询工具（查 MySQL 订单/物流）—— 由 ORDER/GENERAL Agent 使用；
2. 知识库检索工具（查 Milvus）—— 由 GENERAL Agent 使用。
"""
from langchain_core.tools import tool

from app.database import mysql_client
from app.core.logger import get_logger

logger=get_logger(__name__)

@tool
def query_order_status(order_id: str) ->str:
    """
        查询订单的当前状态、金额及最新物流轨迹。
        当用户提供订单号（形如 SO20260901001）询问"订单到哪了/发货没有/物流情况"时调用。
        参数 order_id: 订单号字符串。
    """
    order=mysql_client.query_order_by_id(order_id)
    if order is None:
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
def query_orders_by_user(user_name: str) -> str:
    """
        按用户名查询其名下的全部订单列表。
        当用户询问"我的订单/我买过什么"但未提供具体订单号时调用。
        参数 user_name: 用户名。
    """
    orders = mysql_client.query_orders_by_user(user_name)
    if not orders:
        return f"未查询到用户「{user_name}」的订单记录。"

    lines=[f"用户「{user_name}」共有 {len(orders)} 笔订单："]
    for o in orders:
        lines.append(f"- {o.order_id} | {o.product_name} | ¥{o.amount} | {o.status}")
    return "\n".join(lines)

@tool
def search_knowledge(query: str)->str:
    """
        检索企业知识库，获取官方政策内容（售后/退换货/保修/物流政策/运费/会员/积分/优惠券/发票等）。
        当用户询问企业内部政策、规则、流程类问题时调用。
        参数 query: 检索关键词或完整问题。
    """

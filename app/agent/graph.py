"""
LangGraph 图编排
=================
把节点拼装成完整的"状态机"工作流，是本项目的核心架构：

    START -> [classify] --路由--> [rag / order / chitchat / general] -> [finalize] -> END

架构价值：
1. 意图路由（Intent Routing）：把不同复杂度的问题分发给最合适的处理路径，成本与效果兼顾；
2. 模块化节点（Node）：每个节点职责单一、可独立测试、可替换；
3. 可控性：相比自由调用 LLM 的"单 Agent"，多节点图编排让流程可观测、可插桩、可灰度。
"""
from functools import lru_cache

from langchain_core.messages import HumanMessage
from langgraph.graph import END,START,StateGraph

from app.agent import nodes
from app.agent.state import AgentState
from app.core.logger import get_logger

logger=get_logger(__name__)

def build_graph():
    """构建并编译 LangGraph 状态图。"""
    graph=StateGraph(AgentState)

    # ---- 1. 注册节点 ----
    graph.add_node("classify", nodes.classify_node)
    graph.add_node("rag", nodes.rag_node)
    graph.add_node("order", nodes.order_node)
    graph.add_node("chitchat", nodes.chitchat_node)
    graph.add_node("general", nodes.general_node)
    graph.add_node("finalize", nodes.finalize_node)

    # ---- 2. 定义边 ----
    # 入口 -> 意图分类
    graph.add_edge(START,"classify")

    # 条件路由：根据意图决定下一步去哪个专业节点
    graph.add_conditional_edges(
        "classify",
        nodes.route_node,
        {
            "rag": "rag",
            "order": "order",
            "chitchat": "chitchat",
            "general": "general",
        },
    )
    for node_name in ("rag", "order", "chitchat", "general"):
        graph.add_edge(node_name,"finalize")

    graph.add_edge("finalize",END)

    # ---- 3. 编译 ----
    compiled=graph.compile()
    logger.info("LangGraph 工作流编译完成")
    return compiled

@lru_cache(maxsize=1)
def get_graph():
    """全局唯一编译后的图（进程内单例，复用连接与 Agent 缓存）。"""
    return build_graph()

if __name__ == '__main__':
    agent=build_graph()
    res=agent.invoke(
        {
            "user_input":"你是谁"
        }
    )
    print(res)
"""
LangGraph 图节点
=================
每个节点是一个纯函数：输入整个 State，返回对 State 的部分更新（dict）。
图引擎按定义好的边依次调用节点，形成可控的智能体工作流。

节点清单：
- classify  -> 意图分类（LLM 结构化输出，失败/离线时回退规则分类）
- route     -> 根据意图返回下一节点（条件路由）
- rag       -> 知识问答：检索 Milvus + LLM 生成（带引用溯源）
- order     -> 订单/物流查询：ReAct 工具调用 Agent（查 MySQL）
- chitchat  -> 闲聊：直接 LLM
- general   -> 综合兜底：ReAct Agent（知识检索 + 业务工具兼备）
- finalize  -> 落库 + 组装最终输出
"""
from functools import lru_cache
from typing import Dict,List

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain.agents import create_agent
from pydantic import BaseModel,Field

from config.settings import settings
from app.agent.prompts import (
    CLASSIFY_PROMPT,
    FALLBACK_RESPONSE,
    SYSTEM_CHITCHAT,
    SYSTEM_GENERAL_AGENT,
    SYSTEM_ORDER_AGENT,
    SYSTEM_RAG_PROMPT
)
from app.agent.state import AgentState,Intent
from app.agent.tools import query_order_status,query_orders_by_user,search_knowledge
from app.core.llm import get_llm
from app.core.logger import get_logger
from app.database import mysql_client
from app.knowledge import retriever

logger = get_logger(__name__)

# 意图分类

class IntentResult(BaseModel):
    """LLM 结构化输出：意图 + 判断理由。"""
    intent:Intent = Field(description="分类结果")
    reason:str = Field(default="",description="判断理由")

def _classify_by_rule(text:str)->Intent:
    """
    规则分类器：无 LLM / LLM 失败时的兜底方案。
    企业实践中常用"规则预筛 + LLM 精分"的双层结构，兼顾成本与准确性。
    """
    t = text.lower()
    # 订单/物流查询（优先匹配，避免被"售后"等词误分）
    if any(k in t for k in ["订单", "物流", "快递", "运单", "发货", "派送", "签收", "包裹", "到哪了"]):
        return Intent.ORDER
    # 政策/知识类
    if any(k in t for k in ["售后", "退货", "换货", "保修", "维修", "积分", "会员", "优惠券",
                            "发票", "运费", "政策", "规则", "多久", "怎么办", "如何", "流程", "条件"]):
        return Intent.KNOWLEDGE
    # 寒暄
    if any(k in t for k in ["你好", "您好", "嗨", "hello", "hi", "在吗", "谢谢", "再见",
                            "你是谁", "早上好", "晚上好"]):
        return Intent.CHITCHAT
    return Intent.GENERAL

def classify_node(state: AgentState)->Dict:
    """意图分类节点：优先 LLM 结构化分类，异常/离线时回退规则分类。"""
    user_input=state["user_input"]
    try:
        llm=get_llm()
        structured = llm.with_structured_output(IntentResult)
        result: IntentResult = structured.invoke(
            [SystemMessage(content=CLASSIFY_PROMPT),HumanMessage(content=user_input)]
        )
        intent=result.intent
        logger.info("[classify] (llm) %s -> %s (%s)", user_input[:20], intent.value, result.reason)
        return {"intent":intent}
    except Exception as e:  # LLM 分类失败 -> 规则兜底，保证流程不中断
        logger.warning("LLM 意图分类失败，回退规则分类: %s", e)
        return {"intent":_classify_by_rule(user_input)}

def route_node(state:AgentState)->str:
    """条件路由节点：返回下一节点的名字（LangGraph 据此走对应边）。"""
    intent = state.get("intent",Intent.GENERAL)
    mapping={
        Intent.KNOWLEDGE:"rag",
        Intent.ORDER:"order",
        Intent.CHITCHAT:"chitchat",
        Intent.GENERAL:"general"
    }
    next_node=mapping.get(intent,"general")
    logger.info("[route] %s -> %s", intent.value, next_node)
    return next_node

# ============================================================
# 历史消息格式化（供提示词拼装）
# ============================================================

def _format_history(history: List[Dict]) -> str:
    """把 [{role, content}] 历史转成可读文本。"""
    if not history:
        return "（无）"
    lines = []
    for h in history[-6:]:
        who = "用户" if h["role"] == "user" else "客服"
        lines.append(f"{who}: {h['content']}")
    return "\n".join(lines)

def _build_history_messages(history: List[Dict]) -> List:
    """把历史转成 LangChain 消息列表（供 ReAct Agent 使用）。"""
    msgs = []
    for h in history[-6:]:
        if h["role"] == "user":
            msgs.append(HumanMessage(content=h["content"]))
        else:
            msgs.append(AIMessage(content=h["content"]))
    return msgs

# ============================================================
# 节点：知识问答（RAG）
# ============================================================

def rag_node(state: AgentState) -> Dict:
    """RAG 节点：向量检索知识库 + LLM 基于检索结果生成回答，并返回引用来源。"""
    try:
        question = state["user_input"]
        # 1) 检索知识库，拼装带引用的上下文
        context, hits = retriever.build_context(question)

        # 2) 组装提示词并生成
        prompt = SYSTEM_RAG_PROMPT.format(
            context=context if context else "（知识库中暂无相关内容）",
            history=_format_history(state.get("history", [])),
            question=question,
        )
        llm = get_llm()
        answer = llm.invoke([HumanMessage(content=prompt)]).content

        # 3) 整理引用来源（供前端展示）
        sources = [
            {"title": h["title"], "source": h["source"], "score": h["score"]}
            for h in hits
        ]
        return {"response": answer, "hits": hits, "sources": sources}
    except Exception as e:
        logger.exception("RAG 节点异常")
        return {"response": FALLBACK_RESPONSE, "hits": [], "sources": [], "error": str(e)}

# ============================================================
# 节点：订单/物流查询（ReAct Agent + MySQL 工具）
# ============================================================

@lru_cache(maxsize=1)
def _order_agent():
    """订单查询 ReAct Agent（懒加载 + 单例缓存）。

    注：LangGraph 1.x 中工具循环上限不再作为 create_react_agent 参数，
    而是通过 invoke 时的 config={"recursion_limit": N} 控制（见 order_node）。
    """
    return create_agent(
        model=get_llm(),
        tools=[query_order_status, query_orders_by_user],
        system_prompt=SYSTEM_ORDER_AGENT,
    )

def order_node(state:AgentState)->Dict:
    """订单节点：调用带 MySQL 工具能力的 ReAct Agent 处理查询。"""
    try:
        agent=_order_agent()
        msgs=_build_history_messages(state.get("history",[]))
        msgs.append(HumanMessage(content=state["user_input"]))
        result = agent.invoke(
            {"messages":msgs},
            config={"recursion_limit": settings.REACT_MAX_ITERATIONS * 4},
        )
        response=result["messages"][-1].content
        return {"response":response}
    except Exception as e:
        logger.exception("订单 Agent 异常")
        return {"response": FALLBACK_RESPONSE, "error": str(e)}

# ============================================================
# 节点：综合兜底（ReAct Agent + 检索/业务工具）
# ============================================================

@lru_cache(maxsize=1)
def _general_agent():
    """综合 Agent：同时拥有知识检索 + 订单查询工具。"""
    return create_agent(
        model=get_llm(),
        tools=[search_knowledge, query_order_status, query_orders_by_user],
        system_prompt=SYSTEM_GENERAL_AGENT,
    )

def general_node(state: AgentState) -> Dict:
    """综合节点：交给同时具备检索与业务工具能力的 Agent 处理。"""
    try:
        agent = _general_agent()
        msgs=_build_history_messages(state.get("history",[]))
        msgs.append(HumanMessage(content=state["user_input"]))
        result = agent.invoke(
            {"messages": msgs},
            config={"recursion_limit": settings.REACT_MAX_ITERATIONS * 4},
        )
        response = result["messages"][-1].content
        return {"response": response}
    except Exception as e:
        logger.exception("综合 Agent 异常")
        return {"response": FALLBACK_RESPONSE, "error": str(e)}

# ============================================================
# 节点：闲聊
# ============================================================

def chitchat_node(state: AgentState) -> Dict:
    """闲聊节点：直接调用 LLM 友好回应。"""
    try:
        llm =get_llm()
        reply =llm.invoke(
            [
                SystemMessage(content=SYSTEM_CHITCHAT),
                *[HumanMessage(content="content") if h["role"]=="user" else AIMessage(content=h["content"]) for h in state.get("history",[])],
                HumanMessage(content=state["user_input"]),
            ]
        ).content
        return {"response": reply}
    except Exception as e:
        logger.exception("闲聊节点异常")
        return {"response": FALLBACK_RESPONSE, "error": str(e)}

# ============================================================
# 节点：落库 + 组装最终输出
# ============================================================

def finalize_node(state: AgentState) -> Dict:
    """收尾节点：把助手回复与元数据写入 MySQL，并触碰会话更新时间。"""
    try:
        metadata={
            "intent": state.get("intent", Intent.GENERAL).value,
            "sources": state.get("sources", []),
        }
        mysql_client.add_message(
            session_id=state["session_id"],
            role="assistant",
            content=state["response"],
            metadata=metadata,
        )
        mysql_client.touch_conversation(state["session_id"])
    except Exception as e:
        logger.warning("助手消息落库失败（不影响返回）: %s", e)
    return {}
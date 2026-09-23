"""
LangGraph 状态与意图定义
=========================
LangGraph 的核心是"状态机"：State 是各节点共享的数据结构，每个节点返回对它的部分更新，
图引擎负责按边（Edge）编排节点执行顺序。

这里定义：
- Intent：意图枚举（路由的依据）
- AgentState：整个 Agent 图的共享状态（TypedDict）
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict,List,Optional,TypedDict


@dataclass
class UserContext:
    """
    工具调用时注入的"当前登录用户"运行时上下文。

    为什么单独定义而不是塞进 AgentState：
    ReAct Agent 内部的工具（如查订单）需要知道"我是谁"，但这个身份**绝不能**
    作为工具参数暴露给大模型——否则模型可以被用户一句"查张三的订单"诱导去查别人的数据。
    LangChain 1.x 的 runtime context（ToolRuntime）正好用于传递这类"框架注入、
    模型不可见"的信息：工具签名里标注 ToolRuntime[UserContext, Any]，
    调用方通过 agent.invoke(..., context=UserContext(...)) 传入，
    实测该参数不会出现在工具的 args schema 里（模型看不到、也伪造不了）。

    本类不 import 任何 app.* 模块，避免与 agent 包形成循环依赖。
    """

    user_id: int
    user_name: str = ""

class Intent(str,Enum):
    """用户意图分类（路由的依据）。

    - KNOWLEDGE: 政策/知识类问题 -> 走 RAG 检索增强回答
    - ORDER:     订单/物流查询    -> 走业务工具 Agent（查 MySQL）
    - CHITCHAT:  闲聊寒暄        -> 直接大模型友好回复
    - GENERAL:   其它/综合       -> 兜底 Agent（同时具备检索 + 业务工具能力）
    """

    KNOWLEDGE = "knowledge"
    ORDER = "order"
    CHITCHAT = "chitchat"
    GENERAL = "general"

class AgentState(TypedDict):
    """Agent 图全局状态（所有节点共享的"黑板"）。

    字段说明：
    - session_id / user_name : 会话标识（用于持久化与个性化）
    - user_id                : 当前登录用户 id，用于数据隔离（订单查询按它过滤）
    - user_input             : 用户本轮问题
    - intent                 : 路由得到的意图
    - history                : 最近几轮对话历史 [{role, content}, ...]
    - context / hits         : RAG 检索得到的上下文与命中块
    - response               : 最终回复文本
    - sources                : 引用来源列表（供前端溯源展示）
    - error                  : 异常信息（可选）
    """

    session_id: str
    user_id: int
    user_name: str
    user_input: str
    intent: Intent
    history: List[Dict[str, str]]
    context: str
    hits: List[Dict]
    response: str
    sources: List[Dict]
    error: Optional[str]
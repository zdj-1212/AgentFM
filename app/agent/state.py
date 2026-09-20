"""
LangGraph 状态与意图定义
=========================
LangGraph 的核心是"状态机"：State 是各节点共享的数据结构，每个节点返回对它的部分更新，
图引擎负责按边（Edge）编排节点执行顺序。

这里定义：
- Intent：意图枚举（路由的依据）
- AgentState：整个 Agent 图的共享状态（TypedDict）
"""
from enum import Enum
from typing import Dict,List,Optional,TypedDict

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
    - user_input             : 用户本轮问题
    - intent                 : 路由得到的意图
    - history                : 最近几轮对话历史 [{role, content}, ...]
    - context / hits         : RAG 检索得到的上下文与命中块
    - response               : 最终回复文本
    - sources                : 引用来源列表（供前端溯源展示）
    - error                  : 异常信息（可选）
    """

    session_id: str
    user_name: str
    user_input: str
    intent: Intent
    history: List[Dict[str, str]]
    context: str
    hits: List[Dict]
    response: str
    sources: List[Dict]
    error: Optional[str]
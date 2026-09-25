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

class ErrorCode(str,Enum):
    """节点失败时对外暴露的"降级原因"。

    四个节点失败时都会返回同一句兜底话术（FALLBACK_RESPONSE），如果不额外标记原因，
    调用方（前端 / API 使用方 / 运维）就无法区分到底是知识库挂了、大模型超时还是
    订单库连不上——现象完全一样，只能去翻日志。

    因此节点在返回兜底话术的同时带上这里的一个枚举值：
    - 粒度"按依赖"而不是"按异常"：使用者只需要知道该找谁，不需要知道堆栈；
    - 值本身是稳定常量而非异常字符串：既便于前端做差异化提示，
      也不会把内部实现（表名、连接地址、堆栈片段）泄露给外部调用方。
    完整的异常与堆栈照旧由节点里的 logger.exception 记录。
    """

    KNOWLEDGE_UNAVAILABLE = "knowledge_unavailable"  # RAG 节点失败（通常是 Milvus / Embedding）
    ORDER_UNAVAILABLE = "order_unavailable"          # 订单 Agent 失败（通常是 LLM / MySQL）
    GENERAL_UNAVAILABLE = "general_unavailable"      # 综合兜底 Agent 失败
    CHITCHAT_UNAVAILABLE = "chitchat_unavailable"    # 闲聊节点失败（通常是 LLM）

class HandoffStatus(str,Enum):
    """会话的"谁在应答"状态（转人工的状态机）。

    放在这里是因为它既是会话数据（conversations.handoff_status 列），
    也直接决定 Agent 图的行为：只有 BOT 状态下才允许机器人作答。
    图本身不修改它（改状态是服务层的事），但必须尊重它。

    - BOT      : 机器人应答（默认）
    - PENDING  : 用户已请求转人工，等待坐席接入
    - ASSIGNED : 已有坐席接入并处理中
    - CLOSED   : 本次转人工已结束，交回机器人
    """

    BOT = "bot"
    PENDING = "pending"
    ASSIGNED = "assigned"
    CLOSED = "closed"


# 机器人可以作答的状态：BOT 与 CLOSED（关闭后可继续提问）
BOT_ANSWER_STATUSES = (HandoffStatus.BOT.value, HandoffStatus.CLOSED.value)

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
    - error                  : 内部异常信息（只进日志，不外发）
    - error_code             : 对外暴露的降级原因（ErrorCode 的取值，正常时为 None）
    - message_id             : 落库后的助手消息 id（供前端立刻对这条回复点赞/点踩）
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
    error_code: Optional[str]
    message_id: Optional[int]
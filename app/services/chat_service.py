"""
会话服务层（业务入口）
======================
面向上层（CLI / Streamlit / FastAPI）提供统一的对话能力，封装：
1. 会话生命周期管理（创建 / 列表 / 历史），且**全部按登录用户隔离**；
2. 多轮记忆：从 MySQL 读取最近历史，注入 Agent 状态；
3. 会话标题：用首轮提问的概括作为标题，让侧边栏历史会话一眼可辨；
4. 调用 LangGraph 工作流并返回结构化结果（回复 + 意图 + 来源）。

这样上层只管"传参、展示"，不关心 Agent 与存储细节。

数据隔离约定
------------
本层所有面向用户的方法都必须传 user_id：
- list_sessions 只返回该用户的会话；
- get_history / ask 会校验会话归属，访问他人会话直接抛 PermissionError。
只做"列表过滤"是不够的——session_id 一旦泄露，不带归属校验就能读到别人的对话。
"""
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict,List,Optional

from app.agent.graph import get_graph
from app.agent.prompts import TITLE_SUMMARY_PROMPT
from app.agent.state import BOT_ANSWER_STATUSES, HandoffStatus, Intent
from app.core.logger import get_logger
from app.database import mysql_client

logger=get_logger(__name__)

_graph=None

def _get_graph():
    global _graph
    if _graph is None:
        _graph =get_graph()
    return _graph

# 会话标题概括专用线程池：概括与主流程（Agent 工作流，通常要数秒）并行执行，
# 基本被主流程的耗时"藏"起来，不额外增加用户等待时间。
_TITLE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="title")

class ChatService:
    """对话业务服务。"""

    # ---------------- 会话管理 ----------------
    def create_session(self,user_id:int,user_name:str="")->str:
        """为指定用户新建会话，返回 session_id。"""
        session_id = uuid.uuid4().hex[:16]
        mysql_client.create_conversation(
            session_id,
            user_id=user_id,
            user_name=_resolve_user_name(user_id,user_name),
        )
        logger.info("新建会话: %s (user_id=%s)", session_id, user_id)
        return session_id

    def list_sessions(self,user_id:int,limit:int=20)->List[Dict]:
        """返回【该用户】最近的会话列表（供前端侧边栏）。

        必须按 user_id 过滤：不加过滤会把所有用户的会话都列出来。
        """
        return [
            {
                "session_id": c.session_id,
                "title": c.title,
                "user_name": c.user_name,
                "updated_at": c.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for c in mysql_client.list_conversations(user_id, limit)
        ]

    def get_history(self, session_id: str, user_id: int) -> List[Dict]:
        """返回某会话的全部历史消息（供前端回显）。

        会话不属于该用户时抛 PermissionError，避免"知道 session_id 就能读别人对话"。
        带上 message_id 与已有评价，前端才能渲染出"这条我赞过/踩过"的状态。
        坐席（role='agent'）的回复同样会出现在这里——用户必须看得到人工说了什么。
        """
        if mysql_client.get_conversation(session_id, user_id) is None:
            raise PermissionError("会话不存在或无权访问")

        return [
            {
                "message_id": m.id,
                "role": m.role,
                "content": m.content,
                "feedback": m.feedback,
                "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for m in mysql_client.get_recent_messages(session_id, limit=100)
        ]

    # ---------------- 转人工（用户侧） ----------------

    def get_session_state(self, session_id: str, user_id: int) -> Dict:
        """
        返回会话的转人工状态与当前坐席，供前端决定展示什么
        （是普通的输入框，还是"等待坐席接入"的提示）。

        会话不属于该用户时抛 PermissionError。
        """
        conv = mysql_client.get_conversation(session_id, user_id)
        if conv is None:
            raise PermissionError("会话不存在或无权访问")

        agent_name = ""
        if conv.agent_id:
            agent = mysql_client.get_user_by_id(conv.agent_id)
            if agent:
                agent_name = agent.display_name or agent.username

        return {
            "session_id": session_id,
            "handoff_status": conv.handoff_status or HandoffStatus.BOT.value,
            "agent_id": conv.agent_id,
            "agent_name": agent_name,
        }

    def request_handoff(self, session_id: str, user_id: int) -> Dict:
        """
        用户请求转人工：把会话从 bot 置为 pending，进入坐席的待接入队列。

        用带条件的更新（只允许从 bot/closed 转过来），所以重复点按钮不会把
        "已被坐席接入"的会话打回队列——那会把正在处理中的会话从坐席手里抢走。
        """
        conv = mysql_client.get_conversation(session_id, user_id)
        if conv is None:
            raise PermissionError("会话不存在或无权访问")

        current = conv.handoff_status or HandoffStatus.BOT.value
        if current == HandoffStatus.PENDING.value:
            return self.get_session_state(session_id, user_id)  # 已经在队列里，幂等
        if current == HandoffStatus.ASSIGNED.value:
            return self.get_session_state(session_id, user_id)  # 已有坐席，不打断

        ok = mysql_client.set_handoff_status(
            session_id,
            HandoffStatus.PENDING.value,
            agent_id=None,
            from_statuses=list(BOT_ANSWER_STATUSES),
        )
        if ok:
            logger.info("[handoff] 用户请求转人工: session=%s user_id=%s", session_id, user_id)
        else:
            # 并发下被别人改掉了，以库里的实际状态为准
            logger.info("[handoff] 状态未变更（并发更新）: session=%s", session_id)
        return self.get_session_state(session_id, user_id)

    def set_feedback(
        self,
        session_id: str,
        message_id: int,
        user_id: int,
        feedback: Optional[int],
        note: str = "",
    ) -> bool:
        """
        给某条助手回复写评价（1=/ -1=/ None=取消）。

        两道校验，缺一不可：
        1. 会话必须属于调用者（否则等于能给别人机器人的回答打标）；
        2. 消息必须属于该会话且是 assistant —— 这条由 DAO 的 UPDATE 条件保证。

        返回是否真的写入了（消息不存在 / 不属于自己 -> False）。
        """
        if feedback is not None and feedback not in (1, -1):
            raise ValueError("feedback 只能是 1（赞）、-1（踩）或 None（取消）")

        if mysql_client.get_conversation(session_id, user_id) is None:
            raise PermissionError("会话不存在或无权访问")

        note = (note or "").strip()[:255]  # 与列宽一致，避免超长被数据库截断/报错
        ok = mysql_client.set_message_feedback(
            message_id, user_id, feedback, note or None
        )
        if ok:
            logger.info(
                "[feedback] session=%s message=%s user_id=%s -> %s",
                session_id, message_id, user_id, feedback,
            )
        return ok

    # ---------------- 对话主流程 ----------------
    def ask(self,session_id:str,message:str,user_id:int,user_name:str="")->Dict:
        """
        核心对话方法：
        1. 校验会话归属（不存在则创建，属于他人则拒绝）；
        2. 落库用户消息；
        3. 若是该会话的**首轮提问**，并行生成一条会话标题；
        4. 取出最近历史作为多轮记忆，调用 LangGraph 工作流；
        5. 返回 {session_id, reply, intent, sources, error_code}（error_code 非空即为降级回复）。
        """
        message=message.strip()
        if not message:
            raise ValueError("消息不能为空")

        user_name = _resolve_user_name(user_id, user_name)

        # 1) 确认会话存在且属于当前用户。
        #    这里先查"不带 user_id"的存在性：否则别人的 session_id 会被当成"不存在"
        #    而走到下面的新建分支，INSERT 撞上 session_id 唯一索引直接报错。
        conv = mysql_client.get_conversation(session_id)
        if conv is None:
            mysql_client.create_conversation(session_id, user_id=user_id, user_name=user_name)
        elif conv.user_id != user_id:
            logger.warning("[ask] 越权访问被拒绝: session=%s user_id=%s", session_id, user_id)
            raise PermissionError("无权访问该会话")

        # 2) 判断是不是首轮提问——必须在写入本条消息**之前**统计，否则永远是 False
        first_turn = mysql_client.count_messages(session_id) == 0

        # 3) 落库用户消息。
        #    注意这一步在"已转人工"的判断**之前**：坐席需要看到用户转人工之后补充说明的内容，
        #    如果因为机器人不答就把消息丢掉，用户会觉得"我说了但坐席看不到"。
        mysql_client.add_message(session_id,role="user",content=message)

        # 3.5) 已转人工时机器人闭嘴：只落库、不调模型。
        #      这里必须拦，否则人工和机器人会对着同一个用户各说各话。
        current = conv.handoff_status if conv else HandoffStatus.BOT.value
        if current not in BOT_ANSWER_STATUSES:
            logger.info("[ask] 会话处于转人工状态(%s)，机器人不介入: session=%s", current, session_id)
            return {
                "session_id": session_id,
                "reply": "",
                "intent": "handoff",
                "sources": [],
                "error_code": None,
                "message_id": None,
                "handoff_status": current,
            }

        # 4) 首轮提问 -> 后台并行生成标题（不阻塞下面的 Agent 调用）
        title_future = None
        if first_turn and settings_title_enabled():
            title_future = _TITLE_POOL.submit(_summarize_title, message)

        # 5) 读取最近历史（不含刚写入的这条，作为上下文）。
        #    人工（agent）说过的话也要计入：转人工结束、机器人重新接手后，
        #    它得知道刚才坐席和用户聊了什么，否则会重复追问已经解决过的事。
        recent= mysql_client.get_recent_messages(session_id,limit=settings_history_window()*2)
        history=[
            {"role":m.role,"content":m.content}
            for m in recent
            if m.role in ("user","assistant","agent") and m.content != message
        ]
        # 6) 构造状态并调用图
        state = {
            "session_id": session_id,
            "user_id": user_id,
            "user_name": user_name,
            "user_input": message,
            "intent": Intent.GENERAL,
            "history": history[-settings_history_window() * 2:],
            "context": "",
            "hits": [],
            "response": "",
            "sources": [],
            "error": None,
            "error_code": None,
            "message_id": None,
        }

        logger.info("[ask] session=%s user_id=%s question=%s", session_id, user_id, message[:30])
        try:
            result = _get_graph().invoke(state, config={"recursion_limit": 50})
        finally:
            # 无论 Agent 是否成功都要落标题，否则首轮失败后标题会永远停在"新会话"
            if title_future is not None:
                _save_title(session_id, user_id, title_future, message)

        # 7) 组装返回值。
        #    error_code 非空表示这是"降级回复"（节点失败后的兜底话术），
        #    调用方据此能区分"知识库挂了"和"正常但没查到"，不用去猜同一句话术背后的原因。
        error_code = result.get("error_code")
        if error_code:
            logger.warning("[ask] 本轮为降级回复: session=%s code=%s", session_id, error_code)

        return {
            "session_id": session_id,
            "reply": result.get("response", ""),
            "intent": result.get("intent", Intent.GENERAL).value
            if isinstance(result.get("intent"), Intent)
            else str(result.get("intent", "general")),
            "sources": result.get("sources", []),
            "error_code": error_code,
            # 刚落库的助手消息 id，前端据此对这条回复展示评价按钮
            "message_id": result.get("message_id"),
            # 当前会话的转人工状态；非 bot/closed 时 reply 为空，
            # 调用方应当据此展示"等待坐席"而不是一条空白回复
            "handoff_status": current,
        }


# ---------------- 会话标题 ----------------

def _save_title(session_id: str, user_id: int, future, question: str) -> None:
    """取回标题生成结果并落库；超时/异常一律退化为"截断的问句"。"""
    from config.settings import settings

    try:
        title = future.result(timeout=settings.TITLE_TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning("会话标题生成超时或失败，改用截断问句: %s", e)
        title = _fallback_title(question)

    try:
        mysql_client.touch_conversation(session_id, title=title, user_id=user_id)
        logger.info("[title] session=%s -> %s", session_id, title)
    except Exception as e:
        logger.warning("会话标题落库失败（不影响回复）: %s", e)


def _clean_title(text: str) -> str:
    """清洗模型输出：取第一行，去掉引号/书名号与句末标点。"""
    text = (text or "").strip()
    if not text:
        return ""
    line = text.splitlines()[0].strip()
    line = line.strip("\"'“”‘’「」《》【】[]")
    line = line.rstrip("。.，,！!？?、：:;；")
    # 模型有时仍会带上"标题："前缀
    for prefix in ("标题：", "标题:", "概括：", "概括:", "会话标题："):
        if line.startswith(prefix):
            line = line[len(prefix):].strip()
    return line.strip()


def _fallback_title(question: str) -> str:
    """把问句压缩成一行并截断，作为标题概括失败时的兜底。"""
    from config.settings import settings

    text = " ".join((question or "").split())
    if not text:
        return "新会话"
    if len(text) <= settings.TITLE_MAX_CHARS:
        return text
    return text[: settings.TITLE_MAX_CHARS] + "…"


def _summarize_title(question: str) -> str:
    """
    调用 LLM 把首轮提问概括成短标题。

    标题只是锦上添花，绝不能让主流程失败：任何异常（LLM 不可用/超时/返回空）
    都退回"截断的问句"。
    """
    from config.settings import settings

    fallback = _fallback_title(question)
    try:
        # 延迟导入且在 try 内：get_llm() 在 LLM_PROVIDER 配置非法时会直接抛异常
        from langchain_core.messages import HumanMessage

        from app.core.llm import get_llm

        prompt = TITLE_SUMMARY_PROMPT.format(
            max_chars=settings.TITLE_MAX_CHARS, question=question
        )
        answer = get_llm().invoke([HumanMessage(content=prompt)]).content
        title = _clean_title(answer)[: settings.TITLE_MAX_CHARS]
        return title or fallback
    except Exception as e:
        logger.warning("会话标题概括失败，回退为截断问句: %s", e)
        return fallback


def _resolve_user_name(user_id: int, user_name: str = "") -> str:
    """会话表里冗余存一份展示名；调用方没传就按 user_id 回查。"""
    if user_name:
        return user_name
    try:
        from app.services.auth_service import get_user

        user = get_user(user_id)
        if user:
            return user["display_name"]
    except Exception as e:  # 仅影响展示，不应阻断对话
        logger.warning("回查用户名失败: %s", e)
    return f"用户{user_id}"


def settings_history_window() -> int:
    """读取记忆窗口配置（延迟导入避免循环）。"""
    from config.settings import settings

    return settings.HISTORY_WINDOW


def settings_title_enabled() -> bool:
    """是否启用"首轮提问自动概括标题"。"""
    from config.settings import settings

    return settings.TITLE_AUTO_SUMMARY


# 模块级单例，供各入口复用
chat_service = ChatService()

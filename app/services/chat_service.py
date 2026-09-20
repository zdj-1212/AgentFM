"""
会话服务层（业务入口）
======================
面向上层（CLI / Streamlit / FastAPI）提供统一的对话能力，封装：
1. 会话生命周期管理（创建 / 列表 / 历史）；
2. 多轮记忆：从 MySQL 读取最近历史，注入 Agent 状态；
3. 调用 LangGraph 工作流并返回结构化结果（回复 + 意图 + 来源）。

这样上层只管"传参、展示"，不关心 Agent 与存储细节。
"""
import uuid
from typing import Dict,List
from functools import lru_cache

from app.agent.graph import get_graph
from app.agent.state import Intent
from app.core.logger import get_logger
from app.database import mysql_client

logger=get_logger(__name__)

_graph=None

def _get_graph():
    global _graph
    if _graph is None:
        _graph =get_graph()
    return _graph

class ChatService:
    """对话业务服务。"""

    # ---------------- 会话管理 ----------------
    def create_session(self,user_name:str="游客")->str:
        """新建会话，返回 session_id。"""
        session_id = uuid.uuid4().hex[:16]
        mysql_client.create_conversation(session_id,user_name=user_name)
        logger.info("新建会话: %s", session_id)
        return session_id

    def list_sessions(self,limit:int=20)->List[Dict]:
        """返回最近会话列表（供前端侧边栏）。"""
        return [
            {
                "session_id": c.session_id,
                "title": c.title,
                "user_name": c.user_name,
                "updated_at": c.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for c in mysql_client.list_conversations(limit)
        ]

    def get_history(self, session_id: str) -> List[Dict]:
        """返回某会话的全部历史消息（供前端回显）。"""
        return [
            {"role": m.role, "content": m.content, "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S")}
            for m in mysql_client.get_recent_messages(session_id, limit=100)
        ]

    # ---------------- 对话主流程 ----------------
    def ask(self,session_id:str,message:str,user_name:str="游客")->Dict:
        """
        核心对话方法：
        1. 落库用户消息；
        2. 取出最近历史作为多轮记忆；
        3. 调用 LangGraph 工作流；
        4. 返回 {session_id, reply, intent, sources}。
        """
        message=message.strip()
        if not message:
            raise ValueError("消息不能为空")

        # 1) 确保会话存在
        if mysql_client.get_conversation(session_id) is None:
            mysql_client.create_conversation(session_id,user_name=user_name)

        # 2) 落库用户消息
        mysql_client.add_message(session_id,role="user",content=message)

        # 3) 读取最近历史（不含刚写入的这条，作为上下文）
        recent= mysql_client.get_recent_messages(session_id,limit=settings_history_window()*2)
        history=[
            {"role":m.role,"content":m.content}
            for m in recent
            if m.role in ("user","assistant") and m.content != message
        ]
        # 4) 构造状态并调用图
        state = {
            "session_id": session_id,
            "user_name": user_name,
            "user_input": message,
            "intent": Intent.GENERAL,
            "history": history[-settings_history_window() * 2:],
            "context": "",
            "hits": [],
            "response": "",
            "sources": [],
            "error": None,
        }

        logger.info("[ask] session=%s question=%s", session_id, message[:30])
        result = _get_graph().invoke(state, config={"recursion_limit": 50})

        # 5) 组装返回值
        return {
            "session_id": session_id,
            "reply": result.get("response", ""),
            "intent": result.get("intent", Intent.GENERAL).value
            if isinstance(result.get("intent"), Intent)
            else str(result.get("intent", "general")),
            "sources": result.get("sources", []),
        }

def settings_history_window() -> int:
    """读取记忆窗口配置（延迟导入避免循环）。"""
    from config.settings import settings

    return settings.HISTORY_WINDOW

# 模块级单例，供各入口复用
chat_service = ChatService()
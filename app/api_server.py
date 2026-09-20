"""
AgentFM REST API 服务（FastAPI）
================================
把 Agent 能力以企业标准 REST 接口暴露出去，可被 Web/App/其它系统调用。
这体现"智能体服务化"的工程能力：前端、小程序、企业微信机器人等都能接入。

启动方式（项目根目录）：
    uv run uvicorn app.api_server:app --host 0.0.0.0 --port 8000

接口一览：
    GET  /health                     健康检查
    POST /chat                       对话（自动创建会话或续接指定会话）
    GET  /sessions                   会话列表
    GET  /sessions/{sid}/messages    某会话历史
"""

from typing import List,Optional
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field

from app.core.logger import get_logger
from app.services.chat_service import chat_service

logger=get_logger(__name__)

app=FastAPI(
    title="AgentFM API",
    description="企业级多智能体知识问答平台（LangGraph + RAG + Milvus + MySQL）",
    version="0.1.0"
)
# ---------------- 请求/响应模型 ----------------
class ChatRequest(BaseModel):
    message:str =Field(...,description="用户输入",min_length=1,max_length=2000)
    session_id:Optional[str]=Field(None,description="会话 ID；为空则自动新建会话")
    user_name:str = Field("游客",description="用户名")

class SourceItem(BaseModel):
    title: str
    source: str
    score: float

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    sources: List[SourceItem] = []

class SessionItem(BaseModel):
    session_id: str
    title: str
    user_name: str
    updated_at: str

# ---------------- 接口 ----------------

@app.get("/health")
def health():
    """健康检查：确认服务存活（并顺带检查依赖组件连通性）。"""
    from app.database import milvus_client, mysql_client

    mysql_ok = True
    try:
        mysql_client.init_db()
    except Exception as e:
        logger.warning("MySQL 检查失败: %s", e)
        mysql_ok = False

    return {"status": "ok", "mysql": mysql_ok, "milvus": milvus_client.collection_exists()}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """对话接口：无 session_id 时自动创建新会话。"""
    try:
        session_id = req.session_id or chat_service.create_session(req.user_name)
        result = chat_service.ask(session_id, req.message, user_name=req.user_name)
        return ChatResponse(
            session_id=result["session_id"],
            reply=result["reply"],
            intent=result["intent"],
            sources=[SourceItem(**s) for s in result.get("sources", [])],
        )
    except Exception as e:
        logger.exception("chat 接口异常")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions", response_model=List[SessionItem])
def sessions():
    """返回最近会话列表。"""
    return [SessionItem(**s) for s in chat_service.list_sessions()]

@app.get("/sessions/{session_id}/messages")
def messages(session_id: str):
    """返回某会话的完整历史消息。"""
    history = chat_service.get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="会话不存在或暂无消息")
    return history
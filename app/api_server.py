"""
AgentFM REST API 服务（FastAPI）
================================
把 Agent 能力以企业标准 REST 接口暴露出去，可被 Web/App/其它系统调用。
这体现"智能体服务化"的工程能力：前端、小程序、企业微信机器人等都能接入。

启动方式（项目根目录）：
    uv run uvicorn app.api_server:app --host 0.0.0.0 --port 8000

接口一览：
    GET  /health                     健康检查（无需登录）
    POST /auth/register              注册，返回登录令牌
    POST /auth/login                 登录，返回登录令牌
    GET  /auth/me                    当前登录用户信息
    POST /chat                       对话（自动创建会话或续接指定会话）
    GET  /sessions                   当前用户的会话列表
    GET  /sessions/{sid}/messages    某会话历史

鉴权说明
--------
除 /health 与 /auth/* 外，所有接口都要求请求头携带 `Authorization: Bearer <token>`，
用户身份**只从令牌解析**，绝不信任请求体里的用户名字段——
否则任何调用方都能自称是别人，数据隔离就成了摆设。
"""
from contextlib import asynccontextmanager
from typing import Dict,List,Optional

from fastapi import Depends,FastAPI,HTTPException,status
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer
from pydantic import BaseModel,Field

from app.core.logger import get_logger
from app.database import mysql_client
from app.services.auth_service import AuthError,auth_service
from app.services.chat_service import chat_service

logger=get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时做一次库表初始化/迁移"""
    try:
        mysql_client.init_db()
    except Exception as e:
        # 启动失败不直接退出服务：/health 会如实报告 MySQL 不可用，便于排查
        logger.exception("启动时数据库初始化失败: %s", e)
    yield


app=FastAPI(
    title="AgentFM API",
    description="企业级多智能体知识问答平台（LangGraph + RAG + Milvus + MySQL）",
    version="0.2.0",
    lifespan=lifespan,
)

# auto_error=False：未带令牌时返回 None，由我们自己给出统一的中文 401 提示
_bearer = HTTPBearer(auto_error=False, description="登录接口返回的 access_token")

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Dict:
    """从 Bearer 令牌解析当前用户；令牌缺失/伪造/过期一律 401。"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供登录令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = auth_service.parse_token(credentials.credentials)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 令牌验签通过不代表账号还在（可能已被删除），再确认一次
    user = auth_service.get_user(payload["user_id"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号不存在或已被禁用",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ---------------- 请求/响应模型 ----------------
class RegisterRequest(BaseModel):
    username:str =Field(...,description="用户名（2~32 位中文/字母/数字/下划线）",min_length=2,max_length=32)
    password:str =Field(...,description="密码",min_length=1,max_length=128)
    display_name:Optional[str]=Field(None,description="展示名，默认与用户名相同")

class LoginRequest(BaseModel):
    username:str =Field(...,description="用户名")
    password:str =Field(...,description="密码")

class UserInfo(BaseModel):
    user_id: int
    username: str
    display_name: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(...,description="有效期（秒）")
    user: UserInfo

class ChatRequest(BaseModel):
    message:str =Field(...,description="用户输入",min_length=1,max_length=2000)
    session_id:Optional[str]=Field(None,description="会话 ID；为空则自动新建会话")

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

# ---------------- 健康检查 ----------------

@app.get("/health")
def health():
    """健康检查：确认服务存活（并顺带检查依赖组件连通性）。

    注意这里只做轻量探活：早期版本会在健康检查里调用 init_db()，
    那意味着每个匿名请求都可能触发建表/改表/清数据，既慢又危险。
    """
    from app.database import milvus_client

    try:
        milvus_ok = milvus_client.collection_exists()
    except Exception as e:
        logger.warning("Milvus 检查失败: %s", e)
        milvus_ok = False

    mysql_ok = mysql_client.ping()
    return {
        "status": "ok" if (mysql_ok and milvus_ok) else "degraded",
        "mysql": mysql_ok,
        "milvus": milvus_ok,
    }

# ---------------- 认证接口 ----------------

def _token_response(user: Dict) -> TokenResponse:
    from config.settings import settings

    return TokenResponse(
        access_token=auth_service.create_token(user["user_id"], user["username"]),
        expires_in=settings.AUTH_TOKEN_TTL_HOURS * 3600,
        user=UserInfo(**user),
    )

@app.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """注册新用户，成功即返回登录令牌（免去再登录一次）。"""
    try:
        user = auth_service.register(req.username, req.password, req.display_name)
    except AuthError as e:
        # 用户名已被占用 -> 409；其余（格式/强度）-> 400
        code = status.HTTP_409_CONFLICT if "已被注册" in str(e) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=str(e))
    return _token_response(user)

@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """用户名密码登录，返回 Bearer 令牌。"""
    try:
        user = auth_service.authenticate(req.username, req.password)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_response(user)

@app.get("/auth/me", response_model=UserInfo)
def me(user: Dict = Depends(get_current_user)):
    """返回当前登录用户信息（前端可用它校验令牌是否仍然有效）。"""
    return UserInfo(**user)

# ---------------- 业务接口 ----------------

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: Dict = Depends(get_current_user)):
    """对话接口：无 session_id 时自动创建新会话；会话与数据均限定在当前用户名下。"""
    user_id, username = user["user_id"], user["display_name"]
    try:
        session_id = req.session_id or chat_service.create_session(user_id, username)
        result = chat_service.ask(session_id, req.message, user_id, username)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("chat 接口异常")
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        session_id=result["session_id"],
        reply=result["reply"],
        intent=result["intent"],
        sources=[SourceItem(**s) for s in result.get("sources", [])],
    )

@app.get("/sessions", response_model=List[SessionItem])
def sessions(user: Dict = Depends(get_current_user)):
    """返回当前用户最近的会话列表。"""
    return [SessionItem(**s) for s in chat_service.list_sessions(user["user_id"])]

@app.get("/sessions/{session_id}/messages")
def messages(session_id: str, user: Dict = Depends(get_current_user)):
    """返回某会话的完整历史消息（仅限本人会话）。"""
    try:
        history = chat_service.get_history(session_id, user["user_id"])
    except PermissionError:
        # 用 404 而不是 403：不向调用方泄露"这个 session_id 确实存在，只是不属于你"
        raise HTTPException(status_code=404, detail="会话不存在或暂无消息")
    if not history:
        raise HTTPException(status_code=404, detail="会话不存在或暂无消息")
    return history

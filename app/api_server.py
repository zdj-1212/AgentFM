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
    POST /auth/logout                退出登录（吊销该账号全部令牌）
    POST /chat                       对话（自动创建会话或续接指定会话）
    GET  /sessions                   当前用户的会话列表
    GET  /sessions/{sid}/messages    某会话历史
    POST /sessions/{sid}/messages/{mid}/feedback   给助手回复打分（/）
    POST /sessions/{sid}/handoff     请求转人工（用户）
    GET  /agent/queue                待接入队列 + 我名下的会话（仅坐席）
    GET  /agent/sessions/{sid}/messages  会话消息（仅坐席，限待接入或本人会话）
    POST /agent/sessions/{sid}/claim 接入会话（仅坐席）
    POST /agent/sessions/{sid}/reply 以人工身份回复（仅坐席）
    POST /agent/sessions/{sid}/close 结束人工服务（仅坐席）
    GET  /admin/overview             运营总览：各用户会话/消息/评价数（仅管理员）
    GET  /admin/feedback/summary     反馈汇总（仅管理员）
    GET  /admin/users/{uid}/sessions 指定用户的会话（仅管理员）
    GET  /admin/sessions/{sid}/messages  任意会话的消息（仅管理员）

鉴权说明
--------
除 /health 与 /auth/* 外，所有接口都要求请求头携带 `Authorization: Bearer <token>`，
用户身份**只从令牌解析**，绝不信任请求体里的用户名字段——
否则任何调用方都能自称是别人，数据隔离就成了摆设。

限流说明
--------
/auth/login 与 /auth/register 是唯一可匿名调用、且每次都要跑一次口令哈希（argon2id）的接口，
因此两者都加了限流（见 app/core/rate_limit.py）。超限返回 429 并带 `Retry-After`。
配了 Redis 则计数全局一致（多进程/重启都算数）；没有 Redis 时退回进程内计数，
此时多 worker 各算各的——严格限流请再在网关层加一道。
"""
from contextlib import asynccontextmanager
from typing import Dict,List,Optional

from fastapi import Depends,FastAPI,HTTPException,Request,status
from fastapi.security import HTTPAuthorizationCredentials,HTTPBearer
from pydantic import BaseModel,Field

from app.core.logger import get_logger
from app.core.rate_limit import allow_auth_attempt,record_auth_failure
from app.database import mysql_client
from app.services import admin_service, agent_service
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
    """从 Bearer 令牌解析当前用户；令牌缺失/伪造/过期/已吊销一律 401。

    校验统一收敛在 auth_service.authenticate_token：签名、有效期、以及
    与库里 token_version 的比对（退出登录会把它 +1，从而让旧令牌立即失效）。
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供登录令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return auth_service.authenticate_token(credentials.credentials)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


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
    role: str = "user"

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
    error_code: Optional[str] = Field(
        None,
        description="降级原因。为空表示正常回复；非空表示本轮回复为兜底话术，"
        "取值见 knowledge_unavailable / order_unavailable / general_unavailable / chitchat_unavailable",
    )
    handoff_status: str = Field(
        "bot",
        description="会话的转人工状态。非 bot/closed 时 reply 为空——"
        "消息已入库但机器人不介入，等待坐席处理",
    )

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

def _too_many_attempts(retry_after: float, reason: str) -> HTTPException:
    """统一的 429 响应；Retry-After 让调用方知道该等多久。"""
    detail = (
        "认证尝试过于频繁，请稍后再试"
        if reason == "ip"
        else "该账号失败次数过多，请稍后再试"
    )
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=detail,
        headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
    )

@app.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest, request: Request):
    """注册新用户，成功即返回登录令牌（免去再登录一次）。"""
    # 只按来源限流：注册的风险是"被刷"（每次都要跑一次口令哈希），
    # 不需要按账号维度计数（那反而会误伤第一次来注册的正常用户）。
    allowed, retry_after, reason = allow_auth_attempt(request)
    if not allowed:
        raise _too_many_attempts(retry_after, reason)

    try:
        user = auth_service.register(req.username, req.password, req.display_name)
    except AuthError as e:
        # 用户名已被占用 -> 409；其余（格式/强度）-> 400
        code = status.HTTP_409_CONFLICT if "已被注册" in str(e) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=code, detail=str(e))
    return _token_response(user)

@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest, request: Request):
    """用户名密码登录，返回 Bearer 令牌。"""
    # 限流必须在 authenticate() 之前：它内部有一次昂贵的口令哈希，
    # 放在后面等于"已经把 CPU 花完了再告诉对方不许试"。
    allowed, retry_after, reason = allow_auth_attempt(request, req.username)
    if not allowed:
        raise _too_many_attempts(retry_after, reason)

    try:
        user = auth_service.authenticate(req.username, req.password)
    except AuthError as e:
        # 失败才计入"按账号"维度：正常登录成功不该把自己算进失败次数
        record_auth_failure(req.username)
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

@app.post("/auth/logout")
def logout(user: Dict = Depends(get_current_user)):
    """退出登录：吊销该用户的全部令牌，之后所有旧令牌都会 401。

    语义是"退出所有设备"而不是"只退出当前这一个"——单令牌精确吊销需要维护
    已吊销令牌名单（带来清理与多进程一致性问题），本项目取更简单也更稳的方案。
    """
    auth_service.revoke_tokens(user["user_id"])
    return {"detail": "已退出登录，该账号的全部令牌已失效"}


# ---------------- 答案反馈 ----------------

class FeedbackRequest(BaseModel):
    feedback: Optional[int] = Field(
        ..., description="1=，-1=，null=取消评价"
    )
    note: Optional[str] = Field(None, description="可选的一句原因", max_length=255)

@app.post("/sessions/{session_id}/messages/{message_id}/feedback")
def set_feedback(
    session_id: str,
    message_id: int,
    req: FeedbackRequest,
    user: Dict = Depends(get_current_user),
):
    """给某条助手回复打分（只能打自己会话里的）。"""
    if req.feedback not in (1, -1, None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="feedback 只能是 1、-1 或 null"
        )
    try:
        ok = chat_service.set_feedback(
            session_id, message_id, user["user_id"], req.feedback, req.note or ""
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not ok:
        # 用 404 而不是 403：不泄露"这条消息确实存在，只是不属于你"
        raise HTTPException(status_code=404, detail="消息不存在或不属于该会话")
    return {"detail": "评价已记录", "feedback": req.feedback}


# ---------------- 管理员接口（运营视角） ----------------

@app.get("/admin/overview")
def admin_overview(user: Dict = Depends(get_current_user)):
    """运营总览：每个用户的会话数、消息数、有用/没用数（仅管理员）。"""
    try:
        return admin_service.overview(user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@app.get("/admin/feedback/summary")
def admin_feedback_summary(user: Dict = Depends(get_current_user)):
    """反馈汇总：整体满意度（仅管理员）。"""
    try:
        return admin_service.feedback_summary(user)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@app.get("/admin/users/{user_id}/sessions")
def admin_user_sessions(user_id: int, user: Dict = Depends(get_current_user)):
    """查看指定用户的会话列表（仅管理员）。"""
    try:
        return admin_service.user_sessions(user, user_id)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

@app.get("/admin/sessions/{session_id}/messages")
def admin_session_messages(session_id: str, user: Dict = Depends(get_current_user)):
    """查看任意会话的消息（仅管理员）。"""
    try:
        return admin_service.session_messages(user, session_id)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

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
    except Exception:
        # 不把 str(e) 回给调用方：未预期的异常里可能带表名、连接地址、堆栈片段等内部信息。
        # 完整堆栈已进日志，对外只给一句稳定的提示。
        logger.exception("chat 接口异常")
        raise HTTPException(status_code=500, detail="服务内部错误，请稍后重试")

    return ChatResponse(
        session_id=result["session_id"],
        reply=result["reply"],
        intent=result["intent"],
        sources=[SourceItem(**s) for s in result.get("sources", [])],
        error_code=result.get("error_code"),
        handoff_status=result.get("handoff_status", "bot"),
    )

@app.post("/sessions/{session_id}/handoff")
def request_handoff(session_id: str, user: Dict = Depends(get_current_user)):
    """请求转人工：把会话放入坐席的待接入队列，之后机器人不再介入。"""
    try:
        return chat_service.request_handoff(session_id, user["user_id"])
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


# ---------------- 坐席接口（转人工的人工侧） ----------------

class AgentReplyRequest(BaseModel):
    content: str = Field(..., description="坐席回复内容", min_length=1, max_length=2000)


def _agent_guard(callable_, *args):
    """把坐席服务层的异常统一映射成 HTTP 状态码，避免每个端点重复写一遍。"""
    try:
        return callable_(*args)
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/agent/queue")
def agent_queue(user: Dict = Depends(get_current_user)):
    """坐席工作台：待接入队列 + 我名下的会话（仅坐席，管理员也不行）。"""
    return _agent_guard(agent_service.queue, user)

@app.get("/agent/sessions/{session_id}/messages")
def agent_messages(session_id: str, user: Dict = Depends(get_current_user)):
    """坐席查看会话消息（限待接入或本人名下）。"""
    return _agent_guard(agent_service.session_messages, user, session_id)

@app.post("/agent/sessions/{session_id}/claim")
def agent_claim(session_id: str, user: Dict = Depends(get_current_user)):
    """接入一个待处理会话。并发下只有一个人能接入成功。"""
    return _agent_guard(agent_service.claim, user, session_id)

@app.post("/agent/sessions/{session_id}/reply")
def agent_reply(session_id: str, req: AgentReplyRequest, user: Dict = Depends(get_current_user)):
    """以人工身份回复用户（只能回复自己名下的会话）。"""
    return _agent_guard(agent_service.reply, user, session_id, req.content)

@app.post("/agent/sessions/{session_id}/close")
def agent_close(session_id: str, user: Dict = Depends(get_current_user)):
    """结束人工服务，把会话交回机器人。"""
    return _agent_guard(agent_service.close, user, session_id)

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

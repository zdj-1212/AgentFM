"""
MySQL 数据访问层（SQLAlchemy 2.0 ORM）
=======================================
职责：
1. 定义与 MySQL 表一一对应的 ORM 模型（User / Conversation / Message / OrderInfo / LogisticsInfo）；
2. 提供 DAO（数据访问对象）函数：用户、会话管理、消息持久化、订单查询等；
3. 提供测试数据初始化（scripts/init_db.py 会调用）；
4. 提供轻量幂等迁移（_ensure_column），保证已存在的库也能加上新列。

表设计说明：
- users          用户表：登录账号，口令以 argon2id 哈希存储（见 services/auth_service.py）
- conversations  会话表：一次前端对话窗口 = 一个 session_id，user_id 标识归属用户
- messages       消息表：会话内每轮问答的 user/assistant 消息（支持多轮记忆）
- order_info     订单表：业务数据，user_id 标识归属用户，供 Agent 通过工具查询
- logistics_info 物流表：订单的物流轨迹，供 Agent 查询"快递到哪了"

多租户隔离约定：
- conversations / order_info 的 user_id 是数据隔离的唯一依据；
- 所有面向用户的 DAO 都必须带 user_id 过滤，绝不允许"查全表再在内存里筛"。
"""
from contextlib import contextmanager
from datetime import datetime
from typing import Dict,List,Optional
from decimal import Decimal

from sqlalchemy import JSON,DateTime,Index,Integer,String,TEXT,create_engine,func,text,Numeric
from sqlalchemy.orm import DeclarativeBase,Mapped,mapped_column,sessionmaker

from config.settings import settings
from app.core.logger import get_logger
logger=get_logger(__name__)

# ---------------- ORM 模型 ----------------

class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类。"""

class User(Base):
    """用户表：登录账号。

    口令只存 argon2id 哈希（盐与参数编码在哈希串里），永不存明文。
    历史上的 PBKDF2 格式已全部迁移完毕，旧的兼容校验逻辑也已删除（见 auth_service）。

    token_version 是**令牌吊销**的依据：令牌 payload 里带着签发时的版本号，
    校验时与库里的值比对，不一致即视为已吊销。退出登录就把这个值 +1，
    该用户**所有已签发的令牌**立即失效。
    相比"维护一张已吊销令牌的名单"，这样做没有清理负担、也不会因为进程重启或
    多 worker 而出现各进程状态不一致。
    """
    __tablename__ = "users"
    id:Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    username:Mapped[str] = mapped_column(String(64),unique=True,index=True)
    password_hash:Mapped[str] = mapped_column(String(255))
    display_name:Mapped[str] = mapped_column(String(64),default="")
    token_version: Mapped[int] = mapped_column(Integer,default=0,server_default="0")
    # 角色：'user' 普通用户 / 'admin' 管理员（可跨用户查看会话与反馈）。
    # 只能由后台/种子数据设置，**注册接口永远不接受这个字段**（见 auth_service.register——
    # 允许客户端指定角色等于让所有人一键自封管理员）。
    role: Mapped[str] = mapped_column(String(16),default="user",server_default="user")
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.now)

class Conversation(Base):
    """会话表：一个 session_id 代表一个对话。

    user_id 是数据隔离的依据：侧边栏只列当前用户的会话，且按 session_id 取历史时也要校验归属。
    声明为 Optional 是为了与"给已有表加列"的 ALTER（只能先加 NULL 列）保持一致，
    非空约束由 DAO 层保证（create_conversation 必须传 user_id）。

    转人工（handoff_status / agent_id）是**会话级**状态，不是消息级：
    一个会话在任一时刻只会处于"机器人应答 / 等待坐席 / 坐席处理中 / 已关闭"之一，
    用状态机表达比在每条消息上打标记更不容易出现自相矛盾的状态。
    """
    __tablename__ = "conversations"
    id:Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    session_id:Mapped[str] = mapped_column(String(64),unique=True,index=True)
    user_id:Mapped[Optional[int]] = mapped_column(Integer,index=True,nullable=True)
    user_name:Mapped[str] = mapped_column(String(64),default="游客")
    title: Mapped[str] = mapped_column(String(255),default="新会话")
    # bot / pending / assigned / closed，见 app/agent/state.py 的 HandoffStatus
    handoff_status: Mapped[str] = mapped_column(
        String(16), default="bot", server_default="bot", index=True
    )
    # 当前接待的坐席（users.id）；未接入时为 NULL。
    # 认领用"带条件的原子 UPDATE"实现，所以这个字段同时就是并发互斥的凭据。
    agent_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.now,onupdate=datetime.now)

class Message(Base):
    """消息表：会话内每条消息（user / assistant）。

    feedback 是用户对**助手回复**的评价：1=，-1=，NULL=未评价。
    用户消息不会有评价，所以这两个字段对 role='user' 的行恒为 NULL。
    反馈攒起来就是最有价值的东西——它是后面做评测集、判断"哪类问题答不好"的原始信号。
    """
    __tablename__ = "messages"
    __table_args__ = (Index("idx_session_created", "session_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64),index=True)
    role: Mapped[str] = mapped_column(String(20)) #role/assistant
    content: Mapped[str] = mapped_column(TEXT)
    # metadata 存 JSON：可记录意图、引用来源等结构化信息（如 sources）
    # 注意：Python 属性名不能用 metadata（SQLAlchemy 保留字），故用 meta
    meta:Mapped[Optional[dict]] = mapped_column("meta",JSON,nullable=True)
    feedback: Mapped[Optional[int]] = mapped_column(Integer,nullable=True)
    feedback_note: Mapped[Optional[str]] = mapped_column(String(255),nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.now)

class OrderInfo(Base):
    """订单表：演示业务工具查询的静态数据。"""

    __tablename__ = "order_info"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(32), unique=True)
    # 订单归属用户；user_name 保留作展示（下单人姓名与登录名可以不同）
    user_id: Mapped[Optional[int]] = mapped_column(Integer, index=True, nullable=True)
    user_name: Mapped[str] = mapped_column(String(64), index=True)
    product_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))  # 待支付/待发货/已发货/已完成/已退款
    amount: Mapped[Decimal] = mapped_column(Numeric(precision=10, scale=2))  # 金额（元）
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

class LogisticsInfo(Base):
    """物流表：订单的物流轨迹，一条记录 = 一个物流节点。"""

    __tablename__ = "logistics_info"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    node_time: Mapped[str] = mapped_column(String(64))  # 节点时间，如 "2026-09-01 10:30"
    description: Mapped[str] = mapped_column(String(255))  # 节点描述
    location: Mapped[str] = mapped_column(String(128))  # 所在城市

# ---------------- 引擎与会话工厂 ----------------

_engine=create_engine(
    settings.mysql_url,
    pool_size=10,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=settings.DEBUG
)

_SessionLocal=sessionmaker(bind=_engine,autoflush=False,expire_on_commit=False)

@contextmanager
def session_scope():
    """会话上下文管理器：自动提交/回滚/关闭，避免资源泄漏。"""
    session=_SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def ensure_database():
    """
    确保目标数据库存在：连接 MySQL 服务器（不指定库），
    若 MYSQL_DATABASE 不存在则自动创建（utf8mb4 支持中文）。
    企业实践：初始化脚本应"幂等"，可重复执行。
    """
    from sqlalchemy import create_engine as _create_engine, text as _text

    server_url=(
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/?charset=utf8mb4"
    )
    engine= _create_engine(server_url)
    try:
        with engine.connect() as conn:
            conn.execute(
                _text(
                    f"CREATE DATABASE IF NOT EXISTS `{settings.MYSQL_DATABASE}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            conn.commit()
        logger.info("数据库 %s 已就绪", settings.MYSQL_DATABASE)
    finally:
        engine.dispose()

def _ensure_column(table: str, column: str, ddl: str) -> bool:
    """
    幂等加列：information_schema 里查不到该列才执行 ALTER，返回是否真的执行了 DDL。

    为什么需要它：Base.metadata.create_all() 只建"不存在的表"，
    每次升级新增字段都必须显式 ALTER，否则老库永远缺列。
    """
    with _engine.connect() as conn:
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.columns "
                "WHERE table_schema = DATABASE() AND table_name = :t AND column_name = :c"
            ),
            {"t": table, "c": column},
        ).scalar()
        if exists:
            return False
        conn.execute(text(ddl))
        conn.commit()
    logger.info("已为表 %s 新增列 %s", table, column)
    return True


def purge_legacy_conversations() -> int:
    """
    清理"无主"旧会话：user_id 为空的行是加列之前遗留的数据，没有归属用户，无法安全展示给任何人。
    先删消息再删会话（messages 按 session_id 关联，没有外键约束，不受删除顺序影响，
    但先删子表语义更清晰）。返回清理的会话数。
    """
    with session_scope() as s:
        orphan_ids = [
            row[0]
            for row in s.query(Conversation.session_id)
            .filter(Conversation.user_id.is_(None))
            .all()
        ]
        if not orphan_ids:
            return 0
        s.query(Message).filter(Message.session_id.in_(orphan_ids)).delete(
            synchronize_session=False
        )
        deleted = (
            s.query(Conversation)
            .filter(Conversation.user_id.is_(None))
            .delete(synchronize_session=False)
        )
    logger.info("已清理 %s 个无归属的历史会话（旧数据无 user_id）", deleted)
    return deleted


def ping() -> bool:
    """
    轻量探活：只执行 SELECT 1，不做任何 DDL / 数据变更。
    """
    try:
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.warning("MySQL 探活失败: %s", e)
        return False


def init_db():
    """建库 → 建表 → 补列 → 清理无主旧数据 → 写入演示账号/订单（幂等，可重复执行）。"""
    ensure_database()
    Base.metadata.create_all(_engine)

    # 给已存在的库补上新增列（新库由 create_all 直接建好，这两条会自然跳过）
    _ensure_column(
        "conversations",
        "user_id",
        "ALTER TABLE conversations ADD COLUMN user_id INT NULL, ADD INDEX idx_conv_user (user_id)",
    )
    _ensure_column(
        "order_info",
        "user_id",
        "ALTER TABLE order_info ADD COLUMN user_id INT NULL, ADD INDEX idx_order_user (user_id)",
    )
    # 令牌吊销：已存在的库里补上版本列，存量用户默认 0（DEFAULT 让已有行直接可读，
    # 不需要额外的回填 UPDATE）
    _ensure_column(
        "users",
        "token_version",
        "ALTER TABLE users ADD COLUMN token_version INT NOT NULL DEFAULT 0",
    )
    # 角色分级：存量用户一律是普通用户（DEFAULT 'user' 直接让已有行可读）
    _ensure_column(
        "users",
        "role",
        "ALTER TABLE users ADD COLUMN role VARCHAR(16) NOT NULL DEFAULT 'user'",
    )
    # 转人工：会话级状态 + 接待坐席
    _ensure_column(
        "conversations",
        "handoff_status",
        "ALTER TABLE conversations ADD COLUMN handoff_status VARCHAR(16) NOT NULL DEFAULT 'bot', "
        "ADD INDEX idx_conv_handoff (handoff_status)",
    )
    _ensure_column(
        "conversations",
        "agent_id",
        "ALTER TABLE conversations ADD COLUMN agent_id INT NULL, "
        "ADD INDEX idx_conv_agent (agent_id)",
    )
    # 答案反馈：可空，未评价即 NULL
    _ensure_column(
        "messages", "feedback", "ALTER TABLE messages ADD COLUMN feedback INT NULL"
    )
    _ensure_column(
        "messages",
        "feedback_note",
        "ALTER TABLE messages ADD COLUMN feedback_note VARCHAR(255) NULL",
    )

    # 隔离上线：旧会话没有归属用户，直接清空（产品决策）
    purge_legacy_conversations()

    # 演示账号先于订单创建，seed_demo_data 才能把订单挂到对应用户名下
    seed_demo_users()
    seed_demo_data()

    logger.info("MySQL 表结构与演示数据初始化完成")

# ---------------- 用户 DAO ----------------

def create_user(username: str, password_hash: str, display_name: str = "") -> User:
    """新建用户；用户名重复会抛 IntegrityError（services 层已提前查重并给出友好提示）。

    **刻意不接受 role 参数**：这个函数是注册接口的唯一落库出口，一旦开放角色入参，
    注册请求就能自封管理员。角色只能由种子数据或后台直接改库设置。
    """
    with session_scope() as s:
        user = User(
            username=username,
            password_hash=password_hash,
            display_name=display_name or username,
        )
        s.add(user)
        return user

def get_user_by_username(username: str) -> Optional[User]:
    """按登录名查用户；不存在返回 None。"""
    with session_scope() as s:
        return s.query(User).filter(User.username == username).first()

def get_user_by_id(user_id: int) -> Optional[User]:
    """按主键查用户；不存在返回 None。"""
    with session_scope() as s:
        return s.query(User).filter(User.id == user_id).first()

def bump_token_version(user_id: int) -> Optional[int]:
    """
    令牌版本 +1（= 吊销该用户全部已签发令牌），返回新版本号；用户不存在返回 None。

    这里用的是 SQL 的原子自增，而不是"先查出来 +1 再写回"：后者在并发下会丢更新
    （两个请求同时读到 5、都写 6，等于只加了一次，于是就有一批本该失效的令牌继续能用）。
    """
    with session_scope() as s:
        affected = (
            s.query(User)
            .filter(User.id == user_id)
            .update({User.token_version: User.token_version + 1}, synchronize_session=False)
        )
        if not affected:
            return None
        return s.query(User.token_version).filter(User.id == user_id).scalar()

def delete_user(user_id: int) -> int:
    """
    连根删除一个用户：先删其名下所有消息与会话，再删用户本身。
    返回删除的会话数。

    主要用于测试账号收尾（见 scripts/smoke_test.py）——冒烟测试每次都会注册一个
    一次性账号，不清理的话演示库里会越积越多无名账号。
    """
    with session_scope() as s:
        session_ids = [
            row[0]
            for row in s.query(Conversation.session_id)
            .filter(Conversation.user_id == user_id)
            .all()
        ]
        if session_ids:
            s.query(Message).filter(Message.session_id.in_(session_ids)).delete(
                synchronize_session=False
            )
        deleted = (
            s.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .delete(synchronize_session=False)
        )
        s.query(User).filter(User.id == user_id).delete(synchronize_session=False)
        return deleted

def delete_conversation(session_id: str) -> bool:
    """删除一个会话及其全部消息；会话不存在返回 False。同样用于测试收尾。"""
    with session_scope() as s:
        conv = s.query(Conversation).filter(Conversation.session_id == session_id).first()
        if conv is None:
            return False
        s.query(Message).filter(Message.session_id == session_id).delete(
            synchronize_session=False
        )
        s.delete(conv)
        return True

# ---------------- 会话 DAO ----------------

def create_conversation(
    session_id: str, user_id: int, user_name: str = "游客", title: str = "新会话"
) -> Conversation:
    """新建一个会话记录（user_id 必传，归属关系不可缺省）。"""
    with session_scope() as s:
        conv = Conversation(
            session_id=session_id, user_id=user_id, user_name=user_name, title=title
        )
        s.add(conv)
        return conv

def get_conversation(session_id: str, user_id: Optional[int] = None) -> Optional[Conversation]:
    """
    按 session_id 查询会话；不存在返回 None。

    传入 user_id 时同时校验归属：会话不属于该用户时同样返回 None，
    调用方据此拒绝访问（防止拿到 session_id 就能读别人对话的越权）。
    """
    with session_scope() as s:
        q = s.query(Conversation).filter(Conversation.session_id == session_id)
        if user_id is not None:
            q = q.filter(Conversation.user_id == user_id)
        return q.first()

def list_conversations(user_id: int, limit: int = 20) -> List[Conversation]:
    """按更新时间倒序返回"该用户"最近的会话列表（前端侧边栏用）。

    必须带 user_id 过滤——这是多租户隔离的核心，漏掉就会把所有人的会话都列出来。
    """
    with session_scope() as s:
        return (
            s.query(Conversation)
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .all()
        )

def list_conversations_by_user_ids(user_ids: List[int], limit: int = 50) -> List[Conversation]:
    """返回指定的多个用户的会话（管理员总览下钻用）。

    只接受显式的 user_id 列表，不提供"不传就查全表"的默认行为：
    这种"省略参数 = 全量"的口子一旦存在，早晚会被某次误调用变成越权查询。
    """
    if not user_ids:
        return []
    with session_scope() as s:
        return (
            s.query(Conversation)
            .filter(Conversation.user_id.in_(user_ids))
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .all()
        )

def set_handoff_status(
    session_id: str,
    status: str,
    agent_id: Optional[int] = None,
    from_statuses: Optional[List[str]] = None,
) -> bool:
    """
    更新会话的转人工状态。返回是否真的改到了行。

    传 from_statuses 时是"带条件的更新"（CAS）：只有当前状态在允许集合里才生效。
    这是防止状态被并发改乱的关键——例如两个坐席同时认领，只有一个人能成功。
    """
    with session_scope() as s:
        q = s.query(Conversation).filter(Conversation.session_id == session_id)
        if from_statuses:
            q = q.filter(Conversation.handoff_status.in_(from_statuses))
        affected = q.update(
            {Conversation.handoff_status: status, Conversation.agent_id: agent_id},
            synchronize_session=False,
        )
        return bool(affected)

def list_conversations_by_status(statuses: List[str], limit: int = 50) -> List[Conversation]:
    """按转人工状态列出会话（坐席的待接入队列用），按最后更新倒序。"""
    with session_scope() as s:
        return (
            s.query(Conversation)
            .filter(Conversation.handoff_status.in_(statuses))
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .all()
        )

def list_conversations_by_agent(agent_id: int, statuses: List[str], limit: int = 50) -> List[Conversation]:
    """列出某坐席名下的会话。"""
    with session_scope() as s:
        return (
            s.query(Conversation)
            .filter(
                Conversation.agent_id == agent_id,
                Conversation.handoff_status.in_(statuses),
            )
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .all()
        )

def touch_conversation(session_id: str, title: str = None, user_id: Optional[int] = None) -> None:
    """更新会话的更新时间（每次问答后调用）；传 title 时一并更新标题。

    传 user_id 时只更新属于该用户的会话，避免越权改写他人会话。
    """
    with session_scope() as s:
        q = s.query(Conversation).filter(Conversation.session_id == session_id)
        if user_id is not None:
            q = q.filter(Conversation.user_id == user_id)
        conv = q.first()
        if conv:
            conv.updated_at = datetime.now()
            if title:
                # 列宽 String(255)，留出余地但不再用早期硬编码的 50
                conv.title = title[:200]

def count_messages(session_id: str, role: Optional[str] = None) -> int:
    """统计会话内消息条数（可按 role 过滤）。用于判断"这是不是首轮提问"。"""
    with session_scope() as s:
        q = s.query(Message).filter(Message.session_id == session_id)
        if role:
            q = q.filter(Message.role == role)
        return q.count()

# ---------------- 消息 DAO ----------------

def add_message(session_id: str, role: str, content: str, meta: Optional[dict] = None) -> Message:
    """
    向会话写入一条消息（role: user / assistant）。

    注意参数名是 meta 而不是 metadata：SQLAlchemy 的 DeclarativeBase 自带一个类级
    `metadata` 属性，若形参也叫 metadata，传进来会被静默 setattr 成未映射的实例属性
    （不报错，但数据永远不落库）。ORM 上的 JSON 列因此也命名为 meta。
    这里直接把 dict 交给 JSON 列，不要自己 json.dumps（否则会二次编码成字符串）。
    """
    with session_scope() as s:
        msg=Message(
            session_id=session_id,
            role=role,
            content=content,
            meta=meta,
        )
        s.add(msg)
        return msg

def get_recent_messages(session_id: str, limit: int = 10) -> List[Message]:
    """返回某会话最近 N 条消息（用于多轮记忆，取时间倒序后翻转）。"""
    with session_scope() as s:
        rows=(
            s.query(Message)
            .filter(Message.session_id==session_id)
            .order_by(Message.id.desc())
            .limit(limit)
            .all()
        )
        return list(reversed(rows))

def set_message_feedback(
    message_id: int,
    user_id: int,
    feedback: Optional[int],
    note: Optional[str] = None,
) -> bool:
    """
    给某条助手消息写评价（1=/ -1=/ None=取消评价）。

    归属校验直接写进 UPDATE 的 WHERE 里（messages 关联 conversations 再比对 user_id）：
    用一条语句同时完成"校验 + 写入"，比"先查出来判断、再更新"更不容易被绕过——
    后者中间多出一个可以忘记判断的分支。返回是否真的改到了行。
    """
    with session_scope() as s:
        session_ids = [
            row[0]
            for row in s.query(Conversation.session_id).filter(Conversation.user_id == user_id).all()
        ]
        if not session_ids:
            return False
        affected = (
            s.query(Message)
            .filter(
                Message.id == message_id,
                Message.session_id.in_(session_ids),
                # 只允许评价助手回复：给用户自己的提问点赞没有意义
                Message.role == "assistant",
            )
            .update(
                {Message.feedback: feedback, Message.feedback_note: note},
                synchronize_session=False,
            )
        )
        return bool(affected)

def update_password_hash(user_id: int, password_hash: str) -> bool:
    """
    更新口令哈希（用于把旧格式的哈希就地升级为新算法，见 auth_service.authenticate）。
    返回是否真的改到了行。
    """
    with session_scope() as s:
        affected = (
            s.query(User)
            .filter(User.id == user_id)
            .update({User.password_hash: password_hash}, synchronize_session=False)
        )
        return bool(affected)

def get_message(message_id: int) -> Optional[Message]:
    """按主键取消息（仅供管理员下钻等只读场景使用，调用方须自行确认权限）。"""
    with session_scope() as s:
        return s.query(Message).filter(Message.id == message_id).first()

# ---------------- 管理员统计 ----------------

def user_feedback_stats() -> List[Dict]:
    """
    按用户汇总：会话数、消息数、数、数。供管理员总览页使用。

    用一次 GROUP BY 查询取全量，而不是"先列用户再逐个用户查三遍"（N+1）。
   /只统计助手消息上的评价。
    """
    with session_scope() as s:
        conversations = dict(
            s.query(Conversation.user_id, func.count(Conversation.id))
            .group_by(Conversation.user_id)
            .all()
        )
        messages = dict(
            s.query(Conversation.user_id, func.count(Message.id))
            .join(Message, Message.session_id == Conversation.session_id)
            .group_by(Conversation.user_id)
            .all()
        )
        thumbs = dict(
            (
                (user_id, feedback),
                count,
            )
            for user_id, feedback, count in s.query(
                Conversation.user_id, Message.feedback, func.count(Message.id)
            )
            .join(Message, Message.session_id == Conversation.session_id)
            .filter(Message.feedback.isnot(None))
            .group_by(Conversation.user_id, Message.feedback)
            .all()
        )

        result = []
        for user in s.query(User).order_by(User.id.asc()).all():
            result.append(
                {
                    "user_id": user.id,
                    "username": user.username,
                    "display_name": user.display_name or user.username,
                    "role": user.role or "user",
                    "sessions": int(conversations.get(user.id, 0)),
                    "messages": int(messages.get(user.id, 0)),
                    "up": int(thumbs.get((user.id, 1), 0)),
                    "down": int(thumbs.get((user.id, -1), 0)),
                }
            )
        return result

# ---------------- 订单/物流 DAO（供 Agent 工具调用） ----------------

def query_order_by_id(order_id: str) -> Optional[OrderInfo]:
    """按订单号查询订单。"""
    with session_scope() as s:
        return s.query(OrderInfo).filter(OrderInfo.order_id == order_id).first()


def query_orders_by_user(user_id: int) -> List[OrderInfo]:
    """按用户 id 查询其全部订单。

    身份必须来自登录态（由工具通过 InjectedState 注入），不能由模型从问句里猜——
    否则用户说一句"查张三的订单"就能越权看到别人的订单。
    """
    with session_scope() as s:
        return (
            s.query(OrderInfo)
            .filter(OrderInfo.user_id == user_id)
            .order_by(OrderInfo.created_at.desc())
            .all()
        )


def query_logistics(order_id: str) -> List[LogisticsInfo]:
    """按订单号查询物流轨迹（按时间升序）。"""
    with session_scope() as s:
        return (
            s.query(LogisticsInfo)
            .filter(LogisticsInfo.order_id == order_id)
            .order_by(LogisticsInfo.node_time.asc())
            .all()
        )

# ---------------- 测试数据初始化 ----------------

# 演示账号：用户名 -> 展示名。密码统一为 DEMO_PASSWORD。
# 这些账号同时是演示订单的归属人（见 seed_demo_data 里的回填逻辑）。
DEMO_USERS = [
    ("张三", "张三"),
    ("李四", "李四"),
    ("王五", "王五"),
    # 坐席账号：转人工功能需要有人"接入"，单独一个账号比让普通用户兼任更清楚
    ("客服小李", "客服小李"),
]
DEMO_PASSWORD = "123456"

# 演示角色分配：三种角色各一个账号，方便分别登录体验。
# 张三同时是订单归属人，用同一个账号就能演示"我的会话"和"运营总览"；
# 李四/王五保持普通用户，用来验证隔离对普通用户依然严格；
# 客服小李是坐席，只会看到等待接入的会话与自己名下的会话。
DEMO_ADMIN_USERNAME = "张三"
DEMO_AGENT_USERNAME = "客服小李"


def seed_demo_users() -> None:
    """写入演示账号（幂等：已存在的用户名跳过），并修正演示账号的角色。"""
    from app.services.auth_service import hash_password  # 延迟导入，避免与 services 层循环依赖

    created = 0
    with session_scope() as s:
        existing = {row[0] for row in s.query(User.username).all()}
        for username, display_name in DEMO_USERS:
            if username in existing:
                continue
            s.add(
                User(
                    username=username,
                    password_hash=hash_password(DEMO_PASSWORD),
                    display_name=display_name,
                    # 角色在种子里显式指定；注册接口永远走默认的 'user'
                    role=_demo_role(username),
                )
            )
            created += 1

        # 已存在的演示账号也要补上角色：否则在"角色功能上线前"就已经建好的库里，
        # 张三会一直停留在 user（进不去运营总览）、客服小李也当不上坐席。
        promoted = 0
        for username in (DEMO_ADMIN_USERNAME, DEMO_AGENT_USERNAME):
            promoted += (
                s.query(User)
                .filter(User.username == username, User.role != _demo_role(username))
                .update({User.role: _demo_role(username)}, synchronize_session=False)
            )
    if created:
        logger.info("已创建 %s 个演示账号（密码均为 %s）", created, DEMO_PASSWORD)
    else:
        logger.info("演示账号已存在，跳过初始化")
    if promoted:
        logger.info("已修正演示账号的角色（管理员/坐席）")


def _demo_role(username: str) -> str:
    """演示账号 → 角色。只有这两个名字有特殊角色，其余一律普通用户。"""
    if username == DEMO_ADMIN_USERNAME:
        return "admin"
    if username == DEMO_AGENT_USERNAME:
        return "agent"
    return "user"


def seed_demo_data() -> None:
    """写入演示用的订单与物流数据，并把订单挂到对应用户名下（幂等，可重复执行）。"""
    with session_scope() as s:
        if s.query(OrderInfo).count() > 0:
            logger.info("订单演示数据已存在，跳过初始化")
        else:
            _insert_demo_orders(s)
            # 必须显式 flush：本工程 sessionmaker 设了 autoflush=False，
            # 否则下面那条 UPDATE 看不到刚 add 还没落库的订单，回填会全部落空。
            s.flush()

        # 回填/绑定订单归属：按"下单人姓名 == 登录名"把订单挂到用户 id 上。
        # 这一步对"刚建的新库"和"加列之前就存在的老库"都有效，所以放在插入之外单独执行。
        bound = s.execute(
            text(
                "UPDATE order_info o JOIN users u ON o.user_name = u.username "
                "SET o.user_id = u.id WHERE o.user_id IS NULL"
            )
        ).rowcount
        if bound:
            logger.info("已为 %s 笔订单绑定所属用户", bound)


def _insert_demo_orders(s) -> None:
    """插入演示订单与物流节点（调用方保证只在订单表为空时执行）。"""
    orders = [
        OrderInfo(order_id="SO20260901001", user_name="张三", product_name="云翼智能手表 S1", status="运输中", amount=1299.0),
        OrderInfo(order_id="SO20260901002", user_name="张三", product_name="云翼降噪耳机 Pro", status="已签收", amount=699.0),
        OrderInfo(order_id="SO20260902001", user_name="李四", product_name="云翼家用路由器 AX3000", status="待发货", amount=399.0),
        OrderInfo(order_id="SO20260902002", user_name="李四", product_name="云翼 65W 氮化镓充电器", status="已退款", amount=129.0),
        OrderInfo(order_id="SO20260903001", user_name="王五", product_name="云翼 4K 高清摄像头", status="已完成", amount=459.0),
    ]
    s.add_all(orders)

    logistics = [
        LogisticsInfo(order_id="SO20260901001", node_time="2026-09-01 09:00", description="商家已发货", location="上海"),
        LogisticsInfo(order_id="SO20260901001", node_time="2026-09-02 15:30", description="到达转运中心", location="苏州"),
        LogisticsInfo(order_id="SO20260901001", node_time="2026-09-03 08:45", description="派送中", location="南通"),
        LogisticsInfo(order_id="SO20260901002", node_time="2026-08-28 10:00", description="已签收", location="南通"),
    ]
    s.add_all(logistics)
    logger.info("订单/物流演示数据初始化完成（5 单 + 4 条物流）")
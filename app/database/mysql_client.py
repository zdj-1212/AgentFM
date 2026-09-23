"""
MySQL 数据访问层（SQLAlchemy 2.0 ORM）
=======================================
职责：
1. 定义与 MySQL 表一一对应的 ORM 模型（User / Conversation / Message / OrderInfo / LogisticsInfo）；
2. 提供 DAO（数据访问对象）函数：用户、会话管理、消息持久化、订单查询等；
3. 提供测试数据初始化（scripts/init_db.py 会调用）；
4. 提供轻量幂等迁移（_ensure_column），保证已存在的库也能加上新列。

表设计说明：
- users          用户表：登录账号，passwords 以 PBKDF2 哈希存储（见 services/auth_service.py）
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
from typing import List,Optional
from decimal import Decimal

from sqlalchemy import JSON,DateTime,Index,Integer,String,TEXT,create_engine,text,Numeric
from sqlalchemy.orm import DeclarativeBase,Mapped,mapped_column,sessionmaker

from config.settings import settings
from app.core.logger import get_logger
logger=get_logger(__name__)

# ---------------- ORM 模型 ----------------

class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类。"""

class User(Base):
    """用户表：登录账号。

    口令只存 PBKDF2 哈希（格式 pbkdf2_sha256$迭代次数$盐$摘要），永不存明文。
    """
    __tablename__ = "users"
    id:Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    username:Mapped[str] = mapped_column(String(64),unique=True,index=True)
    password_hash:Mapped[str] = mapped_column(String(255))
    display_name:Mapped[str] = mapped_column(String(64),default="")
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.now)

class Conversation(Base):
    """会话表：一个 session_id 代表一个对话。

    user_id 是数据隔离的依据：侧边栏只列当前用户的会话，且按 session_id 取历史时也要校验归属。
    声明为 Optional 是为了与"给已有表加列"的 ALTER（只能先加 NULL 列）保持一致，
    非空约束由 DAO 层保证（create_conversation 必须传 user_id）。
    """
    __tablename__ = "conversations"
    id:Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    session_id:Mapped[str] = mapped_column(String(64),unique=True,index=True)
    user_id:Mapped[Optional[int]] = mapped_column(Integer,index=True,nullable=True)
    user_name:Mapped[str] = mapped_column(String(64),default="游客")
    title: Mapped[str] = mapped_column(String(255),default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(DateTime,default=datetime.now,onupdate=datetime.now)

class Message(Base):
    """消息表：会话内每条消息（user / assistant）。"""
    __tablename__ = "messages"
    __table_args__ = (Index("idx_session_created", "session_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64),index=True)
    role: Mapped[str] = mapped_column(String(20)) #role/assistant
    content: Mapped[str] = mapped_column(TEXT)
    # metadata 存 JSON：可记录意图、引用来源等结构化信息（如 sources）
    # 注意：Python 属性名不能用 metadata（SQLAlchemy 保留字），故用 meta
    meta:Mapped[Optional[dict]] = mapped_column("meta",JSON,nullable=True)
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

    # 隔离上线：旧会话没有归属用户，直接清空（产品决策）
    purge_legacy_conversations()

    # 演示账号先于订单创建，seed_demo_data 才能把订单挂到对应用户名下
    seed_demo_users()
    seed_demo_data()

    logger.info("MySQL 表结构与演示数据初始化完成")

# ---------------- 用户 DAO ----------------

def create_user(username: str, password_hash: str, display_name: str = "") -> User:
    """新建用户；用户名重复会抛 IntegrityError（services 层已提前查重并给出友好提示）。"""
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
]
DEMO_PASSWORD = "123456"


def seed_demo_users() -> None:
    """写入演示账号（幂等：已存在的用户名跳过）。"""
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
                )
            )
            created += 1
    if created:
        logger.info("已创建 %s 个演示账号（密码均为 %s）", created, DEMO_PASSWORD)
    else:
        logger.info("演示账号已存在，跳过初始化")


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
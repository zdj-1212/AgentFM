"""
MySQL 数据访问层（SQLAlchemy 2.0 ORM）
=======================================
职责：
1. 定义与 MySQL 表一一对应的 ORM 模型（Conversation / Message / OrderInfo / LogisticsInfo）；
2. 提供 DAO（数据访问对象）函数：会话管理、消息持久化、订单查询等；
3. 提供测试数据初始化（scripts/init_db.py 会调用）。

表设计说明（对应 scripts/init_db.sql 的语义）：
- conversations  会话表：一次前端对话窗口 = 一个 session_id
- messages       消息表：会话内每轮问答的 user/assistant 消息（支持多轮记忆）
- order_info     订单表：业务数据，供 Agent 通过工具查询（演示"业务工具调用"）
- logistics_info 物流表：订单的物流轨迹，供 Agent 查询"快递到哪了"
"""
import json
from contextlib import contextmanager
from datetime import datetime
from typing import List,Optional
from decimal import Decimal

from sqlalchemy import JSON,DateTime,Index,String,TEXT,create_engine,Numeric
from sqlalchemy.orm import DeclarativeBase,Mapped,mapped_column,sessionmaker

from config.settings import settings
from app.core.logger import get_logger
logger=get_logger(__name__)

# ---------------- ORM 模型 ----------------

class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式基类。"""

class Conversation(Base):
    """会话表：一个 session_id 代表一个对话。"""
    __tablename__ = "conversations"
    id:Mapped[int] = mapped_column(primary_key=True,autoincrement=True)
    session_id:Mapped[str] = mapped_column(String(64),unique=True,index=True)
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

def init_db():
    """建库（若不存在）并按 ORM 模型建表（若表不存在）。"""
    ensure_database()
    Base.metadata.create_all(_engine)
    logger.info("MySQL 表结构初始化完成（若不存在则创建）")

# ---------------- 会话 DAO ----------------

def create_conversation(session_id: str, user_name: str = "游客",title: str="新会话")->Conversation:
    """新建一个会话记录。"""
    with session_scope() as s:
        conv = Conversation(session_id=session_id,user_name=user_name,title=title)
        s.add(conv)
        return conv

def get_conversation(session_id: str) -> Optional[Conversation]:
    """按 session_id 查询会话；不存在返回 None。"""
    with session_scope() as s:
        return s.query(Conversation).filter(Conversation.session_id == session_id).first()

def list_conversations(limit: int = 20) -> List[Conversation]:
    """按更新时间倒序返回最近的会话列表（前端侧边栏用）。"""
    with session_scope() as s:
        return (
            s.query(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .all()
        )

def touch_conversation(session_id: str, title: str = None) -> None:
    """更新会话的更新时间（每次问答后调用）。"""
    with session_scope() as s:
        conv = s.query(Conversation).filter(Conversation.session_id == session_id).first()
        if conv:
            conv.updated_at = datetime.now()
            if title:
                conv.title = title[:50]

# ---------------- 消息 DAO ----------------

def add_message(session_id: str, role: str, content: str, metadata: Optional[dict] = None) -> Message:
    """向会话写入一条消息（role: user / assistant）。"""
    with session_scope() as s:
        msg=Message(
            session_id=session_id,
            role=role,
            content=content,
            metadata=json.dumps(metadata,ensure_ascii=False) if metadata else None,
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


def query_orders_by_user(user_name: str) -> List[OrderInfo]:
    """按用户名查询其全部订单。"""
    with session_scope() as s:
        return (
            s.query(OrderInfo)
            .filter(OrderInfo.user_name == user_name)
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

def seed_demo_data() -> None:
    """写入演示用的订单与物流数据（幂等：已存在则跳过）。"""
    with session_scope() as s:
        if s.query(OrderInfo).count() > 0:
            logger.info("订单演示数据已存在，跳过初始化")
            return

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
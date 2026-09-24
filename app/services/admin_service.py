"""
管理员服务（运营视角）
======================
这里集中放**能跨用户读数据**的能力：用户总览、任意用户的会话与消息、反馈汇总。

为什么单独开一个模块，而不是往 chat_service 里加几个方法：

1. **权限边界要能一眼审完。** chat_service 的每条路径都带 user_id 过滤，是"只看自己"的
   世界；admin_service 恰好相反，是"能看所有人"。把两种截然不同的信任级别混在同一个类里，
   迟早会有人复制一段代码却漏掉了权限检查。分开之后，"哪些能力需要管理员"是一份可以
   从头读到尾的清单。

2. **强制走同一个闸门。** 本模块的每个对外方法**第一行**都是 `require_admin(user)`。
   新增方法时照着写，漏了会非常显眼。

注意：本模块只读（统计/查询），不提供"代用户提问""改用户数据"这类写操作——
那类能力会显著放大管理员账号被拿下的后果，需要单独设计（审计日志等），本项目不做。
"""
from typing import Dict, List, Optional

from app.core.logger import get_logger
from app.database import mysql_client
from app.services.auth_service import require_admin

logger = get_logger(__name__)


def overview(user: Optional[Dict]) -> List[Dict]:
    """运营总览：每个用户的会话数、消息数、👍/👎 数（管理员可见）。"""
    require_admin(user)
    return mysql_client.user_feedback_stats()


def user_sessions(user: Optional[Dict], target_user_id: int, limit: int = 50) -> List[Dict]:
    """查看**指定用户**的会话列表（管理员下钻用）。"""
    require_admin(user)
    return [
        {
            "session_id": c.session_id,
            "user_id": c.user_id,
            "title": c.title,
            "updated_at": c.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for c in mysql_client.list_conversations_by_user_ids([target_user_id], limit)
    ]


def session_messages(user: Optional[Dict], session_id: str) -> List[Dict]:
    """
    查看任意会话的消息（管理员用）。

    这里**刻意不做归属校验**——跨用户查看正是这个接口存在的意义。
    但因为它绕过了隔离，所以入口必须先过 require_admin；
    而且返回内容里不带任何令牌/口令类字段，只有消息本身。
    """
    require_admin(user)

    conversation = mysql_client.get_conversation(session_id)
    if conversation is None:
        raise ValueError("会话不存在")

    return [
        {
            "message_id": m.id,
            "role": m.role,
            "content": m.content,
            "feedback": m.feedback,
            "feedback_note": m.feedback_note,
            "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for m in mysql_client.get_recent_messages(session_id, limit=100)
    ]


def feedback_summary(user: Optional[Dict]) -> Dict:
    """反馈汇总：整体 👍/👎 比例与"最近被踩的回复"，用于定位答得不好的地方。"""
    require_admin(user)

    rows = mysql_client.user_feedback_stats()
    up = sum(r["up"] for r in rows)
    down = sum(r["down"] for r in rows)
    total = up + down
    return {
        "up": up,
        "down": down,
        "total": total,
        # 没有任何评价时不要给出 0.0，那是"全被踩"的歧义表达
        "satisfaction": round(up / total, 4) if total else None,
    }

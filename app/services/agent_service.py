"""
坐席服务（转人工的人工侧）
==========================
用户请求转人工后，由**坐席**（role='agent'）接入会话并以人工身份回复。
本模块集中放坐席的全部能力：待接入队列、认领、回复、结束。

与 admin_service 的关系（两者刻意分开）：

- `admin_service` 是**只读**的运营视角，能看到所有人的数据，但**不能代替坐席说话**；
- `agent_service` 能写（回复用户），但**只能碰自己名下的会话**。

为什么不让 admin 兼任坐席：管理员掌握的是查看与分析能力，坐席掌握的是对用户说话的能力。
合成一个角色后，一个被盗的管理员账号就能直接向任意用户发消息，风险面明显更大。
所以本模块的每个对外方法第一行都是 `require_agent(user)`，**不放过管理员**。

并发
----
认领用"带条件的原子 UPDATE"（见 mysql_client.set_handoff_status 的 from_statuses）实现：
两个坐席同时点接入，只有一个人能改到行。这样就不用引入分布式锁，
也不会出现"两个坐席都在回复同一个人"的尴尬。
"""
from typing import Dict, List, Optional

from app.agent.state import HandoffStatus
from app.core.logger import get_logger
from app.database import mysql_client
from app.services.auth_service import require_agent

logger = get_logger(__name__)

# 人工回复在 messages.role 里用 'agent'，与机器人的 'assistant' 区分开。
# 前端靠它决定气泡上写"客服"还是"坐席 xxx"，也靠它决定要不要显示/
# （对人工回复打分没有意义，打分的对象是机器人的回答质量）。
AGENT_MESSAGE_ROLE = "agent"


def _summary(conv) -> Dict:
    """把会话行转成队列里要展示的字段。"""
    return {
        "session_id": conv.session_id,
        "user_id": conv.user_id,
        "user_name": conv.user_name,
        "title": conv.title,
        "handoff_status": conv.handoff_status or HandoffStatus.BOT.value,
        "agent_id": conv.agent_id,
        "updated_at": conv.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


def queue(user: Optional[Dict]) -> Dict:
    """
    坐席的工作台数据：待接入队列 + 我名下的会话。

    一次返回两组而不是两个接口，是因为坐席界面本来就要同时展示它们——
    分成两次请求只会让"认领后列表没刷新"这类问题更容易出现。
    """
    require_agent(user)
    agent_id = user["user_id"]
    pending = mysql_client.list_conversations_by_status([HandoffStatus.PENDING.value])
    mine = mysql_client.list_conversations_by_agent(
        agent_id, [HandoffStatus.ASSIGNED.value]
    )
    return {
        "pending": [_summary(c) for c in pending],
        "mine": [_summary(c) for c in mine],
    }


def claim(user: Optional[Dict], session_id: str) -> Dict:
    """
    认领一个待接入会话。已在自己名下则幂等返回。

    并发安全：更新条件里带了 `handoff_status == 'pending'`，
    所以两个人同时认领时只有一个人会改到行；另一个会拿到 409 语义（这里抛 PermissionError）。
    """
    require_agent(user)
    agent_id = user["user_id"]

    conv = mysql_client.get_conversation(session_id)
    if conv is None:
        raise ValueError("会话不存在")

    if (conv.handoff_status or HandoffStatus.BOT.value) == HandoffStatus.ASSIGNED.value:
        if conv.agent_id == agent_id:
            return _summary(conv)  # 自己已经接了，重复点击无害
        raise PermissionError("该会话已被其他坐席接入")

    ok = mysql_client.set_handoff_status(
        session_id,
        HandoffStatus.ASSIGNED.value,
        agent_id=agent_id,
        from_statuses=[HandoffStatus.PENDING.value],
    )
    if not ok:
        # 条件更新没命中：在这两步之间状态被改了（被别人抢走，或用户/系统改了状态）
        raise PermissionError("该会话已被其他坐席接入或状态已变更")

    logger.info("[坐席] %s 接入会话 %s", user["username"], session_id)
    refreshed = mysql_client.get_conversation(session_id)
    return _summary(refreshed)


def reply(user: Optional[Dict], session_id: str, content: str) -> Dict:
    """
    坐席回复：以人工身份落库一条 role='agent' 的消息。

    只能回复**自己名下**且处于 assigned 的会话——否则坐席 A 能往坐席 B 正在处理的
    会话里插话，用户会看到两个"客服"各说各的。
    """
    require_agent(user)
    agent_id = user["user_id"]
    content = (content or "").strip()
    if not content:
        raise ValueError("回复内容不能为空")

    conv = mysql_client.get_conversation(session_id)
    if conv is None:
        raise ValueError("会话不存在")
    if conv.handoff_status != HandoffStatus.ASSIGNED.value or conv.agent_id != agent_id:
        raise PermissionError("该会话不在你名下，无法回复")

    msg = mysql_client.add_message(session_id, role=AGENT_MESSAGE_ROLE, content=content)
    mysql_client.touch_conversation(session_id, user_id=conv.user_id)
    logger.info("[坐席] %s 回复了会话 %s", user["username"], session_id)

    return {
        "session_id": session_id,
        "message_id": msg.id,
        "agent_name": user.get("display_name") or user.get("username", ""),
        "created_at": msg.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


def close(user: Optional[Dict], session_id: str) -> Dict:
    """
    结束本次人工服务，把会话交回机器人（状态置为 closed）。

    只有会话归属者能关。closed 与 bot 一样允许机器人作答（见 BOT_ANSWER_STATUSES），
    区别只在语义：closed 表示"刚结束一轮人工服务"，便于运营侧统计。
    """
    require_agent(user)
    agent_id = user["user_id"]

    conv = mysql_client.get_conversation(session_id)
    if conv is None:
        raise ValueError("会话不存在")
    if conv.agent_id != agent_id:
        raise PermissionError("该会话不在你名下")

    ok = mysql_client.set_handoff_status(
        session_id,
        HandoffStatus.CLOSED.value,
        agent_id=agent_id,
        from_statuses=[HandoffStatus.ASSIGNED.value],
    )
    if not ok:
        raise PermissionError("会话状态已变更，无法结束")

    logger.info("[坐席] %s 结束会话 %s 的人工服务", user["username"], session_id)
    return {"session_id": session_id, "handoff_status": HandoffStatus.CLOSED.value}


def session_messages(user: Optional[Dict], session_id: str) -> List[Dict]:
    """
    坐席查看会话消息。只允许看自己名下的会话，或还在队列里等待接入的会话——
    坐席需要先看几眼对话内容才能判断要不要接。
    """
    require_agent(user)
    agent_id = user["user_id"]

    conv = mysql_client.get_conversation(session_id)
    if conv is None:
        raise ValueError("会话不存在")

    status = conv.handoff_status or HandoffStatus.BOT.value
    allowed = (status == HandoffStatus.PENDING.value) or (conv.agent_id == agent_id)
    if not allowed:
        raise PermissionError("无权查看该会话")

    return [
        {
            "message_id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for m in mysql_client.get_recent_messages(session_id, limit=100)
    ]

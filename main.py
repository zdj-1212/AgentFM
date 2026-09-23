"""
AgentFM 命令行对话入口
======================
最简单的人机交互方式：在终端里直接和 Agent 对话。
适合快速体验 / 联调，也适合在没有图形界面的服务器上运行。

启动后需要先登录（或注册），会话与订单数据都按登录账号隔离。

用法（项目根目录）：
    uv run python main.py
"""
import getpass
from typing import Dict,Optional

from app.core.logger import get_logger
from app.services.auth_service import AuthError, auth_service
from app.services.chat_service import chat_service

logger = get_logger(__name__)

# 意图 -> 中文名（展示用）
INTENT_NAMES = {
    "knowledge": "知识问答(RAG)",
    "order": "订单查询(工具Agent)",
    "chitchat": "闲聊",
    "general": "综合(兜底Agent)",
}


def login() -> Optional[Dict]:
    """
    终端登录流程：先尝试登录，失败后可选直接注册新账号。

    直接回车即使用演示账号（张三 / 123456），方便快速体验。
    返回用户信息字典；用户放弃登录时返回 None。
    """
    print("请输入登录信息（直接回车使用演示账号：张三 / 123456）")
    username = input("用户名: ").strip() or "张三"
    # getpass 不回显密码，避免明文留在终端历史里
    password = getpass.getpass("密码: ") or "123456"

    try:
        user = auth_service.authenticate(username, password)
        print(f"[系统] 登录成功，欢迎 {user['display_name']}！\n")
        return user
    except AuthError as e:
        print(f"[系统] {e}")

    if input(f"是否用「{username}」注册一个新账号？(y/N): ").strip().lower() == "y":
        try:
            user = auth_service.register(username, password)
            print(f"[系统] 注册成功，欢迎 {user['display_name']}！")
            print("[提示] 演示订单属于 张三 / 李四 / 王五，新账号查不到订单属正常现象。\n")
            return user
        except AuthError as e:
            print(f"[系统] 注册失败：{e}")

    return None


def main() -> None:
    print("=" * 50)
    print("  AgentFM 企业智能客服 · 命令行演示")
    print("  输入 'exit' 或 'quit' 退出；输入 'new' 开启新会话")
    print("=" * 50)

    user = login()
    if user is None:
        print("[系统] 未登录，已退出。")
        return

    user_id, user_name = user["user_id"], user["display_name"]
    session_id = chat_service.create_session(user_id, user_name)
    print(f"[系统] 已创建会话：{session_id}\n")

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[系统] 再见！")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("[系统] 再见！")
            break
        if user_input.lower() == "new":
            session_id = chat_service.create_session(user_id, user_name)
            print(f"[系统] 已开启新会话：{session_id}")
            continue
        if not user_input:
            continue

        try:
            result = chat_service.ask(session_id, user_input, user_id, user_name)
            print(f"[意图] {INTENT_NAMES.get(result['intent'], result['intent'])}")
            print(f"客服 > {result['reply']}")
            if result.get("sources"):
                print("[来源] " + "、".join(s["title"] for s in result["sources"]))
            print("-" * 50)
        except Exception as e:
            logger.exception("对话失败")
            print(f"[系统] 出错了：{e}\n")


if __name__ == "__main__":
    main()

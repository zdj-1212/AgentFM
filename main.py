"""
AgentFM 命令行对话入口
======================
最简单的人机交互方式：在终端里直接和 Agent 对话。
适合快速体验 / 联调，也适合在没有图形界面的服务器上运行。

用法（项目根目录）：
    uv run python main.py
"""
from app.core.logger import get_logger
from app.services.chat_service import chat_service

logger = get_logger(__name__)

# 意图 -> 中文名（展示用）
INTENT_NAMES = {
    "knowledge": "知识问答(RAG)",
    "order": "订单查询(工具Agent)",
    "chitchat": "闲聊",
    "general": "综合(兜底Agent)",
}


def main() -> None:
    print("=" * 50)
    print("  AgentFM 企业智能客服 · 命令行演示")
    print("  输入 'exit' 或 'quit' 退出；输入 'new' 开启新会话")
    print("=" * 50)

    session_id = chat_service.create_session(user_name="CLI用户")
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
            session_id = chat_service.create_session(user_name="CLI用户")
            print(f"[系统] 已开启新会话：{session_id}")
            continue
        if not user_input:
            continue

        try:
            result = chat_service.ask(session_id, user_input)
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

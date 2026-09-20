"""
AgentFM 网页对话入口（Streamlit）
=================================
面向业务人员/用户的图形化界面：
- 左侧：会话管理（新建会话、历史会话列表）
- 主区：类微信的对话气泡 + 引用来源展示

启动方式（项目根目录）：
    uv run streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# 把项目根目录加入 sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import streamlit as st

from app.services.chat_service import chat_service

# ---------- 页面基础配置 ----------
st.set_page_config(page_title="AgentFM 企业智能客服", page_icon="🤖", layout="wide")

# 意图中文名
INTENT_NAMES = {
    "knowledge": "📚 知识问答",
    "order": "📦 订单查询",
    "chitchat": "💬 闲聊",
    "general": "🧭 综合问答",
}


# ---------- 初始化会话状态 ----------
def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = chat_service.create_session(user_name="游客")
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content, intent, sources}]


init_state()


# ---------- 侧边栏：会话管理 ----------
with st.sidebar:
    st.title("🤖 AgentFM")
    st.caption("企业级多智能体客服（LangGraph + RAG + Milvus + MySQL）")

    if st.button("🆕 新建会话", use_container_width=True):
        st.session_state.session_id = chat_service.create_session(user_name="游客")
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("历史会话")
    sessions = chat_service.list_sessions(limit=15)
    for s in sessions:
        # 点击会话 -> 切换并回显历史
        if st.button(s["title"], key=s["session_id"], use_container_width=True):
            st.session_state.session_id = s["session_id"]
            history = chat_service.get_history(s["session_id"])
            st.session_state.messages = [
                {
                    "role": h["role"],
                    "content": h["content"],
                }
                for h in history
            ]
            st.rerun()

    st.divider()
    st.caption("当前会话: " + st.session_state.session_id[:8])


# ---------- 主区：对话气泡 ----------
st.title("💬 企业智能客服")

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 引用来源"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['title']}**（{src['source']}，相似度 {src['score']}）")

# 输入框
if prompt := st.chat_input("请输入你的问题，例如：七天无理由退货条件是什么？/ 我的订单到哪里了？"):
    # 1) 展示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2) 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                result = chat_service.ask(st.session_state.session_id, prompt)
            except Exception as e:
                result = {"reply": f"出错了：{e}", "intent": "error", "sources": []}

        st.markdown(result["reply"])
        st.caption(f"🛰️ 路由：{INTENT_NAMES.get(result['intent'], result['intent'])}")
        if result.get("sources"):
            with st.expander("📎 引用来源"):
                for src in result["sources"]:
                    st.markdown(f"- **{src['title']}**（{src['source']}，相似度 {src['score']}）")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["reply"],
            "intent": result["intent"],
            "sources": result.get("sources", []),
        }
    )

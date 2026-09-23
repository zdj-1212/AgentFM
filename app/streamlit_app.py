"""
AgentFM 网页对话入口（Streamlit）
=================================
面向业务人员/用户的图形化界面：
- 登录闸门：未登录只能看到登录/注册页，登录后所有数据按用户隔离
- 左侧：当前用户、会话管理（新建会话、历史会话列表）
- 主区：类微信的对话气泡 + 引用来源展示

历史会话的标题来自该会话**首轮提问的概括**（由 chat_service 生成），
因此侧边栏可以直接看出每个会话聊的是什么，而不是清一色的"新会话"。

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

from app.services.auth_service import AuthError, auth_service
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


# ---------- 登录 / 注册页 ----------
def render_auth_page() -> None:
    """未登录时展示的登录/注册界面。"""
    st.title("🤖 AgentFM 企业智能客服")
    st.caption("请先登录。每个账号只能看到自己的会话与订单数据。")

    login_tab, register_tab = st.tabs(["登录", "注册"])

    with login_tab:
        # 用 st.form 把输入框包起来：只有点"登录"按钮才会触发一次 rerun，
        # 否则每次敲键都会重跑脚本（甚至误触发登录逻辑）。
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            submitted = st.form_submit_button("登录", type="primary", width="stretch")

        if submitted:
            try:
                st.session_state.user = auth_service.authenticate(username, password)
            except AuthError as e:
                st.error(str(e))
            else:
                _reset_user_state()
                st.rerun()

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("用户名", placeholder="2~32 位中文、字母、数字或下划线")
            new_password = st.text_input("密码", type="password", placeholder="至少 6 位")
            confirm_password = st.text_input("确认密码", type="password", placeholder="再输入一次")
            registered = st.form_submit_button("注册并登录", type="primary", width="stretch")

        if registered:
            if new_password != confirm_password:
                st.error("两次输入的密码不一致")
            else:
                try:
                    st.session_state.user = auth_service.register(new_username, new_password)
                except AuthError as e:
                    st.error(str(e))
                else:
                    _reset_user_state()
                    # 提示语放到下一轮渲染，避免被紧随其后的 rerun 冲掉
                    st.session_state.flash = (
                        "✅ 注册成功，已自动登录。"
                        "（演示订单属于 张三 / 李四 / 王五，新账号暂时查不到订单，属正常现象）"
                    )
                    st.rerun()

    st.divider()
    st.caption("演示账号：张三 / 李四 / 王五，密码均为 123456")


def _reset_user_state() -> None:
    """切换用户时清掉上一个用户残留的会话与消息，避免前端串号。"""
    st.session_state.session_id = None
    st.session_state.messages = []


# ---------- 登录闸门 ----------
# 必须在任何业务调用之前：下面的 init_state() 会写库建会话，
# 未登录就执行等于给匿名访问者建数据。
if "user" not in st.session_state:
    render_auth_page()
    st.stop()

current_user = st.session_state.user
USER_ID = current_user["user_id"]
DISPLAY_NAME = current_user["display_name"]


# ---------- 初始化会话状态 ----------
def init_state() -> None:
    if not st.session_state.get("session_id"):
        st.session_state.session_id = chat_service.create_session(USER_ID, DISPLAY_NAME)
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content, intent, sources}]


init_state()

# 登录/注册成功后的提示
if flash := st.session_state.pop("flash", None):
    st.success(flash)


# ---------- 侧边栏：用户信息 + 会话管理 ----------
with st.sidebar:
    st.title("🤖 AgentFM")
    st.caption("企业级多智能体客服（LangGraph + RAG + Milvus + MySQL）")
    st.caption(f"👤 当前用户：{DISPLAY_NAME}")

    if st.button("🆕 新建会话", width="stretch"):
        st.session_state.session_id = chat_service.create_session(USER_ID, DISPLAY_NAME)
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("历史会话")
    sessions = chat_service.list_sessions(USER_ID, limit=15)
    if not sessions:
        st.caption("暂无历史会话，去提问吧～")
    for s in sessions:
        # 标题 = 该会话首轮提问的概括（见 chat_service._summarize_title）
        is_current = s["session_id"] == st.session_state.session_id
        label = f"{'▸ ' if is_current else ''}{s['title']}"
        # 点击会话 -> 切换并回显历史
        if st.button(label, key=s["session_id"], width="stretch", help=f"最后更新：{s['updated_at']}"):
            st.session_state.session_id = s["session_id"]
            try:
                history = chat_service.get_history(s["session_id"], USER_ID)
            except PermissionError as e:
                st.error(str(e))
            else:
                st.session_state.messages = [
                    {
                        "role": h["role"],
                        "content": h["content"],
                    }
                    for h in history
                ]
            st.rerun()

    st.divider()
    if st.button("退出登录", width="stretch"):
        # 必须整体清空：残留的 session_id / messages 会让下一个登录者看到上一个用户的数据
        st.session_state.clear()
        st.rerun()

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
                result = chat_service.ask(
                    st.session_state.session_id, prompt, USER_ID, DISPLAY_NAME
                )
            except PermissionError as e:
                result = {"reply": f"⛔ {e}", "intent": "error", "sources": []}
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

    # 3) 首轮提问会生成会话标题，刷新侧边栏让新标题立刻可见
    st.rerun()

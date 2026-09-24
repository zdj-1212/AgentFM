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

from app.core.logger import get_logger
from app.services import admin_service
from app.services.auth_service import AuthError, auth_service
from app.services.chat_service import chat_service

logger = get_logger(__name__)

# ---------- 页面基础配置 ----------
st.set_page_config(page_title="AgentFM 企业智能客服", layout="wide")

# 意图中文名
INTENT_NAMES = {
    "knowledge": "知识问答",
    "order": "订单查询",
    "chitchat": "闲聊",
    "general": "综合问答",
}

# 降级提示：节点失败时回复都是同一句兜底话术，光看那句话分不清是哪个依赖出了问题。
# 这里按 error_code 给出各自的说明，让用户知道该重试还是该报修。
ERROR_HINTS = {
    "knowledge_unavailable": "⚠️ 知识库暂时不可用，本条为降级回复。请稍后重试。",
    "order_unavailable": "⚠️ 订单查询服务暂时不可用，本条为降级回复。请稍后重试。",
    "general_unavailable": "⚠️ 智能客服暂时不可用，本条为降级回复。请稍后重试。",
    "chitchat_unavailable": "⚠️ 智能客服暂时不可用，本条为降级回复。请稍后重试。",
}


# ---------- 登录 / 注册页 ----------
def render_auth_page() -> None:
    """未登录时展示的登录/注册界面。"""
    st.title("AgentFM 企业智能客服")
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
    st.title("AgentFM")
    st.caption("企业级多智能体客服（LangGraph + RAG + Milvus + MySQL）")
    is_admin = auth_service.is_admin(current_user)
    st.caption(f" 当前用户：{DISPLAY_NAME}" + ("　 管理员" if is_admin else ""))

    if st.button("新建会话", width="stretch"):
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
                        "message_id": h["message_id"],
                        "feedback": h["feedback"],
                    }
                    for h in history
                ]
            st.rerun()

    st.divider()
    if st.button(
        "退出登录",
        width="stretch",
        help="退出后该账号已签发的所有登录令牌都会失效，其它设备需重新登录",
    ):
        # 先吊销服务端令牌，再清本地状态。
        # 网页端本身不发令牌（直接走服务层），所以这一步不是给自己用的——
        # 它是为了让"退出登录"对 REST API 那边签发的令牌同样生效，
        # 否则网页上点了退出，之前发出的令牌还能继续用到过期为止。
        try:
            auth_service.revoke_tokens(USER_ID)
        except Exception as e:
            logger.warning("吊销令牌失败（不阻断退出）: %s", e)
        # 必须整体清空：残留的 session_id / messages 会让下一个登录者看到上一个用户的数据
        st.session_state.clear()
        st.rerun()

    st.caption("当前会话: " + st.session_state.session_id[:8])


# ---------- 主区：对话气泡 ----------

FEEDBACK_LABELS = {1: "👍 已赞", -1: "👎 已踩", None: "未评价"}


def _save_feedback(msg: dict, feedback, note: str = "") -> None:
    """写评价并同步本地状态，然后重跑让按钮状态立刻更新。"""
    mid = msg.get("message_id")
    if mid is None:
        st.warning("这条回复尚未落库，暂时无法评价")
        return
    try:
        chat_service.set_feedback(
            st.session_state.session_id, mid, USER_ID, feedback, note
        )
    except (PermissionError, ValueError) as e:
        st.error(str(e))
        return
    msg["feedback"] = feedback
    if note:
        msg["feedback_note"] = note
    st.rerun()


def _render_feedback(msg: dict) -> None:
    """在助手回复下方渲染 👍/👎；点踩后可补一句原因（这一步是选填）。"""
    if msg.get("message_id") is None:
        return

    current = msg.get("feedback")
    col_up, col_down, col_state = st.columns([1, 1, 5])
    if col_up.button(
        "👍", key=f"fbup_{msg['message_id']}",
        type="primary" if current == 1 else "secondary",
        help="这条回答有帮助",
    ):
        _save_feedback(msg, 1 if current != 1 else None)
    if col_down.button(
        "👎", key=f"fbdown_{msg['message_id']}",
        type="primary" if current == -1 else "secondary",
        help="这条回答没帮助，可再补充原因",
    ):
        _save_feedback(msg, -1 if current != -1 else None)
    col_state.caption(f"反馈：{FEEDBACK_LABELS.get(current, '未评价')}")

    # 点了踩才展开原因框：form 里的输入不会每敲一个字就重跑整个页面
    if current == -1:
        with st.form(key=f"fbnote_{msg['message_id']}", border=False):
            note = st.text_input(
                "哪里不对？（选填）",
                value=msg.get("feedback_note") or "",
                key=f"fbtxt_{msg['message_id']}",
            )
            if st.form_submit_button("保存原因"):
                _save_feedback(msg, -1, note.strip())


def _render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("error_code"):
            st.warning(ERROR_HINTS.get(msg["error_code"], "⚠️ 本条为降级回复。"))
        if msg.get("sources"):
            with st.expander("📎 引用来源"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['title']}**（{src['source']}，相似度 {src['score']}）")
        if msg["role"] == "assistant":
            _render_feedback(msg)


def render_chat() -> None:
    """我的对话：历史气泡 + 输入框。"""
    for msg in st.session_state.messages:
        _render_message(msg)

    if not (prompt := st.chat_input(
        "请输入你的问题，例如：七天无理由退货条件是什么？/ 我的订单到哪里了？"
    )):
        return

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
        if result.get("error_code"):
            st.warning(ERROR_HINTS.get(result["error_code"], "⚠️ 本条为降级回复。"))
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
            "error_code": result.get("error_code"),
            "message_id": result.get("message_id"),
            "feedback": None,
        }
    )

    # 3) 首轮提问会生成会话标题，刷新侧边栏让新标题立刻可见
    st.rerun()


def render_admin_console() -> None:
    """
    运营工作台：所有用户的会话/消息/评价一览，可下钻到某个用户的会话与消息。

    注意这里的每一次数据获取都要经过 admin_service 的 require_admin 闸门——
    UI 上"只有管理员看得到这个页签"不算保护，服务层必须独立再判一次。
    """
    st.subheader("运营总览")

    try:
        stats = admin_service.overview(current_user)
        summary = admin_service.feedback_summary(current_user)
    except PermissionError as e:
        st.error(str(e))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("用户数", len(stats))
    c2.metric("会话数", sum(r["sessions"] for r in stats))
    c3.metric("消息数", sum(r["messages"] for r in stats))
    c4.metric(
        "好评率",
        f"{summary['satisfaction'] * 100:.0f}%" if summary["satisfaction"] is not None else "暂无评价",
        help=f"👍 {summary['up']} / 👎 {summary['down']}",
    )

    if not stats:
        st.info("还没有任何用户数据")
        return

    st.caption("点击任意一行下钻查看该用户的会话与消息")
    event = st.dataframe(
        [
            {
                "用户": r["display_name"],
                "角色": "管理员" if r["role"] == "admin" else "普通用户",
                "会话数": r["sessions"],
                "消息数": r["messages"],
                "👍": r["up"],
                "👎": r["down"],
            }
            for r in stats
        ],
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="admin_overview_table",
    )

    picked = event.selection.rows if event and event.selection else []
    if not picked:
        return

    target = stats[picked[0]]
    if not target["sessions"]:
        st.info(f"{target['display_name']} 还没有任何会话")
        return

    st.divider()
    st.subheader(f" {target['display_name']} 的会话")
    try:
        sessions = admin_service.user_sessions(current_user, target["user_id"])
    except PermissionError as e:
        st.error(str(e))
        return

    labels = {s["session_id"]: f"{s['title']}（{s['updated_at']}）" for s in sessions}
    chosen = st.selectbox(
        "选择一个会话查看消息",
        options=list(labels),
        format_func=lambda sid: labels[sid],
        key="admin_session_pick",
    )
    try:
        messages = admin_service.session_messages(current_user, chosen)
    except (PermissionError, ValueError) as e:
        st.error(str(e))
        return

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                mark = FEEDBACK_LABELS.get(m["feedback"], "未评价")
                note = f"｜原因：{m['feedback_note']}" if m["feedback_note"] else ""
                st.caption(f"{m['created_at']}　反馈：{mark}{note}")


# ---------- 主区 ----------
st.title(" 企业智能客服")

# 管理员多一个"运营工作台"页签；普通用户界面完全不变。
# 角色来自服务端会话信息，不来自任何前端输入。
if is_admin:
    tab_chat, tab_admin = st.tabs([" 对话", " 运营工作台"])
    with tab_chat:
        render_chat()
    with tab_admin:
        render_admin_console()
else:
    render_chat()

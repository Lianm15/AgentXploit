import streamlit as st
from api_client import ApiClient
import time


def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title="AgentXploit", layout="wide")

client = ApiClient()

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "start_time" not in st.session_state:
    st.session_state.start_time = None

if "end_time" not in st.session_state:
    st.session_state.end_time = None


# load correct css
if st.session_state.session_id is None:
    load_css("styles/home.css")
else:
    load_css("styles/chat.css")


# ==========================
# HOME PAGE
# ==========================

if st.session_state.session_id is None:

    st.markdown(
        '<h1 class="page-title">AgentXploit</h1>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="page-subtitle">Configure your jailbreak testing parameters</p>',
        unsafe_allow_html=True
    )

    try:
        models = client.get_models()
    except Exception:
        st.error("Backend not reachable.")
        st.stop()

    selected_model = st.selectbox("Target Model", models)

    success_criteria = st.text_area("Success Criteria")

    max_attempts = st.number_input(
        "Maximum Attempts",
        min_value=1,
        value=50
    )

    if st.button("Start Test"):

        if not success_criteria.strip():
            st.warning("Enter success criteria")

        else:

            session_id = client.initialize(
                selected_model,
                success_criteria,
                max_attempts,
            )

            client.start_attack(session_id)

            st.session_state.session_id = session_id
            st.session_state.start_time = time.time()
            st.session_state.end_time = None

            st.rerun()


# ==========================
# TEST PAGE
# ==========================

else:

    st.markdown(
        '<div class="chat-header-title">AgentXploit</div>',
        unsafe_allow_html=True
    )

    session_id = st.session_state.session_id

    status_response = client.get_status(session_id)
    status = status_response["status"]

    # save finish time
    if status in ["finished", "failed", "success_found"]:
        if st.session_state.end_time is None:
            st.session_state.end_time = time.time()

    # timer calculation
    if st.session_state.end_time is not None:
        elapsed = int(st.session_state.end_time - st.session_state.start_time)
    else:
        elapsed = int(time.time() - st.session_state.start_time)

    col1, col2, col3, col4 = st.columns([2,2,2,2])

    with col1:
        st.markdown(
            '<div class="brand-name">AgentXploit</div>',
            unsafe_allow_html=True
        )

    with col2:

        color = {
            "running": ("rgba(34,197,94,0.15)", "#22c55e"),
            "paused": ("rgba(234,179,8,0.15)", "#eab308"),
            "finished": ("rgba(99,102,241,0.15)", "#6366f1"),
            "failed": ("rgba(239,68,68,0.15)", "#ef4444")
        }

        bg, border = color.get(status, ("rgba(255,255,255,0.1)", "#aaa"))

        st.markdown(
            f"""
            <div class="status-pill"
            style="--pill-bg:{bg}; --pill-border:{border}; --pill-color:{border}">
            {status.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <span class="stat-label">Session</span>
            <span class="stat-value">{session_id[:8]}</span>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <span class="stat-label">Elapsed</span>
            <span class="stat-value">{elapsed}s</span>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ==========================
    # CONTROL BUTTONS
    # ==========================

    if status not in ["finished", "failed", "success_found"]:

        left, right = st.columns([8,2])

        with right:

            c1, c2, c3 = st.columns(3)

            with c1:
                if st.button("Pause"):
                    client.session_control(session_id, "pause")
                    st.rerun()

            with c2:
                if st.button("Resume"):
                    client.session_control(session_id, "resume")
                    st.rerun()

            with c3:
                if st.button("Stop"):
                    client.session_control(session_id, "stop")
                    st.rerun()

    st.divider()

    # ==========================
    # CHAT TRANSCRIPT
    # ==========================

    transcript_response = client.get_transcript(session_id)

    if isinstance(transcript_response, dict) and "transcript" in transcript_response:
        transcript = transcript_response["transcript"]
    else:
        transcript = transcript_response

    st.markdown('<div class="chat-area">', unsafe_allow_html=True)

    if not transcript:

        st.markdown(
            '<div class="empty-chat">Waiting for messages...</div>',
            unsafe_allow_html=True
        )

    else:

        for msg in transcript:

            sender = msg["sender"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")

            if sender == "attacker":
                avatar_class = "avatar avatar-ax"
                name = "AgentXploit"
                avatar_text = "AX"

            elif sender == "target":
                avatar_class = "avatar avatar-ai"
                name = "Target Model"
                avatar_text = "AI"

            else:
                avatar_class = "avatar avatar-ai"
                name = "Judge"
                avatar_text = "J"

            st.markdown(f"""
        <div class="msg-card">

        <div class="msg-header">
        <div class="{avatar_class}">
        {avatar_text}
        </div>

        <div class="msg-meta">
        <span class="msg-name">{name}</span>
        <span class="msg-time">{timestamp}</span>
        </div>
        </div>

        <div class="msg-body">
        {content}
        </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ==========================
    # FINISH BUTTON
    # ==========================

    if status in ["finished", "failed", "success_found"]:

        left, center, right = st.columns([3,2,3])

        with center:

            if st.button(
                "Finish Test",
                type="primary",
                use_container_width=True
            ):
                st.session_state.session_id = None
                st.session_state.start_time = None
                st.session_state.end_time = None
                st.rerun()

    # ==========================
    # AUTO REFRESH
    # ==========================

    if status not in ["finished", "failed", "success_found"]:
        time.sleep(1)
        st.rerun()
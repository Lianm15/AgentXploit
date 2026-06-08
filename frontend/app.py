import streamlit as st
from api_client import ApiClient
import time
import html
import re


def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(page_title="AgentXploit", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stHeader"], header  { display: none !important; }
    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"]      { display: none !important; }
    [data-testid="stDeployButton"]    { display: none !important; }
    [data-testid="stToolbar"]         { display: none !important; }
    #MainMenu                         { display: none !important; }
    section[data-testid="stMain"] > div { padding-top: 0 !important; }
    </style>
""",
    unsafe_allow_html=True,
)

client = ApiClient()


def normalize_message_content(content: str) -> str:
    """Prepare model output so HTML payloads render instead of showing as raw code blocks."""
    if not content:
        return ""

    # Unescape HTML entities
    normalized = html.unescape(content).strip()

    # Remove code fence wrappers (```html ... ```)
    fence_match = re.search(
        r"```(?:html)?\s*(.*?)\s*```", normalized, flags=re.IGNORECASE | re.DOTALL
    )
    if fence_match:
        return fence_match.group(1).strip()

    return normalized


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

    _, nav_col = st.columns([8, 2])
    with nav_col:
        st.page_link("pages/stats.py", label="Statistics")

    st.markdown(
        '<h1 class="page-title">Agent<span>Xploit</span></h1>', unsafe_allow_html=True
    )

    st.markdown(
        '<p class="page-subtitle">Automated jailbreak &amp; adversarial testing for local models</p>',
        unsafe_allow_html=True,
    )

    try:
        models = client.get_models()
    except Exception:
        st.error("Backend not reachable.")
        st.stop()

    with st.container(border=True):

        st.markdown(
            '<div class="form-card-head">Test Configuration</div>',
            unsafe_allow_html=True,
        )

        selected_model = st.selectbox("Target model", models)

        success_criteria = st.text_area(
            "Success criteria",
            placeholder=(
                "Describe what a successful jailbreak looks like - e.g. "
                '"The model gives step-by-step instructions it would normally refuse."'
            ),
            height=130,
        )

        max_attempts = st.number_input("Maximum attempts", min_value=1, value=50)

        st.markdown('<div class="launch-wrapper">', unsafe_allow_html=True)
        start_clicked = st.button("Launch Test")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<p class="disclaimer">For authorized security research and model evaluation only.</p>',
        unsafe_allow_html=True,
    )

    if start_clicked:

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
        '<div class="chat-header-title">AgentXploit</div>', unsafe_allow_html=True
    )

    session_id = st.session_state.session_id

    status_response = client.get_status(session_id)
    status = status_response["status"]

    is_active = status in ["running", "paused"]
    is_finished = status in ["finished", "failed", "success_found"]

    # save finish time
    if status in ["finished", "failed", "success_found"]:
        if st.session_state.end_time is None:
            st.session_state.end_time = time.time()

    # timer calculation
    if st.session_state.end_time is not None:
        elapsed = int(st.session_state.end_time - st.session_state.start_time)
    else:
        elapsed = int(time.time() - st.session_state.start_time)

    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 4])

    with col1:
        st.markdown('<div class="brand-name">AgentXploit</div>', unsafe_allow_html=True)

    with col2:

        color = {
            "running": ("rgba(34,197,94,0.15)", "#22c55e"),
            "paused": ("rgba(234,179,8,0.15)", "#eab308"),
            "finished": ("rgba(99,102,241,0.15)", "#6366f1"),
            "failed": ("rgba(239,68,68,0.15)", "#ef4444"),
            "success_found": ("rgba(34,197,94,0.15)", "#22c55e"),
        }

        bg, border = color.get(status, ("rgba(255,255,255,0.1)", "#aaa"))

        st.markdown(
            f"""
            <div class="status-pill"
            style="--pill-bg:{bg}; --pill-border:{border}; --pill-color:{border}">
            {status.upper()}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <span class="stat-label">Session</span>
            <span class="stat-value">{session_id[:8]}</span>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <span class="stat-label">Elapsed</span>
            <span class="stat-value">{elapsed}s</span>
            """,
            unsafe_allow_html=True,
        )

    with col5:

        if is_active:

                b1, b2, b3 = st.columns(3)

                with b1:
                    if st.button("Pause", use_container_width=True):
                        client.session_control(session_id, "pause")
                        st.rerun()

                with b2:
                    if st.button("Resume", use_container_width=True):
                        client.session_control(session_id, "resume")
                        st.rerun()

                with b3:
                    if st.button("Stop", use_container_width=True):
                        client.session_control(session_id, "stop")
                        st.rerun()

        else:
            st.markdown(
                '<div style="height:38px;"></div>',
                unsafe_allow_html=True
            )
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
            unsafe_allow_html=True,
        )

    else:

        for msg in transcript:

            sender = msg["sender"]
            raw_content = msg["content"]
            timestamp = msg.get("timestamp", "")

            if sender == "attacker":
                avatar_class = "avatar avatar-ax"
                card_class = "msg-card msg-card-attacker"
                name = "AgentXploit"
                avatar_text = "AX"

            elif sender == "target":
                avatar_class = "avatar avatar-target"
                card_class = "msg-card msg-card-target"
                name = "Target Model"
                avatar_text = "AI"

            else:
                avatar_class = "avatar avatar-judge"
                card_class = "msg-card msg-card-judge"
                name = "Judge"
                avatar_text = "J"

            # Normalize and prepare content
            content = normalize_message_content(raw_content)

            # Render header and metadata
            header_html = f"""<div class="{card_class}">
<div class="msg-header">
<div class="{avatar_class}">{avatar_text}</div>
<div class="msg-meta">
<span class="msg-name">{name}</span>
<span class="msg-time">{timestamp}</span>
</div>
</div>
<div class="msg-body">"""

            st.markdown(header_html, unsafe_allow_html=True)
            st.markdown(content, unsafe_allow_html=True)
            st.markdown("</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ==========================
    # FINISH BUTTON
    # ==========================

    if is_finished:

        left, center, right = st.columns([3, 2, 3])

        with center:

            if st.button("Finish Test", type="primary", use_container_width=True):
                st.session_state.session_id = None
                st.session_state.start_time = None
                st.session_state.end_time = None
                st.rerun()

    # ==========================
    # AUTO REFRESH
    # ==========================

    if status in ["running", "paused"]:
        time.sleep(1)
        st.rerun()

"""Chat / live-testing page."""

import time

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import api_client
from components.message_card import render as render_card
from utils.css import load_css

# ── Status pill configuration ──────────────────────────────────────────────────
# Maps status → (emoji, label, bg-color, border-color, text-color)
_PILL: dict[str, tuple] = {
    "active":  ("🟢", "Testing Active",  "#f0fdf4", "#bbf7d0", "#15803d"),
    "paused":  ("🟡", "Testing Paused",  "#fefce8", "#fde047", "#92400e"),
    "stopped": ("🔴", "Testing Stopped", "#fef2f2", "#fca5a5", "#991b1b"),
    "idle":    ("⚪", "Idle",             "#f9fafb", "#e5e7eb", "#6b7280"),
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _elapsed() -> str:
    """Return a HH:MM:SS string of active (non-paused) elapsed time."""
    if st.session_state.start_time is None:
        return "00:00:00"

    deducted = st.session_state.total_paused_secs

    if st.session_state.status == "paused" and st.session_state.pause_start_time:
        s = int(st.session_state.pause_start_time - st.session_state.start_time - deducted)
    else:
        s = int(time.time() - st.session_state.start_time - deducted)

    s = max(s, 0)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _sync_messages() -> None:
    """
    Fetch the latest transcript from the backend and update session state.
    Silently no-ops if the backend is unreachable or the session is not active.
    """
    session_id = st.session_state.get("session_id")
    if not session_id or st.session_state.status != "active":
        return
    try:
        st.session_state.messages = api_client.get_messages(session_id)
    except Exception:
        pass  # keep showing last known messages; don't crash the UI


# ── Page ───────────────────────────────────────────────────────────────────────

def show() -> None:
    status = st.session_state.status

    # Auto-refresh every second only while the test is actively running.
    if status == "active":
        st_autorefresh(interval=1000, key="chat_refresh")

    # Fetch latest messages from backend on every active refresh.
    _sync_messages()

    # Load static CSS from files.
    load_css("styles/common.css", "styles/chat.css")

    # Inject the only dynamic piece: CSS variables for the status pill colors.
    _dot, _label, _bg, _border, _color = _PILL.get(status, _PILL["idle"])
    st.markdown(
        f"<style>:root {{"
        f"  --pill-bg: {_bg};"
        f"  --pill-border: {_border};"
        f"  --pill-color: {_color};"
        f"}}</style>",
        unsafe_allow_html=True,
    )

    model_name = st.session_state.target_model.split(" (")[0]
    msgs       = st.session_state.messages

    # ── Navbar ─────────────────────────────────────────────────────────────────
    nav_l, nav_c, nav_r = st.columns([3, 5, 4])

    with nav_l:
        st.markdown(
            f'<div style="padding:0.4rem 0;">'
            f'  <span class="brand-name">AgentXploit</span>&nbsp;&nbsp;'
            f'  <span class="status-pill">{_dot} {_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with nav_c:
        st.markdown(
            f'<div style="display:flex;align-items:center;padding:0.3rem 0;">'
            f'  <div style="text-align:center;">'
            f'    <span class="stat-label">Target Model</span>'
            f'    <span class="stat-value">{model_name}</span>'
            f'  </div>'
            f'  <span class="stat-divider"></span>'
            f'  <div style="text-align:center;">'
            f'    <span class="stat-label">Elapsed Time</span>'
            f'    <span class="stat-value">{_elapsed()}</span>'
            f'  </div>'
            f'  <span class="stat-divider"></span>'
            f'  <div style="text-align:center;">'
            f'    <span class="stat-label">Messages</span>'
            f'    <span class="stat-value">{len(msgs)}</span>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with nav_r:
        # Button availability per status:
        #   active  → Pause ✓  Stop ✓
        #   paused  → Resume ✓  Stop ✓
        #   stopped → (none — user clicks Finish Test to configure a new run)
        dis_pause  = status != "active"
        dis_resume = status != "paused"
        dis_stop   = status in ("idle", "stopped")

        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("Pause",  key="btn_pause",  disabled=dis_pause,
                         use_container_width=True):
                st.session_state.status = "paused"
                st.session_state.pause_start_time = time.time()
                st.rerun()

        with b2:
            if st.button("Resume", key="btn_resume", disabled=dis_resume,
                         use_container_width=True):
                if st.session_state.pause_start_time:
                    st.session_state.total_paused_secs += (
                        time.time() - st.session_state.pause_start_time
                    )
                    st.session_state.pause_start_time = None
                st.session_state.status = "active"
                st.rerun()

        with b3:
            if st.button("Stop",   key="btn_stop",   disabled=dis_stop,
                         use_container_width=True):
                st.session_state.status = "stopped"
                st.rerun()

    st.markdown(
        '<hr style="margin:0;border-color:rgba(255,255,255,0.08);">',
        unsafe_allow_html=True,
    )

    # ── Message feed ───────────────────────────────────────────────────────────
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)

    if not msgs:
        st.markdown(
            '<div class="empty-chat">Waiting for test messages…</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in msgs:
            st.markdown(render_card(msg, model_name), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Finish Test ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    _, center_col, _ = st.columns([3, 2, 3])

    with center_col:
        if st.button(
            "Finish Test",
            key="btn_finish",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page       = "home"
            st.session_state.session_id = None
            st.session_state.messages   = []
            st.session_state.status     = "idle"
            st.session_state.start_time = None
            st.rerun()

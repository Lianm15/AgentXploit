"""Home / configuration page."""

import time

import httpx
import streamlit as st

import api_client
from state import MODELS
from utils.css import load_css


def show() -> None:
    load_css("styles/common.css", "styles/home.css")

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="header-badge"><span class="badge">AI Red-Teaming Suite</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="page-title">AgentXploit</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">Configure your jailbreak testing parameters</p>',
        unsafe_allow_html=True,
    )

    # ── Configuration form ────────────────────────────────────────────────────
    with st.form("config_form"):

        st.markdown('<p class="field-label">Target Model</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="field-desc">Select the AI model you want to test for vulnerabilities</p>',
            unsafe_allow_html=True,
        )
        target_model = st.selectbox(
            "Target Model",
            ["Select a model..."] + MODELS,
            label_visibility="collapsed",
        )

        st.markdown('<hr class="card-divider">', unsafe_allow_html=True)

        st.markdown('<p class="field-label">Success Criteria</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="field-desc">Define what constitutes a successful jailbreak attempt</p>',
            unsafe_allow_html=True,
        )
        success_criteria = st.text_area(
            "Success Criteria",
            placeholder=(
                "Example: Model generates harmful content, bypasses safety guidelines, "
                "reveals system prompts, produces unethical responses..."
            ),
            height=120,
            label_visibility="collapsed",
        )

        st.markdown('<hr class="card-divider">', unsafe_allow_html=True)

        st.markdown('<p class="field-label">Maximum Attempts</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="field-desc">Number of attack attempts per jailbreak category</p>',
            unsafe_allow_html=True,
        )
        max_attempts = st.number_input(
            "Maximum Attempts",
            min_value=1,
            max_value=500,
            value=50,
            step=1,
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Start Test")

    st.markdown(
        '<p class="disclaimer">Ensure you have authorization before testing any AI system</p>',
        unsafe_allow_html=True,
    )

    # ── Handle submission ─────────────────────────────────────────────────────
    if submitted:
        if target_model == "Select a model...":
            st.error("Please select a target model before starting the test.")
            return
        if not success_criteria.strip():
            st.error("Please define success criteria before starting the test.")
            return

        with st.spinner("Initializing session…"):
            try:
                session_id = api_client.initialize_session(
                    target_model, success_criteria, int(max_attempts)
                )
            except httpx.ConnectError:
                st.error(
                    "Cannot reach the backend. "
                    "Make sure the FastAPI server is running on http://localhost:8000"
                )
                return
            except Exception as exc:
                st.error(f"Failed to initialize session: {exc}")
                return

        st.session_state.session_id        = session_id
        st.session_state.target_model      = target_model
        st.session_state.success_criteria  = success_criteria
        st.session_state.max_attempts      = int(max_attempts)
        st.session_state.start_time        = time.time()
        st.session_state.status            = "active"
        st.session_state.total_paused_secs = 0.0
        st.session_state.pause_start_time  = None
        st.session_state.messages          = []
        st.session_state.page              = "chat"
        st.rerun()

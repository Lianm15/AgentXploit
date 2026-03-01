"""
Central session-state definitions and initialization.

Import ``init_session_state`` in app.py and call it once at startup.
Other modules import ``MODELS`` as needed.
"""

import streamlit as st

# ── Available target models ────────────────────────────────────────────────────
MODELS: list[str] = [
    "GPT-4o (OpenAI)",
    "GPT-4 Turbo (OpenAI)",
    "GPT-3.5 Turbo (OpenAI)",
    "Claude 3.5 Sonnet (Anthropic)",
    "Claude 3 Opus (Anthropic)",
    "Claude 3 Haiku (Anthropic)",
    "Gemini 1.5 Pro (Google)",
    "Gemini 1.5 Flash (Google)",
    "Llama 3 70B (Meta)",
    "Mistral Large (Mistral AI)",
    "Mixtral 8x7B (Mistral AI)",
    "Command R+ (Cohere)",
]

# ── Session-state schema with defaults ────────────────────────────────────────
# Each message in ``messages`` is a dict with keys:
#   sender:      "agent" | "model"
#   time:        str   (formatted, e.g. "2:34:12 PM")
#   text:        str
#   attack_type: str | None

_DEFAULTS: dict = {
    "page":               "home",   # "home" | "chat"
    "session_id":         None,     # str | None  — set after POST /api/initialize
    "target_model":       "",
    "success_criteria":   "",
    "max_attempts":       50,
    "start_time":         None,     # float | None  (time.time() at test start)
    "messages":           [],
    "status":             "idle",   # "idle" | "active" | "paused" | "stopped"
    "total_paused_secs":  0.0,      # accumulated seconds spent paused
    "pause_start_time":   None,     # float | None  (time.time() when paused)
}


def init_session_state() -> None:
    """Idempotently set all session-state keys to their default values."""
    for key, value in _DEFAULTS.items():
        st.session_state.setdefault(key, value)

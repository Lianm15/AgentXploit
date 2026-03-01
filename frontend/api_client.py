"""
HTTP client for the AgentXploit FastAPI backend.

All communication with the backend goes through this module so that the
views never import backend code directly and the base URL is configured
in one place.
"""

from datetime import datetime

import httpx

BACKEND_URL = "http://localhost:8000"

# ── Sender name mapping ────────────────────────────────────────────────────────
# Backend uses "gemini" / "target_llm"; the UI card component uses "agent" / "model".
_SENDER_MAP: dict[str, str] = {
    "gemini":     "agent",
    "target_llm": "model",
}


def _format_timestamp(raw: str) -> str:
    """Convert a SQLite timestamp string to a human-readable time string."""
    try:
        dt = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%-I:%M:%S %p")   # e.g. "2:34:07 PM"
    except (ValueError, TypeError):
        return raw


# ── Public API ─────────────────────────────────────────────────────────────────

def initialize_session(
    target_model: str,
    success_criteria: str,
    max_attempts: int,
) -> str:
    """
    Call POST /api/initialize and return the session_id string.

    Raises:
        httpx.ConnectError   – backend is not reachable
        httpx.HTTPStatusError – backend returned a non-2xx response
    """
    response = httpx.post(
        f"{BACKEND_URL}/api/initialize",
        json={
            "target_model":    target_model,
            "success_criteria": success_criteria,
            "max_attempts":    max_attempts,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()["session_id"]


def get_messages(session_id: str) -> list[dict]:
    """
    Call GET /api/{session_id}/messages and return a list of message dicts
    already formatted for the UI message-card component:

        {
            "sender":      "agent" | "model",
            "time":        str,        # e.g. "2:34:07 PM"
            "text":        str,
            "attack_type": None,       # populated by backend in future
        }

    Returns an empty list when the backend reports 404 (no messages yet).

    Raises:
        httpx.ConnectError    – backend is not reachable
        httpx.HTTPStatusError – backend returned a non-2xx / non-404 response
    """
    response = httpx.get(
        f"{BACKEND_URL}/api/{session_id}/messages",
        timeout=10.0,
    )

    if response.status_code == 404:
        return []

    response.raise_for_status()

    return [
        {
            "sender":      _SENDER_MAP.get(msg["sender"], "model"),
            "time":        _format_timestamp(msg["timestamp"]),
            "text":        msg["content"],
            "attack_type": msg.get("attack_type"),   # None until backend adds it
        }
        for msg in response.json()["transcript"]
    ]

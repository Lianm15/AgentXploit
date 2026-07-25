import os
import requests


class ApiClient:
    """Thin HTTP wrapper around the FastAPI backend - every Streamlit page
    talks to the backend only through this class, never via raw `requests` calls."""

    def __init__(self, base_url=None):
        self.base_url = (base_url or os.getenv("BACKEND_URL", "http://127.0.0.1:8000")).rstrip("/")

    def _request(self, method, path, **kwargs):
        """Send a request and raise on any non-2xx response.

        Centralizes URL-building and error handling so each public method below
        only needs to describe its own endpoint shape.
        """
        res = requests.request(method, f"{self.base_url}{path}", **kwargs)
        res.raise_for_status()
        return res

    def _request_allow_404(self, method, path, **kwargs):
        """Like `_request`, but returns None on 404 instead of raising.

        Session-scoped endpoints 404 legitimately when a session hasn't reached
        that state yet (e.g. no messages, no summary) - that's not an error.
        """
        res = requests.request(method, f"{self.base_url}{path}", **kwargs)
        if res.status_code == 404:
            return None
        res.raise_for_status()
        return res.json()

    def get_models(self):
        return self._request("GET", "/api/models").json()["models"]

    def initialize(self, target_model, success_criteria, max_attempts, mode="standard"):
        res = self._request(
            "POST",
            "/api/initialize",
            json={
                "target_model": target_model,
                "success_criteria": success_criteria,
                "max_attempts": max_attempts,
                "mode": mode,
            },
        )
        return res.json()["session_id"]

    def start_attack(self, session_id):
        self._request("POST", f"/api/{session_id}/start")

    def get_transcript(self, session_id):
        data = self._request_allow_404("GET", f"/api/{session_id}/messages")
        return data["transcript"] if data is not None else []

    def get_status(self, session_id: str):
        return self._request("GET", f"/api/{session_id}/status").json()

    def session_control(self, session_id, control):
        return self._request("POST", f"/api/{session_id}/control", json={"action": control}).json()

    def get_stats(self):
        return self._request("GET", "/api/stats").json()

    def get_intelligence(self, session_id: str):
        return self._request_allow_404("GET", f"/api/{session_id}/intelligence")

    def get_intelligence_summary(self):
        return self._request("GET", "/api/intelligence/summary").json()

    def get_priors(self, target_model: str):
        # {"priors": {technique: avg_compliance}, "session_count": N}
        return self._request("GET", f"/api/priors/{target_model}").json()

    def get_summary(self, session_id: str):
        return self._request_allow_404("GET", f"/api/{session_id}/summary")

    def get_scorer_status(self) -> dict:
        # Best-effort only: a cosmetic status badge, never worth failing the page load over.
        try:
            return self._request("GET", "/api/scorer/status", timeout=2).json()
        except Exception:
            return {"scorer": "unknown", "model": None}

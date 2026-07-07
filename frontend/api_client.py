import os
import requests

class ApiClient:
    def __init__(self, base_url=None):
        self.base_url = (base_url or os.getenv("BACKEND_URL", "http://127.0.0.1:8000")).rstrip("/")

    def get_models(self):
        res = requests.get(f"{self.base_url}/api/models")
        res.raise_for_status()
        return res.json()["models"]

    def initialize(self, target_model, success_criteria, max_attempts,mode="standard"):
        res = requests.post(
            f"{self.base_url}/api/initialize",
            json={
                "target_model": target_model,
                "success_criteria": success_criteria,
                "max_attempts": max_attempts,
                "mode": mode
            },
        )
        res.raise_for_status()
        return res.json()["session_id"]

    def start_attack(self, session_id):
        res = requests.post(f"{self.base_url}/api/{session_id}/start")
        res.raise_for_status()

    def get_transcript(self, session_id):
        res = requests.get(f"{self.base_url}/api/{session_id}/messages")
        if res.status_code == 404:
            return []
        res.raise_for_status()
        return res.json()["transcript"]

    def get_status(self, session_id: str):
        url = f"{self.base_url}/api/{session_id}/status"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def session_control(self, session_id, control):
        res = requests.post(
            f"{self.base_url}/api/{session_id}/control",
            json={"action": control}
        )
        res.raise_for_status()
        return res.json()

    def get_stats(self):
        res = requests.get(f"{self.base_url}/api/stats")
        res.raise_for_status()
        return res.json()

    def get_intelligence(self, session_id: str):
        res = requests.get(f"{self.base_url}/api/{session_id}/intelligence")
        if res.status_code == 404:
            return None
        res.raise_for_status()
        return res.json()

    def get_intelligence_summary(self):
        res = requests.get(f"{self.base_url}/api/intelligence/summary")
        res.raise_for_status()
        return res.json()

    def get_priors(self, target_model: str):
        res = requests.get(f"{self.base_url}/api/priors/{target_model}")
        res.raise_for_status()
        return res.json()  # {"priors": {technique: avg_compliance}, "session_count": N}

    def get_summary(self, session_id: str):
        res = requests.get(f"{self.base_url}/api/{session_id}/summary")
        if res.status_code == 404:
            return None
        res.raise_for_status()
        return res.json()

    def get_scorer_status(self) -> dict:
        try:
            res = requests.get(f"{self.base_url}/api/scorer/status", timeout=2)
            res.raise_for_status()
            return res.json()
        except Exception:
            return {"scorer": "unknown", "model": None}
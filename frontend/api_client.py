import requests

class ApiClient:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def get_models(self):
        res = requests.get(f"{self.base_url}/api/models")
        res.raise_for_status()
        return res.json()["models"]

    def initialize(self, target_model, success_criteria, max_attempts):
        res = requests.post(
            f"{self.base_url}/api/initialize",
            json={
                "target_model": target_model,
                "success_criteria": success_criteria,
                "max_attempts": max_attempts,
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

    def session_control(self, session_id, action):
        res = requests.post(
            f"{self.base_url}/api/{session_id}/control",
            json={"action": action}
        )
        res.raise_for_status()
        return res.json()
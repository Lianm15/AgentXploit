import base64
import os


def get_ollama_auth():
    username = os.getenv("OLLAMA_USERNAME", "").strip()
    password = os.getenv("OLLAMA_PASSWORD", "").strip()

    if username and password:
        return username, password

    return None


def get_ollama_headers():
    auth = get_ollama_auth()
    if not auth:
        return {}

    username, password = auth
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}

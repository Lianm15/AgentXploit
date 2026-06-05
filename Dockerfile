# Shared base: installs all Python dependencies.
# Both backend and frontend stages inherit from here — pip install runs only once.
FROM python:3.12-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Backend stage: FastAPI served by uvicorn ──────────────────────────────────
FROM base AS backend

COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Frontend stage: Streamlit web UI ─────────────────────────────────────────
FROM base AS frontend

COPY frontend/ .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
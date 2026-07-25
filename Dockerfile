# Shared base: installs all Python dependencies.
# Both backend and frontend stages inherit from here — pip install runs only once.
FROM python:3.12-slim AS base

WORKDIR /app

# Build tools needed by garak's transitive deps (tokenizers, sentencepiece, etc.)
# curl is needed to fetch rustup below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# base2048 (a PyRIT dependency) has no prebuilt wheel for this platform and
# compiles a Rust extension from source — Debian's packaged cargo/rustc are too
# old for it, so install a current toolchain via rustup instead.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model so the embedding scorer is always
# available at runtime without requiring outbound network access from the container.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

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
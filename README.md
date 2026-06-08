# AgentXploit

AgentXploit is a Python project with:
- a FastAPI backend (`backend/`)
- a Streamlit frontend (`frontend/`)

## Prerequisites

### Docker (recommended)

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- A Gemini API key (Google AI Studio)

### Manual setup

- Python 3.10+ (3.11 recommended)
- `pip`
- Ollama installed and running locally
- A Gemini API key

## Project Structure

```text
AgentXploit/
├── backend/
│   ├── main.py
│   ├── routes.py
│   ├── logic.py
│   ├── gemini.py
│   └── database.py
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── styles/
├── requirements.txt
└── README.md
```

## Run with Docker Compose (recommended)

Docker Compose starts all three services — Ollama, backend, and frontend — with a single command.

### 1) Create the environment file

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 2) Build and start all services

```bash
docker compose up --build
```

### How the startup sequence works

Docker Compose starts services in a strict order to guarantee the app is fully ready when the frontend opens:

```
1. ollama        → starts the local LLM server
2. ollama-init   → waits for ollama, then downloads all 5 target models
3. backend       → waits for ollama-init to finish, then starts the API
4. frontend      → waits for backend to be healthy, then opens the UI
```

**First run** downloads the Ollama image, builds the Python images, and pulls all 5 models.
This can take **5–20 minutes** depending on your internet speed (models are 1–4 GB each).

**Subsequent runs** skip the downloads — models are cached in the `ollama_data` Docker volume and startup takes only a few seconds.

To watch model download progress in real time:

```bash
docker compose logs -f ollama-init
```

### 3) Open the app

Once the frontend is ready, open:

| Service    | URL                             |
|------------|---------------------------------|
| Frontend   | http://localhost:8501           |
| Statistics | http://localhost:8501/stats     |
| Backend    | http://localhost:8000           |
| Swagger    | http://localhost:8000/docs      |

### Add more target models

To pull additional models into the running Ollama container:

```bash
docker compose exec ollama ollama pull <model-name>
```

Example: `docker compose exec ollama ollama pull llama3.1`

Browse available models at [ollama.com/library](https://ollama.com/library).

### Stop all services

```bash
docker compose down
```

> **Data persistence:** Session history (`db_data` volume) and downloaded models (`ollama_data` volume) survive restarts.
> To wipe everything and start fresh: `docker compose down -v`

---

## Manual Setup

Run these commands from the project root (`AgentXploit/`).

### 1) Create and activate virtual environment

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows (Command Prompt)
```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Create environment file for backend

Create a file at `backend/.env` with:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

## Run Manually (Use 2 Terminals)

You need **two separate terminals**: one for backend and one for frontend.

> In both terminals, activate the same virtual environment first.

Before starting backend/frontend, ensure Ollama is running.

## Required: Start Ollama

Start Ollama first:

```bash
ollama serve
```

Verify models are available:

```bash
ollama list
```

---

### Terminal 1: Backend (FastAPI)

```powershell
cd backend
..\.venv\Scripts\Activate.ps1
uvicorn main:app --port 8000 --reload
```

Backend URLs:
- API base: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

---

### Terminal 2: Frontend (Streamlit)

```powershell
cd frontend
..\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Frontend URL (default):
- `http://localhost:8501`

## Stop / Exit

- Stop servers: press `Ctrl + C` in each terminal
- Deactivate venv when done:

```bash
deactivate
```

## Common Issues

**Docker**
- Frontend not loading yet: the startup sequence is still running — models are still downloading. Run `docker compose logs -f ollama-init` to watch progress.
- Port already in use (`8000` or `8501`): another process is using that port. Stop it or change the host port in `docker-compose.yml` (e.g. `"8080:8000"`).
- Models missing after restart: volumes are intact unless you ran `docker compose down -v`. Run `docker compose logs ollama-init` to verify downloads completed.
- `GEMINI_API_KEY` not set: make sure you created a `.env` file from `.env.example` in the project root before running `docker compose up`.

**Manual / general**
- `503 UNAVAILABLE` from Gemini: temporary Gemini service overload; retry after a short delay.
- `Connection refused` for `127.0.0.1:8000`: backend is not running.
- Empty models list: Ollama is not running or no local models are installed.
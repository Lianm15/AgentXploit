# AgentXploit

AgentXploit is a Python project with:
- a FastAPI backend (`backend/`)
- a Streamlit frontend (`frontend/`)

## Prerequisites

Install these on your PC before running the project:

1. Python 3.10+ (3.11 recommended)
2. `pip`
3. Ollama (for local model listing via `http://localhost:11434`)
4. A Gemini API key (Google AI Studio)

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

## Setup

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

## Run the App (Use 2 Terminals)

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

- `503 UNAVAILABLE` from Gemini: temporary Gemini service overload; retry after a short delay.
- `Connection refused` for `127.0.0.1:8000`: backend is not running.
- Empty models list: Ollama is not running or no local models are installed.
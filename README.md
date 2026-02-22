# AgentXploit

A FastAPI-based application for agent-driven security exploitation and analysis.

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

## Setup Instructions

### 1. Create a Virtual Environment

#### On Windows (Command Prompt or PowerShell):
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Requirements

Once the virtual environment is activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

This will install:
- FastAPI
- Uvicorn (ASGI server)
- Pydantic
- Pydantic Settings
- Google Generative AI
- HTTPX

### 3. Configure Environment Variables

Create a `.env` file in the `app/` directory with any necessary environment variables (if not already present).

### 4. Run the FastAPI Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`

#### Useful flags:
- `--reload`: Auto-reload on code changes (development mode)
- `--host 0.0.0.0`: Listen on all network interfaces
- `--port 8000`: Specify a custom port

### 5. Access the API Documentation

Once the server is running, you can access:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Deactivating the Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

## Project Structure

```
AgentXploit/
├── app/
│   ├── .env           # Environment variables
│   ├── main.py        # FastAPI application entry point
│   ├── routes.py      # API routes
│   ├── logic.py       # Core logic
│   └── __pycache__/   # Python cache
├── requirements.txt   # Project dependencies
└── README.md         # This file
```
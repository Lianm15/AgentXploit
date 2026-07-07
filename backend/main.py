import os
from fastapi import FastAPI
from routes import router
from database import create_tables
from logic import HealthStatus
from fastapi.middleware.cors import CORSMiddleware
from scorer_instance import scorer as _scorer

app = FastAPI(title="AgentXploit", description="Automated jailbreak testing")

app.include_router(router)

create_tables()

@app.get("/", response_model=HealthStatus)
def health_check() -> HealthStatus:
    scorer_path = "embedding (all-MiniLM-L6-v2)" if _scorer._available else "heuristic (keyword fallback)"
    return HealthStatus(status=f"AgentXploit is running | scorer: {scorer_path}")

# CORS_ORIGINS can be a comma-separated list; defaults to the Streamlit frontend port.
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
from fastapi import FastAPI
from routes import router
from database import create_tables
from logic import HealthStatus
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AgentXploit", description="Automated jailbreak testing")

app.include_router(router)

create_tables()

@app.get("/", response_model=HealthStatus)
def health_check() -> HealthStatus:
    return HealthStatus(status="AgentXploit is running")

# CORS_ORIGINS can be a comma-separated list; defaults to the Streamlit frontend port.
_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

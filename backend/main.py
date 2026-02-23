from fastapi import FastAPI
from routes import router
from database import create_tables
from logic import HealthStatus
app = FastAPI(title="AgentXploit", description="Automated jailbreak testing")

app.include_router(router)

create_tables() 

@app.get("/", response_model=HealthStatus)
def health_check() -> HealthStatus:
    return HealthStatus(status="AgentXploit is running")
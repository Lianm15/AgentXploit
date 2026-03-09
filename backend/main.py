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

# Enable CORS: Allows the frontend at http://localhost:3000 to communicate with this API, permitting all methods and credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

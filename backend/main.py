from fastapi import FastAPI
from routes import router

app = FastAPI(title="AgentXploit", description="Automated jailbreak testing")

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "AgentXploit is running"}
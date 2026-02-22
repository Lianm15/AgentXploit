from fastapi import APIRouter, HTTPException
from logic import AttackConfig, AttackResult, process_attack, initialize_test
from pydantic import BaseModel
from typing import List   

router = APIRouter(prefix="/api") 

#the request body for initialize
class InitializeRequest(BaseModel):
    target_model: str
    success_criteria: List[str]
    max_attempts: int

@router.post("/test/initialize")
async def initialize(request: InitializeRequest):
    try:
        return initialize_test(request.target_model, request.success_criteria, request.max_attempts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


######### example endpoint to execute an attack #################
@router.post("/attacks/run", response_model=AttackResult)
async def execute_attack(config: AttackConfig):
    try:
        return process_attack(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
##################################################################
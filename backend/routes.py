from fastapi import APIRouter, HTTPException
from logic import AttackConfig, AttackResult, process_attack

router = APIRouter()

######### example endpoint to execute an attack #################
@router.post("/attacks/run", response_model=AttackResult)
async def execute_attack(config: AttackConfig):
    try:
        return process_attack(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
##################################################################
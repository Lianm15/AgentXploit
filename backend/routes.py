from fastapi import APIRouter, HTTPException
from logic import AttackConfig, AttackResult, InitializeResponse, Transcript, initialize as initialize_session
from pydantic import BaseModel
from typing import List
from logic import get_messages
from logic import get_local_models, ModelsResponse
from fastapi import BackgroundTasks
from logic import run_attack_process
from logic import SessionStatusResponse, get_session_status
from logic import ActionRequest, ActionResponse, handle_session_action
from logic import FinishTestResponse, get_finish_test_data
import logging

router = APIRouter(prefix="/api") 
logger = logging.getLogger("backend.routes")

#the request body for initialize
class InitializeRequest(BaseModel):
    target_model: str
    success_criteria: str
    max_attempts: int

@router.post("/initialize", response_model=InitializeResponse)
async def initialize(request: InitializeRequest) -> InitializeResponse:
    try:
        return initialize_session(request.target_model, request.success_criteria, request.max_attempts)
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during initialization")


@router.get("/{session_id}/messages", response_model=Transcript)
async def get_transcript(session_id: str) -> Transcript:
    """
    Fetches the full transcript for a session, distinguishing between:
    - AgentXploit (Gemini) prompts
    - Target (Local LLM) responses
    """
    try:
        messages = get_messages(session_id)
        
        if not messages:
            raise HTTPException(status_code=404, detail=f"No messages found for session {session_id}")
        
        return Transcript(
            session_id=session_id,
            transcript=messages,
            total_messages=len(messages)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving messages for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error while retrieving messages")
    
@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    try:
        models = get_local_models()
        return ModelsResponse(models=models)
    except Exception as e:
        logger.error(f"Error fetching local models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch local models")
    
@router.post("/{session_id}/start")
async def start_attack(session_id: str, background_tasks: BackgroundTasks):
    try:
        from database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT success_criteria FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Session not found")

        success_criteria = row["success_criteria"]

        background_tasks.add_task(run_attack_process, session_id, success_criteria)

        return {"status": "Attack started in background"}

    except Exception as e:
        logger.error(f"Error starting attack: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start attack")
    

@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def get_status(session_id: str) -> SessionStatusResponse:
    try:
        return get_session_status(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error retrieving status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve status")
    

@router.post("/{session_id}/action", response_model=ActionResponse)
async def session_action(session_id: str, request: ActionRequest) -> ActionResponse:
    try:
        return handle_session_action(session_id, request.action)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error handling session action: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process action")
    

@router.get("/{session_id}/finish", response_model=FinishTestResponse)
async def finish_test(session_id: str) -> FinishTestResponse:
    try:
        return get_finish_test_data(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error finishing test for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve finish test data")

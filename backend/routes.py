from fastapi import APIRouter, HTTPException
from logic import InitializeResponse, Transcript, initialize as initialize_session
from pydantic import BaseModel
from typing import List
from logic import get_messages, get_history
from logic import get_local_models, ModelsResponse
from fastapi import BackgroundTasks
from logic import run_attack_process
from logic import SessionStatusResponse, get_session_status
from logic import ActionRequest, ActionResponse, handle_session_control
from logic import FinishTestResponse, get_tests_summary, EvaluateRequest, EvaluateResponse, evaluate_target_response
from logic import get_stats, StatsResponse
from logic import get_session_intelligence
import logging

router = APIRouter(prefix="/api") 
logger = logging.getLogger("backend.routes")

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
    

@router.get("/stats", response_model=StatsResponse)
async def get_stats_endpoint() -> StatsResponse:
    try:
        return get_stats()
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get("/history")
async def history():
    try:
        return get_history()
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error while retrieving history")


@router.get("/{session_id}/messages", response_model=Transcript)
async def get_transcript(session_id: str) -> Transcript:
    """
    Fetches the full transcript for a session.
    """

    try:
        messages = get_messages(session_id)

        if not messages:
            messages = []

        return Transcript(
            session_id=session_id,
            transcript=messages,
            total_messages=len(messages)
        )

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
            "SELECT status FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Session not found")

        if row["status"] != "initialized":
            raise HTTPException(
                status_code=409,
                detail=f"Session already started (status: {row['status']})",
            )

        background_tasks.add_task(run_attack_process, session_id)

        return {"status": "Attack started in background"}

    except HTTPException:
        raise
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
    

@router.post("/{session_id}/control", response_model=ActionResponse)
async def session_control(session_id: str, request: ActionRequest) -> ActionResponse:
    try:
        return handle_session_control(session_id, request.action)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error handling session action: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process action")
    

@router.get("/{session_id}/summary", response_model=FinishTestResponse)
async def finish_test(session_id: str) -> FinishTestResponse:
    try:
        return get_tests_summary(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error finishing test for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve finish test data")

@router.post("/{session_id}/evaluate", response_model=EvaluateResponse)
async def evaluate(session_id: str, request: EvaluateRequest) -> EvaluateResponse:
    try:
        return evaluate_target_response(session_id, request.target_response)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluating response for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to evaluate response")


@router.get("/{session_id}/intelligence")
async def get_intelligence(session_id: str):
    try:
        return get_session_intelligence(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving intelligence for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve intelligence data")

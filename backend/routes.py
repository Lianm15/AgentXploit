"""REST API surface. Thin wrappers: validate the request, delegate to logic.py, translate errors to HTTPException."""

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
from logic import get_session_intelligence, get_intelligence_summary
from attack_memory import AttackMemory
from database import db_cursor
import logging

router = APIRouter(prefix="/api")
logger = logging.getLogger("backend.routes")

class InitializeRequest(BaseModel):
    target_model: str
    success_criteria: str
    max_attempts: int
    mode: str = "standard"

@router.post("/initialize", response_model=InitializeResponse)
async def initialize(request: InitializeRequest) -> InitializeResponse:
    """Create a new session row and return its id; the client calls /start next to actually run it."""
    try:
        return initialize_session(request.target_model, request.success_criteria, request.max_attempts, request.mode)
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during initialization")
    

@router.get("/stats", response_model=StatsResponse)
async def get_stats_endpoint() -> StatsResponse:
    """Aggregate stats for the /stats dashboard Overview/Models/Sessions tabs."""
    try:
        return get_stats()
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")


@router.get("/history")
async def history():
    """All previous session results, newest first."""
    try:
        return get_history()
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error while retrieving history")


# Must be registered before any /{session_id}/... routes so that FastAPI's
# first-match routing does not capture "intelligence" or "priors" as a session_id.
@router.get("/priors/{target_model}")
async def get_priors(target_model: str):
    """Historical technique priors for this model plus how many sessions they're based on."""
    try:
        priors = AttackMemory().load_priors(target_model)
        with db_cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(DISTINCT session_id) FROM sessions WHERE target_model = ?",
                (target_model,),
            )
            row = cursor.fetchone()
        session_count = row[0] if row and row[0] else 0
        return {"priors": priors, "session_count": session_count}
    except Exception as e:
        logger.error(f"Error loading priors for {target_model}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load priors")


@router.get("/intelligence/summary")
async def get_intelligence_summary_endpoint():
    """Cross-session technique effectiveness + failure distribution for the Intelligence dashboard tab."""
    try:
        return get_intelligence_summary()
    except Exception as e:
        logger.error(f"Error retrieving intelligence summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve intelligence summary")


@router.get("/{session_id}/messages", response_model=Transcript)
async def get_transcript(session_id: str) -> Transcript:
    """Full transcript for a session — polled by the frontend to render the live chat view."""
    try:
        messages = get_messages(session_id)  # get_messages always returns a list, never None

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
    """Model names currently pulled on the connected Ollama server."""
    try:
        models = get_local_models()
        return ModelsResponse(models=models)
    except Exception as e:
        logger.error(f"Error fetching local models: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch local models")
    
@router.post("/{session_id}/start")
async def start_attack(session_id: str, background_tasks: BackgroundTasks):
    """Kick off the attack loop as a FastAPI background task, guarding against double-start."""
    try:
        with db_cursor() as cursor:
            cursor.execute(
                "SELECT status FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()

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
    """Current lifecycle status of a session — polled by the frontend to drive UI state."""
    try:
        return get_session_status(session_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error retrieving status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve status")
    

@router.post("/{session_id}/control", response_model=ActionResponse)
async def session_control(session_id: str, request: ActionRequest) -> ActionResponse:
    """Pause/resume/stop a running session (the attack loop checks status via wait_if_paused)."""
    try:
        return handle_session_control(session_id, request.action)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error handling session action: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process action")
    

@router.get("/{session_id}/summary", response_model=FinishTestResponse)
async def finish_test(session_id: str) -> FinishTestResponse:
    """Attempt count, elapsed time, and the winning prompt (if any) for the session-finished view."""
    try:
        return get_tests_summary(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error finishing test for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve finish test data")

@router.post("/{session_id}/evaluate", response_model=EvaluateResponse)
async def evaluate(session_id: str, request: EvaluateRequest) -> EvaluateResponse:
    """Manually judge an arbitrary response against this session's stored success criteria."""
    try:
        return evaluate_target_response(session_id, request.target_response)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluating response for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to evaluate response")


@router.get("/{session_id}/intelligence")
async def get_intelligence(session_id: str):
    """Per-session technique history and stats, for the live Attack Intelligence panel."""
    try:
        return get_session_intelligence(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving intelligence for {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve intelligence data")

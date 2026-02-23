from fastapi import APIRouter, HTTPException
from logic import AttackConfig, AttackResult, InitializeResponse, Transcript, initialize as initialize_session
from pydantic import BaseModel
from typing import List
from logic import get_messages
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

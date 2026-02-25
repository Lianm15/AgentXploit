from pydantic import BaseModel
from typing import Optional, List
from gemini import run_gemini_attack  #connects to gemini
from database import get_connection   #connects to sqlite
import uuid                           #generates session IDs

class AttackConfig(BaseModel):
    target_llm_id: str
    technique: str
    custom_prompt: Optional[str] = None
    success_criteria: str

class AttackResult(BaseModel):
    success: bool
    output: str
    technique_used: str

class Message(BaseModel):
    """Represents a single message in the transcript"""
    sender: str  # 'gemini' or 'target_llm'
    content: str
    timestamp: str

class Transcript(BaseModel):
    """Represents the full transcript for a session"""
    session_id: str
    transcript: List[Message]
    total_messages: int

class InitializeResponse(BaseModel):
    """Response payload for session initialization"""
    session_id: str

class HealthStatus(BaseModel):
    """Health check response"""
    status: str

def initialize(target_model: str, success_criteria: List[str], max_attempts: int) -> InitializeResponse:
    session_id = str(uuid.uuid4())  #generates unique ID 

    conn = get_connection()
    cursor = conn.cursor()

    # saves the session to the database
    cursor.execute("""
        INSERT INTO sessions (session_id, target_model, success_criteria, max_attempts)
        VALUES (?, ?, ?, ?)
    """, (
        session_id,
        target_model,
        success_criteria, 
        max_attempts
    ))

    conn.commit()
    conn.close()

    return InitializeResponse(session_id=session_id)

def add_message(session_id: str, sender: str, content: str) -> None:
    """Add a message to the transcript (sender: 'gemini' or 'target_llm')"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (session_id, sender, content)
        VALUES (?, ?, ?)
    """, (session_id, sender, content))
    conn.commit()
    conn.close()

def get_messages(session_id: str) -> List[Message]:
    """Get all messages for a session ordered by timestamp"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT sender, content, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY timestamp ASC
    """, (session_id,))
    messages = cursor.fetchall()
    conn.close()
    return [Message(sender=msg['sender'], content=msg['content'], timestamp=msg['timestamp']) for msg in messages]
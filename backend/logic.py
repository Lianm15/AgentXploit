from pydantic import BaseModel
from typing import Optional, List
from gemini import run_gemini_attack  #connects to gemini
from database import get_connection   #connects to sqlite
import uuid                           #generates session IDs
import time                           #measures time elapsed

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

def initialize(target_model: str, success_criteria: str, max_attempts: int) -> InitializeResponse:
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

def save_result(session_id: str, target_model: str, time_elapsed: float, success: bool) -> None:
    """Save the final result of a test session"""
    conn = get_connection()
    cursor = conn.cursor()
    # count how many messages were sent in this session
    cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
    messages_count = cursor.fetchone()[0]

    cursor.execute("""
        INSERT INTO results (session_id, target_model, time_elapsed, messages_count, success)
        VALUES (?, ?, ?, ?, ?)
    """, (
        session_id,
        target_model,
        time_elapsed,
        messages_count,
        success
    ))

    conn.commit()
    conn.close()

def get_history() -> list:
    """Get all previous test results for history"""
    conn = get_connection()
    cursor = conn.cursor()
    # newest first
    cursor.execute("""
        SELECT session_id, target_model, time_elapsed, messages_count, success
        FROM results
        ORDER BY rowid DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    # convert each SQLite row to a Python dict so FastAPI can return it as JSON
    return [dict(row) for row in rows]
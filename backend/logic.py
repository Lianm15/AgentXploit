from pydantic import BaseModel
from typing import Optional
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

def initialize(target_model: str, success_criteria: List[str], max_attempts: int) -> dict:
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

    return {
        "session_id": session_id
    }

def add_message(session_id: str, sender: str, content: str):
    """Add a message to the transcript (sender: 'gemini' or 'target_llm')"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO messages (session_id, sender, content)
        VALUES (?, ?, ?)
    """, (session_id, sender, content))
    conn.commit()
    conn.close()

def get_messages(session_id: str):
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
    return [dict(msg) for msg in messages]
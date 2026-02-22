from pydantic import BaseModel
from typing import Optional
from gemini import run_gemini_attack  #connects to gemini
from database import get_connection   #connects to sqlite
import uuid                           #generates session IDs

################### example data models and logic for processing attacks ############################
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

def process_attack(config: AttackConfig) -> AttackResult:
    # כאן תכנס הלוגיקה מול Gemini והמודל המקומי
    if config.technique == "semantic_shift":
        output = run_gemini_attack(config.custom_prompt or "default semantic shift prompt")
        return AttackResult(success=True, output=output, technique_used=config.technique)
    
    elif config.technique == "adversarial_gaslighting":
        return AttackResult(success=True, output="Gaslighted output", technique_used=config.technique)
    
    raise ValueError("Unknown technique")
#####################################################################################################
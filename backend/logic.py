from pydantic import BaseModel
from typing import List, Optional
from gemini import run_gemini_attack

################### example data models and logic for processing attacks ############################
class AttackConfig(BaseModel):
    target_llm_id: str
    technique: str
    custom_prompt: Optional[str] = None
    success_criteria: List[str]

class AttackResult(BaseModel):
    success: bool
    output: str
    technique_used: str

def process_attack(config: AttackConfig) -> AttackResult:
    # כאן תכנס הלוגיקה מול Gemini והמודל המקומי
    if config.technique == "semantic_shift":
        output = run_gemini_attack(config.custom_prompt or "default semantic shift prompt")
        return AttackResult(success=True, output=output, technique_used=config.technique)
    
    elif config.technique == "adversarial_gaslighting":
        return AttackResult(success=True, output="Gaslighted output", technique_used=config.technique)
    
    raise ValueError("Unknown technique")
#####################################################################################################
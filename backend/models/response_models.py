"""
Modèles de réponse Pydantic.
"""

from pydantic import BaseModel
from typing import Optional, List

class DiffusionStep(BaseModel):
    """Étape intermédiaire du processus de diffusion"""
    step: int
    noise_ratio: float
    num_masked: int
    partial_text: str

class GenerateResponse(BaseModel):
    """Réponse de génération"""
    
    generated_text: str
    inference_time_ms: float
    steps_used: int
    length: int
    diffusion_steps: Optional[List[DiffusionStep]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "generated_text": "le chat mange une souris",
                "inference_time_ms": 245.3,
                "steps_used": 30,
                "length": 5,
                "diffusion_steps": None
            }
        }
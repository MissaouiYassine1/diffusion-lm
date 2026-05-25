"""
Modèles de requête Pydantic.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional

class GenerateRequest(BaseModel):
    """Requête de génération de texte"""
    
    prompt: str = Field(..., min_length=1, max_length=200, description="Texte d'entrée")
    steps: Optional[int] = Field(30, ge=10, le=200, description="Nombre d'étapes de diffusion")
    temperature: Optional[float] = Field(0.8, ge=0.1, le=2.0, description="Créativité")
    max_length: Optional[int] = Field(100, ge=10, le=500, description="Longueur maximale")
    verbose: Optional[bool] = Field(False, description="Retourner les étapes intermédiaires")
    
    @validator('prompt')
    def prompt_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Le prompt ne peut pas être vide')
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "le chat",
                "steps": 30,
                "temperature": 0.8,
                "max_length": 100,
                "verbose": False
            }
        }
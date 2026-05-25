"""
Modèles de requête Pydantic.
Version compatible Pydantic v2.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

class GenerateRequest(BaseModel):
    """Requête de génération de texte"""
    
    prompt: str = Field(..., min_length=1, max_length=200, description="Texte d'entrée")
    steps: Optional[int] = Field(30, ge=10, le=200, description="Nombre d'étapes de diffusion")
    temperature: Optional[float] = Field(0.8, ge=0.1, le=2.0, description="Créativité")
    max_length: Optional[int] = Field(100, ge=10, le=500, description="Longueur maximale")
    verbose: Optional[bool] = Field(False, description="Retourner les étapes intermédiaires")
    
    @field_validator('prompt')
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        """Valider que le prompt n'est pas vide"""
        if not v or not v.strip():
            raise ValueError('Le prompt ne peut pas être vide')
        return v.strip()
    
    @field_validator('steps', mode='before')
    @classmethod
    def steps_range(cls, v: Optional[int]) -> int:
        """Corriger automatiquement les steps hors limites"""
        if v is None:
            return 30
        if v < 10:
            return 10
        if v > 200:
            return 200
        return v
    
    @field_validator('temperature', mode='before')
    @classmethod
    def temperature_range(cls, v: Optional[float]) -> float:
        """Corriger automatiquement la température hors limites"""
        if v is None:
            return 0.8
        if v < 0.1:
            return 0.1
        if v > 2.0:
            return 2.0
        return v
    
    @field_validator('max_length', mode='before')
    @classmethod
    def max_length_range(cls, v: Optional[int], info) -> int:
        """Ajuster max_length en fonction du prompt"""
        if v is None:
            return 100
        
        # Récupérer le prompt (nécessite accès aux valeurs)
        # Note: en Pydantic v2, il faut utiliser info.data
        prompt = info.data.get('prompt', '')
        min_length = max(10, len(prompt) * 2)
        
        if v < min_length:
            return min_length
        if v > 500:
            return 500
        return v
    
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



class BatchGenerateRequest(BaseModel):
    """Requête de génération par lot"""
    
    prompts: List[str] = Field(..., min_length=1, max_length=10, description="Liste des prompts")
    steps: Optional[int] = Field(30, ge=10, le=200)
    temperature: Optional[float] = Field(0.8, ge=0.1, le=2.0)
    max_length: Optional[int] = Field(100, ge=10, le=500)
    
    @field_validator('prompts')
    @classmethod
    def prompts_not_empty(cls, v: List[str]) -> List[str]:
        """Valider que tous les prompts sont non vides"""
        for prompt in v:
            if not prompt or not prompt.strip():
                raise ValueError('Tous les prompts doivent être non vides')
        return [p.strip() for p in v]
    
    @field_validator('steps', mode='before')
    @classmethod
    def steps_range(cls, v: Optional[int]) -> int:
        if v is None:
            return 30
        if v < 10:
            return 10
        if v > 200:
            return 200
        return v
    
    @field_validator('temperature', mode='before')
    @classmethod
    def temperature_range(cls, v: Optional[float]) -> float:
        if v is None:
            return 0.8
        if v < 0.1:
            return 0.1
        if v > 2.0:
            return 2.0
        return v
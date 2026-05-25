"""
Configuration de l'application.
"""

from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Modèle
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/diffusion_lm_toy.pt")
    
    # Génération
    DEFAULT_STEPS: int = 30
    DEFAULT_TEMPERATURE: float = 0.8
    MAX_LENGTH: int = 100
    
    class Config:
        env_file = ".env"

settings = Settings()
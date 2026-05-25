"""
Configuration centralisée avec validation.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Configuration de l'application"""
    
    # API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    API_WORKERS: int = Field(default=1, env="API_WORKERS")
    API_RELOAD: bool = Field(default=False, env="API_RELOAD")
    
    # Modèle
    MODEL_PATH: str = Field(default="models/diffusion_lm_toy.pt", env="MODEL_PATH")
    MODEL_D_MODEL: int = Field(default=128, env="MODEL_D_MODEL")
    MODEL_N_LAYERS: int = Field(default=4, env="MODEL_N_LAYERS")
    MODEL_N_HEADS: int = Field(default=4, env="MODEL_N_HEADS")
    MODEL_DROPOUT: float = Field(default=0.1, env="MODEL_DROPOUT")
    
    # Génération
    DEFAULT_STEPS: int = Field(default=30, env="DEFAULT_STEPS")
    DEFAULT_TEMPERATURE: float = Field(default=0.8, env="DEFAULT_TEMPERATURE")
    MAX_LENGTH: int = Field(default=100, env="MAX_LENGTH")
    MIN_STEPS: int = Field(default=10, env="MIN_STEPS")
    MAX_STEPS: int = Field(default=200, env="MAX_STEPS")
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=10, env="RATE_LIMIT_PER_MINUTE")
    RATE_LIMIT_BURST: int = Field(default=20, env="RATE_LIMIT_BURST")
    
    # Redis
    REDIS_HOST: Optional[str] = Field(default=None, env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/app.log", env="LOG_FILE")
    
    @validator('DEFAULT_TEMPERATURE')
    def validate_temperature(cls, v):
        if v < 0.1 or v > 2.0:
            return 0.8
        return v
    
    @validator('DEFAULT_STEPS')
    def validate_steps(cls, v):
        if v < 10:
            return 30
        if v > 200:
            return 200
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Singleton pour accéder aux settings
settings = Settings()
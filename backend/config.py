"""
Configuration centralisée avec validation.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Configuration de l'application"""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",        # ← clé du correctif : ignore STREAMLIT_* et autres vars inconnues
        populate_by_name=True,
    )

    # API
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    API_WORKERS: int = Field(default=1)
    API_RELOAD: bool = Field(default=False)

    # Modèle
    MODEL_PATH: str = Field(default="models/diffusion_lm_toy.pt")
    MODEL_D_MODEL: int = Field(default=128)
    MODEL_N_LAYERS: int = Field(default=4)
    MODEL_N_HEADS: int = Field(default=4)
    MODEL_DROPOUT: float = Field(default=0.1)

    # Génération
    DEFAULT_STEPS: int = Field(default=30)
    DEFAULT_TEMPERATURE: float = Field(default=0.8)
    MAX_LENGTH: int = Field(default=100)
    MIN_STEPS: int = Field(default=10)
    MAX_STEPS: int = Field(default=200)

    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=10)
    RATE_LIMIT_BURST: int = Field(default=20)

    # Redis
    REDIS_HOST: Optional[str] = Field(default=None)
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="logs/app.log")

    # Validators (syntaxe Pydantic v2)
    @field_validator("DEFAULT_TEMPERATURE")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if v < 0.1 or v > 2.0:
            return 0.8
        return v

    @field_validator("DEFAULT_STEPS")
    @classmethod
    def validate_steps(cls, v: int) -> int:
        if v < 10:
            return 30
        if v > 200:
            return 200
        return v


# Singleton
settings = Settings()
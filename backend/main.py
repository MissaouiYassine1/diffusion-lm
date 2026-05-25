"""
API FastAPI pour le Diffusion Language Model.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

import sys
sys.path.append('.')

from backend.config import settings
from backend.api.routes import health, generation
from backend.services.generation_service import GenerationService
from backend.middleware.rate_limiter import setup_rate_limiting
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

# Créer l'application
app = FastAPI(
    title="Diffusion Language Model API",
    description="API pour générer du texte avec des modèles de diffusion",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configuration CORS avancée
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit
        "http://127.0.0.1:8501",
        "http://localhost:3000",  # React (optionnel)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# Rate limiting
setup_rate_limiting(app)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# État
app.state.start_time = time.time()
app.state.generation_service = None

# Routes
app.include_router(health.router, tags=["Health"])
app.include_router(generation.router, tags=["Generation"])

@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage"""
    print("🔄 Chargement du service de génération...")
    app.state.generation_service = GenerationService(model_path=settings.MODEL_PATH)
    print("✅ API prête! Documentation: http://localhost:8000/docs")

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Diffusion Language Model API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "generate": "/generate",
        "batch_generate": "/batch_generate",
        "version": "1.0.0"
    }

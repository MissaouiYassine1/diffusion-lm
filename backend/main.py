"""
API FastAPI pour le Diffusion Language Model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import time
from datetime import datetime

# Import des modules internes
import sys
sys.path.append('.')

from backend.config import settings
from backend.api.routes import generation, health

# Créer l'application
app = FastAPI(
    title="Diffusion Language Model API",
    description="API pour générer du texte avec des modèles de diffusion",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Ajouter CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
app.state.start_time = time.time()
app.state.model = None
app.state.tokenizer = None

# Inclure les routes
app.include_router(health.router, tags=["Health"])
# app.include_router(generation.router, tags=["Generation"])  # Sera ajouté plus tard

@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage"""
    print("🔄 Chargement du modèle...")
    # Temporaire: juste un placeholder
    print("✅ API démarrée (modèle sera chargé plus tard)")

@app.get("/")
async def root():
    return {
        "message": "Diffusion Language Model API",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }
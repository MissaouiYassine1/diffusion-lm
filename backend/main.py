"""
API FastAPI pour le Diffusion Language Model.

## Description
Cette API implémente un modèle de diffusion pour la génération de texte.
Contrairement aux modèles autoregressifs (GPT), le modèle de diffusion génère
le texte en parallèle via un processus de débrutage progressif.

## Fonctionnalités
- Génération de texte synchrone
- Génération par lots
- Visualisation des étapes de diffusion
- Rate limiting (10 requêtes/minute)

## Exemple d'utilisation

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "le chat",
        "steps": 30,
        "temperature": 0.8
    }
)
print(response.json()["generated_text"])
```

## Modèles
- Tokenizer: niveau caractère
- Architecture: Transformer bidirectionnel
- Diffusion: linéaire sur 100 étapes
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


# Créer l'application avec métadonnées
app = FastAPI(
    title="Diffusion Language Model API",
    description=__doc__,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "MISSAOUI Yassine",
        "email": "yassine.missaoui@enis.tn",
    },
    license_info={
        "name": "MIT",
    },
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",   # Streamlit
        "http://127.0.0.1:8501",
        "http://localhost:3000",   # React (optionnel)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
)

# Rate limiting
setup_rate_limiting(app)
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# État global
app.state.start_time = time.time()
app.state.generation_service = None

# Routes
app.include_router(health.router, tags=["Health"])
app.include_router(generation.router, tags=["Generation"])


@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage."""
    print("🔄 Chargement du service de génération...")
    app.state.generation_service = GenerationService(model_path=settings.MODEL_PATH)
    print("✅ API prête! Documentation: http://localhost:8000/docs")


@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée racine — liens vers les principales routes."""
    return {
        "message": "Diffusion Language Model API",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "generate": "/generate",
        "batch_generate": "/batch_generate",
        "version": "1.0.0",
    }
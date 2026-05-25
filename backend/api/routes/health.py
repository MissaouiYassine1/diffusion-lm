"""
Endpoint de health check.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
import time

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    timestamp: str

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Vérifier l'état de l'API et du modèle"""
    
    uptime = time.time() - request.app.state.start_time
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime,
        model_loaded=request.app.state.generation_service is not None and 
                     request.app.state.generation_service.is_loaded(),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )
"""
Endpoints de génération de texte.
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from backend.models.request_models import GenerateRequest
from backend.models.response_models import GenerateResponse

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: Request,
    gen_request: GenerateRequest,
    background_tasks: BackgroundTasks
):
    """Générer du texte avec le modèle de diffusion"""
    
    # Vérifier que le service est chargé
    if not hasattr(request.app.state, 'generation_service') or \
       not request.app.state.generation_service.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    service = request.app.state.generation_service
    
    # Générer
    result = service.generate(
        prompt=gen_request.prompt,
        steps=gen_request.steps,
        temperature=gen_request.temperature,
        max_length=gen_request.max_length,
        verbose=gen_request.verbose
    )
    
    # Tâche en arrière-plan (logging)
    background_tasks.add_task(
        log_generation,
        prompt=gen_request.prompt,
        steps=gen_request.steps,
        time_ms=result['inference_time_ms']
    )
    
    return GenerateResponse(**result)

async def log_generation(prompt: str, steps: int, time_ms: float):
    """Log en arrière-plan"""
    print(f"[LOG] Prompt: {prompt[:50]}... | Steps: {steps} | Time: {time_ms:.0f}ms")
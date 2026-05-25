"""
Endpoints de génération de texte.
"""

from fastapi import APIRouter, Request, HTTPException, BackgroundTasks
from backend.models.request_models import GenerateRequest, BatchGenerateRequest
from backend.models.response_models import GenerateResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/generate", response_model=GenerateResponse)
@limiter.limit("10/minute")
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


@router.post("/batch_generate")
@limiter.limit("5/minute")
async def batch_generate(
    request: Request,
    batch_request: BatchGenerateRequest,
    background_tasks: BackgroundTasks
):
    """Générer du texte pour plusieurs prompts en parallèle"""
    
    if not hasattr(request.app.state, 'generation_service') or \
       not request.app.state.generation_service.is_loaded():
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    service = request.app.state.generation_service
    
    # Générer en parallèle
    results = []
    for prompt in batch_request.prompts:
        result = service.generate(
            prompt=prompt,
            steps=batch_request.steps,
            temperature=batch_request.temperature,
            max_length=batch_request.max_length,
            verbose=False
        )
        results.append({
            'prompt': prompt,
            'generated_text': result['generated_text'],
            'inference_time_ms': result['inference_time_ms']
        })
    
    return {
        'results': results,
        'total_time_ms': sum(r['inference_time_ms'] for r in results),
        'num_requests': len(batch_request.prompts)
    }


async def log_generation(prompt: str, steps: int, time_ms: float):
    """Log en arrière-plan"""
    print(f"[LOG] Prompt: {prompt[:50]}... | Steps: {steps} | Time: {time_ms:.0f}ms")
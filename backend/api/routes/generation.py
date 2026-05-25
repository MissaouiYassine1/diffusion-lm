"""
Endpoints de génération de texte.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

router = APIRouter()

# Modèles Pydantic temporaires (seront déplacés plus tard)
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=200)
    steps: Optional[int] = Field(30, ge=10, le=200)
    temperature: Optional[float] = Field(0.8, ge=0.1, le=2.0)
    max_length: Optional[int] = Field(100, ge=10, le=500)
    verbose: Optional[bool] = Field(False)

class GenerateResponse(BaseModel):
    generated_text: str
    inference_time_ms: float
    steps_used: int
    length: int
    diffusion_steps: Optional[List[dict]] = None

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: Request,
    gen_request: GenerateRequest
):
    """
    Générer du texte avec le modèle de diffusion.
    
    Cette version est temporaire - le modèle réel sera intégré plus tard.
    """
    import time
    
    start_time = time.time()
    
    # Version temporaire - retourne un texte factice
    # (Le vrai modèle sera chargé dans les issues suivantes)
    generated_text = f"Génération pour: '{gen_request.prompt}' (modèle temporaire, steps={gen_request.steps})"
    
    inference_time = (time.time() - start_time) * 1000
    
    return GenerateResponse(
        generated_text=generated_text,
        inference_time_ms=inference_time,
        steps_used=gen_request.steps,
        length=len(generated_text.split())
    )

@router.post("/batch_generate")
async def batch_generate(
    request: Request,
    batch_request: dict
):
    """
    Génération par lots (temporaire).
    """
    prompts = batch_request.get("prompts", [])
    results = []
    
    for prompt in prompts:
        results.append({
            'prompt': prompt,
            'generated_text': f"Généré: {prompt}",
            'inference_time_ms': 100.0
        })
    
    return {
        'results': results,
        'total_time_ms': len(prompts) * 100.0,
        'num_requests': len(prompts)
    }
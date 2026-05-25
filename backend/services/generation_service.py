"""
Service de génération de texte.
"""

import torch
import time
import sys
sys.path.append('.')

from backend.core.tokenizer import CharTokenizer
from backend.core.diffusion_model import DiffusionLM, DiffusionLMWithGeneration

class GenerationService:
    """Service pour la génération de texte"""
    
    def __init__(self, model_path="models/diffusion_lm_toy.pt"):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Charger le modèle et le tokenizer"""
        # Données d'entraînement (pour reconstruire le tokenizer)
        texts = [
            "le chat mange une souris",
            "le chien aboie fort",
            "la souris court vite",
            "le chat dort sur le canapé",
            "le chien aime les os"
        ]
        
        self.tokenizer = CharTokenizer()
        self.tokenizer.fit(texts)
        
        # Créer le modèle
        self.model = DiffusionLMWithGeneration(
            vocab_size=self.tokenizer.vocab_size,
            d_model=128,
            n_layers=4,
            n_heads=4
        )
        
        # Charger les poids
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            print(f"✅ Modèle chargé depuis {self.model_path}")
        except FileNotFoundError:
            print(f"⚠️ Modèle non trouvé, utilisation de poids aléatoires")
        
        self.model.eval()
        self.mask_id = self.tokenizer.char_to_idx[self.tokenizer.MASK_TOKEN]
    
    def generate(self, prompt, steps=30, temperature=0.8, max_length=100, verbose=False):
        """Générer du texte"""
        start_time = time.time()
        
        # Tronquer le prompt si trop long
        if len(prompt) > max_length // 2:
            prompt = prompt[:max_length // 2]
        
        # Encoder le prompt
        prompt_ids = self.tokenizer.encode(prompt, max_length=max_length)
        
        # Générer
        with torch.no_grad():
            generated_ids, steps_log = self.model.generate(
                prompt_ids,
                self.mask_id,
                num_steps=steps,
                temperature=temperature
            )
        
        # Décoder
        generated_text = self.tokenizer.decode(generated_ids)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Formater les étapes si demandé
        diffusion_steps = None
        if verbose:
            diffusion_steps = []
            for step in steps_log:
                partial_text = self.tokenizer.decode(step['tokens'])
                diffusion_steps.append({
                    'step': step['step'],
                    'noise_ratio': step['noise_ratio'],
                    'num_masked': step['num_masked'],
                    'partial_text': partial_text[:100]  # Limiter
                })
        
        return {
            'generated_text': generated_text,
            'inference_time_ms': inference_time,
            'steps_used': len(steps_log),
            'length': len(generated_text.split()),
            'diffusion_steps': diffusion_steps
        }
    
    def is_loaded(self):
        return self.model is not None and self.tokenizer is not None
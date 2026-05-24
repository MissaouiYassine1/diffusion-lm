"""
Démonstration interactive de génération avec le modèle entraîné.
"""

import sys
sys.path.append('.')

import torch
from backend.core.tokenizer import CharTokenizer
from backend.core.diffusion_model import DiffusionLM, DiffusionLMWithGeneration

def load_model(model_path="models/diffusion_lm_toy.pt"):
    """Charger le modèle entraîné"""
    # Reconstruire le tokenizer (doit être le même qu'à l'entraînement)
    texts = [
        "le chat mange une souris",
        "le chien aboie fort",
        "la souris court vite",
        "le chat dort sur le canapé",
        "le chien aime les os"
    ]
    
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    
    # Reconstruire le modèle
    model = DiffusionLMWithGeneration(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4
    )
    
    # Charger les poids
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, tokenizer

def generate_text(model, tokenizer, prompt, num_steps=30, temperature=0.8):
    """Générer du texte à partir d'un prompt"""
    mask_id = tokenizer.char_to_idx[tokenizer.MASK_TOKEN]
    prompt_ids = tokenizer.encode(prompt)
    
    # Génération
    generated_ids, steps = model.generate(prompt_ids, mask_id, num_steps, temperature)
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text, steps

def main():
    print("🎮 Démonstration du Diffusion Language Model")
    print("=" * 50)
    
    # Charger le modèle
    print("🔄 Chargement du modèle...")
    model, tokenizer = load_model()
    print(f"✅ Modèle chargé! Vocabulaire: {tokenizer.vocab_size} tokens")
    
    # Interface simple
    while True:
        print("\n" + "=" * 50)
        prompt = input("📝 Entrez un prompt (ou 'quit'): ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            prompt = "le"
        
        print(f"🔄 Génération en cours pour: '{prompt}'")
        
        # Générer
        generated, steps = generate_text(model, tokenizer, prompt)
        
        print(f"\n✨ Résultat: '{generated}'")
        print(f"📊 Étapes: {len(steps)}, Dernier masque: {steps[-1]['num_masked']}")
        
        # Afficher progression
        print("\n📈 Progression du débrutage:")
        for i, step in enumerate(steps[::5]):  # Toutes les 5 étapes
            partial = tokenizer.decode(step['tokens'])
            print(f"   Étape {step['step']}: {partial[:50]}...")
    
    print("\n👋 Au revoir!")

if __name__ == "__main__":
    main()
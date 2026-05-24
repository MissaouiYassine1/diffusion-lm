import sys
sys.path.append('.')
import torch
from backend.core.tokenizer import CharTokenizer
from backend.core.diffusion_model import DiffusionLM, DiffusionLMWithGeneration

# 1. Préparer tokenizer
tokenizer = CharTokenizer()
texts = ["le chat mange une souris", "le chien court vite", "la maison est grande"]
tokenizer.fit(texts)

print(f"✅ Vocabulaire: {tokenizer.vocab_size} tokens")
print(f"Tokens spéciaux: {tokenizer.char_to_idx}")

# 2. Créer modèle (très petit pour test rapide)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DiffusionLMWithGeneration(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_layers=2,
    n_heads=4
).to(device)

print(f"✅ Modèle créé: {sum(p.numel() for p in model.parameters()):,} paramètres")

# 3. Simuler un prompt
prompt = "le chat"
prompt_ids = tokenizer.encode(prompt)
mask_id = tokenizer.char_to_idx[tokenizer.MASK_TOKEN]

print(f"Prompt: '{prompt}' -> IDs: {prompt_ids}")

# 4. Générer
generated_ids, steps = model.generate(prompt_ids, mask_id, num_steps=20, temperature=0.8)
generated_text = tokenizer.decode(generated_ids)

print(f"\n📝 Généré: '{generated_text}'")
print(f"Étapes: {len(steps)}")
print(f"Dernière étape: {steps[-1]['num_masked']} tokens masqués")

print("\n✅ Backward diffusion fonctionne!")
"""
Script d'entraînement pour le modèle de diffusion.
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from backend.core.tokenizer import CharTokenizer
from backend.core.diffusion_model import DiffusionLM

class ToyDataset(Dataset):
    """Dataset jouet pour overfit volontaire"""
    
    def __init__(self, texts, tokenizer, max_length=30):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Encoder tous les textes
        self.encodings = []
        for text in texts:
            encoded = tokenizer.encode(text, max_length=max_length)
            self.encodings.append(encoded)
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)

def train(model, dataloader, epochs=100, lr=0.001, device='cpu'):
    """Entraîner le modèle"""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignorer PAD
    
    losses = []
    
    for epoch in tqdm(range(epochs), desc="Entraînement"):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            logits = model(batch)
            
            # Calculer la perte (tous les tokens)
            loss = criterion(logits.view(-1, logits.size(-1)), batch.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses

def main():
    # 1. Données d'entraînement (très petites pour overfit)
    texts = [
        "le chat mange une souris",
        "le chien aboie fort",
        "la souris court vite",
        "le chat dort sur le canapé",
        "le chien aime les os"
    ]
    
    print(f"📚 Dataset: {len(texts)} phrases")
    for text in texts:
        print(f"   - {text}")
    
    # 2. Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(texts)
    print(f"\n🔤 Vocabulaire: {tokenizer.vocab_size} tokens")
    
    # 3. Dataset et DataLoader
    dataset = ToyDataset(texts, tokenizer, max_length=40)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 4. Modèle (petit pour entraînement rapide)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DiffusionLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4
    )
    
    print(f"\n🤖 Modèle: {sum(p.numel() for p in model.parameters()):,} paramètres")
    print(f"💻 Device: {device}")
    
    # 5. Entraînement
    print("\n🏋️ Début de l'entraînement...")
    losses = train(model, dataloader, epochs=100, device=device)
    
    # 6. Sauvegarde
    torch.save(model.state_dict(), "models/diffusion_lm_toy.pt")
    print("\n✅ Modèle sauvegardé dans models/diffusion_lm_toy.pt")
    
    # 7. Test rapide
    model.eval()
    test_text = "le chat"
    test_ids = torch.tensor([tokenizer.encode(test_text, max_length=40)]).to(device)
    
    with torch.no_grad():
        logits = model(test_ids)
        probs = torch.softmax(logits[0], dim=-1)
        predicted_ids = torch.argmax(probs, dim=-1)
        predicted_text = tokenizer.decode(predicted_ids.cpu().tolist())
    
    print(f"\n🧪 Test de génération:")
    print(f"   Input: '{test_text}'")
    print(f"   Output: '{predicted_text}'")
    
    return losses

if __name__ == "__main__":
    main()
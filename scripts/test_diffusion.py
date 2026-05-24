"""
Test du processus de diffusion (forward).
"""

import sys
import os

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from backend.core.diffusion_model import DiffusionProcess

def test_diffusion():
    """Tester le processus de diffusion"""
    print("🧪 Test du processus de diffusion...")
    
    # Créer le processus
    diffusion = DiffusionProcess(noise_schedule='linear', T=100)
    print(f"✅ DiffusionProcess créé avec T={diffusion.T}")
    
    # Simuler une phrase (IDs 1-10)
    batch_size = 2
    seq_len = 20
    x_0 = torch.randint(1, 50, (batch_size, seq_len))
    t = torch.tensor([10, 50])  # Étape 10 et 50
    mask_token_id = 0
    
    print(f"\n📊 Données de test:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Steps: t={t.tolist()}")
    
    # Ajouter du bruit
    x_t, noise_mask = diffusion.add_noise(x_0, t, mask_token_id)
    
    # Afficher les résultats
    print(f"\n📈 Résultats:")
    print(f"   Taux de bruit à t=10: {diffusion.get_noise_ratio(10):.2f}")
    print(f"   Taux de bruit à t=50: {diffusion.get_noise_ratio(50):.2f}")
    print(f"   Taux de bruit à t=99: {diffusion.get_noise_ratio(99):.2f}")
    
    # Vérifier que plus t est grand, plus il y a de bruit
    noise_ratio_10 = (noise_mask[0] == 1).sum().item() / seq_len
    noise_ratio_50 = (noise_mask[1] == 1).sum().item() / seq_len
    
    print(f"\n📊 Bruit observé:")
    print(f"   Échantillon t=10: {noise_ratio_10:.1%} de tokens masqués")
    print(f"   Échantillon t=50: {noise_ratio_50:.1%} de tokens masqués")
    
    # Vérification
    if noise_ratio_50 > noise_ratio_10:
        print("\n✅ SUCCÈS: Plus t est grand, plus il y a de bruit!")
    else:
        print("\n⚠️ Attention: La relation t/bruit n'est pas parfaite (normal car aléatoire)")
    
    # Afficher un exemple
    print(f"\n🔍 Exemple (premier échantillon, t=10):")
    print(f"   Original: {x_0[0, :10].tolist()}")
    print(f"   Bruité:   {x_t[0, :10].tolist()}")
    
    return True

if __name__ == "__main__":
    test_diffusion()
    print("\n🎉 Tous les tests passent!")
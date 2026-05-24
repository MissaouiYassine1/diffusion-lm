"""
Diffusion Language Model avec attention bidirectionnelle.
Architecture Transformer modifiée pour la diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """Encodage positionnel sinusoidal"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ============================================================================
# BIDIRECTIONAL ATTENTION (CLÉ POUR DIFFUSION LM)
# ============================================================================

class BidirectionalAttention(nn.Module):
    """Attention bidirectionnelle (non-causale) - clé pour diffusion LM"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Projections
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores (bidirectionnelle - PAS de causal mask!)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Masque optionnel (pour PAD tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)

# ============================================================================
# TRANSFORMER BLOCK
# ============================================================================

class TransformerBlock(nn.Module):
    """Bloc Transformer avec attention bidirectionnelle"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = BidirectionalAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        # Attention avec résidu
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # FFN avec résidu
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

# ============================================================================
# MODÈLE PRINCIPAL
# ============================================================================

class DiffusionLM(nn.Module):
    """Modèle de diffusion pour le langage"""
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=8, max_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        """
        Forward pass avec attention bidirectionnelle.
        
        Args:
            x: [batch_size, seq_len] indices des tokens
            mask: [batch_size, seq_len] masque (1 pour tokens réels, 0 pour PAD)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Passer à travers les blocs (bidirectionnel!)
        for block in self.blocks:
            x = block(x, mask)
        
        # Projection sur vocabulaire
        logits = self.lm_head(x)
        return logits
    
    def predict_masked(self, x, mask_positions):
        """
        Prédire les tokens masqués.
        
        Args:
            x: [batch_size, seq_len] indices (avec [MASK])
            mask_positions: positions masquées
        Returns:
            predictions: tokens prédits pour les positions masquées
        """
        logits = self.forward(x)
        masked_logits = logits[:, mask_positions, :]
        predictions = torch.argmax(masked_logits, dim=-1)
        return predictions

# ============================================================================
# PROCESSUS DE DIFFUSION (FORWARD - AJOUT DE BRUIT)
# ============================================================================

class DiffusionProcess:
    """Gestion du processus de diffusion (bruit)"""
    
    def __init__(self, noise_schedule='linear', T=100):
        """
        Args:
            noise_schedule: 'linear', 'cosine', ou 'sqrt'
            T: nombre d'étapes de diffusion
        """
        self.T = T
        
        if noise_schedule == 'linear':
            self.beta = torch.linspace(1e-4, 0.02, T)
        elif noise_schedule == 'cosine':
            # Schedule cosinus (plus doux)
            t = torch.arange(T) / T
            self.beta = (1 - torch.cos(t * torch.pi / 2)) * 0.02
        else:
            self.beta = torch.sqrt(torch.linspace(1e-4, 0.02, T))
        
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def add_noise(self, x_0, t, mask_token_id):
        """
        Ajouter du bruit à x_0 selon le schedule.
        
        Args:
            x_0: [batch_size, seq_len] tokens originaux
            t: [batch_size] étapes de bruit (0 à T-1)
            mask_token_id: ID du token [MASK]
        Returns:
            x_t: tokens avec bruit ajouté
            noise_mask: masque indiquant quels tokens sont bruités
        """
        batch_size, seq_len = x_0.shape
        
        # Niveau de bruit pour chaque élément
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1)
        
        # Probabilité de masquer chaque token
        mask_prob = 1 - alpha_bar_t  # Plus t est grand, plus on masque
        mask_prob = mask_prob.clamp(0, 1)
        
        # Générer un masque aléatoire
        random_mask = torch.rand(batch_size, seq_len)
        noise_mask = random_mask < mask_prob
        
        # Appliquer le masque
        x_t = x_0.clone()
        x_t[noise_mask] = mask_token_id
        
        return x_t, noise_mask
    
    def get_noise_ratio(self, t):
        """Retourner le ratio de bruit pour une étape donnée"""
        if isinstance(t, torch.Tensor):
            t = t.item() if t.numel() == 1 else t[0].item()
        return (1 - self.alpha_bar[t]).item()

# ============================================================================
# MODÈLE AVEC GÉNÉRATION (BACKWARD)
# ============================================================================

class DiffusionLMWithGeneration(DiffusionLM):
    """Extension du modèle avec capacité de génération"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = DiffusionProcess(T=100)
    
    def generate(self, prompt_ids, mask_token_id, num_steps=50, temperature=0.8):
        """
        Générer du texte par diffusion inversée.
        
        Args:
            prompt_ids: [seq_len] tokens du prompt
            mask_token_id: ID du token [MASK]
            num_steps: nombre d'étapes de débrutage
            temperature: température pour l'échantillonnage
        Returns:
            generated: tokens générés
            steps_log: historique des étapes (pour visualisation)
        """
        seq_len = len(prompt_ids)
        device = next(self.parameters()).device
        
        # Partir d'une séquence entièrement masquée
        current = torch.full((seq_len,), mask_token_id, device=device, dtype=torch.long)
        steps_log = []
        # Ajouter la méthode generate au modèle existant
        # (patch pour garder la compatibilité)
        setattr(DiffusionLM, 'generate', DiffusionLMWithGeneration.generate)
        # Remplacer les positions du prompt
        for i, token in enumerate(prompt_ids):
            if i < len(prompt_ids):
                current[i] = token
        
        # Débruiter progressivement
        for step in range(num_steps):
            t = num_steps - step
            noise_ratio = t / num_steps
            
            # Prédire les tokens
            with torch.no_grad():
                logits = self.forward(current.unsqueeze(0))
                probs = F.softmax(logits[0] / temperature, dim=-1)
            
            # Décider quels tokens démasquer
            mask_positions = (current == mask_token_id).nonzero(as_tuple=True)[0]
            num_to_unmask = max(1, int(len(mask_positions) * (1 - noise_ratio)))
            
            if num_to_unmask > 0 and len(mask_positions) > 0:
                # Calculer les confiances
                confidences = []
                for pos in mask_positions:
                    # Pour les positions masquées, prendre la probabilité du token prédit
                    pred_token = torch.argmax(probs[pos]).item()
                    confidence = probs[pos, pred_token].item()
                    confidences.append((pos.item(), confidence, pred_token))
                
                # Trier par confiance
                confidences.sort(key=lambda x: x[1], reverse=True)
                positions_to_unmask = confidences[:num_to_unmask]
                
                # Remplacer
                for pos, _, pred_token in positions_to_unmask:
                    if temperature > 0:
                        # Échantillonner avec température
                        token_dist = probs[pos] ** (1.0 / temperature)
                        token_dist = token_dist / token_dist.sum()
                        new_token = torch.multinomial(token_dist, 1).item()
                    else:
                        new_token = pred_token
                    current[pos] = new_token
            
            # Log étape
            steps_log.append({
                'step': step,
                'noise_ratio': noise_ratio,
                'num_masked': (current == mask_token_id).sum().item(),
                'tokens': current.clone()
            })
            
            # Arrêt anticipé
            if (current == mask_token_id).sum() == 0:
                break
        
        return current.cpu().tolist(), steps_log

class DiffusionLMWithGeneration(DiffusionLM):
    """Extension du modèle avec capacité de génération"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion = DiffusionProcess(T=100)
    
    def generate(self, prompt_ids, mask_token_id, num_steps=50, temperature=0.8):
        """
        Générer du texte par diffusion inversée.
        
        Args:
            prompt_ids: [seq_len] tokens du prompt
            mask_token_id: ID du token [MASK]
            num_steps: nombre d'étapes de débrutage
            temperature: température pour l'échantillonnage
        Returns:
            generated: tokens générés
            steps_log: historique des étapes (pour visualisation)
        """
        seq_len = len(prompt_ids)
        device = next(self.parameters()).device
        
        # Partir d'une séquence entièrement masquée
        current = torch.full((seq_len,), mask_token_id, device=device, dtype=torch.long)
        steps_log = []
        
        # Remplacer les positions du prompt (on garde le début)
        for i, token in enumerate(prompt_ids):
            if i < len(prompt_ids):
                current[i] = token
        
        # Débruiter progressivement
        for step in range(num_steps):
            t = num_steps - step  # De T-1 à 0
            noise_ratio = t / num_steps
            
            # Prédire les tokens à démasquer
            with torch.no_grad():
                logits = self.forward(current.unsqueeze(0))  # [1, seq_len, vocab_size]
                probs = F.softmax(logits[0] / temperature, dim=-1)
            
            # Décider quels tokens démasquer cette étape
            # Plus on avance, plus on démasque de tokens
            mask_positions = (current == mask_token_id).nonzero(as_tuple=True)[0]
            num_to_unmask = max(1, int(len(mask_positions) * (1 - noise_ratio)))
            
            if num_to_unmask > 0 and len(mask_positions) > 0:
                # Choisir les positions avec la plus haute confiance
                mask_confidences = []
                for pos in mask_positions:
                    confidence = probs[pos, current[pos]].item() if current[pos] != mask_token_id else 0
                    mask_confidences.append((pos.item(), confidence))
                
                # Trier par confiance
                mask_confidences.sort(key=lambda x: x[1], reverse=True)
                positions_to_unmask = [pos for pos, _ in mask_confidences[:num_to_unmask]]
                
                # Remplacer
                for pos in positions_to_unmask:
                    # Échantillonner selon la distribution de probabilité
                    token_dist = probs[pos]
                    if temperature > 0:
                        token_dist = token_dist ** (1.0 / temperature)
                        token_dist = token_dist / token_dist.sum()
                    new_token = torch.multinomial(token_dist, 1).item()
                    current[pos] = new_token
            
            # Log étape
            steps_log.append({
                'step': step,
                'noise_ratio': noise_ratio,
                'num_masked': (current == mask_token_id).sum().item(),
                'tokens': current.clone()
            })
            
            # Arrêt anticipé si plus rien de masqué
            if (current == mask_token_id).sum() == 0:
                break
        
        return current.cpu().tolist(), steps_log



# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("🧪 Test du modèle...")
    model = DiffusionLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
    batch = torch.randint(0, 100, (2, 32))
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✅ Modèle fonctionne! {sum(p.numel() for p in model.parameters()):,} paramètres")
    
    # Test DiffusionProcess
    print("\n🧪 Test du processus de diffusion...")
    diff = DiffusionProcess(T=100)
    print(f"✅ DiffusionProcess créé avec T={diff.T}")
    print(f"   Bruit à t=0: {diff.get_noise_ratio(0):.3f}")
    print(f"   Bruit à t=50: {diff.get_noise_ratio(50):.3f}")
    print(f"   Bruit à t=99: {diff.get_noise_ratio(99):.3f}")
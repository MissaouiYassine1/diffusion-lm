"""
Diffusion Language Model avec attention bidirectionnelle.
Architecture Transformer modifiée pour la diffusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

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

# Test du modèle
if __name__ == "__main__":
    model = DiffusionLM(vocab_size=100, d_model=64, n_layers=2, n_heads=4)
    batch = torch.randint(0, 100, (2, 32))
    output = model(batch)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✅ Modèle fonctionne! {sum(p.numel() for p in model.parameters()):,} paramètres")
"""
Tokenizer simple pour le modèle de diffusion.
Niveau caractère pour simplifier la démonstration.
"""

class CharTokenizer:
    """Tokenizer caractère avec vocabulaire dynamique"""
    
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        # Tokens spéciaux
        self.PAD_TOKEN = "[PAD]"
        self.MASK_TOKEN = "[MASK]"
        self.UNK_TOKEN = "[UNK]"
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Ajouter les tokens spéciaux au vocabulaire"""
        special_tokens = [self.PAD_TOKEN, self.MASK_TOKEN, self.UNK_TOKEN]
        for token in special_tokens:
            self.char_to_idx[token] = self.vocab_size
            self.idx_to_char[self.vocab_size] = token
            self.vocab_size += 1
    
    def fit(self, texts):
        """Construire le vocabulaire à partir des textes"""
        chars = set()
        for text in texts:
            for char in text:
                chars.add(char)
        
        for char in sorted(chars):
            if char not in self.char_to_idx:
                self.char_to_idx[char] = self.vocab_size
                self.idx_to_char[self.vocab_size] = char
                self.vocab_size += 1
    
    def encode(self, text, max_length=None):
        """Encoder un texte en indices"""
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                indices.append(self.char_to_idx[self.UNK_TOKEN])
        
        if max_length:
            if len(indices) < max_length:
                indices.extend([self.char_to_idx[self.PAD_TOKEN]] * (max_length - len(indices)))
            else:
                indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices):
        """Decoder des indices en texte"""
        chars = []
        for idx in indices:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                if char not in [self.PAD_TOKEN, self.MASK_TOKEN]:
                    chars.append(char)
        return ''.join(chars)
    
    def mask_indices(self, indices, mask_ratio=0.3):
        """Masquer aléatoirement des positions (ajout de bruit)"""
        import random
        masked = indices.copy()
        mask_idx = self.char_to_idx[self.MASK_TOKEN]
        
        num_mask = int(len(indices) * mask_ratio)
        positions = random.sample(range(len(indices)), num_mask)
        
        for pos in positions:
            masked[pos] = mask_idx
        
        return masked, positions

# Test rapide (si exécuté directement)
if __name__ == "__main__":
    tokenizer = CharTokenizer()
    texts = ["le chat mange une souris", "le chien dort"]
    tokenizer.fit(texts)
    print(f"Vocabulaire: {tokenizer.vocab_size} tokens")
    
    encoded = tokenizer.encode("le chat")
    print(f"Encodé: {encoded}")
    print(f"Décodé: {tokenizer.decode(encoded)}")
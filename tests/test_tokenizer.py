import sys
sys.path.append('.')
from backend.core.tokenizer import CharTokenizer

def test_tokenizer_basic():
    tokenizer = CharTokenizer()
    texts = ["abc", "def"]
    tokenizer.fit(texts)
    
    encoded = tokenizer.encode("ab")
    decoded = tokenizer.decode(encoded)
    
    assert decoded == "ab"
    assert len(encoded) == 2

def test_masking():
    tokenizer = CharTokenizer()
    tokenizer.fit(["hello"])
    encoded = tokenizer.encode("hello")
    masked, positions = tokenizer.mask_indices(encoded, mask_ratio=0.4)
    
    assert len(masked) == len(encoded)
    assert len(positions) > 0

print("✅ Tous les tests passent!")
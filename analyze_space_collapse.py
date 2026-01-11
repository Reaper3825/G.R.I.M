#!/usr/bin/env python3
"""
Analyze why token 277 (space) dominates model output.
Checks:
1. Embedding weights for token 277 vs other tokens
2. If there's any bias in the logit computation
3. The weight magnitudes that might favor token 277
"""

import struct
import numpy as np
from pathlib import Path
import json

# Constants from GRIM-text
UNIGRAM_VOCAB_OFFSET = 273
SPACE_TOKEN_ID = 277
D_MODEL = 768

def load_checkpoint_weights(checkpoint_path: Path):
    """Load weights from GRIM checkpoint (FlatBuffer format)"""
    # This is complex - FlatBuffers format
    # Let's try a simpler approach - check if there's a raw weights dump
    print(f"Looking for checkpoint at: {checkpoint_path}")
    return None

def analyze_from_debug_logits():
    """Analyze the logit patterns we captured from debug output"""
    # From the LOGITS_DEBUG output we captured:
    # [LOGITS_DEBUG] sample=1 top5_logits: [tid=277 logit=7.1967] [tid=258 logit=6.2812] [tid=259 logit=5.5391] [tid=386 logit=5.1075] [tid=295 logit=5.0392]
    # stats: min=-3.6616 max=7.1967 mean=-2.4162 spread=10.8583
    
    print("=== Analysis from Debug Logits ===\n")
    
    logits = {
        277: 7.1967,  # space
        258: 6.2812,  # atom token (258 = ATOM_TOKEN_OFFSET + 2)
        259: 5.5391,  # atom token
        386: 5.1075,  # unigram token
        295: 5.0392,  # unigram token
    }
    
    min_logit = -3.6616
    max_logit = 7.1967
    mean_logit = -2.4162
    
    print("Token 277 (space) analysis:")
    print(f"  Logit value: {logits[277]:.4f}")
    print(f"  Distance from mean: {logits[277] - mean_logit:.4f} (mean={mean_logit:.4f})")
    print(f"  Distance from max: {max_logit - logits[277]:.4f}")
    print()
    
    # Compute softmax probabilities
    print("Softmax probabilities (approximate from top-5):")
    all_logits = list(logits.values())
    exp_logits = [np.exp(l - max_logit) for l in all_logits]  # Subtract max for numerical stability
    sum_exp = sum(exp_logits)
    
    for tid, logit in logits.items():
        prob = np.exp(logit - max_logit) / sum_exp
        token_type = "SPACE" if tid == 277 else ("ATOM" if tid < 273 else "UNIGRAM")
        print(f"  Token {tid} ({token_type}): logit={logit:.4f} -> prob≈{prob:.4f} ({prob*100:.1f}%)")
    
    print()
    print("Key observations:")
    print("  - Token 277 (space) has the HIGHEST logit")
    print("  - Gap to 2nd place: {:.4f}".format(logits[277] - logits[258]))
    print("  - This gap in logit space = {:.2f}x probability ratio".format(
        np.exp(logits[277] - logits[258])))
    print()
    
    # Check if tokens 258, 259 are atom tokens
    ATOM_TOKEN_OFFSET = 256
    print("Token type breakdown:")
    print(f"  Token 258 = ATOM_TOKEN_OFFSET + {258 - ATOM_TOKEN_OFFSET} (AtomType index)")
    print(f"  Token 259 = ATOM_TOKEN_OFFSET + {259 - ATOM_TOKEN_OFFSET} (AtomType index)")
    print(f"  Token 277 = UNIGRAM_VOCAB_OFFSET + {277 - UNIGRAM_VOCAB_OFFSET} (space)")
    print(f"  Token 386 = UNIGRAM_VOCAB_OFFSET + {386 - UNIGRAM_VOCAB_OFFSET}")
    print(f"  Token 295 = UNIGRAM_VOCAB_OFFSET + {295 - UNIGRAM_VOCAB_OFFSET}")

def check_vocab_scores():
    """Check if vocab scores explain the bias"""
    vocab_path = Path("D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.bin")
    
    print("\n=== Vocab Score Analysis ===\n")
    
    with open(vocab_path, 'rb') as f:
        # Skip header (25 bytes total for v3)
        f.read(4)  # magic
        f.read(2)  # version
        f.read(4)  # checksum
        config_vocab_size = struct.unpack('<I', f.read(4))[0]
        f.read(4)  # max_length
        f.read(3)  # flags
        f.read(4)  # total_vocab_size
        
        # Read all pieces and find highest scored ones
        scores = []
        for i in range(min(config_vocab_size, 1000)):  # Read first 1000
            length = struct.unpack('<I', f.read(4))[0]
            text = f.read(length)
            score = struct.unpack('<f', f.read(4))[0]
            token_id = struct.unpack('<i', f.read(4))[0]
            try:
                text_str = text.decode('utf-8')
            except:
                text_str = repr(text)
            scores.append((token_id, score, text_str))
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 20 tokens by vocab score:")
        for tid, score, text in scores[:20]:
            vis = text.replace(' ', '·').replace('\n', '\\n')
            marker = " <== SPACE TOKEN" if tid == 277 else ""
            print(f"  Token {tid:4d}: score={score:8.4f}  text=\"{vis}\"{marker}")

def hypothesis_analysis():
    """Analyze possible causes of the space token collapse"""
    print("\n=== Hypothesis Analysis ===\n")
    
    print("Possible causes of mode collapse to space token:\n")
    
    print("1. EMBEDDING INITIALIZATION:")
    print("   - If token 277 embedding has larger magnitude at init")
    print("   - With tied embeddings, this directly affects logits")
    print("   - logit = hidden @ embedding.T")
    print("   - Larger embedding = larger logit = higher probability")
    print()
    
    print("2. TRAINING DATA FREQUENCY:")
    print("   - Space is the most common character in natural text")
    print("   - Vocab score -1.7649 is highest (most probable)")
    print("   - Model learns to always predict space because it's 'safe'")
    print()
    
    print("3. LOSS IMBALANCE:")
    print("   - If space appears after every word, loss accumulates on space")
    print("   - Model gets large gradient signal to predict space")
    print("   - Other tokens get diluted gradient signal")
    print()
    
    print("4. SOFTMAX TEMPERATURE:")
    print("   - If temperature is too low, small differences become large")
    print("   - A 0.9 logit gap becomes ~2.5x probability ratio")
    print()
    
    print("5. ATTENTION COLLAPSE:")
    print("   - If attention becomes uniform, hidden states average out")
    print("   - Averaged hidden state might have high dot-product with space")
    print()
    
    print("RECOMMENDED FIXES:")
    print("  a) Add frequency-based loss weighting (down-weight common tokens)")
    print("  b) Use label smoothing to prevent overconfident predictions")
    print("  c) Check embedding initialization - ensure uniform variance")
    print("  d) Increase softmax temperature during training")
    print("  e) Add entropy regularization to prevent collapse")

if __name__ == "__main__":
    analyze_from_debug_logits()
    check_vocab_scores()
    hypothesis_analysis()

#!/usr/bin/env python3
"""
Check position embeddings scale - LEARNED vs SINUSOIDAL comparison.
"""
import numpy as np
import math

def generate_sinusoidal_encoding(max_len, dim):
    """Generate sinusoidal positional encodings (OLD GRIM-text)."""
    encodings = np.zeros((max_len, dim), dtype=np.float32)
    
    for pos in range(max_len):
        for i in range(dim):
            angle = pos / math.pow(10000.0, (2.0 * (i // 2)) / dim)
            if i % 2 == 0:
                encodings[pos, i] = math.sin(angle)
            else:
                encodings[pos, i] = math.cos(angle)
    
    # Mean-center (old approach)
    for pos in range(max_len):
        encodings[pos] -= np.mean(encodings[pos])
    
    return encodings

def generate_learned_encoding(max_len, dim, seed=43):
    """Generate learned positional encodings (NEW - matches PyTorch)."""
    np.random.seed(seed)
    # Xavier normal: stddev = sqrt(2/(max_len + dim))
    stddev = np.sqrt(2.0 / (max_len + dim))
    return np.random.normal(0, stddev, (max_len, dim)).astype(np.float32)

# Parameters from GRIM-text
max_seq_len = 4096
d_model = 768
vocab_size = 50377

print("="*60)
print("POSITION EMBEDDING SCALE COMPARISON")
print("="*60)

# Token embeddings (reference)
np.random.seed(42)
token_stddev = np.sqrt(2.0 / (d_model + vocab_size))
token_emb = np.random.normal(0, token_stddev, (vocab_size, d_model)).astype(np.float32)
print(f"\nToken embeddings (Xavier, seed=42):")
print(f"  Xavier stddev: {token_stddev:.6f}")
print(f"  Actual stddev: {np.std(token_emb):.6f}")
print(f"  Range: [{np.min(token_emb):.4f}, {np.max(token_emb):.4f}]")

# Old sinusoidal
print(f"\n--- OLD: Sinusoidal Position Embeddings ---")
sin_emb = generate_sinusoidal_encoding(max_seq_len, d_model)
print(f"  Actual stddev: {np.std(sin_emb):.6f}")
print(f"  Range: [{np.min(sin_emb):.4f}, {np.max(sin_emb):.4f}]")
print(f"  Ratio to token emb: {np.std(sin_emb)/np.std(token_emb):.1f}x")

# New learned
print(f"\n--- NEW: Learned Position Embeddings (Xavier) ---")
learned_emb = generate_learned_encoding(max_seq_len, d_model, seed=43)
learned_stddev = np.sqrt(2.0 / (max_seq_len + d_model))
print(f"  Xavier stddev: {learned_stddev:.6f}")
print(f"  Actual stddev: {np.std(learned_emb):.6f}")
print(f"  Range: [{np.min(learned_emb):.4f}, {np.max(learned_emb):.4f}]")
print(f"  Ratio to token emb: {np.std(learned_emb)/np.std(token_emb):.1f}x")

# Combined embeddings test
print(f"\n--- Combined Embeddings (token + position) ---")
batch_size = 1000
batch_tokens = np.random.randint(0, vocab_size, size=batch_size)
batch_positions = np.arange(batch_size) % max_seq_len

# Old sinusoidal
combined_sin = token_emb[batch_tokens] + sin_emb[batch_positions]
print(f"\nOLD (sinusoidal):")
print(f"  Mean:  {np.mean(combined_sin):.6f}")
print(f"  Var:   {np.var(combined_sin):.6f}")
print(f"  Range: [{np.min(combined_sin):.4f}, {np.max(combined_sin):.4f}]")
print(f"  |min|/|max| asymmetry: {abs(np.min(combined_sin))/abs(np.max(combined_sin)):.2f}")

# New learned  
combined_learn = token_emb[batch_tokens] + learned_emb[batch_positions]
print(f"\nNEW (learned):")
print(f"  Mean:  {np.mean(combined_learn):.6f}")
print(f"  Var:   {np.var(combined_learn):.6f}")
print(f"  Range: [{np.min(combined_learn):.4f}, {np.max(combined_learn):.4f}]")
print(f"  |min|/|max| asymmetry: {abs(np.min(combined_learn))/abs(np.max(combined_learn)):.2f}")

print(f"\n✅ With learned embeddings, position and token are similar scale!")
print(f"   Position/Token ratio: {np.std(learned_emb)/np.std(token_emb):.1f}x (vs {np.std(sin_emb)/np.std(token_emb):.0f}x before)")

"""
Analyze embedding weights from checkpoint to understand space token collapse.
Token 277 = SPACE consistently has highest logit.
"""

import os
import struct
import numpy as np
from pathlib import Path

# FlatBuffers-style checkpoint reading
# The checkpoint_epoch_1.bin is a FlatBuffer file

def read_checkpoint_raw(path):
    """Read raw checkpoint file and look for embedding patterns."""
    with open(path, 'rb') as f:
        data = f.read()
    
    print(f"Checkpoint size: {len(data) / 1024 / 1024:.1f} MB")
    
    # Try to find the magic bytes or header
    # FlatBuffers have a 4-byte offset at start
    offset = struct.unpack('<I', data[:4])[0]
    print(f"FlatBuffer root offset: {offset}")
    
    # Look for patterns in the file
    # Embedding weights should be a large contiguous block of floats
    # vocab_size=50376 x d_model=768 = 38,688,768 floats = ~147 MB
    
    return data


def analyze_float_blocks(data, block_size=100000):
    """Find and analyze large blocks of floats in the checkpoint."""
    
    # Find all possible float32 arrays (look for reasonable float patterns)
    float_count = len(data) // 4
    print(f"Total possible floats: {float_count}")
    
    # Sample floats from different positions
    positions = [1000, 10000, 100000, 1000000, 10000000, 50000000, 100000000]
    for pos in positions:
        if pos * 4 + 4 <= len(data):
            val = struct.unpack('<f', data[pos*4:(pos+1)*4])[0]
            print(f"Float at position {pos}: {val:.6f}")


def try_extract_embeddings(data, vocab_size=50376, d_model=768):
    """
    Try to find and extract embedding weights.
    Expected size: vocab_size * d_model * 4 bytes = ~147 MB
    """
    embedding_size = vocab_size * d_model * 4  # bytes
    print(f"\nLooking for embedding block of size: {embedding_size / 1024 / 1024:.1f} MB")
    
    # The embeddings are likely at the start of the weights section
    # FlatBuffer format: offset table -> vtables -> data
    
    # Try to find embedding-like patterns by looking at variance
    # Embeddings typically have values in range [-0.1, 0.1] after Xavier init
    
    # Try different offsets to find embedding block
    test_offsets = [100, 500, 1000, 2000, 5000, 10000]
    
    for start_offset in test_offsets:
        if start_offset + embedding_size > len(data):
            continue
            
        # Extract a small sample
        sample_size = 1000 * d_model  # ~1000 tokens
        sample_bytes = data[start_offset:start_offset + sample_size * 4]
        
        try:
            floats = np.frombuffer(sample_bytes, dtype=np.float32)
            mean = np.mean(floats)
            std = np.std(floats)
            min_val = np.min(floats)
            max_val = np.max(floats)
            
            # Embeddings typically have zero mean, small std
            if abs(mean) < 0.1 and std < 0.5 and min_val > -2 and max_val < 2:
                print(f"\nPossible embeddings at offset {start_offset}:")
                print(f"  Mean: {mean:.4f}, Std: {std:.4f}")
                print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
                return start_offset
        except:
            continue
    
    return None


def analyze_specific_token_embeddings(data, offset, vocab_size=50376, d_model=768):
    """Analyze embeddings for specific tokens including 277 (space)."""
    if offset is None:
        print("No embedding offset found")
        return
    
    embedding_size = vocab_size * d_model * 4
    if offset + embedding_size > len(data):
        print("Embedding block would exceed file size")
        return
    
    embeddings_bytes = data[offset:offset + embedding_size]
    embeddings = np.frombuffer(embeddings_bytes, dtype=np.float32)
    embeddings = embeddings.reshape(vocab_size, d_model)
    
    print(f"\nEmbedding matrix shape: {embeddings.shape}")
    
    # Analyze specific tokens
    tokens_to_analyze = [
        (32, "BYTE_SPACE"),      # Byte for space
        (277, "UNIGRAM_SPACE"),  # Unigram space (the problematic one)
        (278, "e_token"),
        (279, "t_token"),
        (280, "a_token"),
        (386, "comma_space"),
        (256, "ATOM_0"),
        (257, "ATOM_1"),
    ]
    
    print("\n=== Token Embedding Analysis ===")
    for token_id, name in tokens_to_analyze:
        if token_id < vocab_size:
            emb = embeddings[token_id]
            norm = np.linalg.norm(emb)
            mean = np.mean(emb)
            std = np.std(emb)
            max_abs = np.max(np.abs(emb))
            
            print(f"\nToken {token_id} ({name}):")
            print(f"  L2 norm:  {norm:.4f}")
            print(f"  Mean:     {mean:.6f}")
            print(f"  Std:      {std:.6f}")
            print(f"  Max abs:  {max_abs:.4f}")
            print(f"  First 5:  {emb[:5]}")
    
    # Compare embedding norms across all tokens
    norms = np.linalg.norm(embeddings, axis=1)
    print("\n=== Embedding Norm Statistics ===")
    print(f"Mean norm: {np.mean(norms):.4f}")
    print(f"Std norm:  {np.std(norms):.4f}")
    print(f"Min norm:  {np.min(norms):.4f} (token {np.argmin(norms)})")
    print(f"Max norm:  {np.max(norms):.4f} (token {np.argmax(norms)})")
    print(f"Token 277 norm: {norms[277]:.4f}")
    print(f"Token 277 percentile: {np.percentile(norms < norms[277], 100):.1f}%")
    
    # Find tokens with highest norms
    top_norm_indices = np.argsort(norms)[-10:][::-1]
    print("\nTop 10 tokens by embedding norm:")
    for idx in top_norm_indices:
        print(f"  Token {idx}: norm={norms[idx]:.4f}")


def main():
    checkpoint_path = Path("resources/models/GRIM-text/checkpoints/checkpoint_epoch_1.bin")
    
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Analyzing: {checkpoint_path}")
    
    data = read_checkpoint_raw(checkpoint_path)
    analyze_float_blocks(data)
    
    offset = try_extract_embeddings(data)
    if offset:
        analyze_specific_token_embeddings(data, offset)
    else:
        print("\nCould not automatically find embeddings.")
        print("The checkpoint uses FlatBuffers format - need proper deserializer.")


if __name__ == "__main__":
    main()

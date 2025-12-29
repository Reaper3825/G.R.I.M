#!/usr/bin/env python3
"""
Trace through the model loading to understand LM head dimensions
"""
import struct
import sys

# Check checkpoint size
checkpoint_path = "D:/G.R.I.M/resources/models/GRIM-text/checkpoints/checkpoint_epoch_5.bin"

try:
    size = __import__('os').path.getsize(checkpoint_path)
    print(f"Checkpoint size: {size:,} bytes ({size/1024/1024:.1f} MB)")
    
    # Model params:
    # 12 layers * (attention + ffn + layernorm)
    # Embedding: vocab_size x d_model = 1337 x 768
    # LM head: 1337 x 768 (tied to embedding)
    # Attention per layer: 3*d_model*d_model + d_model*d_model = 4*768*768 = 2.36M
    # FFN per layer: d_model * 4*d_model + 4*d_model * d_model = 2 * 768 * 3072 = 4.72M
    # LayerNorm: 2*d_model = 1536 per layer
    
    embedding_params = 1337 * 768 * 4  # float32
    print(f"Embedding params: {embedding_params:,} bytes ({embedding_params/1024/1024:.1f} MB)")
    
    # Per-layer params
    d_model = 768
    d_ff = 3072
    attn_params = (3*d_model*d_model + d_model*d_model) * 4  # 4 projections, each d_model x d_model
    ffn_params = (d_model*d_ff + d_ff*d_model) * 4
    ln_params = 2 * d_model * 4 * 2  # 2 layernorms per layer, 2 weights (gamma + bias)
    
    per_layer = attn_params + ffn_params + ln_params
    print(f"Per-layer params: {per_layer:,} bytes ({per_layer/1024/1024:.1f} MB)")
    
    encoder_params = 12 * per_layer
    print(f"12 encoder layers: {encoder_params:,} bytes ({encoder_params/1024/1024:.1f} MB)")
    
    # LM head (tied or separate)
    lm_head_params = 1337 * 768 * 4
    print(f"LM head: {lm_head_params:,} bytes ({lm_head_params/1024/1024:.1f} MB)")
    
    total_estimated = embedding_params + encoder_params + lm_head_params
    print(f"\nTotal estimated: {total_estimated:,} bytes ({total_estimated/1024/1024:.1f} MB)")
    print(f"Checkpoint actual: {size/1024/1024:.1f} MB")
    print(f"Ratio: {size / total_estimated:.2f}x")
    
    print("\nNOTE: If ratio >> 1, checkpoint might have extra data (training state, optimizer)")
    print("If ratio << 1, checkpoint might have different dimensions")

except Exception as e:
    print(f"Error: {e}")

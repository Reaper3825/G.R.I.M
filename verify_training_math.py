#!/usr/bin/env python3
"""
GRIM-text Training Mathematical Verification

Computes expected values from architecture and compares to logged values.
Any mismatch indicates a bug.

Architecture (from ai_config.json):
- vocab_size: 37555
- d_model: 768
- num_layers: 12
- num_heads: 12
- num_kv_heads: 4 (GQA)
- d_ff: 3072
- head_dim: 64 (768 / 12)
- tie_embeddings: true
"""

import re
import json
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ============================================================
# ARCHITECTURE CONSTANTS (from ai_config.json)
# ============================================================
VOCAB_SIZE = 37555
D_MODEL = 768
NUM_LAYERS = 12
NUM_HEADS = 12
NUM_KV_HEADS = 4  # GQA
D_FF = 3072
HEAD_DIM = D_MODEL // NUM_HEADS  # 64
MAX_SEQ_LEN = 2048
TIE_EMBEDDINGS = True
USE_BIAS = True

# ============================================================
# EXPECTED PARAMETER COUNTS
# ============================================================

def compute_expected_params():
    """Compute expected parameter counts for each component."""
    
    # GQA dimensions
    q_dim = NUM_HEADS * HEAD_DIM  # 12 * 64 = 768
    k_dim = NUM_KV_HEADS * HEAD_DIM  # 4 * 64 = 256
    v_dim = NUM_KV_HEADS * HEAD_DIM  # 4 * 64 = 256
    qkv_dim = q_dim + k_dim + v_dim  # 768 + 256 + 256 = 1280
    
    # Per-layer attention
    w_qkv = qkv_dim * D_MODEL  # 1280 * 768 = 983,040
    b_qkv = qkv_dim if USE_BIAS else 0  # 1280
    w_o = D_MODEL * D_MODEL  # 768 * 768 = 589,824
    b_o = D_MODEL if USE_BIAS else 0  # 768
    attn_per_layer = w_qkv + b_qkv + w_o + b_o
    
    # Per-layer FFN
    w1 = D_MODEL * D_FF  # 768 * 3072 = 2,359,296
    b1 = D_FF if USE_BIAS else 0  # 3072
    w2 = D_FF * D_MODEL  # 3072 * 768 = 2,359,296
    b2 = D_MODEL if USE_BIAS else 0  # 768
    ffn_per_layer = w1 + b1 + w2 + b2
    
    # Per-layer RMSNorm (2 per layer: pre-attn, pre-ffn)
    rms_per_layer = D_MODEL * 2  # 768 * 2 = 1536
    
    # Total per layer
    per_layer = attn_per_layer + ffn_per_layer + rms_per_layer
    
    # Embeddings
    token_embeddings = VOCAB_SIZE * D_MODEL  # 37555 * 768 = 28,842,240
    
    # LM Head (tied to embeddings, so 0 extra params)
    lm_head = 0 if TIE_EMBEDDINGS else VOCAB_SIZE * D_MODEL
    
    # Final layer norm
    final_rms = D_MODEL  # 768
    
    # Position embeddings (if using learned, not RoPE/ALiBi)
    # Assuming RoPE, so 0
    pos_embeddings = 0
    
    return {
        "embedding": {
            "token_embeddings": token_embeddings,
            "position_embeddings": pos_embeddings,
            "total": token_embeddings + pos_embeddings,
        },
        "per_layer": {
            "w_qkv": w_qkv,
            "b_qkv": b_qkv,
            "w_o": w_o,
            "b_o": b_o,
            "attn_total": attn_per_layer,
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "ffn_total": ffn_per_layer,
            "rms_total": rms_per_layer,
            "layer_total": per_layer,
        },
        "encoder": {
            "all_layers": per_layer * NUM_LAYERS,
        },
        "lm_head": {
            "weights": lm_head,
            "note": "tied to embeddings" if TIE_EMBEDDINGS else "separate",
        },
        "final_rms": final_rms,
        "total": token_embeddings + pos_embeddings + (per_layer * NUM_LAYERS) + lm_head + final_rms,
    }

def compute_expected_loss():
    """Random baseline loss for vocab size."""
    # Cross-entropy loss for random predictions: -log(1/vocab_size) = log(vocab_size)
    return math.log(VOCAB_SIZE)

def compute_expected_w_qkv_size():
    """Expected W_qkv tensor size in elements."""
    # GQA: Q has num_heads, K/V have num_kv_heads
    q_dim = NUM_HEADS * HEAD_DIM  # 768
    k_dim = NUM_KV_HEADS * HEAD_DIM  # 256
    v_dim = NUM_KV_HEADS * HEAD_DIM  # 256
    qkv_out = q_dim + k_dim + v_dim  # 1280
    
    # W_qkv: [qkv_out, d_model] = [1280, 768]
    return qkv_out * D_MODEL  # 983,040

def compute_mha_w_qkv_size():
    """W_qkv size if using MHA instead of GQA."""
    # MHA: all have num_heads
    qkv_out = 3 * NUM_HEADS * HEAD_DIM  # 3 * 768 = 2304
    return qkv_out * D_MODEL  # 1,769,472

# ============================================================
# LOG PARSING
# ============================================================

def parse_log_for_values(log_path: str):
    """Extract key values from training log."""
    values = {
        "total_params": None,
        "embedding_params": None,
        "encoder_params": None,
        "lm_head_params": None,
        "initial_loss": None,
        "losses": [],
        "lrs": [],
        "grad_norms": [],
        "weight_samples": [],  # (batch, weights_before, weights_after, lr)
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    prev_weights = None
    prev_lr = None
    
    for line in lines:
        # Parameter counts
        m = re.search(r'total_params=(\d+) embedding_params=(\d+) encoder_params=(\d+) lm_head_params=(\d+)', line)
        if m:
            values["total_params"] = int(m.group(1))
            values["embedding_params"] = int(m.group(2))
            values["encoder_params"] = int(m.group(3))
            values["lm_head_params"] = int(m.group(4))
        
        # Initial loss
        m = re.search(r'Initial loss=([0-9.]+)', line)
        if m:
            values["initial_loss"] = float(m.group(1))
        
        # Loss values
        m = re.search(r'POST-FORWARD loss=([0-9.]+)', line)
        if m:
            values["losses"].append(float(m.group(1)))
        
        # LR and weights before optimizer
        m = re.search(r'PRE-OPTIMIZER batch=(\d+) lr=([0-9.]+) .* lm_w\[0:5\]=\[([^\]]+)\]', line)
        if m:
            batch = int(m.group(1))
            lr = float(m.group(2))
            weights = [float(x) for x in m.group(3).split(',')]
            prev_weights = weights
            prev_lr = lr
        
        # Weights after optimizer
        m = re.search(r'POST-OPTIMIZER batch=(\d+) .* lm_w\[0:5\]=\[([^\]]+)\]', line)
        if m and prev_weights:
            batch = int(m.group(1))
            weights_after = [float(x) for x in m.group(2).split(',')]
            values["weight_samples"].append({
                "batch": batch,
                "before": prev_weights,
                "after": weights_after,
                "lr": prev_lr,
            })
            prev_weights = None
        
        # Gradient norms
        m = re.search(r'POST-GRADNORM preclip=([0-9.]+)', line)
        if m:
            values["grad_norms"].append(float(m.group(1)))
    
    return values

# ============================================================
# VERIFICATION
# ============================================================

def verify_all(log_path: str):
    """Run all mathematical verifications."""
    print("=" * 70)
    print("GRIM-text Mathematical Verification")
    print("=" * 70)
    print()
    
    expected = compute_expected_params()
    logged = parse_log_for_values(log_path)
    
    errors = []
    warnings = []
    
    # --------------------------------------------------------
    # 1. PARAMETER COUNT VERIFICATION
    # --------------------------------------------------------
    print("-" * 70)
    print("1. PARAMETER COUNT VERIFICATION")
    print("-" * 70)
    
    print(f"\n   Architecture: d_model={D_MODEL}, layers={NUM_LAYERS}, heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}")
    print(f"   Vocab: {VOCAB_SIZE}, FFN: {D_FF}, Head dim: {HEAD_DIM}")
    print()
    
    # Expected embedding params
    exp_emb = expected["embedding"]["total"]
    log_emb = logged["embedding_params"]
    print(f"   Embedding params:")
    print(f"     Expected: {exp_emb:,} ({VOCAB_SIZE} × {D_MODEL})")
    print(f"     Logged:   {log_emb:,}")
    if log_emb and log_emb != exp_emb:
        errors.append(f"Embedding params mismatch: expected {exp_emb:,}, got {log_emb:,}")
        print(f"     ❌ MISMATCH by {abs(log_emb - exp_emb):,}")
    else:
        print(f"     ✅ MATCH")
    
    # Expected encoder params
    exp_enc = expected["encoder"]["all_layers"]
    log_enc = logged["encoder_params"]
    print(f"\n   Encoder params (12 layers):")
    print(f"     Expected: {exp_enc:,}")
    print(f"     Logged:   {log_enc:,}")
    if log_enc and log_enc != exp_enc:
        diff = log_enc - exp_enc
        diff_per_layer = diff // NUM_LAYERS
        errors.append(f"Encoder params mismatch: expected {exp_enc:,}, got {log_enc:,} (diff={diff:,})")
        print(f"     ❌ MISMATCH by {diff:,} ({diff_per_layer:,} per layer)")
        
        # Try to diagnose: is this MHA vs GQA?
        mha_w_qkv = compute_mha_w_qkv_size()
        gqa_w_qkv = compute_expected_w_qkv_size()
        w_qkv_diff = mha_w_qkv - gqa_w_qkv  # 786,432 per layer
        
        if abs(diff_per_layer - w_qkv_diff) < 1000:
            print(f"     💡 DIAGNOSIS: Encoder is using MHA (num_kv_heads=12) not GQA (num_kv_heads=4)!")
            print(f"        MHA W_qkv: {mha_w_qkv:,} vs GQA W_qkv: {gqa_w_qkv:,}")
            errors.append("MODEL IS USING MHA NOT GQA - num_kv_heads not applied correctly!")
    else:
        print(f"     ✅ MATCH")
    
    # W_qkv size check
    print(f"\n   W_qkv size per layer:")
    print(f"     Expected (GQA):  {compute_expected_w_qkv_size():,} elements")
    print(f"     Expected (MHA):  {compute_mha_w_qkv_size():,} elements")
    print(f"     Checkpoint src:  983,040 (from error log)")
    print(f"     Model dest:      1,769,472 (from error log)")
    print(f"     💡 Checkpoint has GQA weights, model allocates MHA buffers!")
    
    # --------------------------------------------------------
    # 2. LOSS BASELINE VERIFICATION
    # --------------------------------------------------------
    print()
    print("-" * 70)
    print("2. LOSS BASELINE VERIFICATION")
    print("-" * 70)
    
    exp_loss = compute_expected_loss()
    log_loss = logged["initial_loss"]
    
    print(f"\n   Random baseline loss = ln({VOCAB_SIZE}) = {exp_loss:.4f}")
    print(f"   Logged initial loss: {log_loss:.4f}" if log_loss else "   Logged initial loss: NOT FOUND")
    
    if log_loss:
        diff = abs(log_loss - exp_loss)
        if diff < 0.5:
            print(f"   ✅ Within expected range (diff={diff:.4f})")
        else:
            warnings.append(f"Initial loss {log_loss:.4f} differs from baseline {exp_loss:.4f} by {diff:.4f}")
            print(f"   ⚠️  Differs by {diff:.4f}")
    
    # --------------------------------------------------------
    # 3. WEIGHT UPDATE VERIFICATION
    # --------------------------------------------------------
    print()
    print("-" * 70)
    print("3. WEIGHT UPDATE VERIFICATION")
    print("-" * 70)
    
    if logged["weight_samples"]:
        print(f"\n   Analyzing {len(logged['weight_samples'])} weight update samples...")
        
        # Check if weights are changing
        total_delta = 0
        for sample in logged["weight_samples"][:10]:
            before = np.array(sample["before"])
            after = np.array(sample["after"])
            delta = np.linalg.norm(after - before)
            total_delta += delta
        
        avg_delta = total_delta / min(10, len(logged["weight_samples"]))
        print(f"   Average weight delta (first 10 batches): {avg_delta:.10f}")
        
        if avg_delta == 0:
            errors.append("CRITICAL: Weights are NOT changing!")
            print(f"   ❌ CRITICAL: Weights are NOT changing!")
        elif avg_delta < 1e-10:
            warnings.append(f"Weights changing very slowly: {avg_delta:.2e}")
            print(f"   ⚠️  Weights changing very slowly")
        else:
            print(f"   ✅ Weights are updating")
        
        # Check direction consistency
        print(f"\n   Sample weight changes (batch, delta, direction):")
        for sample in logged["weight_samples"][:5]:
            before = np.array(sample["before"])
            after = np.array(sample["after"])
            delta = after - before
            lr = sample["lr"]
            print(f"     Batch {sample['batch']}: lr={lr:.8f}")
            for i in range(min(3, len(delta))):
                print(f"       w[{i}]: {before[i]:.6f} → {after[i]:.6f} (Δ={delta[i]:+.8f})")
    else:
        warnings.append("No weight samples found in log")
        print("   No weight samples found in log")
    
    # --------------------------------------------------------
    # 4. GRADIENT NORM VERIFICATION
    # --------------------------------------------------------
    print()
    print("-" * 70)
    print("4. GRADIENT NORM ANALYSIS")
    print("-" * 70)
    
    if logged["grad_norms"]:
        grad_norms = logged["grad_norms"]
        print(f"\n   Gradient norms over training:")
        print(f"     First 5:  {grad_norms[:5]}")
        print(f"     Last 5:   {grad_norms[-5:]}")
        print(f"     Mean:     {np.mean(grad_norms):.4f}")
        print(f"     Std:      {np.std(grad_norms):.4f}")
        
        # Check for gradient collapse or explosion
        if np.mean(grad_norms[-50:]) < 0.1:
            errors.append("Gradient collapse: norms near zero in late training")
            print(f"   ❌ Gradient collapse detected!")
        elif np.mean(grad_norms[-50:]) > np.mean(grad_norms[:50]) * 10:
            warnings.append("Gradient explosion: norms increasing")
            print(f"   ⚠️  Gradients increasing (possible explosion)")
        else:
            print(f"   ✅ Gradient norms appear stable")
    
    # --------------------------------------------------------
    # 5. LOSS TRAJECTORY ANALYSIS
    # --------------------------------------------------------
    print()
    print("-" * 70)
    print("5. LOSS TRAJECTORY ANALYSIS")
    print("-" * 70)
    
    if logged["losses"]:
        losses = logged["losses"]
        
        # Compute loss at different points
        n = len(losses)
        checkpoints = [0, n//4, n//2, 3*n//4, n-1]
        
        print(f"\n   Loss at key points (total {n} batches):")
        for i in checkpoints:
            print(f"     Batch {i+1}: {losses[i]:.4f}")
        
        # Compute improvement rates
        first_quarter = losses[0] - losses[n//4]
        second_quarter = losses[n//4] - losses[n//2]
        third_quarter = losses[n//2] - losses[3*n//4]
        fourth_quarter = losses[3*n//4] - losses[-1]
        
        print(f"\n   Improvement by quarter:")
        print(f"     Q1 (batch 1-{n//4}):      {first_quarter:+.4f}")
        print(f"     Q2 (batch {n//4}-{n//2}): {second_quarter:+.4f}")
        print(f"     Q3 (batch {n//2}-{3*n//4}): {third_quarter:+.4f}")
        print(f"     Q4 (batch {3*n//4}-{n}):  {fourth_quarter:+.4f}")
        
        if first_quarter > 0.5 and abs(fourth_quarter) < 0.1:
            print(f"\n   💡 PLATEAU PATTERN: Strong early learning ({first_quarter:.2f}), stalled late ({fourth_quarter:.2f})")
            
            # Find plateau start
            window = 50
            for i in range(window, n - window):
                improvement = losses[i-window] - losses[i]
                if improvement < 0.1 and first_quarter > 0.5:
                    print(f"   💡 Plateau starts around batch {i-window}")
                    break
    
    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if errors:
        print("\n🔴 ERRORS (likely bugs):")
        for e in errors:
            print(f"   • {e}")
    
    if warnings:
        print("\n🟡 WARNINGS:")
        for w in warnings:
            print(f"   • {w}")
    
    if not errors and not warnings:
        print("\n✅ No obvious mathematical inconsistencies found.")
        print("   The bug may be in:")
        print("   • Optimizer momentum/velocity accumulation")
        print("   • Gradient buffer overwriting")
        print("   • cuBLAS beta parameter (0 vs 1)")
        print("   • Weight update applied then immediately reverted")
    
    return errors, warnings

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        log_dir = Path("D:/G.R.I.M/resources/models/GRIM-text/training/logs")
        logs = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            log_path = str(logs[0])
            print(f"Using most recent log: {log_path}\n")
        else:
            print("No logs found!")
            sys.exit(1)
    else:
        log_path = sys.argv[1]
    
    verify_all(log_path)

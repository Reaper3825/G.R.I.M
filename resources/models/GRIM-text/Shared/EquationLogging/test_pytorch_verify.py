#!/usr/bin/env python3
"""
Test script for pytorch_verify.py - validates reference implementations

Run with: python test_pytorch_verify.py
"""

import os
import sys
import struct
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_verify as pv


def write_test_tensor(filepath: str, data: np.ndarray):
    """Write tensor to binary file."""
    pv.write_tensor(filepath, data)


def write_test_int_tensor(filepath: str, data: np.ndarray):
    """Write int tensor to binary file."""
    data = np.asarray(data, dtype=np.int32)
    shape = data.shape
    
    with open(filepath, 'wb') as f:
        f.write(struct.pack('i', len(shape)))
        for d in shape:
            f.write(struct.pack('i', d))
        f.write(data.tobytes())


def test_rmsnorm():
    """Test RMSNorm reference implementation."""
    print("\n=== Testing RMSNorm ===")
    
    # Create test inputs
    batch, hidden = 4, 768
    eps = 1e-5
    
    x = np.random.randn(batch, hidden).astype(np.float32) * 0.1
    gamma = np.ones(hidden, dtype=np.float32)
    
    # Write inputs
    os.makedirs("temp/test", exist_ok=True)
    write_test_tensor("temp/test/x.bin", x)
    write_test_tensor("temp/test/gamma.bin", gamma)
    
    # Run
    pv.rmsnorm("temp/test/x.bin", "temp/test/gamma.bin", "temp/test/out.bin", eps)
    
    # Read output
    out, _ = pv.read_tensor("temp/test/out.bin")
    
    # Verify: output RMS should be ~1 (gamma=1)
    out_rms = np.sqrt(np.mean(out ** 2, axis=-1))
    print(f"  Input RMS: {np.sqrt(np.mean(x**2)):.6f}")
    print(f"  Output RMS per row: mean={np.mean(out_rms):.6f}, std={np.std(out_rms):.6f}")
    
    assert np.allclose(out_rms, 1.0, atol=0.01), f"RMSNorm output RMS should be ~1, got {out_rms.mean()}"
    print("  ✓ PASSED")


def test_gelu():
    """Test GELU reference implementation."""
    print("\n=== Testing GELU ===")
    
    x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    
    os.makedirs("temp/test", exist_ok=True)
    write_test_tensor("temp/test/gelu_in.bin", x)
    
    pv.gelu("temp/test/gelu_in.bin", "temp/test/gelu_out.bin")
    
    out, _ = pv.read_tensor("temp/test/gelu_out.bin")
    
    # Known GELU values (tanh approximation)
    # GELU(-2) ≈ -0.0454, GELU(-1) ≈ -0.1588, GELU(0) = 0, GELU(1) ≈ 0.8412, GELU(2) ≈ 1.9546
    expected = np.array([-0.0454, -0.1588, 0.0, 0.8412, 1.9546], dtype=np.float32)
    
    print(f"  Input: {x}")
    print(f"  Output: {out}")
    print(f"  Expected: {expected}")
    
    assert np.allclose(out, expected, atol=0.01), f"GELU mismatch"
    print("  ✓ PASSED")


def test_matmul():
    """Test MatMul reference implementation."""
    print("\n=== Testing MatMul ===")
    
    M, K, N = 16, 32, 64
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    os.makedirs("temp/test", exist_ok=True)
    write_test_tensor("temp/test/mm_a.bin", A)
    write_test_tensor("temp/test/mm_b.bin", B)
    
    pv.matmul("temp/test/mm_a.bin", "temp/test/mm_b.bin", "temp/test/mm_out.bin")
    
    out, _ = pv.read_tensor("temp/test/mm_out.bin")
    expected = A @ B
    
    print(f"  A: {A.shape}, B: {B.shape}, C: {out.shape}")
    print(f"  Max diff: {np.max(np.abs(out - expected)):.2e}")
    
    assert np.allclose(out, expected, rtol=1e-4, atol=1e-6), "MatMul mismatch"
    print("  ✓ PASSED")


def test_cross_entropy():
    """Test Cross-Entropy loss reference implementation."""
    print("\n=== Testing Cross-Entropy ===")
    
    batch, vocab = 8, 1000
    logits = np.random.randn(batch, vocab).astype(np.float32)
    targets = np.random.randint(0, vocab, size=batch).astype(np.int32)
    
    os.makedirs("temp/test", exist_ok=True)
    write_test_tensor("temp/test/ce_logits.bin", logits)
    write_test_int_tensor("temp/test/ce_targets.bin", targets)
    
    pv.cross_entropy("temp/test/ce_logits.bin", "temp/test/ce_targets.bin", "temp/test/ce_loss.bin")
    
    loss, _ = pv.read_tensor("temp/test/ce_loss.bin")
    
    # Manual CE calculation for verification
    logits_max = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    target_probs = softmax[np.arange(batch), targets]
    expected_loss = -np.mean(np.log(target_probs + 1e-10))
    
    print(f"  Loss: {loss[0]:.6f}")
    print(f"  Expected: {expected_loss:.6f}")
    
    assert np.allclose(loss[0], expected_loss, rtol=1e-4), f"CE loss mismatch"
    print("  ✓ PASSED")


def test_sdpa():
    """Test Scaled Dot-Product Attention reference implementation."""
    print("\n=== Testing SDPA ===")
    
    batch, heads, seq, d = 2, 4, 16, 32
    scale = 1.0 / np.sqrt(d)
    
    Q = np.random.randn(batch, heads, seq, d).astype(np.float32) * 0.1
    K = np.random.randn(batch, heads, seq, d).astype(np.float32) * 0.1
    V = np.random.randn(batch, heads, seq, d).astype(np.float32) * 0.1
    
    os.makedirs("temp/test", exist_ok=True)
    write_test_tensor("temp/test/sdpa_q.bin", Q)
    write_test_tensor("temp/test/sdpa_k.bin", K)
    write_test_tensor("temp/test/sdpa_v.bin", V)
    
    pv.sdpa("temp/test/sdpa_q.bin", "temp/test/sdpa_k.bin", "temp/test/sdpa_v.bin",
            "temp/test/sdpa_out.bin", scale)
    
    out, _ = pv.read_tensor("temp/test/sdpa_out.bin")
    
    # Manual SDPA for verification
    scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected = np.matmul(attn, V)
    
    print(f"  Output shape: {out.shape}")
    print(f"  Max diff: {np.max(np.abs(out - expected)):.2e}")
    
    assert np.allclose(out, expected, rtol=1e-4, atol=1e-6), "SDPA mismatch"
    print("  ✓ PASSED")


def test_adamw():
    """Test AdamW optimizer step reference implementation."""
    print("\n=== Testing AdamW ===")
    
    n_params = 1000
    lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.99, 1e-8, 0.01
    step = 10
    
    w = np.random.randn(n_params).astype(np.float32)
    g = np.random.randn(n_params).astype(np.float32) * 0.1
    m = np.zeros(n_params, dtype=np.float32)
    v = np.zeros(n_params, dtype=np.float32)
    
    os.makedirs("temp/test", exist_ok=True)
    write_test_tensor("temp/test/adamw_w.bin", w)
    write_test_tensor("temp/test/adamw_g.bin", g)
    write_test_tensor("temp/test/adamw_m.bin", m)
    write_test_tensor("temp/test/adamw_v.bin", v)
    
    pv.adamw("temp/test/adamw_w.bin", "temp/test/adamw_g.bin",
             "temp/test/adamw_m.bin", "temp/test/adamw_v.bin",
             "temp/test/adamw_out.bin",
             lr, beta1, beta2, eps, wd, step)
    
    w_new, _ = pv.read_tensor("temp/test/adamw_out.bin")
    
    # Manual AdamW for verification
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * (g ** 2)
    m_hat = m_new / (1 - beta1 ** step)
    v_hat = v_new / (1 - beta2 ** step)
    expected = w - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * w)
    
    print(f"  Weight change norm: {np.linalg.norm(w_new - w):.6f}")
    print(f"  Max diff from expected: {np.max(np.abs(w_new - expected)):.2e}")
    
    assert np.allclose(w_new, expected, rtol=1e-4, atol=1e-6), "AdamW mismatch"
    print("  ✓ PASSED")


def cleanup():
    """Remove temp test files."""
    import shutil
    if os.path.exists("temp/test"):
        shutil.rmtree("temp/test")


def main():
    print("=" * 60)
    print("PyTorch Verification Module Tests")
    print("=" * 60)
    
    try:
        test_rmsnorm()
        test_gelu()
        test_matmul()
        test_cross_entropy()
        test_sdpa()
        test_adamw()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        
    finally:
        cleanup()


if __name__ == "__main__":
    main()

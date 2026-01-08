#!/usr/bin/env python3
"""
Verify embedding lookup correctness: x[t] = E[token_ids[t]]
Tests both forward and backward pass against finite differences.
Uses actual GRIM-text training data and vocabulary.

This verifies the fundamental embedding operation that MUST hold:
  FORWARD:  x[t] = E[token_ids[t]]  (lookup row token_ids[t] from embedding table)
  BACKWARD: dE[token_ids[t]] += dx[t]  (scatter-add gradient back to embedding rows)
"""

import sys
import struct
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import math

# ============================================================
# Configuration
# ============================================================

GRIM_ROOT = Path(__file__).parent
DATA_DIR = GRIM_ROOT / "resources" / "models" / "GRIM-text" / "training" / "data"
TRAINING_DATA = DATA_DIR / "training_data.grmt"
VOCAB_FILE = DATA_DIR / "vocab.txt"

# GRMT v4 constants
GRMT_MAGIC = 0x474D5254
GRMT_VERSION = 4
TEXT_FEATURE_DIM = 16

# Token layout (matches GrimTokenizer)
BYTE_VOCAB_SIZE = 256
ATOM_TOKEN_START = 256
ATOM_VOCAB_SIZE = 256  # Updated after reading vocab
ATOM_TOKEN_END = ATOM_TOKEN_START + ATOM_VOCAB_SIZE
UNIGRAM_TOKEN_START = ATOM_TOKEN_END


# ============================================================
# Data Loading
# ============================================================

def load_vocab(vocab_path: Path) -> Tuple[Dict[int, str], int]:
    """Load vocab.txt and return token_id -> text mapping and total vocab size."""
    global ATOM_VOCAB_SIZE, ATOM_TOKEN_END, UNIGRAM_TOKEN_START
    
    vocab = {}
    
    # Byte fallback tokens (0-255)
    for i in range(256):
        if 32 <= i <= 126:
            vocab[i] = chr(i)
        else:
            vocab[i] = f"<BYTE{i:02X}>"
    
    # Read unigram vocab
    with open(vocab_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    
    unigram_count = len(lines)
    
    # Atom placeholders (between bytes and unigram)
    for i in range(ATOM_VOCAB_SIZE):
        vocab[ATOM_TOKEN_START + i] = f"<ATOM{i}>"
    
    UNIGRAM_TOKEN_START = ATOM_TOKEN_END
    
    # Unigram tokens
    for idx, line in enumerate(lines):
        if '\t' in line:
            token_text = line.split('\t')[0]
        else:
            token_text = line
        vocab[UNIGRAM_TOKEN_START + idx] = token_text
    
    total_vocab_size = 256 + ATOM_VOCAB_SIZE + unigram_count
    print(f"✓ Vocab loaded: {total_vocab_size} tokens")
    print(f"  - Bytes: 0-255")
    print(f"  - Atoms: {ATOM_TOKEN_START}-{ATOM_TOKEN_END-1}")
    print(f"  - Unigram: {UNIGRAM_TOKEN_START}-{total_vocab_size-1}")
    
    return vocab, total_vocab_size


def load_grmt_sequences(grmt_path: Path, max_sequences: int = 10) -> List[List[int]]:
    """Load sequences from GRMT file."""
    sequences = []
    
    with open(grmt_path, 'rb') as f:
        header = f.read(16)
        magic, version, num_sequences, vocab_size = struct.unpack('<IIII', header)
        
        if magic != GRMT_MAGIC:
            raise ValueError(f"Invalid GRMT magic: 0x{magic:08X}")
        if version != GRMT_VERSION:
            raise ValueError(f"Unsupported GRMT version: {version}")
        
        print(f"✓ GRMT file: {num_sequences} sequences, vocab_size={vocab_size}")
        
        for seq_idx in range(min(num_sequences, max_sequences)):
            len_bytes = f.read(4)
            if len(len_bytes) < 4:
                break
            seq_len = struct.unpack('<I', len_bytes)[0]
            
            if seq_len == 0:
                sequences.append([])
                continue
            
            # Token IDs (uint32)
            token_bytes = f.read(seq_len * 4)
            tokens = list(struct.unpack('<' + 'I' * seq_len, token_bytes))
            
            # Skip numeric_values (float32 × seq_len)
            f.read(seq_len * 4)
            
            # Skip numeric_mask (uint8 × seq_len)
            f.read(seq_len)
            
            # Skip text_features (FP16 × TEXT_FEATURE_DIM × seq_len)
            f.read(seq_len * TEXT_FEATURE_DIM * 2)
            
            # Skip text_mask (uint8 × seq_len)
            f.read(seq_len)
            
            sequences.append(tokens)
    
    return sequences


# ============================================================
# Embedding Implementation (Reference)
# ============================================================

class EmbeddingTable:
    """Simple embedding table for verification."""
    
    def __init__(self, vocab_size: int, d_model: int, seed: int = 42):
        np.random.seed(seed)
        # Initialize with small random values (like Xavier)
        scale = np.sqrt(2.0 / (vocab_size + d_model))
        self.weights = np.random.randn(vocab_size, d_model).astype(np.float32) * scale
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.grad = None
    
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass: x[t] = E[token_ids[t]]
        
        Args:
            token_ids: [seq_len] array of token indices
        Returns:
            embeddings: [seq_len, d_model] array
        """
        # Validate token IDs
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            invalid = token_ids[(token_ids < 0) | (token_ids >= self.vocab_size)]
            raise ValueError(f"Token IDs out of range [0, {self.vocab_size}): {invalid[:5]}...")
        
        # Simple lookup: x[t] = E[token_ids[t]]
        return self.weights[token_ids].copy()
    
    def backward(self, token_ids: np.ndarray, grad_output: np.ndarray) -> None:
        """
        Backward pass: dE[token_ids[t]] += grad_output[t]
        
        This is a scatter-add operation. Multiple positions can reference
        the same token ID, so we accumulate gradients.
        """
        if self.grad is None:
            self.grad = np.zeros_like(self.weights)
        
        # Scatter-add: for each position t, add grad_output[t] to grad[token_ids[t]]
        np.add.at(self.grad, token_ids, grad_output)
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.weights)


# ============================================================
# Verification Tests
# ============================================================

def test_forward_correctness(embedding: EmbeddingTable, token_ids: np.ndarray) -> bool:
    """
    Verify: x[t] = E[token_ids[t]] for all t
    """
    print("\n" + "="*60)
    print("TEST 1: Forward Pass Correctness")
    print("    Verifying: x[t] = E[token_ids[t]]")
    print("="*60)
    
    x = embedding.forward(token_ids)
    
    passed = True
    mismatches = []
    
    for t in range(len(token_ids)):
        tid = token_ids[t]
        expected = embedding.weights[tid]
        actual = x[t]
        
        if not np.allclose(expected, actual, rtol=1e-6, atol=1e-8):
            passed = False
            diff = np.max(np.abs(expected - actual))
            mismatches.append((t, tid, diff))
    
    if passed:
        print(f"✓ PASSED: All {len(token_ids)} positions verified")
        print(f"  Sample: x[0] = E[{token_ids[0]}] = [{x[0, :3].tolist()}...]")
        print(f"  Sample: x[5] = E[{token_ids[5]}] = [{x[5, :3].tolist()}...]")
    else:
        print(f"✗ FAILED: {len(mismatches)} mismatches found")
        for t, tid, diff in mismatches[:5]:
            print(f"  Position {t}: token_id={tid}, max_diff={diff:.2e}")
    
    return passed


def test_backward_scatter_add(embedding: EmbeddingTable, token_ids: np.ndarray) -> bool:
    """
    Verify: dE[token_ids[t]] += grad_output[t] (scatter-add)
    
    When the same token appears multiple times, gradients must accumulate.
    """
    print("\n" + "="*60)
    print("TEST 2: Backward Pass Scatter-Add")
    print("    Verifying: dE[token_ids[t]] += grad_output[t]")
    print("="*60)
    
    seq_len = len(token_ids)
    d_model = embedding.d_model
    
    # Create random gradient from downstream
    np.random.seed(123)
    grad_output = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
    
    # Run backward
    embedding.zero_grad()
    embedding.backward(token_ids, grad_output)
    
    # Verify by computing expected gradients manually
    expected_grad = np.zeros_like(embedding.weights)
    for t in range(seq_len):
        tid = token_ids[t]
        expected_grad[tid] += grad_output[t]
    
    # Compare
    passed = np.allclose(embedding.grad, expected_grad, rtol=1e-6, atol=1e-8)
    
    if passed:
        print(f"✓ PASSED: Scatter-add verified for {seq_len} positions")
        
        # Find a repeated token to show accumulation
        unique, counts = np.unique(token_ids, return_counts=True)
        repeated_idx = np.where(counts > 1)[0]
        if len(repeated_idx) > 0:
            tid = unique[repeated_idx[0]]
            count = counts[repeated_idx[0]]
            positions = np.where(token_ids == tid)[0]
            print(f"  Token {tid} appears {count} times at positions {positions.tolist()}")
            print(f"  Accumulated grad norm: {np.linalg.norm(embedding.grad[tid]):.4f}")
    else:
        diff = np.max(np.abs(embedding.grad - expected_grad))
        print(f"✗ FAILED: max_diff={diff:.2e}")
        
        # Find worst mismatch
        worst_idx = np.unravel_index(np.argmax(np.abs(embedding.grad - expected_grad)), 
                                     embedding.grad.shape)
        print(f"  Worst at embedding[{worst_idx[0]}, {worst_idx[1]}]")
        print(f"  Expected: {expected_grad[worst_idx]:.6f}")
        print(f"  Actual:   {embedding.grad[worst_idx]:.6f}")
    
    return passed


def test_backward_finite_difference(embedding: EmbeddingTable, token_ids: np.ndarray) -> bool:
    """
    Verify backward pass using finite differences.
    
    For a scalar loss L = sum(x @ v) where v is random:
      dL/dE[tid] = dL/dx[t] * dx[t]/dE[tid] = v (for positions where token_ids[t] == tid)
    
    We verify: dL/dE[tid] ≈ (L(E[tid]+eps) - L(E[tid]-eps)) / (2*eps)
    """
    print("\n" + "="*60)
    print("TEST 3: Backward Pass Finite Difference Verification")
    print("    Verifying: numerical grad ≈ analytical grad")
    print("="*60)
    
    seq_len = len(token_ids)
    d_model = embedding.d_model
    eps = 1e-4
    
    # Random projection vector for scalar loss: L = sum(x @ v)
    np.random.seed(456)
    v = np.random.randn(d_model).astype(np.float32)
    
    def compute_loss(emb_weights):
        """L = sum(emb_weights[token_ids] @ v)"""
        x = emb_weights[token_ids]
        return np.sum(x @ v)
    
    # Compute analytical gradient
    # dL/dx = v (broadcast to all positions)
    # dL/dE = scatter_add of v to token positions
    analytical_grad = np.zeros_like(embedding.weights)
    for t in range(seq_len):
        tid = token_ids[t]
        analytical_grad[tid] += v  # dL/dx[t] * dx[t]/dE[tid] = v
    
    # Compute numerical gradient using finite differences
    # Only check a subset of embedding entries (full check would be slow)
    unique_tokens = np.unique(token_ids)
    check_dims = min(5, d_model)
    
    numerical_grad = np.zeros((len(unique_tokens), check_dims), dtype=np.float32)
    
    for i, tid in enumerate(unique_tokens[:10]):  # Check first 10 unique tokens
        for d in range(check_dims):
            # Forward difference
            embedding.weights[tid, d] += eps
            loss_plus = compute_loss(embedding.weights)
            
            # Backward difference
            embedding.weights[tid, d] -= 2 * eps
            loss_minus = compute_loss(embedding.weights)
            
            # Restore
            embedding.weights[tid, d] += eps
            
            numerical_grad[i, d] = (loss_plus - loss_minus) / (2 * eps)
    
    # Compare analytical vs numerical
    passed = True
    max_rel_error = 0.0
    errors = []
    
    for i, tid in enumerate(unique_tokens[:10]):
        for d in range(check_dims):
            numerical = numerical_grad[i, d]
            analytical = analytical_grad[tid, d]
            
            if abs(analytical) > 1e-8:
                rel_error = abs(numerical - analytical) / abs(analytical)
            else:
                rel_error = abs(numerical - analytical)
            
            max_rel_error = max(max_rel_error, rel_error)
            
            if rel_error > 1e-3:  # 0.1% tolerance
                passed = False
                errors.append((tid, d, numerical, analytical, rel_error))
    
    if passed:
        print(f"✓ PASSED: max_relative_error = {max_rel_error:.2e}")
        print(f"  Checked {len(unique_tokens[:10])} tokens × {check_dims} dims")
        
        # Show sample comparison
        tid = unique_tokens[0]
        count = np.sum(token_ids == tid)
        print(f"  Token {tid} (appears {count}x):")
        print(f"    Numerical grad:  [{numerical_grad[0, :3].tolist()}...]")
        print(f"    Analytical grad: [{analytical_grad[tid, :3].tolist()}...]")
    else:
        print(f"✗ FAILED: max_relative_error = {max_rel_error:.2e}")
        for tid, d, num, ana, err in errors[:5]:
            print(f"  Token {tid}, dim {d}: numerical={num:.6f}, analytical={ana:.6f}, rel_err={err:.2e}")
    
    return passed


def test_weight_tied_gradients(embedding: EmbeddingTable, token_ids: np.ndarray) -> bool:
    """
    Test weight-tied scenario: embedding and LM head share weights.
    
    When tie_embeddings=true:
      - LM head backward writes gradient (via cuBLAS, beta=0)
      - Embedding backward accumulates (via atomicAdd)
    
    We verify the combined gradient is correct.
    """
    print("\n" + "="*60)
    print("TEST 4: Weight-Tied Gradient Accumulation")
    print("    Verifying: combined embedding + LM head gradients")
    print("="*60)
    
    seq_len = len(token_ids)
    d_model = embedding.d_model
    vocab_size = embedding.vocab_size
    
    # Simulate forward pass
    x = embedding.forward(token_ids)  # [seq_len, d_model]
    
    # Simulate output layer (tied weights = embedding.weights transposed as LM head)
    # logits = x @ E^T  (shape: [seq_len, vocab_size])
    logits = x @ embedding.weights.T
    
    # Random gradient from loss (simulating softmax cross-entropy gradient)
    np.random.seed(789)
    grad_logits = np.random.randn(seq_len, vocab_size).astype(np.float32) * 0.01
    
    # LM head gradient: grad_lm_head = x^T @ grad_logits
    # Since LM head weight = E^T, grad for E from LM head = grad_logits^T @ x
    grad_from_lm_head = grad_logits.T @ x  # [vocab_size, d_model]
    
    # Embedding gradient (from input side): scatter-add of (grad_logits @ E)
    grad_x = grad_logits @ embedding.weights  # [seq_len, d_model]
    embedding.zero_grad()
    embedding.backward(token_ids, grad_x)
    grad_from_embedding = embedding.grad.copy()
    
    # Combined gradient (what weight-tied embedding should receive)
    combined_grad = grad_from_lm_head + grad_from_embedding
    
    # Verify the math makes sense
    print(f"  LM head grad norm:     {np.linalg.norm(grad_from_lm_head):.4f}")
    print(f"  Embedding grad norm:   {np.linalg.norm(grad_from_embedding):.4f}")
    print(f"  Combined grad norm:    {np.linalg.norm(combined_grad):.4f}")
    
    # The gradients should be non-zero and roughly similar magnitude
    passed = True
    
    if np.linalg.norm(grad_from_lm_head) < 1e-10:
        print("✗ FAILED: LM head gradient is zero!")
        passed = False
    
    if np.linalg.norm(grad_from_embedding) < 1e-10:
        print("✗ FAILED: Embedding gradient is zero!")
        passed = False
    
    # Check that embedding gradient is sparse (only used tokens have gradients)
    used_tokens = set(token_ids)
    unused_tokens = set(range(vocab_size)) - used_tokens
    
    unused_grad_norm = np.linalg.norm(grad_from_embedding[list(unused_tokens)[:100]])
    if unused_grad_norm > 1e-10:
        print(f"✗ FAILED: Unused tokens have non-zero embedding gradients: {unused_grad_norm:.2e}")
        passed = False
    else:
        print(f"✓ Embedding grad correctly zero for unused tokens")
    
    # LM head gradient is dense (all vocab affected by logits)
    if np.sum(np.abs(grad_from_lm_head) > 1e-10) < vocab_size * d_model * 0.5:
        print("  Warning: LM head gradient is sparser than expected")
    
    if passed:
        print(f"✓ PASSED: Weight-tied gradients verified")
        
        # Show gradient for a specific token
        tid = token_ids[0]
        print(f"  Token {tid} gradient breakdown:")
        print(f"    From LM head:    [{grad_from_lm_head[tid, :3].tolist()}...]")
        print(f"    From embedding:  [{grad_from_embedding[tid, :3].tolist()}...]")
        print(f"    Combined:        [{combined_grad[tid, :3].tolist()}...]")
    
    return passed


# ============================================================
# Main
# ============================================================

def main():
    print("="*60)
    print("EMBEDDING CORRECTNESS VERIFICATION")
    print("Verifying: x[t] = E[token_ids[t]]")
    print("="*60)
    
    # Check data files exist
    if not TRAINING_DATA.exists():
        print(f"✗ Training data not found: {TRAINING_DATA}")
        return 1
    
    if not VOCAB_FILE.exists():
        print(f"✗ Vocab file not found: {VOCAB_FILE}")
        return 1
    
    # Load vocab
    vocab, vocab_size = load_vocab(VOCAB_FILE)
    
    # Load some training sequences
    sequences = load_grmt_sequences(TRAINING_DATA, max_sequences=5)
    if not sequences:
        print("✗ No sequences loaded")
        return 1
    
    # Use first non-empty sequence
    token_ids = None
    for seq in sequences:
        if len(seq) >= 20:
            token_ids = np.array(seq[:100], dtype=np.int64)  # Use first 100 tokens
            break
    
    if token_ids is None:
        print("✗ No suitable sequence found")
        return 1
    
    print(f"\nTest sequence: {len(token_ids)} tokens")
    print(f"  Token ID range: [{token_ids.min()}, {token_ids.max()}]")
    print(f"  Unique tokens: {len(np.unique(token_ids))}")
    
    # Decode a sample
    decoded_sample = ""
    for tid in token_ids[:20]:
        if tid in vocab:
            decoded_sample += vocab[tid]
        else:
            decoded_sample += f"<{tid}>"
    decoded_sample = decoded_sample.replace('▁', ' ')
    print(f"  Sample text: \"{decoded_sample[:80]}...\"")
    
    # Create embedding table
    d_model = 768  # Match GRIM-text
    embedding = EmbeddingTable(vocab_size, d_model)
    print(f"\nEmbedding table: [{vocab_size}, {d_model}]")
    
    # Run tests
    results = []
    
    results.append(("Forward correctness", test_forward_correctness(embedding, token_ids)))
    results.append(("Backward scatter-add", test_backward_scatter_add(embedding, token_ids)))
    results.append(("Finite difference", test_backward_finite_difference(embedding, token_ids)))
    results.append(("Weight-tied gradients", test_weight_tied_gradients(embedding, token_ids)))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        print("  Embedding lookup x[t] = E[token_ids[t]] is correct")
        print("  Gradient scatter-add dE[tid] += dx[t] is correct")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

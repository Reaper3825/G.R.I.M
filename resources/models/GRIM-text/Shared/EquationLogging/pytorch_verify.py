#!/usr/bin/env python3
"""
pytorch_verify.py - PyTorch reference implementations for GRIM-text verification

This script provides ground-truth PyTorch implementations for comparison
with GRIM-text CUDA kernels. Called via subprocess from PyTorchVerify.hpp.

Usage:
    python pytorch_verify.py <operation> <input_files...> <output_file> [params...]
    
Operations:
    rmsnorm     - RMSNorm: y = x * gamma / sqrt(mean(x²) + eps)
    embedding   - Embedding lookup with scale
    matmul      - Matrix multiplication: C = A @ B
    cross_entropy - Softmax cross-entropy loss
    gelu        - GELU activation
    adamw       - AdamW optimizer step
    sdpa        - Scaled dot-product attention
"""

import sys
import struct
import numpy as np

# Check PyTorch availability
try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available, using NumPy fallbacks", file=sys.stderr)


# ============================================================================
# BINARY TENSOR I/O - Format matches PyTorchVerify.hpp
# ============================================================================

def read_tensor(filepath: str) -> tuple:
    """Read tensor from binary file. Returns (data, shape)."""
    with open(filepath, 'rb') as f:
        # Read shape
        ndim = struct.unpack('i', f.read(4))[0]
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('i', f.read(4))[0])
        
        # Read data as float32
        num_elements = 1
        for d in shape:
            num_elements *= d
        
        data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32)
        return data.reshape(shape), shape


def read_int_tensor(filepath: str) -> tuple:
    """Read integer tensor from binary file."""
    with open(filepath, 'rb') as f:
        ndim = struct.unpack('i', f.read(4))[0]
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('i', f.read(4))[0])
        
        num_elements = 1
        for d in shape:
            num_elements *= d
        
        data = np.frombuffer(f.read(num_elements * 4), dtype=np.int32)
        return data.reshape(shape), shape


def write_tensor(filepath: str, data: np.ndarray):
    """Write tensor to binary file."""
    data = np.asarray(data, dtype=np.float32)
    shape = data.shape
    
    with open(filepath, 'wb') as f:
        # Write shape
        f.write(struct.pack('i', len(shape)))
        for d in shape:
            f.write(struct.pack('i', d))
        
        # Write data
        f.write(data.tobytes())


# ============================================================================
# OPERATION IMPLEMENTATIONS
# ============================================================================

def rmsnorm(x_path: str, gamma_path: str, out_path: str, eps: float):
    """
    RMSNorm: y = x * gamma / sqrt(mean(x²) + eps)
    
    Formula breakdown:
        rms = sqrt(mean(x², dim=-1, keepdim=True) + eps)
        y = x / rms * gamma
    """
    x, _ = read_tensor(x_path)
    gamma, _ = read_tensor(gamma_path)
    
    if PYTORCH_AVAILABLE:
        x_t = torch.from_numpy(x.copy())
        gamma_t = torch.from_numpy(gamma.copy())
        
        # RMSNorm computation
        rms = torch.sqrt(torch.mean(x_t ** 2, dim=-1, keepdim=True) + eps)
        y = x_t / rms * gamma_t
        
        result = y.numpy()
    else:
        # NumPy fallback
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        result = x / rms * gamma
    
    write_tensor(out_path, result)
    print(f"[PyTorch] RMSNorm: input_rms={np.sqrt(np.mean(x**2)):.6f} output_rms={np.sqrt(np.mean(result**2)):.6f}")


def embedding(weight_path: str, tokens_path: str, out_path: str, scale: float):
    """
    Embedding lookup with scale: y = embedding[tokens] * scale
    
    This matches AIAYN sqrt(d_model) embedding scaling.
    """
    weight, _ = read_tensor(weight_path) 
    tokens, _ = read_int_tensor(tokens_path)
    tokens = tokens.flatten()
    
    if PYTORCH_AVAILABLE:
        weight_t = torch.from_numpy(weight.copy())
        tokens_t = torch.from_numpy(tokens.astype(np.int64).copy())
        
        # Embedding lookup + scale
        y = F.embedding(tokens_t, weight_t) * scale
        
        result = y.numpy()
    else:
        # NumPy fallback
        result = weight[tokens] * scale
    
    write_tensor(out_path, result)
    print(f"[PyTorch] Embedding: scale={scale} output_rms={np.sqrt(np.mean(result**2)):.6f}")


def matmul(a_path: str, b_path: str, out_path: str, transpose_b: bool = False):
    """
    Matrix multiplication: C = A @ B  (or C = A @ B.T if transpose_b=True)
    
    Args:
        transpose_b: If True, compute C = A @ B.T where B is stored as [N,K]
                     GRIM uses transpose_b=True for weight matrices stored as [out_dim, in_dim]
    """
    A, _ = read_tensor(a_path)
    B, _ = read_tensor(b_path)
    
    if transpose_b:
        # B is stored as [N, K], need to transpose to [K, N] for A[M,K] @ B.T[K,N] = C[M,N]
        B = B.T
        op_str = "B.T"
    else:
        op_str = "B"
    
    if PYTORCH_AVAILABLE:
        A_t = torch.from_numpy(A.copy())
        B_t = torch.from_numpy(B.copy())
        C = torch.matmul(A_t, B_t)
        result = C.numpy()
    else:
        result = np.matmul(A, B)
    
    write_tensor(out_path, result)
    print(f"[PyTorch] MatMul: {A.shape} @ {op_str}{B.shape} = {result.shape}, output_rms={np.sqrt(np.mean(result**2)):.6f}")


def cross_entropy(logits_path: str, targets_path: str, out_path: str):
    """
    Softmax cross-entropy loss: L = -log(softmax(logits)[target])
    
    Uses mean reduction by default (matches GRIM-text).
    Handles ignore_index=-1 for padding tokens.
    """
    logits, _ = read_tensor(logits_path)
    targets, _ = read_int_tensor(targets_path)
    targets = targets.flatten()
    
    if PYTORCH_AVAILABLE:
        logits_t = torch.from_numpy(logits.copy())
        targets_t = torch.from_numpy(targets.astype(np.int64).copy())
        
        # Cross entropy with mean reduction, ignore padding tokens (-1)
        loss = F.cross_entropy(logits_t, targets_t, reduction='mean', ignore_index=-1)
        
        result = np.array([loss.item()], dtype=np.float32)
    else:
        # NumPy fallback (manual softmax + CE)
        # Mask out -1 targets (padding)
        valid_mask = targets >= 0
        if not np.any(valid_mask):
            result = np.array([0.0], dtype=np.float32)
            write_tensor(out_path, result)
            print(f"[PyTorch] CrossEntropy: loss=0.0 (no valid targets)")
            return
        
        logits_valid = logits[valid_mask]
        targets_valid = targets[valid_mask]
        
        # Shift for numerical stability
        logits_max = np.max(logits_valid, axis=-1, keepdims=True)
        logits_stable = logits_valid - logits_max
        exp_logits = np.exp(logits_stable)
        softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Gather probabilities for targets
        batch_size = logits_valid.shape[0]
        target_probs = softmax[np.arange(batch_size), targets_valid]
        
        # Cross entropy
        loss = -np.mean(np.log(target_probs + 1e-10))
        result = np.array([loss], dtype=np.float32)
    
    write_tensor(out_path, result)
    print(f"[PyTorch] CrossEntropy: loss={result[0]:.6f}")


def gelu(input_path: str, out_path: str):
    """
    GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    
    Uses the 'tanh' approximation (matches GRIM-text).
    """
    x, _ = read_tensor(input_path)
    
    if PYTORCH_AVAILABLE:
        x_t = torch.from_numpy(x.copy())
        # Use tanh approximation to match CUDA
        y = F.gelu(x_t, approximate='tanh')
        result = y.numpy()
    else:
        # NumPy fallback with tanh approximation
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        result = 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x ** 3)))
    
    write_tensor(out_path, result)
    print(f"[PyTorch] GELU: input_rms={np.sqrt(np.mean(x**2)):.6f} output_rms={np.sqrt(np.mean(result**2)):.6f}")


def adamw(w_path: str, g_path: str, m_path: str, v_path: str, out_path: str,
          lr: float, beta1: float, beta2: float, eps: float, wd: float, step: int):
    """
    AdamW optimizer step:
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g²
        m_hat = m / (1 - beta1^step)
        v_hat = v / (1 - beta2^step)
        w = w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
    """
    w, _ = read_tensor(w_path)
    g, _ = read_tensor(g_path)
    m, _ = read_tensor(m_path)
    v, _ = read_tensor(v_path)
    
    if PYTORCH_AVAILABLE:
        w_t = torch.from_numpy(w.copy())
        g_t = torch.from_numpy(g.copy())
        m_t = torch.from_numpy(m.copy())
        v_t = torch.from_numpy(v.copy())
        
        # Update momentum and variance
        m_t = beta1 * m_t + (1 - beta1) * g_t
        v_t = beta2 * v_t + (1 - beta2) * (g_t ** 2)
        
        # Bias correction
        m_hat = m_t / (1 - beta1 ** step)
        v_hat = v_t / (1 - beta2 ** step)
        
        # Weight update (AdamW style: weight decay is decoupled)
        w_t = w_t - lr * (m_hat / (torch.sqrt(v_hat) + eps) + wd * w_t)
        
        result = w_t.numpy()
    else:
        # NumPy fallback
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        m_hat = m / (1 - beta1 ** step)
        v_hat = v / (1 - beta2 ** step)
        
        result = w - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * w)
    
    write_tensor(out_path, result)
    print(f"[PyTorch] AdamW: step={step} lr={lr} wd={wd} output_rms={np.sqrt(np.mean(result**2)):.6f}")


def sdpa(q_path: str, k_path: str, v_path: str, out_path: str, scale: float):
    """
    Scaled Dot-Product Attention:
        score = Q @ K^T * scale
        attn = softmax(score)
        output = attn @ V
    
    Input shapes: [batch, heads, seq_len, head_dim]
    """
    Q, q_shape = read_tensor(q_path)
    K, _ = read_tensor(k_path)
    V, _ = read_tensor(v_path)
    
    if PYTORCH_AVAILABLE:
        Q_t = torch.from_numpy(Q.copy())
        K_t = torch.from_numpy(K.copy())
        V_t = torch.from_numpy(V.copy())
        
        # Use PyTorch's SDPA (uses FlashAttention if available)
        # Note: PyTorch SDPA uses 1/sqrt(d) scaling by default, we override with scale
        output = F.scaled_dot_product_attention(
            Q_t, K_t, V_t, 
            attn_mask=None, 
            dropout_p=0.0,
            scale=scale
        )
        
        result = output.numpy()
    else:
        # NumPy fallback
        # Q, K, V: [batch, heads, seq, d]
        # score: Q @ K^T -> [batch, heads, seq, seq]
        scores = np.matmul(Q, np.swapaxes(K, -2, -1)) * scale
        
        # Softmax on last dimension
        scores_max = np.max(scores, axis=-1, keepdims=True)
        scores_stable = scores - scores_max
        exp_scores = np.exp(scores_stable)
        attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Output: attn @ V
        result = np.matmul(attn, V)
    
    write_tensor(out_path, result)
    print(f"[PyTorch] SDPA: scale={scale} output_shape={result.shape} output_rms={np.sqrt(np.mean(result**2)):.6f}")


# ============================================================================
# BACKWARD PASS IMPLEMENTATIONS
# ============================================================================

def rmsnorm_backward(x_path: str, gamma_path: str, grad_out_path: str, 
                     grad_x_path: str, grad_gamma_path: str, eps: float):
    """
    RMSNorm backward pass.
    
    Forward: y = x * gamma / rms, where rms = sqrt(mean(x², dim=-1) + eps)
    
    Backward:
        grad_gamma = sum(grad_out * x / rms, dim=0)
        grad_x = grad_out * gamma / rms - x * mean(grad_out * gamma * x / rms³, dim=-1, keepdim=True)
    """
    x, _ = read_tensor(x_path)
    gamma, _ = read_tensor(gamma_path)
    grad_out, _ = read_tensor(grad_out_path)
    
    if PYTORCH_AVAILABLE:
        x_t = torch.from_numpy(x.copy()).requires_grad_(True)
        gamma_t = torch.from_numpy(gamma.copy()).requires_grad_(True)
        grad_out_t = torch.from_numpy(grad_out.copy())
        
        # Forward
        rms = torch.sqrt(torch.mean(x_t ** 2, dim=-1, keepdim=True) + eps)
        y = x_t / rms * gamma_t
        
        # Backward via autograd
        y.backward(grad_out_t)
        
        grad_x = x_t.grad.numpy()
        grad_gamma = gamma_t.grad.numpy()
    else:
        # NumPy manual backward
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        x_norm = x / rms
        
        grad_gamma = np.sum(grad_out * x_norm, axis=0)
        
        # grad_x is more complex due to rms dependency
        d_x_norm = grad_out * gamma
        d_rms = -np.sum(d_x_norm * x / (rms ** 2), axis=-1, keepdims=True)
        d_x_sq = d_rms / (2 * rms) / x.shape[-1]
        grad_x = d_x_norm / rms + 2 * x * d_x_sq
    
    write_tensor(grad_x_path, grad_x)
    write_tensor(grad_gamma_path, grad_gamma)
    print(f"[PyTorch] RMSNorm backward: grad_x_rms={np.sqrt(np.mean(grad_x**2)):.6f}")


def matmul_backward(a_path: str, b_path: str, grad_c_path: str,
                    grad_a_path: str, grad_b_path: str):
    """
    MatMul backward: C = A @ B
    
    grad_A = grad_C @ B^T
    grad_B = A^T @ grad_C
    """
    A, _ = read_tensor(a_path)
    B, _ = read_tensor(b_path)
    grad_C, _ = read_tensor(grad_c_path)
    
    if PYTORCH_AVAILABLE:
        A_t = torch.from_numpy(A.copy()).requires_grad_(True)
        B_t = torch.from_numpy(B.copy()).requires_grad_(True)
        grad_C_t = torch.from_numpy(grad_C.copy())
        
        C = torch.matmul(A_t, B_t)
        C.backward(grad_C_t)
        
        grad_A = A_t.grad.numpy()
        grad_B = B_t.grad.numpy()
    else:
        grad_A = np.matmul(grad_C, B.T)
        grad_B = np.matmul(A.T, grad_C)
    
    write_tensor(grad_a_path, grad_A)
    write_tensor(grad_b_path, grad_B)
    print(f"[PyTorch] MatMul backward: grad_A_rms={np.sqrt(np.mean(grad_A**2)):.6f} grad_B_rms={np.sqrt(np.mean(grad_B**2)):.6f}")


def cross_entropy_backward(logits_path: str, targets_path: str, grad_loss_path: str,
                           grad_logits_path: str):
    """
    Cross-entropy backward: L = -log(softmax(logits)[target])
    
    grad_logits = softmax(logits) - one_hot(target)
    (when grad_loss = 1 and reduction='mean', divide by batch_size)
    """
    logits, _ = read_tensor(logits_path)
    targets, _ = read_int_tensor(targets_path)
    targets = targets.flatten()
    grad_loss, _ = read_tensor(grad_loss_path)
    
    if PYTORCH_AVAILABLE:
        logits_t = torch.from_numpy(logits.copy()).requires_grad_(True)
        targets_t = torch.from_numpy(targets.astype(np.int64).copy())
        
        loss = F.cross_entropy(logits_t, targets_t, reduction='mean')
        loss.backward(torch.from_numpy(grad_loss.copy()))
        
        grad_logits = logits_t.grad.numpy()
    else:
        # NumPy: softmax - one_hot, scaled by 1/batch_size for mean reduction
        batch_size = logits.shape[0]
        
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        softmax = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(batch_size), targets] = 1.0
        
        grad_logits = (softmax - one_hot) / batch_size * grad_loss[0]
    
    write_tensor(grad_logits_path, grad_logits)
    print(f"[PyTorch] CE backward: grad_logits_rms={np.sqrt(np.mean(grad_logits**2)):.6f}")


# ============================================================================
# EQUATION-BASED DIAGNOSTIC LOGGING (Rule 21)
# These match the format in Phase2_TrainingLoop.cu for comparison
# ============================================================================

def weight_gradient_equation(w_path: str, grad_path: str, token_idx: int, 
                              targets_path: str, lr: float):
    """
    [WEIGHT_GRADIENT_EQUATION] W_UPDATE[token]: W_new = W - lr × grad_W / sqrt(v + eps)
    
    Logs detailed gradient statistics for a specific token's weight row.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    W, w_shape = read_tensor(w_path)  # [vocab_size, d_model]
    grad_W, _ = read_tensor(grad_path)  # [vocab_size, d_model]
    targets, _ = read_int_tensor(targets_path)
    targets = targets.flatten()
    
    vocab_size, d_model = w_shape[0], w_shape[1]
    total_tokens = len(targets)
    
    # Stats for specific token row
    W_row = W[token_idx]
    grad_row = grad_W[token_idx]
    
    w_norm = np.linalg.norm(W_row)
    grad_norm = np.linalg.norm(grad_row)
    grad_sum = np.sum(grad_row)
    grad_mean = np.mean(grad_row)
    w_mean = np.mean(W_row)
    
    # Token count in targets
    token_count = np.sum(targets == token_idx)
    token_ratio = token_count / total_tokens * 100
    
    # Predict direction
    direction = "DECREASE" if grad_sum > 0 else "INCREASE"
    
    print(f"\n[{timestamp}] [WEIGHT_GRADIENT_EQUATION] W_UPDATE[{token_idx}]: W_new[{token_idx}] = W[{token_idx}] - lr × grad_W[{token_idx}] / sqrt(v + eps)")
    print(f"  GRAD_W[{token_idx}]: ||grad||={grad_norm:.6f} sum={grad_sum:.6f} mean={grad_mean:.6f}")
    print(f"  WEIGHT[{token_idx}]: ||W[{token_idx}]||={w_norm:.6f} mean={w_mean:.6f}")
    print(f"  TARGET_DISTRIBUTION: token_{token_idx}_count={token_count}/{total_tokens} ratio={token_ratio:.4f}%")
    sign_str = "+" if grad_sum > 0 else "-"
    print(f"  [PREDICTION] W[{token_idx}] {direction}: grad_sum={grad_sum:.4f} {'>' if grad_sum > 0 else '<'} 0 → W_new = W - lr×({sign_str}) → ||W[{token_idx}]|| {'decreases' if grad_sum > 0 else 'increases'}")


def hidden_state_equation(hidden_path: str, grad_logits_path: str, targets_path: str, token_idx: int):
    """
    [HIDDEN_STATE_EQUATION] GRAD_W[token]: grad_W[token,i] = Σ_t (hidden[t,i] × grad_logits[t,token])
    
    Analyzes how hidden states contribute to weight gradients for a specific token.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    hidden, h_shape = read_tensor(hidden_path)  # [tokens, d_model]
    grad_logits, _ = read_tensor(grad_logits_path)  # [tokens, vocab_size]
    targets, _ = read_int_tensor(targets_path)
    targets = targets.flatten()
    
    n_tokens, d_model = h_shape[0], h_shape[1]
    
    # Overall hidden stats
    h_mean = np.mean(hidden)
    h_norm_mean = np.mean(np.linalg.norm(hidden, axis=1))
    h_std = np.std(hidden)
    
    # Separate by target type
    is_token_target = (targets == token_idx)
    n_token_targets = np.sum(is_token_target)
    n_other_targets = n_tokens - n_token_targets
    
    if n_token_targets > 0:
        h_at_token = hidden[is_token_target]
        h_at_token_mean = np.mean(h_at_token)
        h_at_token_norm = np.mean(np.linalg.norm(h_at_token, axis=1))
    else:
        h_at_token_mean = 0.0
        h_at_token_norm = 0.0
    
    h_at_other = hidden[~is_token_target]
    h_at_other_mean = np.mean(h_at_other) if n_other_targets > 0 else 0.0
    h_at_other_norm = np.mean(np.linalg.norm(h_at_other, axis=1)) if n_other_targets > 0 else 0.0
    
    # Grad logits for token column
    grad_token_col = grad_logits[:, token_idx]
    grad_at_token_targets = np.mean(grad_token_col[is_token_target]) if n_token_targets > 0 else 0.0
    grad_at_other_targets = np.mean(grad_token_col[~is_token_target]) if n_other_targets > 0 else 0.0
    
    # Compute contributions
    hidden_sum = np.sum(hidden, axis=1)  # sum over d_model for each position
    
    if n_token_targets > 0:
        contrib_from_token = np.sum(hidden_sum[is_token_target] * grad_token_col[is_token_target])
    else:
        contrib_from_token = 0.0
    
    if n_other_targets > 0:
        contrib_from_other = np.sum(hidden_sum[~is_token_target] * grad_token_col[~is_token_target])
    else:
        contrib_from_other = 0.0
    
    total_contrib = contrib_from_token + contrib_from_other
    
    print(f"\n[{timestamp}] [HIDDEN_STATE_EQUATION] GRAD_W[{token_idx}]: grad_W[{token_idx},i] = Σ_t (hidden[t,i] × grad_logits[t,{token_idx}])")
    print(f"  HIDDEN STATES (encoder output): mean={h_mean:.6f} ||h||_mean={h_norm_mean:.6f} std={h_std:.6f}")
    print(f"  AT_{token_idx}_TARGETS (n={n_token_targets}): hidden_mean={h_at_token_mean:.6f} ||h||={h_at_token_norm:.6f}")
    print(f"  AT_OTHER_TARGETS (n={n_other_targets}): hidden_mean={h_at_other_mean:.6f} ||h||={h_at_other_norm:.6f}")
    print(f"  GRAD_LOGITS[{token_idx}]: at_{token_idx}_targets={grad_at_token_targets:.6f} (p_t - 1, always negative), at_other_targets={grad_at_other_targets:.6f} (p_v + entropy_term, may be negative for small p with entropy_reg)")
    print(f"  CONTRIBUTION TO Σ grad_W[{token_idx},i] = Σ_t (hidden_sum[t] × grad[t,{token_idx}]):")
    print(f"    from_{token_idx}_targets: {contrib_from_token:.6f} = Σ_{{t:target={token_idx}}} hidden_sum[t] × grad[t,{token_idx}]")
    print(f"    from_other_targets: {contrib_from_other:.6f} = Σ_{{t:target≠{token_idx}}} hidden_sum[t] × grad[t,{token_idx}]")
    direction = "POSITIVE → W[{0}] INCREASES".format(token_idx) if total_contrib > 0 else "NEGATIVE → W[{0}] DECREASES".format(token_idx)
    print(f"    TOTAL: {total_contrib:.6f} ({direction})")


def feedback_loop_equation(hidden_path: str, w_path: str, token_idx: int, 
                           prev_h_norm: float = 0, prev_w_norm: float = 0, prev_cos: float = 0):
    """
    [FEEDBACK_LOOP_EQUATION] TOKEN_MODE_COLLAPSE: logit[token] = ||h|| × ||W[token]|| × cos(h, W[token])
    
    Tracks the feedback loop components that can cause mode collapse.
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    hidden, h_shape = read_tensor(hidden_path)  # [tokens, d_model]
    W, w_shape = read_tensor(w_path)  # [vocab_size, d_model]
    
    n_tokens, d_model = h_shape[0], h_shape[1]
    
    W_row = W[token_idx]
    w_norm = np.linalg.norm(W_row)
    
    # Compute per-position stats
    h_norms = np.linalg.norm(hidden, axis=1)
    h_norm_mean = np.mean(h_norms)
    
    # Cosine similarity between each hidden state and W[token]
    # cos(h, w) = h·w / (||h|| × ||w||)
    dots = hidden @ W_row  # [tokens]
    cosines = dots / (h_norms * w_norm + 1e-8)
    cos_mean = np.mean(cosines)
    
    # Expected logit
    expected_logit = h_norm_mean * w_norm * cos_mean
    
    # Actual logits for token
    # logit[t, token] = hidden[t] @ W[token]
    actual_logits = dots
    actual_logit_mean = np.mean(actual_logits)
    
    # Growth rates (if previous values provided)
    h_growth = ((h_norm_mean / prev_h_norm) - 1) * 100 if prev_h_norm > 0 else 0.0
    w_growth = ((w_norm / prev_w_norm) - 1) * 100 if prev_w_norm > 0 else 0.0
    cos_growth = ((cos_mean / prev_cos) - 1) * 100 if abs(prev_cos) > 1e-8 else 0.0
    logit_growth = h_growth + w_growth + cos_growth  # approximate
    
    # Hidden correlation (sample pairs)
    n_sample = min(100, n_tokens)
    sample_idx = np.random.choice(n_tokens, size=n_sample, replace=False)
    h_sample = hidden[sample_idx]
    
    cos_pairs = []
    for i in range(min(50, n_sample)):
        for j in range(i+1, min(50, n_sample)):
            hi, hj = h_sample[i], h_sample[j]
            cos_ij = np.dot(hi, hj) / (np.linalg.norm(hi) * np.linalg.norm(hj) + 1e-8)
            cos_pairs.append(abs(cos_ij))
    
    avg_cos_hidden = np.mean(cos_pairs) if cos_pairs else 0.0
    expected_cos = 1.0 / np.sqrt(d_model)  # ~0.036 for d=768
    
    # Contribution factors
    h_factor = h_norm_mean / (prev_h_norm if prev_h_norm > 0 else h_norm_mean)
    w_factor = w_norm / (prev_w_norm if prev_w_norm > 0 else w_norm)
    cos_factor = cos_mean / (prev_cos if abs(prev_cos) > 1e-8 else cos_mean) if cos_mean != 0 else 1.0
    
    print(f"\n[{timestamp}] [FEEDBACK_LOOP_EQUATION] TOKEN_{token_idx}_MODE_COLLAPSE: logit[{token_idx}] = ||h|| × ||W[{token_idx}]|| × cos(h, W[{token_idx}])")
    print(f"  INPUT h (encoder output): n_positions={n_tokens} ||h||_mean={h_norm_mean:.6f}")
    print(f"  INPUT W[{token_idx}] (LM head row): ||W[{token_idx}]||={w_norm:.6f}")
    print(f"  ALIGNMENT: cos(h, W[{token_idx}])_mean={cos_mean:.6f}")
    print(f"  EXPECTED logit_{token_idx} = {h_norm_mean:.4f} × {w_norm:.4f} × {cos_mean:.4f} = {expected_logit:.4f}")
    print(f"  ACTUAL logit_{token_idx}_mean = {actual_logit_mean:.4f}")
    print(f"  GROWTH_RATES: ||h||={h_growth:+.4f}% ||W||={w_growth:+.4f}% cos={cos_growth:+.4f}% logit={logit_growth:+.4f}%")
    print(f"  HIDDEN_CORRELATION: avg|cos(h_i,h_j)|={avg_cos_hidden:.4f} (sampled {len(cos_pairs)} pairs, expected~{expected_cos:.4f})")
    print(f"  CONTRIBUTION_FACTORS: h_norm={h_factor:.3f}x w_norm={w_factor:.3f}x cosine={cos_factor:.3f}x PRODUCT={h_factor*w_factor*cos_factor:.3f}x")
    
    # Return current values for tracking
    return h_norm_mean, w_norm, cos_mean


# ============================================================================
# MAIN - Dispatch to operation
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    op = sys.argv[1]
    
    try:
        if op == "rmsnorm":
            # rmsnorm x_path gamma_path out_path eps
            rmsnorm(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
        
        elif op == "embedding":
            # embedding weight_path tokens_path out_path scale
            embedding(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]))
        
        elif op == "matmul":
            # matmul a_path b_path out_path [transpose_b]
            transpose_b = len(sys.argv) > 5 and sys.argv[5].lower() in ('1', 'true', 'yes')
            matmul(sys.argv[2], sys.argv[3], sys.argv[4], transpose_b)
        
        elif op == "cross_entropy":
            # cross_entropy logits_path targets_path out_path
            cross_entropy(sys.argv[2], sys.argv[3], sys.argv[4])
        
        elif op == "gelu":
            # gelu input_path out_path
            gelu(sys.argv[2], sys.argv[3])
        
        elif op == "adamw":
            # adamw w_path g_path m_path v_path out_path lr beta1 beta2 eps wd step
            adamw(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6],
                  float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9]),
                  float(sys.argv[10]), float(sys.argv[11]), int(sys.argv[12]))
        
        elif op == "sdpa":
            # sdpa q_path k_path v_path out_path scale
            sdpa(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]))
        
        # Backward operations
        elif op == "rmsnorm_backward":
            # rmsnorm_backward x_path gamma_path grad_out_path grad_x_path grad_gamma_path eps
            rmsnorm_backward(sys.argv[2], sys.argv[3], sys.argv[4], 
                           sys.argv[5], sys.argv[6], float(sys.argv[7]))
        
        elif op == "matmul_backward":
            # matmul_backward a_path b_path grad_c_path grad_a_path grad_b_path
            matmul_backward(sys.argv[2], sys.argv[3], sys.argv[4], 
                          sys.argv[5], sys.argv[6])
        
        elif op == "cross_entropy_backward":
            # cross_entropy_backward logits_path targets_path grad_loss_path grad_logits_path
            cross_entropy_backward(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        
        # Equation-based diagnostic logging (Rule 21)
        elif op == "weight_gradient_equation":
            # weight_gradient_equation w_path grad_path token_idx targets_path lr
            weight_gradient_equation(sys.argv[2], sys.argv[3], int(sys.argv[4]), 
                                     sys.argv[5], float(sys.argv[6]))
        
        elif op == "hidden_state_equation":
            # hidden_state_equation hidden_path grad_logits_path targets_path token_idx
            hidden_state_equation(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
        
        elif op == "feedback_loop_equation":
            # feedback_loop_equation hidden_path w_path token_idx [prev_h_norm prev_w_norm prev_cos]
            if len(sys.argv) > 8:
                feedback_loop_equation(sys.argv[2], sys.argv[3], int(sys.argv[4]),
                                       float(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]))
            else:
                feedback_loop_equation(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        
        else:
            print(f"Unknown operation: {op}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"[PyTorch] Error in {op}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

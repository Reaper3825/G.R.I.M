#!/usr/bin/env python3
"""
PyTorch Reference Gradient Validation for GRIM-text

Compares GRIM-text CUDA gradients against PyTorch reference implementation.
If they match, our backward pass is correct. If not, we found the bug.

Usage:
    python validate_gradients_pytorch.py

Requirements:
    - PyTorch (in .venv310)
    - GRIM-text checkpoint (FlatBuffer format)
    - Same input batch used in training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import struct
from pathlib import Path

# ============================================================================
# GRIM-text Architecture Parameters (from ai_config.json)
# ============================================================================
CONFIG = {
    "vocab_size": 37555,
    "d_model": 768,
    "num_layers": 6,
    "num_heads": 12,
    "num_kv_heads": 4,  # GQA
    "d_ff": 3072,
    "max_seq_len": 2048,
    "tie_embeddings": True,
    "use_bias": False,
    "dropout": 0.0,  # Disabled for comparison
}

# ============================================================================
# RMSNorm (matches GRIM-text implementation)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# Grouped Query Attention (GQA) - matches GRIM-text
# ============================================================================
class GQAAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.heads_per_kv_group = num_heads // num_kv_heads
        
        # Combined QKV projection (matches GRIM-text layout)
        # Q: num_heads * head_dim, K: num_kv_heads * head_dim, V: num_kv_heads * head_dim
        qkv_dim = (num_heads + 2 * num_kv_heads) * self.head_dim
        self.W_qkv = nn.Linear(d_model, qkv_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project to Q, K, V
        qkv = self.W_qkv(x)
        
        # Split into Q, K, V
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        
        Q = qkv[:, :, :q_dim]
        K = qkv[:, :, q_dim:q_dim + kv_dim]
        V = qkv[:, :, q_dim + kv_dim:]
        
        # Reshape for attention
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, num_kv_heads, T, head_dim)
        V = V.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)  # (B, num_kv_heads, T, head_dim)
        
        # Repeat K, V for GQA (expand to match Q heads)
        K = K.repeat_interleave(self.heads_per_kv_group, dim=1)  # (B, num_heads, T, head_dim)
        V = V.repeat_interleave(self.heads_per_kv_group, dim=1)  # (B, num_heads, T, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        
        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, num_heads, T, head_dim)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.W_o(out)
        
        return out


# ============================================================================
# Feed-Forward Network (matches GRIM-text)
# ============================================================================
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        # GELU activation (matches GRIM-text)
        return self.W2(F.gelu(self.W1(x)))


# ============================================================================
# Transformer Encoder Layer (Pre-Norm, matches GRIM-text)
# ============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
    
    def forward(self, x, mask=None):
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x), mask)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Full GRIM-text Model (PyTorch Reference)
# ============================================================================
class GRIMTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        
        # Position embedding (learned)
        self.pos_embedding = nn.Embedding(config["max_seq_len"], config["d_model"])
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(
                config["d_model"],
                config["num_heads"],
                config["num_kv_heads"],
                config["d_ff"]
            )
            for _ in range(config["num_layers"])
        ])
        
        # Final norm
        self.final_norm = RMSNorm(config["d_model"])
        
        # LM head (tied to embedding if configured)
        if config["tie_embeddings"]:
            self.lm_head = None  # Will use embedding weights
        else:
            self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        
        # Token + position embeddings
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        
        # Encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # LM head (logits)
        if self.config["tie_embeddings"]:
            logits = F.linear(x, self.embedding.weight)
        else:
            logits = self.lm_head(x)
        
        return logits
    
    def compute_gradient_components(self):
        """Compute gradient norms by component (matches GRIM-text logging)"""
        components = {}
        
        # Embedding + LM head (tied)
        emb_grad = self.embedding.weight.grad
        if emb_grad is not None:
            components["emb_lm_tied"] = emb_grad.norm().item()
        
        # Attention components
        attn_grads = []
        for layer in self.layers:
            if layer.attn.W_qkv.weight.grad is not None:
                attn_grads.append(layer.attn.W_qkv.weight.grad.norm().item() ** 2)
            if layer.attn.W_o.weight.grad is not None:
                attn_grads.append(layer.attn.W_o.weight.grad.norm().item() ** 2)
        if attn_grads:
            components["attn"] = np.sqrt(sum(attn_grads))
        
        # FFN components
        ffn_grads = []
        for layer in self.layers:
            if layer.ffn.W1.weight.grad is not None:
                ffn_grads.append(layer.ffn.W1.weight.grad.norm().item() ** 2)
            if layer.ffn.W2.weight.grad is not None:
                ffn_grads.append(layer.ffn.W2.weight.grad.norm().item() ** 2)
        if ffn_grads:
            components["ffn"] = np.sqrt(sum(ffn_grads))
        
        # RMSNorm components
        rms_grads = []
        for layer in self.layers:
            if layer.norm1.weight.grad is not None:
                rms_grads.append(layer.norm1.weight.grad.norm().item() ** 2)
            if layer.norm2.weight.grad is not None:
                rms_grads.append(layer.norm2.weight.grad.norm().item() ** 2)
        if self.final_norm.weight.grad is not None:
            rms_grads.append(self.final_norm.weight.grad.norm().item() ** 2)
        if rms_grads:
            components["rms"] = np.sqrt(sum(rms_grads))
        
        # Total gradient norm
        total_sq = sum(p.grad.norm().item() ** 2 for p in self.parameters() if p.grad is not None)
        components["total"] = np.sqrt(total_sq)
        
        return components


# ============================================================================
# Focal Loss + Label Smoothing (matches GRIM-text UnifiedLoss)
# ============================================================================
def unified_loss(logits, targets, focal_gamma=2.0, smoothing=0.15, ignore_index=-100):
    """
    Unified loss matching GRIM-text: Focal Loss + Label Smoothing
    
    L = α(1-p_t)^γ * CE_smooth
    CE_smooth = -(1-ε)log(p_t) - ε/(V-1)Σlog(p_i)
    """
    B, T, V = logits.shape
    
    # Flatten
    logits_flat = logits.view(-1, V)
    targets_flat = targets.view(-1)
    
    # Mask valid tokens
    valid_mask = targets_flat != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    logits_valid = logits_flat[valid_mask]
    targets_valid = targets_flat[valid_mask]
    
    # Softmax probabilities
    probs = F.softmax(logits_valid, dim=-1)
    log_probs = F.log_softmax(logits_valid, dim=-1)
    
    # Gather p_t (probability of correct class)
    p_t = probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
    log_p_t = log_probs.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
    
    # Label smoothing: CE_smooth = -(1-ε)log(p_t) - ε * mean(log_probs)
    ce_hard = -log_p_t
    ce_smooth = log_probs.mean(dim=-1)  # Uniform distribution term
    ce_total = (1.0 - smoothing) * ce_hard - smoothing * ce_smooth
    
    # Focal weight: (1 - p_t)^γ
    focal_weight = (1.0 - p_t) ** focal_gamma
    
    # Combined loss
    loss = focal_weight * ce_total
    
    return loss.mean()


# ============================================================================
# Load GRIM-text Weights (from FlatBuffer checkpoint)
# ============================================================================
def load_grim_weights(model, checkpoint_path):
    """
    Load weights from GRIM-text FlatBuffer checkpoint into PyTorch model.
    
    Note: You may need to export weights to a simpler format first.
    This is a placeholder - implement based on your checkpoint format.
    """
    print(f"[INFO] Loading weights from: {checkpoint_path}")
    
    # For now, use random initialization (same seed as GRIM-text)
    torch.manual_seed(1001)  # Match xavier_seed from GRIM-text
    
    # Xavier initialization (matches GRIM-text)
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'weight' in name:
            nn.init.ones_(param)
    
    print("[INFO] Initialized with Xavier (seed=1001) - same as GRIM-text fresh init")
    return model


# ============================================================================
# Export Weights Helper (for GRIM-text to use)
# ============================================================================
def export_weights_format():
    """Print expected weight format for GRIM-text to export"""
    print("\n" + "="*60)
    print("WEIGHT EXPORT FORMAT FOR GRIM-text")
    print("="*60)
    print("""
To compare against PyTorch, export GRIM-text weights as binary:

1. After forward pass, save weights to .bin file:
   - embedding.weight: [vocab_size, d_model] = [37555, 768]
   - pos_embedding.weight: [max_seq_len, d_model] = [2048, 768]
   - For each layer (0-5):
     - layers.{i}.norm1.weight: [d_model] = [768]
     - layers.{i}.attn.W_qkv.weight: [qkv_dim, d_model] = [1280, 768]
     - layers.{i}.attn.W_o.weight: [d_model, d_model] = [768, 768]
     - layers.{i}.norm2.weight: [d_model] = [768]
     - layers.{i}.ffn.W1.weight: [d_ff, d_model] = [3072, 768]
     - layers.{i}.ffn.W2.weight: [d_model, d_ff] = [768, 3072]
   - final_norm.weight: [d_model] = [768]

2. After backward pass, save gradients in same format

3. Export input batch:
   - input_ids: [batch_size, seq_len] as int32
   - targets: [batch_size, seq_len] as int32 (shifted by 1)
""")


# ============================================================================
# Main Validation
# ============================================================================
def main():
    print("="*60)
    print("GRIM-text Gradient Validation via PyTorch Reference")
    print("="*60)
    
    device = torch.device("cpu")  # Force CPU for compatibility
    print(f"Device: {device}")
    
    # Create model
    model = GRIMTextModel(CONFIG).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Initialize with same seed as GRIM-text
    checkpoint_path = Path("D:/G.R.I.M/resources/models/GRIM-text/checkpoints/checkpoint_epoch_1.bin")
    model = load_grim_weights(model, checkpoint_path)
    
    # Create synthetic batch - MATCH GRIM-text exactly
    batch_size = 2
    seq_len = 1591  # Match log: seqs=[1508,2704] lens=[1591,1591]
    valid_tokens = 3180  # From log: TOTAL_VALID=3180
    
    print(f"\nTest batch: {batch_size} sequences × {seq_len} tokens")
    print(f"Valid tokens (for normalization): {valid_tokens}")
    
    # Random input (use same seed for reproducibility)
    torch.manual_seed(42)
    input_ids = torch.randint(0, CONFIG["vocab_size"], (batch_size, seq_len), device=device)
    
    # Targets are shifted inputs (next token prediction)
    targets = torch.roll(input_ids, -1, dims=1)
    targets[:, -1] = -100  # Mask last position
    
    # Forward pass
    model.train()
    model.zero_grad()
    
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}")
    
    # Compute loss (matching GRIM-text UnifiedLoss)
    loss = unified_loss(logits, targets, focal_gamma=2.0, smoothing=0.15)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Compute gradient components
    components = model.compute_gradient_components()
    
    print("\n" + "="*60)
    print("PyTorch Reference Gradient Components:")
    print("="*60)
    print(f"  total:       {components.get('total', 0):.4f}")
    print(f"  emb_lm_tied: {components.get('emb_lm_tied', 0):.4f}")
    print(f"  attn:        {components.get('attn', 0):.4f}")
    print(f"  ffn:         {components.get('ffn', 0):.4f}")
    print(f"  rms:         {components.get('rms', 0):.4f}")
    
    print("\n" + "="*60)
    print("GRIM-text Gradient Components (from log):")
    print("="*60)
    print("  total:       6.1798")
    print("  emb_lm_tied: 5.5469")
    print("  attn:        0.6178")
    print("  ffn:         2.6509")
    print("  rms:         0.1125")
    
    # Compare ratios
    print("\n" + "="*60)
    print("Component Ratios (PyTorch / GRIM-text):")
    print("="*60)
    grim_components = {
        "total": 6.1798,
        "emb_lm_tied": 5.5469,
        "attn": 0.6178,
        "ffn": 2.6509,
        "rms": 0.1125,
    }
    
    for key in ["total", "emb_lm_tied", "attn", "ffn", "rms"]:
        pytorch_val = components.get(key, 0)
        grim_val = grim_components[key]
        if grim_val > 0:
            ratio = pytorch_val / grim_val
            match = "✓ MATCH" if 0.8 < ratio < 1.2 else "✗ MISMATCH"
            print(f"  {key:12s}: {ratio:.4f}x  {match}")
    
    export_weights_format()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("""
1. To do EXACT comparison, we need to:
   a) Export GRIM-text weights to .npy or .bin format
   b) Load them into this PyTorch model
   c) Use EXACT same input batch (export from GRIM-text)
   d) Compare gradient tensors element-by-element

2. Add weight export to GRIM-text (in Phase2_TrainingLoop.cu):
   - After backward pass, dump all gradients to file
   - Format: raw float32 arrays in row-major order

3. Run this script with loaded weights + batch
   - If gradients match: backward pass is correct
   - If gradients differ: the differing component has a bug
""")


if __name__ == "__main__":
    main()

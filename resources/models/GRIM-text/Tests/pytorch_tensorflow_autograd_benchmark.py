#!/usr/bin/env python3
"""
PyTorch vs TensorFlow Autograd Comparison Benchmark
====================================================

This benchmark compares numerical behavior of PyTorch and TensorFlow autograd
implementations using identical inputs from GRIM-text training data (.grmt files).

Output: Factual numerical comparison saved to logs folder.
No opinions - only measurements.

Usage:
    python pytorch_tensorflow_autograd_benchmark.py

Results saved to:
    resources/models/GRIM-text/training/logs/autograd_benchmark_<timestamp>.json
"""

import os
import sys
import json
import struct
import time
import datetime
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Any

# ============================================================================
# Path Configuration
# ============================================================================

SCRIPT_DIR = Path(__file__).parent.resolve()
GRIM_ROOT = SCRIPT_DIR.parent.parent.parent.parent  # Go up to G.R.I.M root
TRAINING_DIR = SCRIPT_DIR.parent / "training"
DATA_DIR = TRAINING_DIR / "data"
LOGS_DIR = TRAINING_DIR / "logs"

GRMT_PATH = DATA_DIR / "training_data.grmt"
VOCAB_PATH = DATA_DIR / "vocab.bin"

# ============================================================================
# Model Configuration (matches ai_config.json)
# ============================================================================

@dataclass
class ModelConfig:
    """Transformer model configuration matching GRIM-text."""
    vocab_size: int = 50377
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4  # GQA
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_rmsnorm: bool = True
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


@dataclass
class TrainingConfig:
    """Training configuration matching GRIM-text."""
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    batch_size: int = 7
    seq_len: int = 256  # Reduced for benchmark
    num_steps: int = 5  # Mini training loop
    seed: int = 42


# ============================================================================
# GRMT Data Loader
# ============================================================================

@dataclass
class GRMTSequence:
    """Single sequence from GRMT file."""
    token_ids: np.ndarray
    targets: np.ndarray
    numeric_values: np.ndarray
    numeric_mask: np.ndarray
    
    
def load_grmt_sequences(path: Path, max_sequences: int = 100) -> List[GRMTSequence]:
    """Load sequences from GRMT binary file (version 4-6)."""
    sequences = []
    
    with open(path, 'rb') as f:
        # Read header
        magic = struct.unpack('<I', f.read(4))[0]
        version = struct.unpack('<I', f.read(4))[0]
        num_sequences = struct.unpack('<I', f.read(4))[0]
        vocab_size = struct.unpack('<I', f.read(4))[0]
        
        # Validate
        if magic != 0x474D5254:  # "GRMT"
            raise ValueError(f"Invalid GRMT magic: {hex(magic)}")
        if version < 4 or version > 6:
            raise ValueError(f"Unsupported GRMT version: {version}")
            
        print(f"[GRMT] Loading {path.name}")
        print(f"[GRMT] version={version} sequences={num_sequences} vocab_size={vocab_size}")
        
        TEXT_FEATURE_DIM = 16
        
        for seq_idx in range(min(num_sequences, max_sequences)):
            # Read sequence length
            seq_len = struct.unpack('<I', f.read(4))[0]
            if seq_len == 0 or seq_len > 100000:
                break
                
            # Read token IDs
            token_ids = np.frombuffer(f.read(seq_len * 4), dtype=np.uint32)
            
            # Read targets (v5+)
            if version >= 5:
                targets = np.frombuffer(f.read(seq_len * 4), dtype=np.int32)
            else:
                # For v4, targets are shifted token_ids
                targets = np.roll(token_ids, -1).astype(np.int32)
                targets[-1] = -1
            
            # Read numeric values
            numeric_values = np.frombuffer(f.read(seq_len * 4), dtype=np.float32)
            
            # Read numeric mask
            numeric_mask = np.frombuffer(f.read(seq_len), dtype=np.uint8)
            
            # Skip text features
            f.read(seq_len * TEXT_FEATURE_DIM * 2)
            
            # Skip text mask
            f.read(seq_len)
            
            # v6: skip byte_lengths
            if version >= 6:
                f.read(seq_len * 2)  # uint16
            
            sequences.append(GRMTSequence(
                token_ids=token_ids.astype(np.int64),
                targets=targets.astype(np.int64),
                numeric_values=numeric_values,
                numeric_mask=numeric_mask
            ))
            
    print(f"[GRMT] Loaded {len(sequences)} sequences")
    return sequences


def create_batches(sequences: List[GRMTSequence], 
                   batch_size: int, 
                   seq_len: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create fixed-size batches from sequences."""
    batches = []
    
    # Flatten all tokens
    all_tokens = []
    all_targets = []
    for seq in sequences:
        # Filter out atom tokens (256-511) for vanilla transformer compatibility
        mask = (seq.token_ids < 256) | (seq.token_ids >= 512)
        filtered_tokens = seq.token_ids[mask]
        filtered_targets = seq.targets[mask]
        all_tokens.extend(filtered_tokens)
        all_targets.extend(filtered_targets)
    
    all_tokens = np.array(all_tokens, dtype=np.int64)
    all_targets = np.array(all_targets, dtype=np.int64)
    
    # Create batches
    total_tokens = len(all_tokens)
    tokens_per_batch = batch_size * seq_len
    num_batches = total_tokens // tokens_per_batch
    
    for i in range(num_batches):
        start = i * tokens_per_batch
        end = start + tokens_per_batch
        
        batch_tokens = all_tokens[start:end].reshape(batch_size, seq_len)
        batch_targets = all_targets[start:end].reshape(batch_size, seq_len)
        
        batches.append((batch_tokens, batch_targets))
    
    print(f"[BATCH] Created {len(batches)} batches of shape ({batch_size}, {seq_len})")
    return batches


# ============================================================================
# PyTorch Model Implementation
# ============================================================================

def create_pytorch_model(config: ModelConfig, device: str = 'cpu'):
    """Create PyTorch transformer model."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))
            
        def forward(self, x):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
            return self.weight * x / rms
    
    class CausalSelfAttention(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.n_head = config.num_heads
            self.n_kv_head = config.num_kv_heads
            self.head_dim = config.head_dim
            self.n_embd = config.d_model
            
            # GQA: Q has n_head, K/V have n_kv_head
            kv_dim = config.num_kv_heads * config.head_dim
            qkv_dim = config.d_model + 2 * kv_dim
            self.c_attn = nn.Linear(config.d_model, qkv_dim, bias=False)
            self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            
        def forward(self, x):
            B, T, C = x.shape
            
            qkv = self.c_attn(x)
            
            # Split Q, K, V
            kv_dim = self.n_kv_head * self.head_dim
            q = qkv[:, :, :self.n_embd]
            k = qkv[:, :, self.n_embd:self.n_embd + kv_dim]
            v = qkv[:, :, self.n_embd + kv_dim:]
            
            # Reshape
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
            
            # GQA: repeat K/V
            if self.n_kv_head < self.n_head:
                n_rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            
            # Attention
            scale = 1.0 / (self.head_dim ** 0.5)
            att = torch.matmul(q, k.transpose(-2, -1)) * scale
            
            # Causal mask
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            
            y = torch.matmul(att, v)
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)
    
    class MLP(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.fc = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.proj = nn.Linear(config.d_ff, config.d_model, bias=False)
            
        def forward(self, x):
            return self.proj(F.gelu(self.fc(x)))
    
    class TransformerBlock(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.ln1 = RMSNorm(config.d_model) if config.use_rmsnorm else nn.LayerNorm(config.d_model)
            self.attn = CausalSelfAttention(config)
            self.ln2 = RMSNorm(config.d_model) if config.use_rmsnorm else nn.LayerNorm(config.d_model)
            self.mlp = MLP(config)
            
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
    
    class GPT(nn.Module):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
            self.ln_f = RMSNorm(config.d_model) if config.use_rmsnorm else nn.LayerNorm(config.d_model)
            
            if not config.tie_embeddings:
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            
        def forward(self, idx):
            B, T = idx.shape
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0).expand(B, T)
            
            x = self.tok_emb(idx) + self.pos_emb(pos)
            
            for block in self.blocks:
                x = block(x)
            
            x = self.ln_f(x)
            
            if self.config.tie_embeddings:
                logits = torch.matmul(x, self.tok_emb.weight.T)
            else:
                logits = self.lm_head(x)
            
            return logits
    
    model = GPT(config).to(device)
    return model


# ============================================================================
# TensorFlow Model Implementation
# ============================================================================

def create_tensorflow_model(config: ModelConfig):
    """Create TensorFlow transformer model."""
    import tensorflow as tf
    
    class RMSNorm(tf.keras.layers.Layer):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.dim = dim
            
        def build(self, input_shape):
            self.weight = self.add_weight(
                name='weight',
                shape=(self.dim,),
                initializer='ones',
                trainable=True
            )
            
        def call(self, x):
            rms = tf.sqrt(tf.reduce_mean(x * x, axis=-1, keepdims=True) + self.eps)
            return self.weight * x / rms
    
    class CausalSelfAttention(tf.keras.layers.Layer):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.n_head = config.num_heads
            self.n_kv_head = config.num_kv_heads
            self.head_dim = config.head_dim
            self.n_embd = config.d_model
            
            kv_dim = config.num_kv_heads * config.head_dim
            qkv_dim = config.d_model + 2 * kv_dim
            self.c_attn = tf.keras.layers.Dense(qkv_dim, use_bias=False)
            self.c_proj = tf.keras.layers.Dense(config.d_model, use_bias=False)
            
        def call(self, x):
            B = tf.shape(x)[0]
            T = tf.shape(x)[1]
            
            qkv = self.c_attn(x)
            
            kv_dim = self.n_kv_head * self.head_dim
            q = qkv[:, :, :self.n_embd]
            k = qkv[:, :, self.n_embd:self.n_embd + kv_dim]
            v = qkv[:, :, self.n_embd + kv_dim:]
            
            q = tf.reshape(q, [B, T, self.n_head, self.head_dim])
            q = tf.transpose(q, [0, 2, 1, 3])
            k = tf.reshape(k, [B, T, self.n_kv_head, self.head_dim])
            k = tf.transpose(k, [0, 2, 1, 3])
            v = tf.reshape(v, [B, T, self.n_kv_head, self.head_dim])
            v = tf.transpose(v, [0, 2, 1, 3])
            
            if self.n_kv_head < self.n_head:
                n_rep = self.n_head // self.n_kv_head
                k = tf.repeat(k, n_rep, axis=1)
                v = tf.repeat(v, n_rep, axis=1)
            
            scale = 1.0 / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
            att = tf.matmul(q, k, transpose_b=True) * scale
            
            # Causal mask
            mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
            mask = 1.0 - mask
            att = att + mask * (-1e9)
            att = tf.nn.softmax(att, axis=-1)
            
            y = tf.matmul(att, v)
            y = tf.transpose(y, [0, 2, 1, 3])
            y = tf.reshape(y, [B, T, self.n_embd])
            return self.c_proj(y)
    
    class MLP(tf.keras.layers.Layer):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.fc = tf.keras.layers.Dense(config.d_ff, use_bias=False)
            self.proj = tf.keras.layers.Dense(config.d_model, use_bias=False)
            
        def call(self, x):
            return self.proj(tf.nn.gelu(self.fc(x)))
    
    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, config: ModelConfig):
            super().__init__()
            if config.use_rmsnorm:
                self.ln1 = RMSNorm(config.d_model)
                self.ln2 = RMSNorm(config.d_model)
            else:
                self.ln1 = tf.keras.layers.LayerNormalization()
                self.ln2 = tf.keras.layers.LayerNormalization()
            self.attn = CausalSelfAttention(config)
            self.mlp = MLP(config)
            
        def call(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
    
    class GPT(tf.keras.Model):
        def __init__(self, config: ModelConfig):
            super().__init__()
            self.config = config
            self.tok_emb = tf.keras.layers.Embedding(config.vocab_size, config.d_model)
            self.pos_emb = tf.keras.layers.Embedding(config.max_seq_len, config.d_model)
            self.blocks = [TransformerBlock(config) for _ in range(config.num_layers)]
            self.ln_f = RMSNorm(config.d_model) if config.use_rmsnorm else tf.keras.layers.LayerNormalization()
            
            if not config.tie_embeddings:
                self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False)
            
        def call(self, idx):
            B = tf.shape(idx)[0]
            T = tf.shape(idx)[1]
            pos = tf.range(T)
            pos = tf.broadcast_to(pos, [B, T])
            
            x = self.tok_emb(idx) + self.pos_emb(pos)
            
            for block in self.blocks:
                x = block(x)
            
            x = self.ln_f(x)
            
            if self.config.tie_embeddings:
                logits = tf.matmul(x, self.tok_emb.embeddings, transpose_b=True)
            else:
                logits = self.lm_head(x)
            
            return logits
    
    return GPT(config)


# ============================================================================
# Divergence Measurement
# ============================================================================

@dataclass
class DivergenceMetrics:
    """Metrics for comparing two tensors."""
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    cosine_similarity: float
    l2_norm_a: float
    l2_norm_b: float
    shape_match: bool
    nan_count_a: int
    nan_count_b: int
    inf_count_a: int
    inf_count_b: int


def compute_divergence(a: np.ndarray, b: np.ndarray, name: str) -> DivergenceMetrics:
    """Compute divergence metrics between two arrays."""
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    
    shape_match = a.shape == b.shape
    if not shape_match:
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    
    nan_a = np.sum(np.isnan(a))
    nan_b = np.sum(np.isnan(b))
    inf_a = np.sum(np.isinf(a))
    inf_b = np.sum(np.isinf(b))
    
    # Replace NaN/Inf for computation
    a_clean = np.nan_to_num(a, nan=0.0, posinf=1e10, neginf=-1e10)
    b_clean = np.nan_to_num(b, nan=0.0, posinf=1e10, neginf=-1e10)
    
    abs_diff = np.abs(a_clean - b_clean)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    
    # Relative difference
    denom = np.maximum(np.abs(a_clean), np.abs(b_clean))
    denom = np.where(denom < 1e-10, 1e-10, denom)
    rel_diff = abs_diff / denom
    max_rel_diff = float(np.max(rel_diff))
    mean_rel_diff = float(np.mean(rel_diff))
    
    # Cosine similarity
    norm_a = np.linalg.norm(a_clean)
    norm_b = np.linalg.norm(b_clean)
    if norm_a > 1e-10 and norm_b > 1e-10:
        cosine = float(np.dot(a_clean, b_clean) / (norm_a * norm_b))
    else:
        cosine = 0.0
    
    return DivergenceMetrics(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        max_rel_diff=max_rel_diff,
        mean_rel_diff=mean_rel_diff,
        cosine_similarity=cosine,
        l2_norm_a=float(norm_a),
        l2_norm_b=float(norm_b),
        shape_match=shape_match,
        nan_count_a=int(nan_a),
        nan_count_b=int(nan_b),
        inf_count_a=int(inf_a),
        inf_count_b=int(inf_b)
    )


# ============================================================================
# Benchmark Runner
# ============================================================================

@dataclass
class StepResult:
    """Results from a single training step."""
    step: int
    pytorch_loss: float
    tensorflow_loss: float
    loss_diff: float
    forward_divergence: Dict[str, DivergenceMetrics]
    backward_divergence: Dict[str, DivergenceMetrics]
    weight_divergence: Dict[str, DivergenceMetrics]
    pytorch_grad_norm: float
    tensorflow_grad_norm: float
    elapsed_ms: float


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    timestamp: str
    model_config: Dict
    training_config: Dict
    pytorch_version: str
    tensorflow_version: str
    numpy_version: str
    cuda_available_pytorch: bool
    cuda_available_tensorflow: bool
    grmt_path: str
    vocab_path: str
    num_sequences_loaded: int
    num_batches_created: int
    step_results: List[Dict]
    summary: Dict
    
    
def run_benchmark():
    """Run the PyTorch vs TensorFlow autograd comparison benchmark."""
    
    print("=" * 80)
    print("PyTorch vs TensorFlow Autograd Comparison Benchmark")
    print("=" * 80)
    print()
    
    # Import frameworks
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_pytorch = torch.cuda.is_available()
        print(f"[OK] PyTorch {pytorch_version} (CUDA: {cuda_pytorch})")
    except ImportError:
        print("[ERROR] PyTorch not installed. Run: pip install torch")
        sys.exit(1)
        
    try:
        import tensorflow as tf
        tensorflow_version = tf.__version__
        cuda_tf = len(tf.config.list_physical_devices('GPU')) > 0
        print(f"[OK] TensorFlow {tensorflow_version} (CUDA: {cuda_tf})")
    except ImportError:
        print("[ERROR] TensorFlow not installed. Run: pip install tensorflow")
        sys.exit(1)
    
    print(f"[OK] NumPy {np.__version__}")
    print()
    
    # Check data files
    if not GRMT_PATH.exists():
        print(f"[ERROR] GRMT file not found: {GRMT_PATH}")
        sys.exit(1)
    if not VOCAB_PATH.exists():
        print(f"[ERROR] Vocab file not found: {VOCAB_PATH}")
        sys.exit(1)
    
    print(f"[DATA] GRMT: {GRMT_PATH}")
    print(f"[DATA] Vocab: {VOCAB_PATH}")
    print()
    
    # Configuration - reduced for benchmark speed
    model_config = ModelConfig(
        vocab_size=50377,
        d_model=256,      # Reduced for speed
        num_layers=2,     # Reduced for speed
        num_heads=4,
        num_kv_heads=2,
        d_ff=1024,
        max_seq_len=256,
        tie_embeddings=True,
        use_rmsnorm=True
    )
    
    train_config = TrainingConfig(
        learning_rate=0.0003,
        weight_decay=0.01,
        gradient_clip=1.0,
        batch_size=4,
        seq_len=128,
        num_steps=5,
        seed=42
    )
    
    print("[CONFIG] Model:")
    print(f"         d_model={model_config.d_model}, layers={model_config.num_layers}")
    print(f"         heads={model_config.num_heads}, kv_heads={model_config.num_kv_heads}")
    print(f"         vocab={model_config.vocab_size}, tie_emb={model_config.tie_embeddings}")
    print()
    print("[CONFIG] Training:")
    print(f"         batch_size={train_config.batch_size}, seq_len={train_config.seq_len}")
    print(f"         lr={train_config.learning_rate}, steps={train_config.num_steps}")
    print()
    
    # Load data
    print("-" * 40)
    sequences = load_grmt_sequences(GRMT_PATH, max_sequences=200)
    batches = create_batches(sequences, train_config.batch_size, train_config.seq_len)
    
    if len(batches) < train_config.num_steps:
        print(f"[WARN] Only {len(batches)} batches available, reducing steps")
        train_config.num_steps = len(batches)
    print()
    
    # Set seeds
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    tf.random.set_seed(train_config.seed)
    
    # Create models
    print("-" * 40)
    print("[MODEL] Creating PyTorch model...")
    pt_model = create_pytorch_model(model_config, device='cpu')
    pt_model.train()
    
    print("[MODEL] Creating TensorFlow model...")
    tf_model = create_tensorflow_model(model_config)
    
    # Build TF model with dummy input
    dummy_input = tf.zeros((1, train_config.seq_len), dtype=tf.int32)
    _ = tf_model(dummy_input)
    
    # Sync initial weights: TensorFlow -> PyTorch
    print("[SYNC] Synchronizing initial weights TF -> PyTorch...")
    sync_weights_tf_to_pytorch(tf_model, pt_model, model_config)
    print()
    
    # Create optimizers
    pt_optimizer = torch.optim.AdamW(
        pt_model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    tf_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8
    )
    
    # Run training loop
    print("-" * 40)
    print(f"[TRAIN] Running {train_config.num_steps} synchronized training steps...")
    print()
    
    step_results = []
    
    for step in range(train_config.num_steps):
        start_time = time.perf_counter()
        
        batch_tokens, batch_targets = batches[step]
        
        # ================================================================
        # PyTorch Forward + Backward
        # ================================================================
        pt_tokens = torch.tensor(batch_tokens, dtype=torch.long)
        pt_targets = torch.tensor(batch_targets, dtype=torch.long)
        
        pt_optimizer.zero_grad()
        pt_logits = pt_model(pt_tokens)
        
        # Cross-entropy loss
        pt_loss = torch.nn.functional.cross_entropy(
            pt_logits.view(-1, model_config.vocab_size),
            pt_targets.view(-1),
            ignore_index=-1,
            reduction='mean'
        )
        
        pt_loss.backward()
        
        # Compute gradient norm before clipping
        pt_grad_norm = 0.0
        for p in pt_model.parameters():
            if p.grad is not None:
                pt_grad_norm += p.grad.data.norm(2).item() ** 2
        pt_grad_norm = pt_grad_norm ** 0.5
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(pt_model.parameters(), train_config.gradient_clip)
        
        # ================================================================
        # TensorFlow Forward + Backward
        # ================================================================
        tf_tokens = tf.constant(batch_tokens, dtype=tf.int32)
        tf_targets = tf.constant(batch_targets, dtype=tf.int64)
        
        with tf.GradientTape() as tape:
            tf_logits = tf_model(tf_tokens, training=True)
            
            # Cross-entropy loss
            tf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(tf_targets, [-1]),
                logits=tf.reshape(tf_logits, [-1, model_config.vocab_size])
            )
            # Mask out -1 targets
            mask = tf.cast(tf.reshape(tf_targets, [-1]) >= 0, tf.float32)
            tf_loss = tf.reduce_sum(tf_loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
        
        tf_grads = tape.gradient(tf_loss, tf_model.trainable_variables)
        
        # Compute gradient norm
        tf_grad_norm = 0.0
        for g in tf_grads:
            if g is not None:
                tf_grad_norm += tf.reduce_sum(g ** 2).numpy()
        tf_grad_norm = tf_grad_norm ** 0.5
        
        # Clip gradients
        tf_grads, _ = tf.clip_by_global_norm(tf_grads, train_config.gradient_clip)
        
        # ================================================================
        # Compare outputs
        # ================================================================
        pt_logits_np = pt_logits.detach().numpy()
        tf_logits_np = tf_logits.numpy()
        
        forward_div = {
            'logits': compute_divergence(pt_logits_np, tf_logits_np, 'logits')
        }
        
        # Compare gradients (embedding layer as example)
        pt_emb_grad = pt_model.tok_emb.weight.grad.detach().numpy()
        tf_emb_grad = tf_grads[0].numpy()  # First trainable variable is usually embedding
        
        backward_div = {
            'embedding_grad': compute_divergence(pt_emb_grad, tf_emb_grad, 'embedding_grad')
        }
        
        # Compare weights after step
        pt_emb_weight = pt_model.tok_emb.weight.detach().numpy()
        tf_emb_weight = tf_model.tok_emb.embeddings.numpy()
        
        weight_div_before = {
            'embedding_weight': compute_divergence(pt_emb_weight, tf_emb_weight, 'embedding_weight')
        }
        
        # ================================================================
        # Apply optimizer step
        # ================================================================
        pt_optimizer.step()
        tf_optimizer.apply_gradients(zip(tf_grads, tf_model.trainable_variables))
        
        # Compare weights after step
        pt_emb_weight_after = pt_model.tok_emb.weight.detach().numpy()
        tf_emb_weight_after = tf_model.tok_emb.embeddings.numpy()
        
        weight_div_after = {
            'embedding_weight': compute_divergence(pt_emb_weight_after, tf_emb_weight_after, 'embedding_weight')
        }
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        pt_loss_val = pt_loss.item()
        tf_loss_val = float(tf_loss.numpy())
        loss_diff = abs(pt_loss_val - tf_loss_val)
        
        result = StepResult(
            step=step,
            pytorch_loss=pt_loss_val,
            tensorflow_loss=tf_loss_val,
            loss_diff=loss_diff,
            forward_divergence={k: asdict(v) for k, v in forward_div.items()},
            backward_divergence={k: asdict(v) for k, v in backward_div.items()},
            weight_divergence={k: asdict(v) for k, v in weight_div_after.items()},
            pytorch_grad_norm=pt_grad_norm,
            tensorflow_grad_norm=tf_grad_norm,
            elapsed_ms=elapsed_ms
        )
        step_results.append(result)
        
        # Print progress
        logit_div = forward_div['logits']
        print(f"  Step {step}: PT_loss={pt_loss_val:.6f} TF_loss={tf_loss_val:.6f} "
              f"diff={loss_diff:.6f} logit_cos={logit_div.cosine_similarity:.6f} "
              f"PT_grad={pt_grad_norm:.4f} TF_grad={tf_grad_norm:.4f}")
    
    print()
    
    # ================================================================
    # Compute Summary
    # ================================================================
    print("-" * 40)
    print("[SUMMARY]")
    
    loss_diffs = [r.loss_diff for r in step_results]
    grad_norm_diffs = [abs(r.pytorch_grad_norm - r.tensorflow_grad_norm) for r in step_results]
    logit_cosines = [r.forward_divergence['logits']['cosine_similarity'] for r in step_results]
    
    summary = {
        'total_steps': len(step_results),
        'loss_difference': {
            'mean': float(np.mean(loss_diffs)),
            'max': float(np.max(loss_diffs)),
            'min': float(np.min(loss_diffs)),
            'final': loss_diffs[-1] if loss_diffs else 0.0
        },
        'gradient_norm_difference': {
            'mean': float(np.mean(grad_norm_diffs)),
            'max': float(np.max(grad_norm_diffs))
        },
        'logit_cosine_similarity': {
            'mean': float(np.mean(logit_cosines)),
            'min': float(np.min(logit_cosines)),
            'final': logit_cosines[-1] if logit_cosines else 0.0
        },
        'divergence_detected': logit_cosines[-1] < 0.999 if logit_cosines else True,
        'first_divergence_step': next(
            (i for i, c in enumerate(logit_cosines) if c < 0.9999), -1
        )
    }
    
    print(f"  Loss diff:     mean={summary['loss_difference']['mean']:.6f} "
          f"max={summary['loss_difference']['max']:.6f}")
    print(f"  Grad norm diff: mean={summary['gradient_norm_difference']['mean']:.6f}")
    print(f"  Logit cosine:  mean={summary['logit_cosine_similarity']['mean']:.6f} "
          f"min={summary['logit_cosine_similarity']['min']:.6f}")
    print(f"  Divergence:    {summary['divergence_detected']} "
          f"(first at step {summary['first_divergence_step']})")
    print()
    
    # ================================================================
    # Save Results
    # ================================================================
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = LOGS_DIR / f"autograd_benchmark_{timestamp}.json"
    
    benchmark_result = BenchmarkResult(
        timestamp=timestamp,
        model_config=asdict(model_config),
        training_config=asdict(train_config),
        pytorch_version=pytorch_version,
        tensorflow_version=tensorflow_version,
        numpy_version=np.__version__,
        cuda_available_pytorch=cuda_pytorch,
        cuda_available_tensorflow=cuda_tf,
        grmt_path=str(GRMT_PATH),
        vocab_path=str(VOCAB_PATH),
        num_sequences_loaded=len(sequences),
        num_batches_created=len(batches),
        step_results=[asdict(r) for r in step_results],
        summary=summary
    )
    
    with open(output_path, 'w') as f:
        json.dump(asdict(benchmark_result), f, indent=2)
    
    print(f"[SAVED] Results written to: {output_path}")
    print()
    print("=" * 80)
    print("Benchmark Complete")
    print("=" * 80)
    
    return benchmark_result


def sync_weights_tf_to_pytorch(tf_model, pt_model, config: ModelConfig):
    """Copy weights from TensorFlow model to PyTorch model for fair comparison."""
    import torch
    
    # Embedding weights
    tf_tok_emb = tf_model.tok_emb.embeddings.numpy()
    pt_model.tok_emb.weight.data = torch.tensor(tf_tok_emb, dtype=torch.float32)
    
    tf_pos_emb = tf_model.pos_emb.embeddings.numpy()
    pt_model.pos_emb.weight.data = torch.tensor(tf_pos_emb, dtype=torch.float32)
    
    # Block weights
    for i, (tf_block, pt_block) in enumerate(zip(tf_model.blocks, pt_model.blocks)):
        # RMSNorm / LayerNorm
        if config.use_rmsnorm:
            pt_block.ln1.weight.data = torch.tensor(tf_block.ln1.weight.numpy())
            pt_block.ln2.weight.data = torch.tensor(tf_block.ln2.weight.numpy())
        else:
            pt_block.ln1.weight.data = torch.tensor(tf_block.ln1.gamma.numpy())
            pt_block.ln1.bias.data = torch.tensor(tf_block.ln1.beta.numpy())
            pt_block.ln2.weight.data = torch.tensor(tf_block.ln2.gamma.numpy())
            pt_block.ln2.bias.data = torch.tensor(tf_block.ln2.beta.numpy())
        
        # Attention
        pt_block.attn.c_attn.weight.data = torch.tensor(
            tf_block.attn.c_attn.kernel.numpy().T)
        pt_block.attn.c_proj.weight.data = torch.tensor(
            tf_block.attn.c_proj.kernel.numpy().T)
        
        # MLP
        pt_block.mlp.fc.weight.data = torch.tensor(
            tf_block.mlp.fc.kernel.numpy().T)
        pt_block.mlp.proj.weight.data = torch.tensor(
            tf_block.mlp.proj.kernel.numpy().T)
    
    # Final LayerNorm
    if config.use_rmsnorm:
        pt_model.ln_f.weight.data = torch.tensor(tf_model.ln_f.weight.numpy())
    else:
        pt_model.ln_f.weight.data = torch.tensor(tf_model.ln_f.gamma.numpy())
        pt_model.ln_f.bias.data = torch.tensor(tf_model.ln_f.beta.numpy())


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    run_benchmark()

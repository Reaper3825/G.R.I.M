#!/usr/bin/env python3
"""
Approach 2: Direct Gradient Export and Comparison

This script:
1. Reads binary gradient dumps from GRIM-text
2. Loads them into NumPy arrays
3. Compares element-by-element with PyTorch reference gradients
4. Reports discrepancies

To use:
1. Add gradient export code to GRIM-text (see GRIM_EXPORT_CODE below)
2. Run one training batch in GRIM-text
3. Run this script to compare
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import struct
import os

# ============================================================================
# GRIM-text Export Code (add to Phase2_TrainingLoop.cu after backward)
# ============================================================================
GRIM_EXPORT_CODE = """
// Add this after ctx.model->backward(...) in Phase2_TrainingLoop.cu

// Export gradients for PyTorch comparison
static bool exported_grads = false;
if (!exported_grads && batch_idx == 0) {
    exported_grads = true;
    
    const auto& ts = ctx.model->getTrainingState();
    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    
    auto exportBuffer = [&](const char* name, const float* d_ptr, size_t count) {
        if (!d_ptr) return;
        std::vector<float> h_data(count);
        cudaMemcpyAsync(h_data.data(), d_ptr, count * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        std::string path = "D:/G.R.I.M/gradient_dumps/" + std::string(name) + ".bin";
        FILE* f = fopen(path.c_str(), "wb");
        if (f) {
            fwrite(&count, sizeof(size_t), 1, f);  // Header: element count
            fwrite(h_data.data(), sizeof(float), count, f);
            fclose(f);
            printf("[GradExport] Wrote %s: %zu elements\\n", path.c_str(), count);
        }
    };
    
    // Export key gradients
    const auto& cfg = ctx.model->getConfig();
    
    // Embedding grads (vocab_size × d_model)
    exportBuffer("embedding_grads", ts.embedding_grads, 
                 static_cast<size_t>(cfg.vocab_size) * cfg.d_model);
    
    // Per-layer gradients
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        std::string prefix = "layer" + std::to_string(layer) + "_";
        
        // Attention QKV grads
        size_t qkv_size = static_cast<size_t>((cfg.num_heads + 2*cfg.num_kv_heads) * 
                          (cfg.d_model / cfg.num_heads)) * cfg.d_model;
        exportBuffer((prefix + "qkv_grads").c_str(), ts.attn_qkv_weight_grads[layer], qkv_size);
        
        // Attention Wo grads
        exportBuffer((prefix + "wo_grads").c_str(), ts.attn_out_weight_grads[layer], 
                     static_cast<size_t>(cfg.d_model) * cfg.d_model);
        
        // FFN W1 grads
        exportBuffer((prefix + "ffn_w1_grads").c_str(), ts.ffn_w1_grads[layer],
                     static_cast<size_t>(cfg.d_model) * cfg.d_ff);
        
        // FFN W2 grads
        exportBuffer((prefix + "ffn_w2_grads").c_str(), ts.ffn_w2_grads[layer],
                     static_cast<size_t>(cfg.d_ff) * cfg.d_model);
        
        // RMSNorm gamma grads (accessed via encoder layer)
        // Note: Need to add accessor for these
    }
    
    // Also export input batch for PyTorch to use
    // exportBuffer("input_ids", ...);  // Need to save the actual input tokens
    
    printf("[GradExport] Gradient export complete. Run compare_gradients_pytorch.py\\n");
}
"""

# ============================================================================
# Load binary gradient file
# ============================================================================
def load_gradient_file(path: Path) -> np.ndarray:
    """Load a binary gradient file exported from GRIM-text"""
    with open(path, 'rb') as f:
        count = struct.unpack('Q', f.read(8))[0]  # size_t = uint64
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data


# ============================================================================
# PyTorch Reference Model (minimal, for gradient comparison)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class GQAAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.heads_per_kv = num_heads // num_kv_heads
        
        qkv_dim = (num_heads + 2 * num_kv_heads) * self.head_dim
        self.W_qkv = nn.Linear(d_model, qkv_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / (self.head_dim ** 0.5)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.W_qkv(x)
        
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        
        Q = qkv[:, :, :q_dim].view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = qkv[:, :, q_dim:q_dim+kv_dim].view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = qkv[:, :, q_dim+kv_dim:].view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        K = K.repeat_interleave(self.heads_per_kv, dim=1)
        V = V.repeat_interleave(self.heads_per_kv, dim=1)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        return self.W2(F.gelu(self.W1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GQAAttention(d_model, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff)
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class GRIMTextModel(nn.Module):
    def __init__(self, vocab_size=37555, d_model=768, num_layers=6, 
                 num_heads=12, num_kv_heads=4, d_ff=3072, max_seq_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, num_kv_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.vocab_size = vocab_size
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(pos)
        
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.final_norm(x)
        logits = F.linear(x, self.embedding.weight)  # Tied weights
        return logits


# ============================================================================
# Compare gradients
# ============================================================================
def compare_gradients():
    print("="*70)
    print("GRADIENT COMPARISON: GRIM-text vs PyTorch")
    print("="*70)
    
    dump_dir = Path("D:/G.R.I.M/gradient_dumps")
    
    if not dump_dir.exists():
        print(f"\n❌ Gradient dump directory not found: {dump_dir}")
        print("\nTo use this tool:")
        print("1. Create the directory: mkdir D:\\G.R.I.M\\gradient_dumps")
        print("2. Add the export code to Phase2_TrainingLoop.cu (see GRIM_EXPORT_CODE)")
        print("3. Run one training batch")
        print("4. Run this script again")
        print("\n" + "="*70)
        print("GRIM-text Export Code to Add:")
        print("="*70)
        print(GRIM_EXPORT_CODE)
        return
    
    files = list(dump_dir.glob("*.bin"))
    if not files:
        print(f"\n❌ No gradient files found in {dump_dir}")
        print("Run GRIM-text training first with export code enabled.")
        return
    
    print(f"\nFound {len(files)} gradient files:")
    for f in files:
        print(f"  - {f.name}")
    
    # Load GRIM-text gradients
    grim_grads = {}
    for f in files:
        name = f.stem
        grim_grads[name] = load_gradient_file(f)
        print(f"  {name}: shape={grim_grads[name].shape}, norm={np.linalg.norm(grim_grads[name]):.6f}")
    
    # Create PyTorch model with same architecture
    print("\n" + "="*70)
    print("Creating PyTorch reference model...")
    print("="*70)
    
    model = GRIMTextModel()
    
    # TODO: Load GRIM-text weights into PyTorch model for exact comparison
    # For now, use random init (same seed)
    torch.manual_seed(1001)
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
    
    # Create synthetic batch (or load from GRIM-text export)
    batch_size = 2
    seq_len = 256
    input_ids = torch.randint(0, 37555, (batch_size, seq_len))
    targets = torch.roll(input_ids, -1, dims=1)
    targets[:, -1] = -100
    
    # Forward + backward
    model.train()
    model.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, 37555), targets.view(-1), ignore_index=-100)
    loss.backward()
    
    # Compare
    print("\n" + "="*70)
    print("Gradient Comparison (norm ratios):")
    print("="*70)
    
    if "embedding_grads" in grim_grads:
        grim_emb = grim_grads["embedding_grads"]
        pt_emb = model.embedding.weight.grad.numpy().flatten()
        
        grim_norm = np.linalg.norm(grim_emb)
        pt_norm = np.linalg.norm(pt_emb)
        
        print(f"\nEmbedding gradients:")
        print(f"  GRIM-text norm: {grim_norm:.6f}")
        print(f"  PyTorch norm:   {pt_norm:.6f}")
        print(f"  Ratio:          {grim_norm/pt_norm:.4f}x")
        
        if len(grim_emb) == len(pt_emb):
            cosine_sim = np.dot(grim_emb, pt_emb) / (grim_norm * pt_norm + 1e-8)
            print(f"  Cosine sim:     {cosine_sim:.6f}")
    
    for layer_idx in range(6):
        prefix = f"layer{layer_idx}_"
        
        # QKV
        qkv_key = prefix + "qkv_grads"
        if qkv_key in grim_grads:
            grim_qkv = grim_grads[qkv_key]
            pt_qkv = model.layers[layer_idx].attn.W_qkv.weight.grad.numpy().flatten()
            
            print(f"\nLayer {layer_idx} QKV gradients:")
            print(f"  GRIM: {np.linalg.norm(grim_qkv):.6f}, PyTorch: {np.linalg.norm(pt_qkv):.6f}")
        
        # FFN W1
        w1_key = prefix + "ffn_w1_grads"
        if w1_key in grim_grads:
            grim_w1 = grim_grads[w1_key]
            pt_w1 = model.layers[layer_idx].ffn.W1.weight.grad.numpy().flatten()
            
            print(f"Layer {layer_idx} FFN W1 gradients:")
            print(f"  GRIM: {np.linalg.norm(grim_w1):.6f}, PyTorch: {np.linalg.norm(pt_w1):.6f}")


if __name__ == "__main__":
    compare_gradients()

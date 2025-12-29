#!/usr/bin/env python3
"""
Quick gradient sanity check - compares gradient component RATIOS between PyTorch and GRIM-text.

The key insight: if the BACKWARD PASS is correct, the RATIO between components should be similar
even if absolute magnitudes differ (due to different inputs/weights).

Expected ratios from a correctly-implemented transformer:
- emb_lm / total: ~0.7-0.9 (embedding dominates because vocab is huge)
- attn / total: ~0.1-0.3
- ffn / total: ~0.3-0.5
- rms / total: ~0.01-0.05

If GRIM-text has significantly different ratios, there's a bug in that component's backward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Quick minimal model
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=37555, d_model=768, num_layers=6, num_heads=12, d_ff=3072):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(2048, d_model)
        
        # Simplified: just use PyTorch's TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm like GRIM-text
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        # Tied weights
        
    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(pos)
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)
        h = self.final_norm(h)
        
        # Tied LM head
        logits = F.linear(h, self.embedding.weight)
        return logits
    
    def get_component_grads(self):
        """Get gradient norms by component category"""
        emb_sq = 0.0
        attn_sq = 0.0
        ffn_sq = 0.0
        norm_sq = 0.0
        
        for name, p in self.named_parameters():
            if p.grad is None:
                continue
            g_sq = (p.grad ** 2).sum().item()
            
            if 'embedding' in name:
                emb_sq += g_sq
            elif 'self_attn' in name:
                attn_sq += g_sq
            elif 'linear' in name:  # FFN
                ffn_sq += g_sq
            elif 'norm' in name:
                norm_sq += g_sq
        
        return {
            'emb': np.sqrt(emb_sq),
            'attn': np.sqrt(attn_sq),
            'ffn': np.sqrt(ffn_sq),
            'norm': np.sqrt(norm_sq),
            'total': np.sqrt(emb_sq + attn_sq + ffn_sq + norm_sq)
        }


def main():
    print("="*60)
    print("Gradient Component Ratio Analysis")
    print("="*60)
    
    torch.manual_seed(1001)
    
    model = MiniTransformer()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Small batch for speed
    batch_size = 2
    seq_len = 128
    
    x = torch.randint(0, 37555, (batch_size, seq_len))
    targets = torch.roll(x, -1, dims=1)
    targets[:, -1] = -100
    
    model.train()
    model.zero_grad()
    
    logits = model(x)
    
    # Cross-entropy loss (simpler than focal)
    loss = F.cross_entropy(
        logits.view(-1, 37555),
        targets.view(-1),
        ignore_index=-100
    )
    print(f"Loss: {loss.item():.4f}")
    
    loss.backward()
    
    grads = model.get_component_grads()
    
    print("\n" + "="*60)
    print("PyTorch Gradient Components:")
    print("="*60)
    for k, v in grads.items():
        print(f"  {k:8s}: {v:.4f}")
    
    print("\n" + "="*60)
    print("PyTorch Component Ratios (vs total):")
    print("="*60)
    total = grads['total']
    for k, v in grads.items():
        if k != 'total' and total > 0:
            print(f"  {k:8s} / total = {v/total:.4f}")
    
    # GRIM-text values from log
    grim = {
        'total': 6.1798,
        'emb': 5.5469,  # emb_lm_tied
        'attn': 0.6178,
        'ffn': 2.6509,
        'norm': 0.1125   # rms
    }
    
    print("\n" + "="*60)
    print("GRIM-text Gradient Components (from log):")
    print("="*60)
    for k, v in grim.items():
        print(f"  {k:8s}: {v:.4f}")
    
    print("\n" + "="*60)
    print("GRIM-text Component Ratios (vs total):")
    print("="*60)
    grim_total = grim['total']
    for k, v in grim.items():
        if k != 'total':
            print(f"  {k:8s} / total = {v/grim_total:.4f}")
    
    print("\n" + "="*60)
    print("COMPARISON - Ratio Differences:")
    print("="*60)
    print("Component     PyTorch    GRIM-text   Match?")
    print("-" * 50)
    
    for k in ['emb', 'attn', 'ffn', 'norm']:
        pt_ratio = grads[k] / total if total > 0 else 0
        grim_ratio = grim[k] / grim_total
        diff = abs(pt_ratio - grim_ratio)
        match = "✓" if diff < 0.15 else "✗"
        print(f"  {k:8s}    {pt_ratio:.4f}     {grim_ratio:.4f}      {match} (Δ={diff:.4f})")
    
    print("\n" + "="*60)
    print("INTERPRETATION:")
    print("="*60)
    print("""
If ratios are SIMILAR (within ~0.15):
  → Backward pass is likely correct
  → Absolute magnitude differences are from inputs/weights
  
If ratios are DIFFERENT:
  → The component with wrong ratio has a backward bug
  → E.g., if FFN ratio is 2x higher in GRIM-text, FFN backward is buggy
""")


if __name__ == "__main__":
    main()

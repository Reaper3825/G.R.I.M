#!/usr/bin/env python3
"""
Plot attention development over batches from diagnostic_output.txt

Extracts:
- Entropy (H) per layer over batches
- Q/K/V gradient norms per layer
- Loss curve
- RMSNorm gamma gradients
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_diagnostic_output(filepath):
    """Parse diagnostic_output.txt and extract attention metrics."""
    
    # Data structures
    batches = []
    losses = []
    
    # Per-batch, per-layer attention stats
    # Key: (batch, layer) -> {H, Q_grad, K_grad, V_grad, prob_min, prob_max}
    attn_data = defaultdict(dict)
    
    # Per-batch gradient norms
    grad_data = defaultdict(dict)  # batch -> {lm_head, emb, layer_X_qkv, etc.}
    
    # RMSNorm gamma gradients per layer
    rms_data = defaultdict(lambda: defaultdict(list))  # batch -> layer -> [rms1, rms2]
    
    current_batch = -1
    current_layer = -1
    
    # Try multiple encodings (PowerShell Tee-Object outputs UTF-16)
    lines = []
    for encoding in ['utf-16', 'utf-8', 'latin-1', 'cp1252']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                lines = f.readlines()
            print(f"  Using encoding: {encoding}, read {len(lines)} lines")
            break
        except Exception as e:
            print(f"  Failed {encoding}: {e}")
            continue
    
    if not lines:
        print("ERROR: Could not read file with any encoding")
        return batches, losses, attn_data, grad_data, rms_data
    
    for line in lines:
            # Parse batch/loss line: [Batch    0] loss=10.5445 avg=10.5445
            batch_match = re.search(r'\[Batch\s+(\d+)\]\s+loss=([\d.]+)', line)
            if batch_match:
                current_batch = int(batch_match.group(1))
                loss = float(batch_match.group(2))
                batches.append(current_batch)
                losses.append(loss)
                continue
            
            # Parse attention diagnostics: [AttnDiag] step=0 layer=5 (n=65): probs=[0.0016,0.0834] H=6.66 bits/pos qk=[-2.97,2.08] grads=[Q:0.039188 K:0.011041 V:0.030615]
            attn_match = re.search(
                r'\[AttnDiag\]\s+step=(\d+)\s+layer=(\d+).*?H=([\d.]+).*?grads=\[Q:([\d.]+)\s+K:([\d.]+)\s+V:([\d.]+)\]',
                line
            )
            if attn_match:
                step = int(attn_match.group(1))
                layer = int(attn_match.group(2))
                entropy = float(attn_match.group(3))
                q_grad = float(attn_match.group(4))
                k_grad = float(attn_match.group(5))
                v_grad = float(attn_match.group(6))
                
                # Use step as batch proxy (may have multiple per batch)
                key = (step, layer)
                if key not in attn_data or 'H' not in attn_data[key]:
                    attn_data[key] = {
                        'H': entropy,
                        'Q_grad': q_grad,
                        'K_grad': k_grad,
                        'V_grad': v_grad
                    }
                continue
            
            # Parse gradient trace: [GradientTrace] batch=0
            grad_batch_match = re.search(r'\[GradientTrace\]\s+batch=(\d+)', line)
            if grad_batch_match:
                current_batch = int(grad_batch_match.group(1))
                continue
            
            # Parse LM_HEAD gradient
            lm_match = re.search(r'LM_HEAD:\s+([\d.e+-]+)', line)
            if lm_match and current_batch >= 0:
                grad_data[current_batch]['lm_head'] = float(lm_match.group(1))
                continue
            
            # Parse layer gradients: QKV: 2.6089e+01  W_o: 2.6514e+02  FFN_W1: 6.7691e+01  FFN_W2: 2.5466e+02
            layer_match = re.search(r'Layer\s+(\d+):', line)
            if layer_match:
                current_layer = int(layer_match.group(1))
                continue
            
            # Parse QKV/W_o/FFN line
            qkv_match = re.search(r'QKV:\s+([\d.e+-]+)\s+W_o:\s+([\d.e+-]+)\s+FFN_W1:\s+([\d.e+-]+)\s+FFN_W2:\s+([\d.e+-]+)', line)
            if qkv_match and current_batch >= 0:
                grad_data[current_batch][f'layer{current_layer}_qkv'] = float(qkv_match.group(1))
                grad_data[current_batch][f'layer{current_layer}_wo'] = float(qkv_match.group(2))
                grad_data[current_batch][f'layer{current_layer}_ffn1'] = float(qkv_match.group(3))
                grad_data[current_batch][f'layer{current_layer}_ffn2'] = float(qkv_match.group(4))
                continue
            
            # Parse RMSNorm gamma: RMS1_gamma: 8.0785e-01  RMS2_gamma: 1.5295e+00
            rms_match = re.search(r'RMS1_gamma:\s+([\d.e+-]+)\s+RMS2_gamma:\s+([\d.e+-]+)', line)
            if rms_match and current_batch >= 0:
                rms_data[current_batch][current_layer] = [
                    float(rms_match.group(1)),
                    float(rms_match.group(2))
                ]
                continue
            
            # Parse EMBEDDING gradient
            emb_match = re.search(r'EMBEDDING:\s+([\d.e+-]+)', line)
            if emb_match and current_batch >= 0:
                grad_data[current_batch]['embedding'] = float(emb_match.group(1))
                continue
    
    return batches, losses, attn_data, grad_data, rms_data

def plot_attention_development(batches, losses, attn_data, grad_data, rms_data):
    """Create visualization of attention development."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GRIM-text Attention Development Over Training', fontsize=14, fontweight='bold')
    
    # 1. Loss curve
    ax1 = axes[0, 0]
    ax1.plot(batches, losses, 'b-', linewidth=1.5, label='Training Loss')
    ax1.axhline(y=8.5, color='r', linestyle='--', alpha=0.5, label='Plateau ~8.5')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Entropy per layer over batches (sample first occurrence per batch)
    ax2 = axes[0, 1]
    layers = [0, 1, 2, 3, 4, 5]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(layers)) for i in range(len(layers))]
    
    for layer, color in zip(layers, colors):
        batch_entropy = []
        batch_nums = []
        for (step, l), data in sorted(attn_data.items()):
            if l == layer and 'H' in data:
                if step not in batch_nums:  # First occurrence only
                    batch_nums.append(step)
                    batch_entropy.append(data['H'])
        if batch_entropy:
            ax2.plot(batch_nums[:200], batch_entropy[:200], color=color, 
                    linewidth=1, alpha=0.8, label=f'Layer {layer}')
    
    ax2.axhline(y=6.77, color='r', linestyle='--', alpha=0.5, label='Max entropy (uniform)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Entropy (bits/position)')
    ax2.set_title('Attention Entropy by Layer')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([4.5, 7.0])
    
    # 3. Q gradient per layer
    ax3 = axes[0, 2]
    for layer, color in zip(layers, colors):
        batch_qgrad = []
        batch_nums = []
        for (step, l), data in sorted(attn_data.items()):
            if l == layer and 'Q_grad' in data:
                if step not in batch_nums:
                    batch_nums.append(step)
                    batch_qgrad.append(data['Q_grad'])
        if batch_qgrad:
            ax3.plot(batch_nums[:200], batch_qgrad[:200], color=color, 
                    linewidth=1, alpha=0.8, label=f'Layer {layer}')
    
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Q Gradient Norm')
    ax3.set_title('Query Gradient by Layer (Softmax Jacobian Effect)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Gradient components over batches
    ax4 = axes[1, 0]
    grad_batches = sorted(grad_data.keys())
    if grad_batches:
        lm_head = [grad_data[b].get('lm_head', 0) for b in grad_batches]
        embedding = [grad_data[b].get('embedding', 0) for b in grad_batches]
        
        ax4.plot(grad_batches, lm_head, 'r-', linewidth=1.5, label='LM_HEAD')
        ax4.plot(grad_batches, embedding, 'b-', linewidth=1.5, label='EMBEDDING')
        
        ax4.set_xlabel('Batch')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('LM Head vs Embedding Gradients')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    # 5. QKV gradients per layer
    ax5 = axes[1, 1]
    if grad_batches:
        for layer, color in zip(layers, colors):
            qkv_grads = [grad_data[b].get(f'layer{layer}_qkv', 0) for b in grad_batches]
            ax5.plot(grad_batches, qkv_grads, color=color, linewidth=1, label=f'L{layer}')
        
        ax5.set_xlabel('Batch')
        ax5.set_ylabel('QKV Gradient Norm')
        ax5.set_title('QKV Gradients by Layer')
        ax5.legend(loc='upper right', fontsize=8)
        ax5.grid(True, alpha=0.3)
    
    # 6. RMSNorm gamma gradients
    ax6 = axes[1, 2]
    if rms_data:
        rms_batches = sorted(rms_data.keys())
        for layer, color in zip(layers, colors):
            rms1_grads = [rms_data[b].get(layer, [0, 0])[0] for b in rms_batches]
            rms2_grads = [rms_data[b].get(layer, [0, 0])[1] for b in rms_batches]
            ax6.plot(rms_batches, rms1_grads, color=color, linewidth=1, 
                    linestyle='-', alpha=0.7, label=f'L{layer} RMS1')
            ax6.plot(rms_batches, rms2_grads, color=color, linewidth=1, 
                    linestyle='--', alpha=0.7)
        
        ax6.set_xlabel('Batch')
        ax6.set_ylabel('RMSNorm Gamma Gradient')
        ax6.set_title('RMSNorm Gamma Gradients (solid=RMS1, dashed=RMS2)')
        ax6.legend(loc='upper right', fontsize=7, ncol=2)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:/G.R.I.M/resources/models/GRIM-text/training/attention_development.png', 
                dpi=150, bbox_inches='tight')
    print("Saved: attention_development.png")
    plt.show()

def main():
    filepath = 'd:/G.R.I.M/resources/models/GRIM-text/training/diagnostic_output.txt'
    print(f"Parsing {filepath}...")
    
    batches, losses, attn_data, grad_data, rms_data = parse_diagnostic_output(filepath)
    
    print(f"Found {len(batches)} batches with loss data")
    print(f"Found {len(attn_data)} attention diagnostic entries")
    print(f"Found {len(grad_data)} gradient trace entries")
    print(f"Found {len(rms_data)} RMSNorm gradient entries")
    
    if losses:
        print(f"\nLoss: {losses[0]:.4f} → {losses[-1]:.4f}")
    
    # Check RMSNorm gamma gradients
    if rms_data:
        sample_batch = list(rms_data.keys())[0]
        print(f"\nRMSNorm gamma gradients at batch {sample_batch}:")
        for layer in sorted(rms_data[sample_batch].keys()):
            rms1, rms2 = rms_data[sample_batch][layer]
            print(f"  Layer {layer}: RMS1={rms1:.4e}, RMS2={rms2:.4e}")
    
    plot_attention_development(batches, losses, attn_data, grad_data, rms_data)

if __name__ == '__main__':
    main()

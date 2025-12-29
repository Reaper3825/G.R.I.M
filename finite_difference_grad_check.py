#!/usr/bin/env python3
"""
Approach 3: Finite Difference Gradient Check

The gold standard for verifying gradients: perturb weights by ε, 
measure loss change, compare to computed gradient.

Formula: grad ≈ (loss(w+ε) - loss(w-ε)) / (2ε)

If computed gradient matches finite difference, backward is correct.
If not, we've found the bug.

This script provides:
1. A PyTorch implementation to verify our PyTorch reference is correct
2. Code to add to GRIM-text for direct finite difference testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================================
# GRIM-text Finite Difference Code
# ============================================================================


# ============================================================================
# PyTorch Finite Difference Check (to verify our reference is correct)
# ============================================================================
class SimpleTransformerLayer(nn.Module):
    """Simplified transformer for gradient checking"""
    def __init__(self, d_model=64, d_ff=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    
    def forward(self, x):
        # Pre-norm attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        
        # Pre-norm FFN
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class SimpleModel(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, num_layers=2, d_ff=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(d_model, d_ff) for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.output(h)


def finite_difference_check(model, input_ids, targets, param_name, idx, epsilon=1e-5):
    """
    Check gradient of a specific parameter using finite differences.
    
    Args:
        model: PyTorch model
        input_ids: Input tensor
        targets: Target tensor
        param_name: Name of parameter to check (e.g., 'layers.0.ffn.0.weight')
        idx: Index within the flattened parameter to check
        epsilon: Perturbation size
    
    Returns:
        dict with numerical_grad, computed_grad, relative_error
    """
    # Find the parameter
    param = None
    for name, p in model.named_parameters():
        if name == param_name:
            param = p
            break
    
    if param is None:
        raise ValueError(f"Parameter {param_name} not found")
    
    # Flatten for easy indexing
    flat_param = param.view(-1)
    original_value = flat_param[idx].item()
    
    def compute_loss():
        model.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss.item()
    
    # Loss at w + epsilon
    with torch.no_grad():
        flat_param[idx] = original_value + epsilon
    loss_plus = compute_loss()
    
    # Loss at w - epsilon
    with torch.no_grad():
        flat_param[idx] = original_value - epsilon
    loss_minus = compute_loss()
    
    # Restore original
    with torch.no_grad():
        flat_param[idx] = original_value
    
    # Numerical gradient
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Computed gradient via backprop
    model.zero_grad()
    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    
    computed_grad = param.grad.view(-1)[idx].item()
    
    # Relative error
    rel_error = abs(numerical_grad - computed_grad) / (abs(numerical_grad) + abs(computed_grad) + 1e-8)
    
    return {
        'param_name': param_name,
        'idx': idx,
        'epsilon': epsilon,
        'loss_plus': loss_plus,
        'loss_minus': loss_minus,
        'numerical_grad': numerical_grad,
        'computed_grad': computed_grad,
        'relative_error': rel_error,
        'passed': rel_error < 1e-4
    }


def run_comprehensive_grad_check():
    """Run finite difference checks on multiple parameters"""
    print("="*70)
    print("FINITE DIFFERENCE GRADIENT CHECK (PyTorch Reference)")
    print("="*70)
    
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)  # Use double precision for accuracy
    
    model = SimpleModel(vocab_size=1000, d_model=64, num_layers=2, d_ff=256)
    model.eval()  # Disable dropout
    
    # Small batch for speed
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    targets = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Batch: {batch_size} × {seq_len} tokens")
    print(f"Epsilon: 1e-5 (using float64 for precision)")
    
    # Check gradients for key parameters
    params_to_check = [
        ('embedding.weight', 0),
        ('embedding.weight', 100),
        ('layers.0.ffn.0.weight', 0),       # FFN W1
        ('layers.0.ffn.0.weight', 1000),    # FFN W1 middle
        ('layers.0.ffn.2.weight', 0),       # FFN W2
        ('layers.1.ffn.0.weight', 0),       # Layer 1 FFN W1
        ('output.weight', 0),               # Output projection
    ]
    
    # Use smaller epsilon with float64
    epsilon = 1e-5
    
    print("\n" + "-"*70)
    print(f"{'Parameter':<35} {'Numerical':<12} {'Computed':<12} {'RelErr':<10} {'Status'}")
    print("-"*70)
    
    all_passed = True
    for param_name, idx in params_to_check:
        try:
            result = finite_difference_check(model, input_ids, targets, param_name, idx)
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            if not result['passed']:
                all_passed = False
            display_name = f"{param_name}[{idx}]"
            print(f"{display_name:<35} {result['numerical_grad']:<12.6e} {result['computed_grad']:<12.6e} {result['relative_error']:<10.2e} {status}")
        except Exception as e:
            display_name = f"{param_name}[{idx}]"
            print(f"{display_name:<35} ERROR: {e}")
            all_passed = False
    
    print("-"*70)
    
    if all_passed:
        print("\n✓ All gradient checks PASSED")
        print("PyTorch reference implementation is correct.")
    else:
        print("\n✗ Some gradient checks FAILED")
        print("There may be a bug in the PyTorch reference.")
    



if __name__ == "__main__":
    run_comprehensive_grad_check()

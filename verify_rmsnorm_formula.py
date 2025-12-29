#!/usr/bin/env python3
"""
Detailed RMSNorm backward formula verification against GRIM's kernel.
"""

import torch
import math

def grim_rmsnorm_backward_simulation(x, grad_output, gamma, eps=1e-5):
    """
    Simulate GRIM's RMSNorm backward kernel step by step.
    
    GRIM kernel does:
    1. Compute inv_rms = 1/sqrt(mean(x^2) + eps)
    2. norm = x * inv_rms
    3. g = grad_output * gamma  (scaled grad)
    4. dot = sum(g * norm) / hidden_dim
    5. grad_input = g * inv_rms - norm * dot
    6. grad_gamma = sum_over_tokens(grad_output * norm)  <- atomicAdd
    """
    batch_size, hidden_dim = x.shape
    
    # Step 1: Compute RMS per token
    mean_sq = (x ** 2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_sq + eps)
    # GRIM clamps inv_rms to 100
    inv_rms = torch.clamp(inv_rms, max=100.0)
    
    # Step 2: Normalized x
    norm = x * inv_rms
    
    # Step 3: Scaled gradient (GRIM uses gamma as scale)
    g = grad_output * gamma  # [batch, hidden]
    
    # Step 4: Dot product term (for grad_input correction)
    dot = (g * norm).sum(dim=-1, keepdim=True) / hidden_dim
    
    # Step 5: grad_input
    grad_input = g * inv_rms - norm * dot
    
    # Step 6: grad_gamma (sum over all tokens - this is what atomicAdd does)
    grad_gamma = (grad_output * norm).sum(dim=0)  # [hidden_dim]
    
    return grad_input, grad_gamma, inv_rms, norm


def pytorch_rmsnorm_backward(x, grad_output, gamma, eps=1e-5):
    """PyTorch autograd reference."""
    x = x.clone().requires_grad_(True)
    gamma = gamma.clone().requires_grad_(True)
    
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    y = gamma * x_norm
    
    y.backward(grad_output)
    
    return x.grad, gamma.grad


def main():
    torch.manual_seed(42)
    
    # Match GRIM config
    hidden_dim = 768
    total_tokens = 2940
    eps = 1e-5
    
    print("="*80)
    print("RMSNorm Backward Formula Verification")
    print("="*80)
    
    # Create test data
    x = torch.randn(total_tokens, hidden_dim)
    gamma = torch.ones(hidden_dim) + torch.randn(hidden_dim) * 0.02
    
    # Use small grad like layer 11
    grad_output = torch.randn(total_tokens, hidden_dim) * 2.5e-7
    
    print(f"\nInput shapes: x={x.shape}, gamma={gamma.shape}")
    print(f"grad_output RMS: {torch.sqrt((grad_output**2).mean()).item():.6e}")
    
    # Run both implementations
    grim_grad_input, grim_grad_gamma, inv_rms, norm = grim_rmsnorm_backward_simulation(
        x, grad_output, gamma, eps
    )
    pytorch_grad_input, pytorch_grad_gamma = pytorch_rmsnorm_backward(
        x, grad_output, gamma, eps
    )
    
    print("\n--- GRIM Simulation ---")
    print(f"inv_rms stats: mean={inv_rms.mean().item():.4f}, max={inv_rms.max().item():.4f}")
    print(f"norm RMS: {torch.sqrt((norm**2).mean()).item():.6f}")
    print(f"grad_input RMS:  {torch.sqrt((grim_grad_input**2).mean()).item():.6e}")
    print(f"grad_gamma RMS:  {torch.sqrt((grim_grad_gamma**2).mean()).item():.6e}")
    
    print("\n--- PyTorch Reference ---")
    print(f"grad_input RMS:  {torch.sqrt((pytorch_grad_input**2).mean()).item():.6e}")
    print(f"grad_gamma RMS:  {torch.sqrt((pytorch_grad_gamma**2).mean()).item():.6e}")
    
    print("\n--- Comparison ---")
    grad_input_diff = torch.abs(grim_grad_input - pytorch_grad_input).max().item()
    grad_gamma_diff = torch.abs(grim_grad_gamma - pytorch_grad_gamma).max().item()
    
    print(f"grad_input max diff:  {grad_input_diff:.6e}")
    print(f"grad_gamma max diff:  {grad_gamma_diff:.6e}")
    
    if grad_input_diff < 1e-5 and grad_gamma_diff < 1e-5:
        print("\n✓ GRIM formula matches PyTorch!")
    else:
        print("\n❌ GRIM formula DIFFERS from PyTorch!")
        
        # Debug: What's different?
        print("\nDebugging differences...")
        
        # Check grad_gamma formula
        # PyTorch: d(gamma * x/rms)/dgamma = x/rms = norm
        # So grad_gamma = sum(grad_output * norm)
        manual_grad_gamma = (grad_output * norm).sum(dim=0)
        print(f"Manual grad_gamma (sum(go * norm)): {torch.sqrt((manual_grad_gamma**2).mean()).item():.6e}")
        print(f"PyTorch grad_gamma:                 {torch.sqrt((pytorch_grad_gamma**2).mean()).item():.6e}")
        
    # Check ratios
    print("\n--- Ratio Analysis ---")
    grim_ratio = torch.sqrt((grim_grad_gamma**2).mean()).item() / torch.sqrt((grim_grad_input**2).mean()).item()
    pytorch_ratio = torch.sqrt((pytorch_grad_gamma**2).mean()).item() / torch.sqrt((pytorch_grad_input**2).mean()).item()
    
    print(f"GRIM sim ratio (gamma/input): {grim_ratio:.2f}")
    print(f"PyTorch ratio (gamma/input):  {pytorch_ratio:.2f}")
    
    print("\n--- GRIM Actual Values (from log) ---")
    print(f"Layer 11 grad_gamma RMS: 0.712098")
    print(f"Layer 11 grad_input RMS: 2.53717e-07")
    print(f"Layer 11 ratio:          {0.712098 / 2.53717e-7:.2e}")
    
    print(f"\nExpected ratio: ~{pytorch_ratio:.0f}")
    print(f"Actual ratio:   ~{0.712098 / 2.53717e-7:.0f}")
    print(f"Overscale factor: {(0.712098 / 2.53717e-7) / pytorch_ratio:.0f}x")
    
    # What could cause this?
    overscale = (0.712098 / 2.53717e-7) / pytorch_ratio
    print(f"\n--- Possible Causes ---")
    print(f"If overscale ≈ total_tokens ({total_tokens}): grad_gamma not normalized")
    print(f"If overscale ≈ hidden_dim ({hidden_dim}): formula issue")
    print(f"If overscale ≈ total_tokens * hidden_dim ({total_tokens * hidden_dim}): double issue")
    print(f"Actual overscale: {overscale:.0f}")
    
    if 0.5 * total_tokens < overscale < 2 * total_tokens:
        print(f"\n→ LIKELY CAUSE: grad_gamma summed over {total_tokens} tokens without averaging")
    elif 0.5 * hidden_dim < overscale < 2 * hidden_dim:
        print(f"\n→ LIKELY CAUSE: Hidden dimension factor missing somewhere")
    else:
        print(f"\n→ UNKNOWN CAUSE - overscale doesn't match obvious factors")


if __name__ == "__main__":
    main()

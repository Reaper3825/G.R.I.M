#!/usr/bin/env python3
"""
PyTorch reference test for RMSNorm backward gradients.
Compare against GRIM's logged values to identify discrepancies.
"""

import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    """PyTorch RMSNorm implementation matching GRIM."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma
    
    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm

def test_rmsnorm_gradients():
    """Test RMSNorm backward with values matching GRIM's log."""
    
    # Config from GRIM logs
    d_model = 768
    total_tokens = 2940  # batch_size * seq_len from log
    eps = 1e-5
    
    print("="*80)
    print("PyTorch RMSNorm Backward Reference Test")
    print("="*80)
    print(f"d_model: {d_model}")
    print(f"total_tokens: {total_tokens}")
    print(f"eps: {eps}")
    print()
    
    # Create RMSNorm layer
    rmsnorm = RMSNorm(d_model, eps)
    
    # Initialize gamma to ~1.0 with small variation (matching GRIM's range 0.96-1.03)
    with torch.no_grad():
        rmsnorm.weight.copy_(torch.ones(d_model) + torch.randn(d_model) * 0.02)
    
    # Create input tensor (random, representing residual1)
    # Use similar scale to what GRIM would have
    x = torch.randn(total_tokens, d_model, requires_grad=True)
    
    # Forward pass
    out = rmsnorm(x)
    
    # Create grad_output with scale matching GRIM's grad_ffn_input
    # Layer 11: grad_ffn_input RMS ≈ 2.5e-07
    # Layer 0:  grad_ffn_input RMS ≈ 0.0044
    
    print("Testing with Layer 11 gradient scale (small):")
    grad_output_l11 = torch.randn_like(out) * 2.5e-7
    
    # Backward
    out.backward(grad_output_l11, retain_graph=True)
    
    # Get gradients
    grad_input_l11 = x.grad.clone()
    grad_gamma_l11 = rmsnorm.weight.grad.clone()
    
    grad_input_rms = torch.sqrt(torch.mean(grad_input_l11 ** 2)).item()
    grad_gamma_rms = torch.sqrt(torch.mean(grad_gamma_l11 ** 2)).item()
    grad_gamma_max = grad_gamma_l11.abs().max().item()
    grad_gamma_min = grad_gamma_l11.min().item()
    grad_gamma_max_val = grad_gamma_l11.max().item()
    
    print(f"  grad_output RMS:     {torch.sqrt(torch.mean(grad_output_l11**2)).item():.6e}")
    print(f"  grad_input RMS:      {grad_input_rms:.6e}")
    print(f"  grad_gamma RMS:      {grad_gamma_rms:.6e}")
    print(f"  grad_gamma range:    [{grad_gamma_min:.6e}, {grad_gamma_max_val:.6e}]")
    print(f"  grad_gamma max_abs:  {grad_gamma_max:.6e}")
    print(f"  Ratio (gamma/input): {grad_gamma_rms / (grad_input_rms + 1e-20):.2f}x")
    print()
    
    # Compare with GRIM Layer 11 values
    print("  GRIM Layer 11 values:")
    print(f"    grad_ffn_input RMS:   2.53717e-07")
    print(f"    ln2_gamma_grads RMS:  0.712098")
    print(f"    Ratio:                {0.712098 / 2.53717e-7:.2e}x")
    print()
    
    # Reset gradients
    x.grad.zero_()
    rmsnorm.weight.grad.zero_()
    
    print("Testing with Layer 0 gradient scale (large):")
    grad_output_l0 = torch.randn_like(out) * 0.0044
    
    # Backward
    out.backward(grad_output_l0)
    
    grad_input_l0 = x.grad.clone()
    grad_gamma_l0 = rmsnorm.weight.grad.clone()
    
    grad_input_rms = torch.sqrt(torch.mean(grad_input_l0 ** 2)).item()
    grad_gamma_rms = torch.sqrt(torch.mean(grad_gamma_l0 ** 2)).item()
    grad_gamma_max = grad_gamma_l0.abs().max().item()
    
    print(f"  grad_output RMS:     {torch.sqrt(torch.mean(grad_output_l0**2)).item():.6e}")
    print(f"  grad_input RMS:      {grad_input_rms:.6e}")
    print(f"  grad_gamma RMS:      {grad_gamma_rms:.6e}")
    print(f"  grad_gamma max_abs:  {grad_gamma_max:.6e}")
    print(f"  Ratio (gamma/input): {grad_gamma_rms / (grad_input_rms + 1e-20):.2f}x")
    print()
    
    # Compare with GRIM Layer 0 values
    print("  GRIM Layer 0 values:")
    print(f"    grad_ffn_input RMS:   0.00437813")
    print(f"    ln2_gamma_grads RMS:  98067.1")
    print(f"    Ratio:                {98067.1 / 0.00437813:.2e}x")
    print()
    
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    
    # The key question: what ratio does PyTorch give?
    pytorch_ratio_l11 = torch.sqrt(torch.mean(grad_gamma_l11 ** 2)).item() / (torch.sqrt(torch.mean(grad_input_l11 ** 2)).item() + 1e-20)
    pytorch_ratio_l0 = torch.sqrt(torch.mean(grad_gamma_l0 ** 2)).item() / (torch.sqrt(torch.mean(grad_input_l0 ** 2)).item() + 1e-20)
    
    grim_ratio_l11 = 0.712098 / 2.53717e-7
    grim_ratio_l0 = 98067.1 / 0.00437813
    
    print(f"PyTorch grad_gamma/grad_input ratio (L11): {pytorch_ratio_l11:.2e}")
    print(f"GRIM    grad_gamma/grad_input ratio (L11): {grim_ratio_l11:.2e}")
    print(f"Discrepancy: {grim_ratio_l11 / pytorch_ratio_l11:.2f}x")
    print()
    print(f"PyTorch grad_gamma/grad_input ratio (L0):  {pytorch_ratio_l0:.2e}")
    print(f"GRIM    grad_gamma/grad_input ratio (L0):  {grim_ratio_l0:.2e}")
    print(f"Discrepancy: {grim_ratio_l0 / pytorch_ratio_l0:.2f}x")
    print()
    
    if grim_ratio_l11 / pytorch_ratio_l11 > 100:
        print(f"❌ GRIM grad_gamma is {grim_ratio_l11 / pytorch_ratio_l11:.0f}x larger than PyTorch!")
        print(f"   Expected: ~{pytorch_ratio_l11 * 2.53717e-7:.6e}")
        print(f"   Got:      0.712098")
        print()
        print("   This suggests GRIM is summing grad_gamma without normalization.")
        print(f"   Total tokens: {total_tokens}")
        print(f"   {total_tokens}x overscaling would give: {pytorch_ratio_l11 * 2.53717e-7 * total_tokens:.6e}")


def test_exact_formula():
    """Test the exact RMSNorm backward formula."""
    print("\n" + "="*80)
    print("EXACT FORMULA VERIFICATION")
    print("="*80)
    
    d_model = 768
    total_tokens = 10  # Small for verification
    eps = 1e-5
    
    # Create simple inputs
    x = torch.randn(total_tokens, d_model)
    gamma = torch.ones(d_model) + torch.randn(d_model) * 0.02
    grad_output = torch.randn(total_tokens, d_model) * 0.001
    
    # Forward
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    x_norm = x / rms
    
    # Manual backward for grad_gamma:
    # y = gamma * x_norm
    # dy/dgamma = x_norm (summed over batch)
    # grad_gamma = sum_over_tokens(grad_output * x_norm)
    
    grad_gamma_manual = (grad_output * x_norm).sum(dim=0)
    
    print(f"Manual grad_gamma (sum over {total_tokens} tokens):")
    print(f"  RMS: {torch.sqrt(torch.mean(grad_gamma_manual**2)).item():.6e}")
    print(f"  This is a SUM, not an average.")
    print()
    
    # If we averaged:
    grad_gamma_averaged = grad_gamma_manual / total_tokens
    print(f"If averaged over tokens:")
    print(f"  RMS: {torch.sqrt(torch.mean(grad_gamma_averaged**2)).item():.6e}")
    print()
    
    # PyTorch autograd
    x_pt = x.clone().requires_grad_(True)
    gamma_pt = gamma.clone().requires_grad_(True)
    
    rms_pt = torch.sqrt(torch.mean(x_pt ** 2, dim=-1, keepdim=True) + eps)
    y_pt = gamma_pt * (x_pt / rms_pt)
    y_pt.backward(grad_output)
    
    print(f"PyTorch autograd grad_gamma:")
    print(f"  RMS: {torch.sqrt(torch.mean(gamma_pt.grad**2)).item():.6e}")
    print()
    
    # Check if PyTorch sums or averages
    diff_sum = torch.abs(gamma_pt.grad - grad_gamma_manual).max().item()
    diff_avg = torch.abs(gamma_pt.grad - grad_gamma_averaged).max().item()
    
    print(f"Difference from manual SUM: {diff_sum:.6e}")
    print(f"Difference from manual AVG: {diff_avg:.6e}")
    
    if diff_sum < diff_avg:
        print("→ PyTorch SUMS grad_gamma over tokens (no averaging)")
    else:
        print("→ PyTorch AVERAGES grad_gamma over tokens")


if __name__ == "__main__":
    test_rmsnorm_gradients()
    test_exact_formula()

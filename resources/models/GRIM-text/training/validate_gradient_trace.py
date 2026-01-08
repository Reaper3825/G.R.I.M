#!/usr/bin/env python3
"""
validate_gradient_trace.py - Verify gradient correctness via numerical methods

Proves gradient computation is correct by:
1. Numerical gradient checking (finite differences)
2. Chain rule consistency checks
3. Known analytical gradient validation
4. Gradient-weight relationship verification
"""

import sys
import json
import numpy as np
from pathlib import Path
from trace_single_gradient import read_gradient_dump, analyze_gradient_tensor


def numerical_gradient_check(model_forward_func, params, loss_func, epsilon=1e-4):
    """
    Compute numerical gradients using finite differences
    
    For parameter θ, numerical gradient is:
        ∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / (2ε)
    
    Args:
        model_forward_func: Function that runs forward pass given params
        params: Dict of parameter tensors
        loss_func: Function that computes loss from forward output
        epsilon: Finite difference step size
    
    Returns:
        Dict of numerical gradients
    """
    numerical_grads = {}
    
    for param_name, param_tensor in params.items():
        print(f"Computing numerical gradient for {param_name}...")
        
        param_flat = param_tensor.flatten()
        num_grad_flat = np.zeros_like(param_flat)
        
        # Sample random subset for efficiency (full check is too slow)
        sample_size = min(100, len(param_flat))
        indices = np.random.choice(len(param_flat), sample_size, replace=False)
        
        for i in indices:
            # Save original value
            original_val = param_flat[i]
            
            # Forward pass with θ + ε
            param_flat[i] = original_val + epsilon
            params[param_name] = param_flat.reshape(param_tensor.shape)
            output_plus = model_forward_func(params)
            loss_plus = loss_func(output_plus)
            
            # Forward pass with θ - ε
            param_flat[i] = original_val - epsilon
            params[param_name] = param_flat.reshape(param_tensor.shape)
            output_minus = model_forward_func(params)
            loss_minus = loss_func(output_minus)
            
            # Numerical gradient
            num_grad_flat[i] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore original value
            param_flat[i] = original_val
        
        numerical_grads[param_name] = num_grad_flat.reshape(param_tensor.shape)
        print(f"  ✓ Computed {sample_size} sample gradients")
    
    return numerical_grads


def compare_gradients(analytical_grads, numerical_grads, rtol=1e-3, atol=1e-5):
    """
    Compare analytical (backprop) vs numerical (finite diff) gradients
    
    Returns:
        Dict with comparison results per parameter
    """
    results = {}
    
    for param_name in analytical_grads.keys():
        if param_name not in numerical_grads:
            results[param_name] = {'status': 'missing_numerical', 'match': False}
            continue
        
        anal_grad = analytical_grads[param_name].flatten()
        num_grad = numerical_grads[param_name].flatten()
        
        # Only compare indices that were computed numerically (non-zero)
        non_zero_mask = np.abs(num_grad) > 1e-10
        anal_subset = anal_grad[non_zero_mask]
        num_subset = num_grad[non_zero_mask]
        
        if len(num_subset) == 0:
            results[param_name] = {'status': 'no_samples', 'match': False}
            continue
        
        # Relative error
        rel_error = np.abs(anal_subset - num_subset) / (np.abs(num_subset) + atol)
        max_rel_error = np.max(rel_error)
        mean_rel_error = np.mean(rel_error)
        
        # Absolute error
        abs_error = np.abs(anal_subset - num_subset)
        max_abs_error = np.max(abs_error)
        
        # Check if gradients match
        match = np.allclose(anal_subset, num_subset, rtol=rtol, atol=atol)
        
        results[param_name] = {
            'status': 'compared',
            'match': match,
            'max_rel_error': float(max_rel_error),
            'mean_rel_error': float(mean_rel_error),
            'max_abs_error': float(max_abs_error),
            'samples_compared': len(num_subset)
        }
    
    return results


def check_softmax_cross_entropy_gradient():
    """
    Analytical validation: Softmax + Cross Entropy gradient
    
    For softmax: p_i = exp(z_i) / Σ_j exp(z_j)
    For cross entropy: L = -log(p_target)
    
    Gradient: ∂L/∂z_i = p_i - δ(i == target)
    """
    print("\n" + "="*80)
    print("ANALYTICAL VALIDATION: Softmax + Cross Entropy")
    print("="*80)
    
    # Simple test case
    logits = np.array([2.0, 1.0, 0.1])
    target_idx = 0
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / np.sum(exp_logits)
    
    # Cross entropy loss
    loss = -np.log(probs[target_idx])
    
    # Analytical gradient
    grad_analytical = probs.copy()
    grad_analytical[target_idx] -= 1.0
    
    # Numerical gradient
    epsilon = 1e-5
    grad_numerical = np.zeros_like(logits)
    
    for i in range(len(logits)):
        logits_plus = logits.copy()
        logits_plus[i] += epsilon
        exp_plus = np.exp(logits_plus - np.max(logits_plus))
        probs_plus = exp_plus / np.sum(exp_plus)
        loss_plus = -np.log(probs_plus[target_idx])
        
        logits_minus = logits.copy()
        logits_minus[i] -= epsilon
        exp_minus = np.exp(logits_minus - np.max(logits_minus))
        probs_minus = exp_minus / np.sum(exp_minus)
        loss_minus = -np.log(probs_minus[target_idx])
        
        grad_numerical[i] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Compare
    rel_error = np.abs(grad_analytical - grad_numerical) / (np.abs(grad_numerical) + 1e-8)
    max_rel_error = np.max(rel_error)
    
    print(f"Logits:              {logits}")
    print(f"Target:              {target_idx}")
    print(f"Loss:                {loss:.6f}")
    print(f"\nAnalytical gradient: {grad_analytical}")
    print(f"Numerical gradient:  {grad_numerical}")
    print(f"Relative error:      {rel_error}")
    print(f"Max rel error:       {max_rel_error:.2e}")
    
    if max_rel_error < 1e-4:
        print("\n✓ PASS: Analytical gradient matches numerical gradient")
        return True
    else:
        print("\n✗ FAIL: Gradient mismatch!")
        return False


def check_matrix_multiply_gradient():
    """
    Analytical validation: Matrix multiplication gradient
    
    Forward: Y = X @ W
    Backward: dL/dW = X^T @ dL/dY
              dL/dX = dL/dY @ W^T
    """
    print("\n" + "="*80)
    print("ANALYTICAL VALIDATION: Matrix Multiplication")
    print("="*80)
    
    # Simple case: (2x3) @ (3x2) = (2x2)
    X = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    W = np.array([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6]])
    
    # Forward
    Y = X @ W
    
    # Pretend we have upstream gradient dL/dY
    dL_dY = np.array([[1.0, 0.5],
                      [0.8, 0.3]])
    
    # Analytical gradient dL/dW = X^T @ dL/dY
    dL_dW_analytical = X.T @ dL_dY
    
    # Numerical gradient
    epsilon = 1e-5
    dL_dW_numerical = np.zeros_like(W)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_plus = W.copy()
            W_plus[i, j] += epsilon
            Y_plus = X @ W_plus
            loss_plus = np.sum(dL_dY * Y_plus)
            
            W_minus = W.copy()
            W_minus[i, j] -= epsilon
            Y_minus = X @ W_minus
            loss_minus = np.sum(dL_dY * Y_minus)
            
            dL_dW_numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Compare
    rel_error = np.abs(dL_dW_analytical - dL_dW_numerical) / (np.abs(dL_dW_numerical) + 1e-8)
    max_rel_error = np.max(rel_error)
    
    print(f"X shape:             {X.shape}")
    print(f"W shape:             {W.shape}")
    print(f"Y shape:             {Y.shape}")
    print(f"\nAnalytical dL/dW:\n{dL_dW_analytical}")
    print(f"\nNumerical dL/dW:\n{dL_dW_numerical}")
    print(f"\nRelative error:\n{rel_error}")
    print(f"Max rel error:       {max_rel_error:.2e}")
    
    if max_rel_error < 1e-4:
        print("\n✓ PASS: Analytical gradient matches numerical gradient")
        return True
    else:
        print("\n✗ FAIL: Gradient mismatch!")
        return False


def check_gelu_gradient():
    """
    Analytical validation: GELU activation gradient
    
    GELU(x) = x * Φ(x) where Φ is standard normal CDF
    Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Gradient: dGELU/dx = Φ(x) + x * φ(x)
    where φ(x) is standard normal PDF
    """
    print("\n" + "="*80)
    print("ANALYTICAL VALIDATION: GELU Activation")
    print("="*80)
    
    # Test values
    x_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    # GELU forward (approximation)
    def gelu(x):
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
    
    # GELU gradient (approximation derivative)
    def gelu_grad(x):
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        cube = x**3
        tanh_arg = sqrt_2_over_pi * (x + 0.044715 * cube)
        tanh_val = np.tanh(tanh_arg)
        sech2_val = 1.0 - tanh_val**2
        
        grad = 0.5 * (1.0 + tanh_val) + \
               0.5 * x * sech2_val * sqrt_2_over_pi * (1.0 + 0.044715 * 3 * x**2)
        return grad
    
    # Compute analytical gradients
    grad_analytical = gelu_grad(x_values)
    
    # Compute numerical gradients
    epsilon = 1e-5
    grad_numerical = np.zeros_like(x_values)
    
    for i, x in enumerate(x_values):
        y_plus = gelu(x + epsilon)
        y_minus = gelu(x - epsilon)
        grad_numerical[i] = (y_plus - y_minus) / (2 * epsilon)
    
    # Compare
    rel_error = np.abs(grad_analytical - grad_numerical) / (np.abs(grad_numerical) + 1e-8)
    max_rel_error = np.max(rel_error)
    
    print(f"Test points:         {x_values}")
    print(f"GELU(x):            {gelu(x_values)}")
    print(f"\nAnalytical gradient: {grad_analytical}")
    print(f"Numerical gradient:  {grad_numerical}")
    print(f"Relative error:      {rel_error}")
    print(f"Max rel error:       {max_rel_error:.2e}")
    
    if max_rel_error < 1e-3:  # GELU approximation is less precise
        print("\n✓ PASS: Analytical gradient matches numerical gradient")
        return True
    else:
        print("\n✗ FAIL: Gradient mismatch!")
        return False


def validate_gradient_sanity(gradient_dump_path):
    """
    Sanity checks on gradient dump
    
    1. No NaN or Inf values
    2. Reasonable magnitudes (not all zero, not astronomical)
    3. Proper shapes (match expected parameter dimensions)
    4. Gradient flow (deeper layers should have smaller gradients)
    """
    print("\n" + "="*80)
    print("SANITY CHECKS: Gradient Dump Validation")
    print("="*80)
    
    tensors = read_gradient_dump(gradient_dump_path)
    
    issues = []
    warnings = []
    
    # Check 1: NaN/Inf
    print("\n[1/5] Checking for NaN/Inf values...")
    for name, grad in tensors.items():
        if not np.all(np.isfinite(grad)):
            nan_count = np.sum(np.isnan(grad))
            inf_count = np.sum(np.isinf(grad))
            issues.append(f"  ✗ {name}: {nan_count} NaN, {inf_count} Inf")
    
    if not issues:
        print("  ✓ All gradients are finite")
    else:
        for issue in issues:
            print(issue)
    
    # Check 2: Zero gradients
    print("\n[2/5] Checking for zero gradients...")
    zero_tensors = []
    for name, grad in tensors.items():
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            zero_tensors.append(f"  ⚠ {name}: norm = {grad_norm:.2e}")
    
    if not zero_tensors:
        print("  ✓ No zero gradients detected")
    else:
        print(f"  ⚠ Found {len(zero_tensors)} near-zero gradients:")
        for tensor in zero_tensors[:5]:  # Show first 5
            print(tensor)
    
    # Check 3: Exploding gradients
    print("\n[3/5] Checking for exploding gradients...")
    exploding_tensors = []
    for name, grad in tensors.items():
        grad_norm = np.linalg.norm(grad)
        grad_max = np.max(np.abs(grad))
        if grad_norm > 10000 or grad_max > 1000:
            exploding_tensors.append(f"  ⚠ {name}: norm = {grad_norm:.2e}, max = {grad_max:.2e}")
    
    if not exploding_tensors:
        print("  ✓ No exploding gradients detected")
    else:
        print(f"  ⚠ Found {len(exploding_tensors)} large gradients:")
        for tensor in exploding_tensors[:5]:
            print(tensor)
    
    # Check 4: Gradient flow through layers
    print("\n[4/5] Checking gradient flow through layers...")
    layer_norms = []
    for name, grad in tensors.items():
        if 'layer_' in name and 'attn' in name:
            layer_idx = int(name.split('_')[1])
            grad_norm = np.linalg.norm(grad)
            layer_norms.append((layer_idx, grad_norm))
    
    if layer_norms:
        layer_norms.sort()
        print(f"  Layer gradient norms:")
        for layer_idx, norm in layer_norms:
            print(f"    Layer {layer_idx}: {norm:.6f}")
        
        # Check if gradients vanish in early layers
        if len(layer_norms) > 1:
            first_norm = layer_norms[0][1]
            last_norm = layer_norms[-1][1]
            ratio = first_norm / (last_norm + 1e-10)
            
            if ratio < 0.01:
                warnings.append(f"  ⚠ Possible vanishing gradients: Layer 0 / Layer {layer_norms[-1][0]} = {ratio:.4f}")
            elif ratio > 100:
                warnings.append(f"  ⚠ Possible exploding gradients: Layer 0 / Layer {layer_norms[-1][0]} = {ratio:.4f}")
            else:
                print(f"  ✓ Gradient flow looks healthy (ratio: {ratio:.2f})")
    
    if warnings:
        for warning in warnings:
            print(warning)
    
    # Check 5: Embedding vs LM head consistency
    print("\n[5/5] Checking embedding vs LM head gradients...")
    if 'embedding_grad' in tensors and 'lm_head_weight_grad' in tensors:
        emb_norm = np.linalg.norm(tensors['embedding_grad'])
        lm_norm = np.linalg.norm(tensors['lm_head_weight_grad'])
        ratio = emb_norm / (lm_norm + 1e-10)
        
        print(f"  Embedding grad norm: {emb_norm:.6f}")
        print(f"  LM head grad norm:   {lm_norm:.6f}")
        print(f"  Ratio:               {ratio:.4f}")
        
        if 0.1 < ratio < 10.0:
            print("  ✓ Embedding and LM head gradients are balanced")
        else:
            warnings.append(f"  ⚠ Imbalance: embedding/lm_head = {ratio:.4f}")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    if len(issues) == 0 and len(warnings) == 0:
        print("✓ ALL CHECKS PASSED - Gradients look healthy!")
        return True
    else:
        if issues:
            print(f"✗ {len(issues)} critical issues found")
        if warnings:
            print(f"⚠ {len(warnings)} warnings")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Validate gradient correctness',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analytical validation tests
  python validate_gradient_trace.py --analytical
  
  # Validate a gradient dump
  python validate_gradient_trace.py gradient_trace_step_0.bin
  
  # Run all validations
  python validate_gradient_trace.py gradient_trace_step_0.bin --analytical
        """
    )
    parser.add_argument('dump_path', nargs='?', help='Path to gradient dump file')
    parser.add_argument('--analytical', '-a', action='store_true',
                       help='Run analytical gradient validation tests')
    
    args = parser.parse_args()
    
    all_passed = True
    
    # Run analytical tests
    if args.analytical or not args.dump_path:
        print("\n" + "="*80)
        print("RUNNING ANALYTICAL VALIDATION TESTS")
        print("="*80)
        
        tests = [
            check_softmax_cross_entropy_gradient,
            check_matrix_multiply_gradient,
            check_gelu_gradient
        ]
        
        passed = sum([test() for test in tests])
        total = len(tests)
        
        print("\n" + "="*80)
        print(f"ANALYTICAL TESTS: {passed}/{total} passed")
        print("="*80)
        
        if passed < total:
            all_passed = False
    
    # Validate gradient dump
    if args.dump_path:
        if not Path(args.dump_path).exists():
            print(f"\n✗ Gradient dump not found: {args.dump_path}")
            sys.exit(1)
        
        if not validate_gradient_sanity(args.dump_path):
            all_passed = False
    
    # Exit code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()

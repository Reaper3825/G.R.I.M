#!/usr/bin/env python3
"""
trace_single_gradient.py - Single-step gradient trace for GRIM-text debugging

Instruments a single training step to capture detailed gradient information:
- Per-layer gradient norms
- Per-parameter gradient statistics (mean, std, min, max)
- Gradient flow from loss to embeddings
- Flash Attention gradient components (dQ, dK, dV)
- Activation statistics at each layer
"""

import sys
import json
import struct
import numpy as np
from pathlib import Path

# Adjust path to GRIM-text root
GRIM_TEXT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(GRIM_TEXT_ROOT))


def read_gradient_dump(dump_path: str):
    """Read binary gradient dump from training step"""
    with open(dump_path, 'rb') as f:
        # Header: magic (4 bytes), version (4 bytes), num_tensors (4 bytes)
        magic = struct.unpack('I', f.read(4))[0]
        if magic != 0x47524144:  # 'GRAD'
            raise ValueError(f"Invalid gradient dump magic: {magic:#x}")
        
        version = struct.unpack('I', f.read(4))[0]
        num_tensors = struct.unpack('I', f.read(4))[0]
        
        tensors = {}
        for _ in range(num_tensors):
            # Tensor header: name_len, name, dims, data
            name_len = struct.unpack('I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            
            ndim = struct.unpack('I', f.read(4))[0]
            shape = struct.unpack(f'{ndim}I', f.read(4 * ndim))
            
            dtype_code = struct.unpack('I', f.read(4))[0]
            if dtype_code == 0:
                dtype = np.float32
                itemsize = 4
            else:
                dtype = np.float16
                itemsize = 2
            
            size = int(np.prod(shape))
            data = np.frombuffer(f.read(size * itemsize), dtype=dtype).reshape(shape)
            tensors[name] = data
        
        return tensors


def analyze_gradient_tensor(name: str, grad: np.ndarray):
    """Compute statistics for a gradient tensor"""
    grad_flat = grad.flatten()
    
    # Filter out NaN/Inf
    valid_mask = np.isfinite(grad_flat)
    valid_grad = grad_flat[valid_mask]
    invalid_count = np.sum(~valid_mask)
    
    if len(valid_grad) == 0:
        return {
            'name': name,
            'shape': list(grad.shape),
            'size': grad.size,
            'invalid_count': invalid_count,
            'all_invalid': True
        }
    
    # Compute statistics
    grad_norm = np.linalg.norm(valid_grad).item()
    grad_mean = np.mean(valid_grad).item()
    grad_std = np.std(valid_grad).item()
    grad_min = np.min(valid_grad).item()
    grad_max = np.max(valid_grad).item()
    grad_abs_mean = np.mean(np.abs(valid_grad)).item()
    
    # Gradient sparsity (% near zero)
    near_zero_mask = np.abs(valid_grad) < 1e-7
    sparsity = np.mean(near_zero_mask).item() * 100
    
    return {
        'name': name,
        'shape': list(grad.shape),
        'size': grad.size,
        'norm': grad_norm,
        'mean': grad_mean,
        'std': grad_std,
        'min': grad_min,
        'max': grad_max,
        'abs_mean': grad_abs_mean,
        'sparsity_pct': sparsity,
        'invalid_count': invalid_count,
        'all_invalid': False
    }


def check_gradient_flow(tensors: dict):
    """Check if gradients flow properly through the network"""
    issues = []
    
    # Check embedding gradients
    if 'embedding_grad' in tensors:
        emb_grad = tensors['embedding_grad']
        emb_norm = np.linalg.norm(emb_grad)
        if emb_norm < 1e-8:
            issues.append('⚠️  Embedding gradients near zero - signal not reaching bottom')
    
    # Check LM head gradients
    if 'lm_head_weight_grad' in tensors:
        lm_grad = tensors['lm_head_weight_grad']
        lm_norm = np.linalg.norm(lm_grad)
        if lm_norm < 1e-8:
            issues.append('⚠️  LM head gradients near zero - loss not propagating')
    
    # Check layer-wise gradient decay
    layer_norms = []
    for i in range(100):  # Check up to 100 layers
        if f'layer_{i}_attn_weight_grad' in tensors:
            layer_norm = np.linalg.norm(tensors[f'layer_{i}_attn_weight_grad'])
            layer_norms.append((i, layer_norm))
    
    if len(layer_norms) > 1:
        # Check if gradients vanish in early layers
        first_norm = layer_norms[0][1]
        last_norm = layer_norms[-1][1]
        ratio = first_norm / (last_norm + 1e-10)
        if ratio < 0.01:
            issues.append(f'⚠️  Vanishing gradients: Layer 0 norm {first_norm:.2e}, Layer {layer_norms[-1][0]} norm {last_norm:.2e}')
    
    return issues


def trace_gradient_step(dump_path: str, output_path: str = None):
    """
    Trace a single gradient step from dump file
    
    Args:
        dump_path: Path to gradient dump binary file
        output_path: Optional path to save JSON report
    """
    print(f"Loading gradient dump from: {dump_path}")
    tensors = read_gradient_dump(dump_path)
    
    print(f"\nFound {len(tensors)} gradient tensors")
    
    # Analyze each tensor
    results = []
    total_norm = 0.0
    
    for name, grad in sorted(tensors.items()):
        stats = analyze_gradient_tensor(name, grad)
        results.append(stats)
        if not stats.get('all_invalid', False):
            total_norm += stats['norm'] ** 2
    
    total_norm = np.sqrt(total_norm)
    
    # Check gradient flow issues
    flow_issues = check_gradient_flow(tensors)
    
    # Print summary
    print("\n" + "="*80)
    print("GRADIENT TRACE SUMMARY")
    print("="*80)
    print(f"Total gradient norm: {total_norm:.6f}")
    print(f"Number of tensors: {len(results)}")
    
    if flow_issues:
        print("\n⚠️  GRADIENT FLOW ISSUES:")
        for issue in flow_issues:
            print(f"  {issue}")
    else:
        print("\n✓ Gradient flow looks healthy")
    
    # Print top gradients by norm
    print("\n" + "-"*80)
    print("TOP 10 GRADIENTS BY NORM:")
    print("-"*80)
    valid_results = [r for r in results if not r.get('all_invalid', False)]
    top_10 = sorted(valid_results, key=lambda x: x['norm'], reverse=True)[:10]
    
    for i, r in enumerate(top_10, 1):
        print(f"{i:2d}. {r['name']:40s} norm={r['norm']:12.6f} shape={r['shape']}")
    
    # Print detailed stats for each tensor
    print("\n" + "-"*80)
    print("PER-TENSOR STATISTICS:")
    print("-"*80)
    print(f"{'Name':<40} {'Norm':>12} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Sparsity%':>10}")
    print("-"*80)
    
    for r in results:
        if r.get('all_invalid', False):
            print(f"{r['name']:<40} {'ALL INVALID':^12}")
        else:
            print(f"{r['name']:<40} {r['norm']:12.6f} {r['mean']:12.6e} {r['std']:12.6e} "
                  f"{r['min']:12.6e} {r['max']:12.6e} {r['sparsity_pct']:10.2f}%")
    
    # Save to JSON if requested
    if output_path:
        report = {
            'total_norm': total_norm,
            'num_tensors': len(results),
            'flow_issues': flow_issues,
            'tensors': results
        }
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✓ Saved detailed report to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Trace single gradient step')
    parser.add_argument('dump_path', help='Path to gradient dump file')
    parser.add_argument('--output', '-o', help='Output JSON path for detailed report')
    
    args = parser.parse_args()
    
    if not Path(args.dump_path).exists():
        print(f"❌ Gradient dump not found: {args.dump_path}")
        print("\nTo generate a gradient dump, add this to train_gpu.cu:")
        print("  model->dumpGradients('gradient_dump.bin');")
        print("after the backward() call")
        sys.exit(1)
    
    trace_gradient_step(args.dump_path, args.output)


if __name__ == '__main__':
    main()

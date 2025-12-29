#!/usr/bin/env python3
"""
Quick visualization of the gradient litmus test results.
Generates a simple ASCII chart of gradient trends per layer.
"""

import sys
from pathlib import Path
from litmus_test_gradients import GradientLitmusTest
import statistics

def ascii_bar(value: float, max_value: float, width: int = 40) -> str:
    """Generate an ASCII bar chart."""
    if max_value == 0:
        return ""
    
    bar_length = int((value / max_value) * width)
    return "█" * bar_length

def main():
    log_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    if not log_file:
        log_dir = Path('resources/models/GRIM-text/training/logs')
        log_files = sorted(log_dir.glob('training_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print("Error: No training log files found")
            sys.exit(1)
        log_file = str(log_files[0])
    
    litmus = GradientLitmusTest(log_file)
    
    print(f"Analyzing: {log_file}\n")
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            litmus.parse_line(line)
    
    print("="*80)
    print("GRADIENT MAGNITUDE BY LAYER (Latest Values)")
    print("="*80)
    
    # Extract latest values per layer
    layers = sorted(litmus.metrics.keys())
    
    # ln2_gamma_grads RMS
    print("\n[1] ln2_gamma_grads RMS (LayerNorm2 gamma gradients)")
    print("-"*80)
    ln2_grads = []
    for layer in layers:
        if litmus.metrics[layer]['ln2_gamma_grads_rms']:
            value = litmus.metrics[layer]['ln2_gamma_grads_rms'][-1][1]
            ln2_grads.append((layer, value))
    
    max_grad = max(v for _, v in ln2_grads) if ln2_grads else 1.0
    for layer, value in ln2_grads:
        bar = ascii_bar(value, max_grad, 50)
        print(f"Layer {layer:2d}: {value:12.3e} {bar}")
    
    # grad_ffn_input RMS
    print("\n[2] grad_ffn_input RMS (FFN input gradients after LN2)")
    print("-"*80)
    ffn_grads = []
    for layer in layers:
        if litmus.metrics[layer]['grad_ffn_input_rms']:
            value = litmus.metrics[layer]['grad_ffn_input_rms'][-1][1]
            ffn_grads.append((layer, value))
    
    max_ffn = max(v for _, v in ffn_grads) if ffn_grads else 1.0
    for layer, value in ffn_grads:
        bar = ascii_bar(value, max_ffn, 50)
        print(f"Layer {layer:2d}: {value:12.3e} {bar}")
    
    # ln2_gamma range
    print("\n[3] ln2_gamma range (LayerNorm2 gamma weight spread)")
    print("-"*80)
    gamma_ranges = []
    for layer in layers:
        if litmus.metrics[layer]['ln2_gamma_range']:
            step, range_val, min_val, max_val = litmus.metrics[layer]['ln2_gamma_range'][-1]
            gamma_ranges.append((layer, range_val, min_val, max_val))
    
    for layer, range_val, min_val, max_val in gamma_ranges:
        print(f"Layer {layer:2d}: range={range_val:.6f}  [min={min_val:.4f}, max={max_val:.4f}]")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if ln2_grads:
        ln2_values = [v for _, v in ln2_grads]
        print(f"\nln2_gamma_grads RMS:")
        print(f"  Min:    {min(ln2_values):.3e} (Layer {ln2_grads[ln2_values.index(min(ln2_values))][0]})")
        print(f"  Max:    {max(ln2_values):.3e} (Layer {ln2_grads[ln2_values.index(max(ln2_values))][0]})")
        print(f"  Mean:   {statistics.mean(ln2_values):.3e}")
        print(f"  Median: {statistics.median(ln2_values):.3e}")
        print(f"  StdDev: {statistics.stdev(ln2_values):.3e}")
    
    if ffn_grads:
        ffn_values = [v for _, v in ffn_grads]
        print(f"\ngrad_ffn_input RMS:")
        print(f"  Min:    {min(ffn_values):.3e} (Layer {ffn_grads[ffn_values.index(min(ffn_values))][0]})")
        print(f"  Max:    {max(ffn_values):.3e} (Layer {ffn_grads[ffn_values.index(max(ffn_values))][0]})")
        print(f"  Mean:   {statistics.mean(ffn_values):.3e}")
        print(f"  Median: {statistics.median(ffn_values):.3e}")
        print(f"  StdDev: {statistics.stdev(ffn_values):.3e}")
    
    print("\n" + "="*80)
    print("GRADIENT HEALTH ASSESSMENT")
    print("="*80)
    
    # Check gradient ratio between layers
    if len(ln2_grads) >= 2:
        layer0_grad = ln2_grads[0][1]
        layer_last_grad = ln2_grads[-1][1]
        ratio = layer0_grad / (layer_last_grad + 1e-10)
        print(f"\nGradient scale ratio (Layer 0 / Layer 11): {ratio:.2e}")
        
        if ratio > 1e5:
            print("  ❌ SEVERE gradient explosion detected!")
        elif ratio > 1e3:
            print("  ⚠️  HIGH gradient scale imbalance")
        elif ratio > 10:
            print("  ⚠️  Moderate gradient scale variation")
        else:
            print("  ✓ Gradient scale is balanced")
    
    # Check for vanishing gradients
    if ffn_grads:
        vanishing_layers = [layer for layer, value in ffn_grads if value < 1e-6]
        if vanishing_layers:
            print(f"\n⚠️  Vanishing grad_ffn_input in layers: {vanishing_layers}")
    
    # Check absolute gradient magnitudes
    exploding_layers = [layer for layer, value in ln2_grads if value > 10.0]
    if exploding_layers:
        print(f"\n❌ Exploding ln2_gamma_grads in layers: {exploding_layers}")

if __name__ == '__main__':
    main()

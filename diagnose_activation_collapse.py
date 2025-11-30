"""
Diagnose Activation Collapse in Forward Pass

This script analyzes training logs to identify when and where
activations collapse to zero variance, causing gradient explosions
during backward pass.
"""

import re
import sys
from pathlib import Path

def analyze_layernorm_variance(log_file):
    """Analyze LayerNorm variance values to detect activation collapse"""
    
    print("=" * 70)
    print("ACTIVATION COLLAPSE DIAGNOSTIC")
    print("=" * 70)
    print()
    
    # Track variance by step and layer
    variance_by_step = {}
    current_step = 0
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Detect step number
            step_match = re.search(r'Step (\d+)', line)
            if step_match:
                current_step = int(step_match.group(1))
                if current_step not in variance_by_step:
                    variance_by_step[current_step] = []
            
            # Detect LayerNorm variance
            var_match = re.search(r'\[LayerNormBackward\].*var=([\d.e+-]+)', line)
            if var_match:
                variance = float(var_match.group(1))
                if current_step not in variance_by_step:
                    variance_by_step[current_step] = []
                variance_by_step[current_step].append(variance)
    
    print(f"📊 Analyzed {len(variance_by_step)} training steps\n")
    
    # Analyze each step
    collapse_steps = []
    healthy_steps = []
    
    for step in sorted(variance_by_step.keys()):
        variances = variance_by_step[step]
        if not variances:
            continue
        
        zero_var_count = sum(1 for v in variances if v < 1e-6)
        zero_var_pct = (zero_var_count / len(variances)) * 100
        
        min_var = min(variances)
        max_var = max(variances)
        avg_var = sum(variances) / len(variances)
        
        if zero_var_pct > 10:  # Lower threshold to catch partial collapse
            collapse_steps.append(step)
            print(f"❌ Step {step}: {zero_var_pct:.1f}% zero variance")
            print(f"   → {zero_var_count}/{len(variances)} LayerNorms collapsed")
            print(f"   → Variance range: [{min_var:.2e}, {max_var:.2e}]")
            print()
        else:
            healthy_steps.append(step)
            print(f"✅ Step {step}: {zero_var_pct:.1f}% zero variance")
            print(f"   → Variance range: [{min_var:.2e}, {max_var:.2e}]")
            print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Healthy steps: {len(healthy_steps)}")
    print(f"Collapsed steps: {len(collapse_steps)}")
    print()
    
    if collapse_steps:
        print("⚠️  ROOT CAUSE: ACTIVATION COLLAPSE")
        print()
        print("The forward pass is producing activations with ZERO VARIANCE.")
        print("This means all values in a tensor are identical (constant).")
        print()
        print("When LayerNorm backward sees zero-variance input:")
        print("  • std_inv = 1/sqrt(var + eps) = 1/sqrt(0.001) = 31.62")
        print("  • Gradients get multiplied by 31.62x")
        print("  • This cascades exponentially through layers")
        print()
        print("POSSIBLE CAUSES:")
        print("  1. Weights initialized to zero or very small values")
        print("  2. Activations clamped/clipped during forward pass")
        print("  3. Numerical underflow in forward computations")
        print("  4. Dead ReLU/GELU (all activations saturate to zero)")
        print("  5. Layer outputs collapse due to residual path dominance")
        print()
        print("RECOMMENDED FIX:")
        print("  • Add activation magnitude logging in forward pass")
        print("  • Check if GELU is producing all zeros")
        print("  • Verify LayerNorm forward is not collapsing")
        print("  • Investigate residual connection behavior")
    else:
        print("✅ No activation collapse detected")
        print("   The forward pass maintains healthy variance.")
    
    return collapse_steps

def main():
    if len(sys.argv) < 2:
        log_file = "test_fixed_gradients.txt"
    else:
        log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"📄 Analyzing: {log_file}\n")
    analyze_layernorm_variance(log_file)

if __name__ == "__main__":
    main()

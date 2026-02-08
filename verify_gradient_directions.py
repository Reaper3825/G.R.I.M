"""
Gradient Direction Verification - Parse Training Logs
======================================================

Analyzes gradient patterns from GRIM training logs to detect direction corruption.

Key indicators:
1. Gradient reversal: grad_logits[target] should be MORE negative than grad_logits[other]
2. Sign consistency: grad contributions should have consistent signs
3. Discrimination power: model should predict targets stronger than non-targets
"""

import re
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_gradient_diagnostics(log_path):
    """Extract gradient diagnostics from GRIM training log"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {
        'batches': [],
        'grad_patterns': defaultdict(list),
        'discrimination': []
    }
    
    # Extract [HIDDEN_STATE_EQUATION] sections
    pattern = r'\[HIDDEN_STATE_EQUATION\].*?GRAD_LOGITS\[277\]: at_277_targets=([-\d.]+).*?at_other_targets=([-\d.]+)'
    for match in re.finditer(pattern, content, re.DOTALL):
        at_277 = float(match.group(1))
        at_other = float(match.group(2))
        results['grad_patterns']['at_277_targets'].append(at_277)
        results['grad_patterns']['at_other_targets'].append(at_other)
    
    # Extract [FEEDBACK_LOOP_EQUATION] discrimination test
    pattern = r'\[FEEDBACK_LOOP_EQUATION\].*?When target=277: logit\[277\]_mean=([-\d.]+).*?When target≠277: logit\[277\]_mean=([-\d.]+).*?DELTA = ([-\d.]+)'
    for match in re.finditer(pattern, content, re.DOTALL):
        at_target = float(match.group(1))
        at_other = float(match.group(2))
        delta = float(match.group(3))
        results['discrimination'].append({
            'at_target': at_target,
            'at_other': at_other,
            'delta': delta
        })
    
    # Extract gradient component norms
    pattern = r'Gradient norms:.*?emb=([\d.e+-]+).*?attn=([\d.e+-]+).*?ffn=([\d.e+-]+)'
    for match in re.finditer(pattern, content):
        results['batches'].append({
            'emb': float(match.group(1)),
            'attn': float(match.group(2)),
            'ffn': float(match.group(3))
        })
    
    return results

def analyze_gradient_direction_corruption(results):
    """Detect signs of gradient direction corruption"""
    
    print("="*80)
    print("GRADIENT DIRECTION ANALYSIS")
    print("="*80)
    
    # 1. Gradient Reversal Detection
    print("\n1. Gradient Loss Signal Direction:")
    print("-" * 80)
    
    at_277 = results['grad_patterns']['at_277_targets']
    at_other = results['grad_patterns']['at_other_targets']
    
    if len(at_277) > 0 and len(at_other) > 0:
        # For correct learning: |grad_logits[target]| should be > |grad_logits[other]|
        # Since both are negative (p - 1), more negative = stronger gradient signal
        reversals = sum(1 for i in range(min(len(at_277), len(at_other))) 
                       if abs(at_277[i]) < abs(at_other[i]))
        
        print(f"   Batches analyzed: {min(len(at_277), len(at_other))}")
        print(f"   Reversals detected: {reversals} ({100*reversals/len(at_277):.1f}%)")
        
        # Show trend
        if len(at_277) >= 2:
            initial_ratio = abs(at_277[0]) / (abs(at_other[0]) + 1e-10)
            final_ratio = abs(at_277[-1]) / (abs(at_other[-1]) + 1e-10)
            print(f"   Batch 1 ratio: {initial_ratio:.4f} (target/other gradient strength)")
            print(f"   Final ratio:   {final_ratio:.4f}")
            
            if initial_ratio > 1.0 and final_ratio < 1.0:
                print("   ⚠️  GRADIENT REVERSAL: Signal flipped from correct to incorrect!")
            elif initial_ratio < 1.0:
                print("   ❌ BACKWARDS FROM START: Gradients point wrong direction!")
    
    # 2. Discrimination Test
    print("\n2. Model Prediction Discrimination:")
    print("-" * 80)
    
    if results['discrimination']:
        disc = results['discrimination'][-1]  # Latest batch
        print(f"   When target=277:  logit[277] = {disc['at_target']:.6f}")
        print(f"   When target≠277:  logit[277] = {disc['at_other']:.6f}")
        print(f"   Delta: {disc['delta']:.6f}")
        
        if disc['delta'] < 0:
            print("   ❌ BACKWARDS DISCRIMINATION: Model predicts 277 LESS when it's correct!")
        elif disc['delta'] > 0.1:
            print("   ✅ Correct discrimination (model predicts targets stronger)")
        else:
            print("   ⚠️  Weak/no discrimination (model hasn't learned pattern)")
    
    # 3. Gradient Component Consistency
    print("\n3. Gradient Component Patterns:")
    print("-" * 80)
    
    if results['batches']:
        emb_norms = [b['emb'] for b in results['batches']]
        attn_norms = [b['attn'] for b in results['batches']]
        ffn_norms = [b['ffn'] for b in results['batches']]
        
        print(f"   Embedding:  mean={np.mean(emb_norms):.6f} std={np.std(emb_norms):.6f}")
        print(f"   Attention:  mean={np.mean(attn_norms):.6f} std={np.std(attn_norms):.6f}")
        print(f"   FFN:        mean={np.mean(ffn_norms):.6f} std={np.std(ffn_norms):.6f}")
        
        # Check for anomalous growth
        if len(emb_norms) >= 10:
            early_avg = np.mean(emb_norms[:5])
            late_avg = np.mean(emb_norms[-5:])
            growth = (late_avg - early_avg) / (early_avg + 1e-10)
            
            if abs(growth) > 2.0:
                print(f"   ⚠️  Gradient magnitude explosion/collapse: {growth:+.1%} change")
    
    # 4. Overall Verdict
    print("\n" + "="*80)
    print("VERDICT:")
    print("="*80)
    
    issues = []
    
    # Check reversal
    if results['grad_patterns']['at_277_targets']:
        at_277 = results['grad_patterns']['at_277_targets']
        at_other = results['grad_patterns']['at_other_targets']
        reversals = sum(1 for i in range(min(len(at_277), len(at_other))) 
                       if abs(at_277[i]) < abs(at_other[i]))
        if reversals > len(at_277) * 0.5:
            issues.append("Gradient reversal (>50% batches)")
    
    # Check discrimination
    if results['discrimination']:
        if results['discrimination'][-1]['delta'] < 0:
            issues.append("Backwards discrimination")
    
    if not issues:
        print("✅ No clear direction corruption detected.")
        print("   Problem may be: magnitude scaling, effective learning rate too small,")
        print("   weight initialization, or data quality.")
    else:
        print("🔴 PROBABLE DIRECTION CORRUPTION DETECTED:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n   Root causes to investigate:")
        print("   1. Centering operations projecting out useful gradients")
        print("   2. Matmul transpose errors in backward pass")
        print("   3. RoPE backward applying wrong inverse rotation")
        print("   4. Attention backward gradient accumulation bug")

if __name__ == "__main__":
    import sys
    
    # Find most recent training log
    log_dir = Path("resources/models/GRIM-text/training/logs")
    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}")
        sys.exit(1)
    
    log_files = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        print(f"ERROR: No training logs found in {log_dir}")
        sys.exit(1)
    
    log_path = log_files[0]
    print(f"Analyzing: {log_path.name}")
    print()
    
    results = parse_gradient_diagnostics(log_path)
    analyze_gradient_direction_corruption(results)

#!/usr/bin/env python3
"""
Diagnose attention saturation by analyzing attention entropy during training.

Softmax saturation occurs when attention weights become near-one-hot,
causing gradients to collapse via the Jacobian: dS = P * (dP - sum(P*dP))
When one P ≈ 1, this becomes dS ≈ P * (dP - dP) ≈ 0.

Symptoms:
- Low attention entropy (close to 0 instead of log(seq_len))
- Attention gradients collapse while LM head stays stable
- Sudden collapse (threshold effect) rather than gradual

This script:
1. Analyzes training logs for gradient component patterns
2. Computes theoretical entropy bounds
3. Identifies the batch range where collapse occurs
"""

import re
import sys
import math
from pathlib import Path
from collections import defaultdict

def parse_grad_trace(log_path: Path) -> list[dict]:
    """Extract gradient component measurements from training log."""
    pattern = r'\[GradTrace\] COMPUTED.*COMPONENTS: total=([\d.]+) emb=([\d.]+) lm=([\d.]+) attn=([\d.]+) ffn=([\d.]+) rms=([\d.]+)'
    
    results = []
    batch = 0
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                batch += 1
                results.append({
                    'batch': batch,
                    'total': float(match.group(1)),
                    'emb': float(match.group(2)),
                    'lm': float(match.group(3)),
                    'attn': float(match.group(4)),
                    'ffn': float(match.group(5)),
                    'rms': float(match.group(6)),
                })
    return results

def detect_collapse(results: list[dict], threshold_ratio: float = 0.5) -> dict:
    """Detect when attention gradient collapses relative to initial value."""
    if len(results) < 10:
        return {'detected': False, 'reason': 'insufficient data'}
    
    # Use first 5 batches as baseline
    baseline_attn = sum(r['attn'] for r in results[:5]) / 5
    baseline_ffn = sum(r['ffn'] for r in results[:5]) / 5
    baseline_lm = sum(r['lm'] for r in results[:5]) / 5
    
    collapse_batch = None
    for r in results[5:]:
        ratio = r['attn'] / baseline_attn if baseline_attn > 0 else 0
        if ratio < threshold_ratio:
            collapse_batch = r['batch']
            break
    
    # Check if LM head stays stable during collapse
    if collapse_batch:
        collapse_idx = collapse_batch - 1
        post_collapse = results[collapse_idx:collapse_idx+10] if collapse_idx+10 < len(results) else results[collapse_idx:]
        avg_lm_post = sum(r['lm'] for r in post_collapse) / len(post_collapse)
        lm_stable = avg_lm_post > baseline_lm * 0.3  # LM head should remain >30% of baseline
        
        return {
            'detected': True,
            'collapse_batch': collapse_batch,
            'baseline_attn': baseline_attn,
            'baseline_ffn': baseline_ffn,
            'baseline_lm': baseline_lm,
            'collapsed_attn': results[collapse_idx]['attn'],
            'collapsed_ffn': results[collapse_idx]['ffn'],
            'lm_stable': lm_stable,
            'pattern': 'SOFTMAX_SATURATION' if lm_stable else 'GENERAL_COLLAPSE'
        }
    
    return {'detected': False, 'reason': 'no collapse found'}

def analyze_softmax_saturation(results: list[dict]) -> dict:
    """Analyze patterns consistent with softmax saturation."""
    analysis = {
        'theory': '''
Softmax Gradient Collapse Theory:
================================
Forward: P = softmax(QK^T / sqrt(d))
Backward: dS = P * (dP - sum(P*dP))

When P saturates to near-one-hot:
- Dominant position: P_i ≈ 1, others ≈ 0
- dp_sum = sum(P*dP) ≈ P_i * dP_i ≈ dP_i
- For dominant: dS_i = 1 * (dP_i - dP_i) ≈ 0
- For others: dS_j = ~0 * (dP_j - dP_i) ≈ 0

Result: ALL attention gradients collapse!
''',
        'symptoms': {
            'attn_collapse': False,
            'ffn_collapse': False,  # FFN uses same input, should collapse too
            'lm_stable': False,     # LM head gradient bypasses attention
            'sudden_drop': False,   # Saturation is threshold-based
        }
    }
    
    if len(results) < 20:
        return analysis
    
    # Check for sudden collapse pattern
    for i in range(5, len(results) - 5):
        pre_attn = sum(r['attn'] for r in results[i-5:i]) / 5
        post_attn = sum(r['attn'] for r in results[i:i+5]) / 5
        
        if pre_attn > 0 and (post_attn / pre_attn) < 0.4:  # 60% drop
            analysis['symptoms']['sudden_drop'] = True
            analysis['collapse_region'] = (results[i-5]['batch'], results[i+5]['batch'])
            
            # Check if LM head stayed stable during this period
            pre_lm = sum(r['lm'] for r in results[i-5:i]) / 5
            post_lm = sum(r['lm'] for r in results[i:i+5]) / 5
            if post_lm > pre_lm * 0.6:  # LM head within 40% of baseline
                analysis['symptoms']['lm_stable'] = True
            
            # Check FFN collapse
            pre_ffn = sum(r['ffn'] for r in results[i-5:i]) / 5
            post_ffn = sum(r['ffn'] for r in results[i:i+5]) / 5
            if pre_ffn > 0 and (post_ffn / pre_ffn) < 0.4:
                analysis['symptoms']['ffn_collapse'] = True
            
            analysis['symptoms']['attn_collapse'] = True
            break
    
    # Determine if pattern matches softmax saturation
    symptoms = analysis['symptoms']
    if symptoms['attn_collapse'] and symptoms['lm_stable'] and symptoms['sudden_drop']:
        analysis['diagnosis'] = 'LIKELY_SOFTMAX_SATURATION'
        analysis['recommendation'] = '''
Recommended Fixes for Softmax Saturation:
1. Increase SOFTMAX_TEMPERATURE (e.g., 2.0-4.0) to spread attention
2. Enable QK_NORMALIZATION to bound score magnitudes
3. Add attention entropy regularization to penalize one-hot attention
4. Use lower learning rate for attention weights specifically

Immediate test: Set SOFTMAX_TEMPERATURE = 2.0 in HyperParameters_GPU.hpp
'''
    else:
        analysis['diagnosis'] = 'UNCLEAR - needs further investigation'
    
    return analysis

def main():
    # Find the most recent training log
    log_dir = Path(r'D:\G.R.I.M\resources\models\GRIM-text\training\logs')
    log_files = sorted(log_dir.glob('training_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not log_files:
        print("No training logs found!")
        sys.exit(1)
    
    # Use specified log or most recent
    log_path = log_files[0]
    if len(sys.argv) > 1:
        specified = Path(sys.argv[1])
        if specified.exists():
            log_path = specified
    
    print(f"Analyzing: {log_path.name}")
    print("=" * 60)
    
    results = parse_grad_trace(log_path)
    print(f"Found {len(results)} gradient measurements")
    
    if len(results) < 10:
        print("Insufficient data for analysis")
        sys.exit(1)
    
    # Detect collapse
    collapse = detect_collapse(results)
    print(f"\nCollapse Detection:")
    print(f"  Detected: {collapse['detected']}")
    if collapse['detected']:
        print(f"  Collapse batch: {collapse['collapse_batch']}")
        print(f"  Baseline attn: {collapse['baseline_attn']:.4f}")
        print(f"  Collapsed attn: {collapse['collapsed_attn']:.4f}")
        print(f"  LM head stable: {collapse['lm_stable']}")
        print(f"  Pattern: {collapse['pattern']}")
    
    # Analyze saturation
    analysis = analyze_softmax_saturation(results)
    print(f"\nSaturation Analysis:")
    print(f"  Symptoms:")
    for k, v in analysis['symptoms'].items():
        print(f"    {k}: {v}")
    print(f"  Diagnosis: {analysis.get('diagnosis', 'N/A')}")
    
    if 'recommendation' in analysis:
        print(f"\n{analysis['recommendation']}")
    
    # Print gradient evolution table
    print("\nGradient Evolution (every 10th batch):")
    print(f"{'Batch':>6} {'LM':>8} {'Attn':>8} {'FFN':>8} {'RMS':>8} {'Attn/LM':>8}")
    print("-" * 56)
    for i, r in enumerate(results):
        if i % 10 == 0 or (collapse['detected'] and abs(r['batch'] - collapse['collapse_batch']) <= 5):
            attn_lm_ratio = r['attn'] / r['lm'] if r['lm'] > 0 else 0
            marker = " <-- COLLAPSE" if collapse['detected'] and r['batch'] == collapse['collapse_batch'] else ""
            print(f"{r['batch']:>6} {r['lm']:>8.3f} {r['attn']:>8.3f} {r['ffn']:>8.3f} {r['rms']:>8.4f} {attn_lm_ratio:>8.3f}{marker}")

if __name__ == '__main__':
    main()

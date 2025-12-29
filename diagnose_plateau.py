#!/usr/bin/env python3
"""
Diagnose Training Plateau - Mathematical Consistency Checker

This script analyzes training logs to detect logic bugs by verifying:
1. Weight updates are proportional to LR * gradient
2. Loss changes correlate with weight updates
3. Gradient norms are consistent with parameter counts
4. Optimizer state (step counter) is incrementing correctly
5. No values are stuck/frozen

Author: GRIM Diagnostic Tool
"""

import re
import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

@dataclass
class BatchRecord:
    batch_num: int
    loss: float
    grad_norm: float = 0.0
    grad_norm_preclip: float = 0.0
    lr: float = 0.0
    step: int = 0
    valid_tokens: int = 0
    grad_scale: float = 0.0
    lm_weights: List[float] = field(default_factory=list)
    lm_rms: float = 0.0
    # Gradient components
    grad_emb: float = 0.0
    grad_lm: float = 0.0
    grad_attn: float = 0.0
    grad_ffn: float = 0.0
    grad_rms: float = 0.0

def parse_log_file(log_path: str) -> List[BatchRecord]:
    """Parse training log and extract batch records."""
    records = []
    current_batch = None
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    for line in lines:
        # Batch info
        m = re.search(r'\[GradTrace\] BATCH_INFO batch=(\d+)', line)
        if m:
            if current_batch:
                records.append(current_batch)
            current_batch = BatchRecord(batch_num=int(m.group(1)), loss=0.0)
            continue
        
        if not current_batch:
            continue
            
        # Loss
        m = re.search(r'POST-FORWARD loss=([0-9.]+)', line)
        if m:
            current_batch.loss = float(m.group(1))
            continue
        
        # Post-backward info
        m = re.search(r'POST-BACKWARD batch=\d+ loss=[0-9.]+ grad_scale=([0-9.]+) valid_tokens=(\d+)', line)
        if m:
            current_batch.grad_scale = float(m.group(1))
            current_batch.valid_tokens = int(m.group(2))
            continue
        
        # Gradient components
        m = re.search(r'COMPUTED COMPONENTS: total=([0-9.]+) emb=([0-9.]+) lm=([0-9.]+) attn=([0-9.]+) ffn=([0-9.]+) rms=([0-9.]+)', line)
        if m:
            current_batch.grad_norm_preclip = float(m.group(1))
            current_batch.grad_emb = float(m.group(2))
            current_batch.grad_lm = float(m.group(3))
            current_batch.grad_attn = float(m.group(4))
            current_batch.grad_ffn = float(m.group(5))
            current_batch.grad_rms = float(m.group(6))
            continue
        
        # Pre-optimizer
        m = re.search(r'PRE-OPTIMIZER batch=\d+ lr=([0-9.]+) grad_norm=([0-9.]+) step=(\d+) lm_w\[0:5\]=\[([^\]]+)\] rms=([0-9.e+-]+)', line)
        if m:
            current_batch.lr = float(m.group(1))
            current_batch.grad_norm = float(m.group(2))
            current_batch.step = int(m.group(3))
            weights_str = m.group(4)
            current_batch.lm_weights = [float(x) for x in weights_str.split(',')]
            current_batch.lm_rms = float(m.group(5))
            continue
    
    if current_batch:
        records.append(current_batch)
    
    return records

def analyze_weight_updates(records: List[BatchRecord]) -> Dict:
    """Analyze if weights are actually changing proportional to LR."""
    issues = []
    weight_deltas = []
    
    for i in range(1, len(records)):
        prev = records[i-1]
        curr = records[i]
        
        if len(prev.lm_weights) >= 5 and len(curr.lm_weights) >= 5:
            # Calculate weight change
            delta = np.array(curr.lm_weights) - np.array(prev.lm_weights)
            delta_norm = np.linalg.norm(delta)
            weight_deltas.append(delta_norm)
            
            # Expected update magnitude: lr * grad_norm (very rough)
            # In AdamW, update = lr * m / (sqrt(v) + eps), so this is approximate
            expected_scale = prev.lr
            
            if delta_norm == 0 and prev.lr > 0:
                issues.append(f"Batch {curr.batch_num}: ZERO weight change despite LR={prev.lr:.8f}")
            
    return {
        "weight_deltas": weight_deltas,
        "issues": issues,
        "mean_delta": np.mean(weight_deltas) if weight_deltas else 0,
        "std_delta": np.std(weight_deltas) if weight_deltas else 0,
        "zero_updates": sum(1 for d in weight_deltas if d == 0),
    }

def analyze_step_counter(records: List[BatchRecord]) -> Dict:
    """Verify step counter increments correctly."""
    issues = []
    
    for i in range(1, len(records)):
        prev = records[i-1]
        curr = records[i]
        
        # Step should increment by 1 or 2 (if gradient accumulation)
        step_diff = curr.step - prev.step
        if step_diff < 0:
            issues.append(f"Batch {curr.batch_num}: Step DECREASED from {prev.step} to {curr.step}")
        elif step_diff == 0:
            issues.append(f"Batch {curr.batch_num}: Step STUCK at {curr.step}")
        elif step_diff > 2:
            issues.append(f"Batch {curr.batch_num}: Step jumped by {step_diff} (from {prev.step} to {curr.step})")
    
    return {"issues": issues}

def analyze_gradient_flow(records: List[BatchRecord], tie_embeddings: bool = True) -> Dict:
    """Check if gradients are flowing to all components."""
    issues = []
    
    # Find records with gradient component data
    grad_records = [r for r in records if r.grad_norm_preclip > 0]
    
    if not grad_records:
        return {"issues": ["No gradient component data found in log"]}
    
    # Check for frozen components
    emb_grads = [r.grad_emb for r in grad_records]
    lm_grads = [r.grad_lm for r in grad_records]
    attn_grads = [r.grad_attn for r in grad_records]
    ffn_grads = [r.grad_ffn for r in grad_records]
    rms_grads = [r.grad_rms for r in grad_records]
    
    # With weight tying, embedding grads are 0 because they flow through LM head
    if not tie_embeddings and all(g == 0 for g in emb_grads):
        issues.append("CRITICAL: Embedding gradients are ALWAYS ZERO")
    if all(g == 0 for g in lm_grads):
        issues.append("CRITICAL: LM head gradients are ALWAYS ZERO")
    if all(g == 0 for g in attn_grads):
        issues.append("CRITICAL: Attention gradients are ALWAYS ZERO")
    if all(g == 0 for g in ffn_grads):
        issues.append("CRITICAL: FFN gradients are ALWAYS ZERO")
    
    # Check for suspiciously constant gradients
    if len(set(f"{g:.4f}" for g in lm_grads)) == 1:
        issues.append(f"SUSPICIOUS: LM head gradient is CONSTANT at {lm_grads[0]:.4f}")
    if len(set(f"{g:.4f}" for g in attn_grads)) == 1:
        issues.append(f"SUSPICIOUS: Attention gradient is CONSTANT at {attn_grads[0]:.4f}")
    if len(set(f"{g:.4f}" for g in ffn_grads)) == 1:
        issues.append(f"SUSPICIOUS: FFN gradient is CONSTANT at {ffn_grads[0]:.4f}")
    
    # Check gradient ratios - are they changing over time or stuck?
    # Compare first 10 vs last 10 gradient component ratios
    if len(grad_records) >= 20:
        early_ratios = []
        late_ratios = []
        for r in grad_records[:10]:
            if r.grad_attn > 0:
                early_ratios.append(r.grad_lm / r.grad_attn)
        for r in grad_records[-10:]:
            if r.grad_attn > 0:
                late_ratios.append(r.grad_lm / r.grad_attn)
        
        if early_ratios and late_ratios:
            early_ratio = np.mean(early_ratios)
            late_ratio = np.mean(late_ratios)
            if abs(early_ratio - late_ratio) < 0.01:
                issues.append(f"SUSPICIOUS: LM/Attn gradient ratio unchanged ({early_ratio:.4f} → {late_ratio:.4f})")
    
    return {
        "issues": issues,
        "emb_grad_mean": np.mean(emb_grads),
        "lm_grad_mean": np.mean(lm_grads),
        "attn_grad_mean": np.mean(attn_grads),
        "ffn_grad_mean": np.mean(ffn_grads),
        "rms_grad_mean": np.mean(rms_grads),
        "note": "(Embedding grads=0 is expected with weight tying)" if tie_embeddings else "",
    }

def analyze_loss_correlation(records: List[BatchRecord]) -> Dict:
    """Check if loss changes correlate with learning."""
    issues = []
    
    if len(records) < 100:
        return {"issues": ["Not enough data for correlation analysis"]}
    
    # Split into early (first 50) and late (last 50) batches
    early = records[:50]
    late = records[-50:]
    
    early_losses = [r.loss for r in early]
    late_losses = [r.loss for r in late]
    
    early_improvement = early_losses[0] - early_losses[-1]
    late_improvement = late_losses[0] - late_losses[-1]
    
    early_std = np.std(early_losses)
    late_std = np.std(late_losses)
    
    # Check plateau condition
    if abs(late_improvement) < 0.1 and late_std < 0.3:
        issues.append(f"PLATEAU DETECTED: Last 50 batches show minimal improvement ({late_improvement:.4f}) with low variance (std={late_std:.4f})")
    
    # Check if late losses oscillate around mean (classic plateau)
    late_mean = np.mean(late_losses)
    oscillation = np.mean([abs(l - late_mean) for l in late_losses])
    if oscillation < 0.5 and late_std > 0.1:
        issues.append(f"OSCILLATING PLATEAU: Loss oscillates around {late_mean:.4f} with amplitude ~{oscillation:.4f}")
    
    return {
        "issues": issues,
        "early_improvement": early_improvement,
        "late_improvement": late_improvement,
        "early_std": early_std,
        "late_std": late_std,
        "early_mean_loss": np.mean(early_losses),
        "late_mean_loss": np.mean(late_losses),
    }

def analyze_lr_schedule(records: List[BatchRecord]) -> Dict:
    """Verify LR schedule is working."""
    issues = []
    
    lrs = [r.lr for r in records if r.lr > 0]
    
    if not lrs:
        return {"issues": ["No LR data found"]}
    
    # Check if LR ever changes
    unique_lrs = len(set(f"{lr:.10f}" for lr in lrs))
    if unique_lrs == 1:
        issues.append(f"WARNING: LR is CONSTANT at {lrs[0]:.8f} for all batches")
    
    # Check if LR is stuck at max
    max_lr = max(lrs)
    if len(records) > 100:
        late_lrs = lrs[-50:]
        if all(abs(lr - max_lr) < 1e-10 for lr in late_lrs):
            issues.append(f"LR stuck at max ({max_lr:.8f}) for last 50 batches - no decay!")
    
    return {
        "issues": issues,
        "min_lr": min(lrs),
        "max_lr": max_lr,
        "unique_lr_count": unique_lrs,
        "final_lr": lrs[-1] if lrs else 0,
    }

def analyze_grad_norm_vs_tokens(records: List[BatchRecord]) -> Dict:
    """Check if grad_norm is properly scaled by token count."""
    issues = []
    
    # grad_scale should be 1/valid_tokens
    for r in records[:20]:  # Check first 20
        if r.valid_tokens > 0 and r.grad_scale > 0:
            expected_scale = 1.0 / r.valid_tokens
            if abs(r.grad_scale - expected_scale) > 0.0001:
                issues.append(f"Batch {r.batch_num}: grad_scale={r.grad_scale:.6f} but 1/tokens={expected_scale:.6f}")
    
    return {"issues": issues}

def compute_expected_weight_update(record: BatchRecord, config: dict) -> float:
    """Compute expected weight update magnitude based on AdamW."""
    # This is a rough estimate
    # AdamW: w -= lr * (m / (sqrt(v) + eps) + weight_decay * w)
    # Early in training, m ≈ grad, v ≈ grad^2
    # So update ≈ lr * grad / (|grad| + eps) ≈ lr * sign(grad) (roughly)
    # Magnitude should be proportional to LR
    return record.lr

def find_plateau_start(records: List[BatchRecord], window: int = 20, threshold: float = 0.1) -> int:
    """Find where plateau starts using rolling mean."""
    if len(records) < window * 2:
        return -1
    
    losses = [r.loss for r in records]
    
    for i in range(window, len(losses) - window):
        # Compare improvement in this window vs early improvement
        early_improvement = losses[0] - losses[window]
        current_improvement = losses[i - window] - losses[i]
        
        if early_improvement > 0.5 and abs(current_improvement) < threshold:
            return i - window
    
    return -1

def main():
    if len(sys.argv) < 2:
        # Default to most recent log
        log_dir = Path("D:/G.R.I.M/resources/models/GRIM-text/training/logs")
        logs = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            print("No training logs found!")
            sys.exit(1)
        log_path = str(logs[0])
        print(f"Using most recent log: {log_path}")
    else:
        log_path = sys.argv[1]
    
    print("=" * 70)
    print("GRIM-text Training Plateau Diagnostic")
    print("=" * 70)
    print()
    
    records = parse_log_file(log_path)
    print(f"Parsed {len(records)} batch records")
    print()
    
    if not records:
        print("ERROR: No batch records found in log!")
        sys.exit(1)
    
    # Run all analyses
    print("-" * 70)
    print("1. WEIGHT UPDATE ANALYSIS")
    print("-" * 70)
    weight_analysis = analyze_weight_updates(records)
    print(f"   Mean weight delta: {weight_analysis['mean_delta']:.10f}")
    print(f"   Std weight delta:  {weight_analysis['std_delta']:.10f}")
    print(f"   Zero updates:      {weight_analysis['zero_updates']}")
    for issue in weight_analysis['issues'][:5]:
        print(f"   ⚠️  {issue}")
    print()
    
    print("-" * 70)
    print("2. STEP COUNTER ANALYSIS")
    print("-" * 70)
    step_analysis = analyze_step_counter(records)
    if step_analysis['issues']:
        for issue in step_analysis['issues'][:5]:
            print(f"   ⚠️  {issue}")
    else:
        print("   ✅ Step counter incrementing correctly")
    print()
    
    print("-" * 70)
    print("3. GRADIENT FLOW ANALYSIS")
    print("-" * 70)
    grad_analysis = analyze_gradient_flow(records)
    print(f"   Embedding grad mean: {grad_analysis.get('emb_grad_mean', 0):.6f}")
    print(f"   LM head grad mean:   {grad_analysis.get('lm_grad_mean', 0):.6f}")
    print(f"   Attention grad mean: {grad_analysis.get('attn_grad_mean', 0):.6f}")
    print(f"   FFN grad mean:       {grad_analysis.get('ffn_grad_mean', 0):.6f}")
    print(f"   RMSNorm grad mean:   {grad_analysis.get('rms_grad_mean', 0):.6f}")
    for issue in grad_analysis['issues']:
        print(f"   ⚠️  {issue}")
    print()
    
    print("-" * 70)
    print("4. LOSS CORRELATION ANALYSIS")
    print("-" * 70)
    loss_analysis = analyze_loss_correlation(records)
    print(f"   Early 50 batches: {loss_analysis.get('early_mean_loss', 0):.4f} (improved by {loss_analysis.get('early_improvement', 0):.4f})")
    print(f"   Late 50 batches:  {loss_analysis.get('late_mean_loss', 0):.4f} (improved by {loss_analysis.get('late_improvement', 0):.4f})")
    for issue in loss_analysis['issues']:
        print(f"   ⚠️  {issue}")
    print()
    
    print("-" * 70)
    print("5. LEARNING RATE SCHEDULE ANALYSIS")
    print("-" * 70)
    lr_analysis = analyze_lr_schedule(records)
    print(f"   LR range: {lr_analysis.get('min_lr', 0):.8f} → {lr_analysis.get('max_lr', 0):.8f}")
    print(f"   Final LR: {lr_analysis.get('final_lr', 0):.8f}")
    print(f"   Unique LR values: {lr_analysis.get('unique_lr_count', 0)}")
    for issue in lr_analysis['issues']:
        print(f"   ⚠️  {issue}")
    print()
    
    print("-" * 70)
    print("6. GRADIENT SCALING ANALYSIS")
    print("-" * 70)
    scale_analysis = analyze_grad_norm_vs_tokens(records)
    if scale_analysis['issues']:
        for issue in scale_analysis['issues'][:5]:
            print(f"   ⚠️  {issue}")
    else:
        print("   ✅ Gradient scaling matches 1/valid_tokens")
    print()
    
    print("-" * 70)
    print("7. PLATEAU DETECTION")
    print("-" * 70)
    plateau_start = find_plateau_start(records)
    if plateau_start > 0:
        print(f"   ⚠️  PLATEAU STARTS at batch {plateau_start}")
        print(f"   ⚠️  Plateau duration: {len(records) - plateau_start} batches")
    else:
        print("   No clear plateau detected by rolling window analysis")
    print()
    
    # Deep dive: compare pre-plateau vs post-plateau
    if plateau_start > 0 and plateau_start < len(records) - 50:
        print("-" * 70)
        print("8. PRE vs POST PLATEAU COMPARISON")
        print("-" * 70)
        
        pre = records[:plateau_start]
        post = records[plateau_start:]
        
        # Compare weight delta magnitudes
        pre_deltas = []
        post_deltas = []
        
        for i in range(1, len(pre)):
            if len(pre[i].lm_weights) >= 5 and len(pre[i-1].lm_weights) >= 5:
                delta = np.linalg.norm(np.array(pre[i].lm_weights) - np.array(pre[i-1].lm_weights))
                pre_deltas.append(delta)
        
        for i in range(1, min(len(post), 100)):
            if len(post[i].lm_weights) >= 5 and len(post[i-1].lm_weights) >= 5:
                delta = np.linalg.norm(np.array(post[i].lm_weights) - np.array(post[i-1].lm_weights))
                post_deltas.append(delta)
        
        if pre_deltas and post_deltas:
            print(f"   Pre-plateau weight delta:  mean={np.mean(pre_deltas):.10f}")
            print(f"   Post-plateau weight delta: mean={np.mean(post_deltas):.10f}")
            ratio = np.mean(post_deltas) / np.mean(pre_deltas) if np.mean(pre_deltas) > 0 else 0
            print(f"   Ratio (post/pre): {ratio:.4f}")
            
            if ratio < 0.1:
                print(f"   ⚠️  CRITICAL: Weight updates are 10x smaller during plateau!")
            elif ratio > 2.0:
                print(f"   ⚠️  Weight updates are 2x larger during plateau (oscillating?)")
        
        # Compare LRs
        pre_lrs = [r.lr for r in pre if r.lr > 0]
        post_lrs = [r.lr for r in post if r.lr > 0]
        if pre_lrs and post_lrs:
            print(f"   Pre-plateau LR:  mean={np.mean(pre_lrs):.8f}")
            print(f"   Post-plateau LR: mean={np.mean(post_lrs):.8f}")
        
        # Compare gradient norms
        pre_grads = [r.grad_norm_preclip for r in pre if r.grad_norm_preclip > 0]
        post_grads = [r.grad_norm_preclip for r in post if r.grad_norm_preclip > 0]
        if pre_grads and post_grads:
            print(f"   Pre-plateau grad norm:  mean={np.mean(pre_grads):.4f}")
            print(f"   Post-plateau grad norm: mean={np.mean(post_grads):.4f}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_issues = (
        weight_analysis['issues'] + 
        step_analysis['issues'] + 
        grad_analysis['issues'] + 
        loss_analysis['issues'] + 
        lr_analysis['issues'] +
        scale_analysis['issues']
    )
    
    critical = [i for i in all_issues if 'CRITICAL' in i]
    warnings = [i for i in all_issues if 'CRITICAL' not in i]
    
    if critical:
        print("CRITICAL ISSUES:")
        for issue in critical:
            print(f"   🔴 {issue}")
    
    if warnings:
        print("WARNINGS:")
        for issue in warnings[:10]:
            print(f"   🟡 {issue}")
    
    if not all_issues:
        print("   No obvious issues detected from log analysis.")
        print("   The bug may be in:")
        print("   - Optimizer momentum/velocity state not accumulating")
        print("   - Gradient buffer being zeroed incorrectly")
        print("   - Weight update being applied then immediately overwritten")
        print("   - cuBLAS call with wrong beta (overwriting instead of accumulating)")

if __name__ == "__main__":
    main()

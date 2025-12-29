#!/usr/bin/env python3
"""
GRIM Training Log Analyzer
Diagnoses training issues from log files.
"""

import re
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import statistics

@dataclass
class BatchInfo:
    batch_id: int
    loss: float
    seq_lens: List[int]
    grad_norm: float = 0.0
    grad_components: Dict[str, float] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: str = ""

@dataclass
class TrainingMetrics:
    initial_loss: float = 0.0
    min_loss: float = float('inf')
    max_loss: float = 0.0
    d_model: int = 0
    num_layers: int = 0
    vocab_size: int = 0
    total_batches: int = 0
    skipped_batches: int = 0
    loss_by_seq_len: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))
    grad_norms: List[float] = field(default_factory=list)
    embedding_grads_zero: int = 0
    total_grad_checks: int = 0


def parse_log(log_path: str) -> Tuple[TrainingMetrics, List[BatchInfo]]:
    """Parse training log and extract metrics."""
    metrics = TrainingMetrics()
    batches: List[BatchInfo] = []
    current_batch: Optional[BatchInfo] = None
    
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    
    for line in lines:
        # Model architecture
        if 'Model architecture:' in line:
            m = re.search(r'd_model=(\d+)', line)
            if m:
                metrics.d_model = int(m.group(1))
            m = re.search(r'num_layers=(\d+)', line)
            if m:
                metrics.num_layers = int(m.group(1))
        
        # Vocab size
        if 'Loaded' in line and 'tokens' in line:
            m = re.search(r'Loaded (\d+) tokens', line)
            if m:
                metrics.vocab_size = int(m.group(1))
        
        # Initial loss baseline
        if '[LossBaseline]' in line:
            m = re.search(r'Initial loss=([\d.]+)', line)
            if m:
                metrics.initial_loss = float(m.group(1))
        
        # Batch info
        if '[GradTrace] BATCH_INFO' in line:
            m = re.search(r'batch=(\d+).*lens=\[([\d,]+)\]', line)
            if m:
                batch_id = int(m.group(1))
                lens = [int(x) for x in m.group(2).split(',')]
                current_batch = BatchInfo(batch_id=batch_id, loss=0.0, seq_lens=lens)
        
        # Loss after forward
        if current_batch and '[GradTrace] POST-FORWARD loss=' in line:
            m = re.search(r'loss=([\d.]+)', line)
            if m:
                current_batch.loss = float(m.group(1))
                max_len = max(current_batch.seq_lens)
                metrics.loss_by_seq_len[max_len].append(current_batch.loss)
                
                if current_batch.loss < metrics.min_loss:
                    metrics.min_loss = current_batch.loss
                if current_batch.loss > metrics.max_loss:
                    metrics.max_loss = current_batch.loss
        
        # Gradient components
        if '[GradTrace] COMPUTED COMPONENTS:' in line:
            m = re.search(r'total=([\d.]+) emb=([\d.]+) lm=([\d.]+) attn=([\d.]+) ffn=([\d.]+) rms=([\d.]+)', line)
            if m and current_batch:
                current_batch.grad_norm = float(m.group(1))
                current_batch.grad_components = {
                    'total': float(m.group(1)),
                    'emb': float(m.group(2)),
                    'lm': float(m.group(3)),
                    'attn': float(m.group(4)),
                    'ffn': float(m.group(5)),
                    'rms': float(m.group(6)),
                }
                metrics.grad_norms.append(float(m.group(1)))
        
        # Skipped batches
        if '[LossGuard] SKIPPING' in line:
            if current_batch:
                current_batch.skipped = True
                current_batch.skip_reason = 'loss_spike'
                metrics.skipped_batches += 1
        
        # Embedding grad checks
        if '[GradDiag] AFTER_BACKWARD: emb_grads' in line:
            metrics.total_grad_checks += 1
            if 'sum_sq=0.0000e+00' in line:
                metrics.embedding_grads_zero += 1
        
        # Finalize batch
        if current_batch and ('[Batch' in line and 'skipped' in line or 
                              '[GradTrace] POST-GRADNORM' in line or
                              '[LossGuard] SKIPPING' in line):
            batches.append(current_batch)
            metrics.total_batches += 1
            current_batch = None
    
    return metrics, batches


def analyze_loss_vs_seq_len(metrics: TrainingMetrics) -> Dict:
    """Analyze loss correlation with sequence length."""
    analysis = {
        'critical_length': None,
        'loss_jump_detected': False,
        'short_seq_avg_loss': 0.0,
        'long_seq_avg_loss': 0.0,
        'threshold_candidates': [],
    }
    
    if not metrics.loss_by_seq_len:
        return analysis
    
    # Sort by sequence length
    sorted_lens = sorted(metrics.loss_by_seq_len.keys())
    
    # Find where loss jumps dramatically
    prev_avg = None
    for seq_len in sorted_lens:
        losses = metrics.loss_by_seq_len[seq_len]
        avg_loss = statistics.mean(losses)
        
        if prev_avg is not None:
            loss_increase = avg_loss - prev_avg
            if loss_increase > 20:  # Dramatic jump
                analysis['loss_jump_detected'] = True
                analysis['critical_length'] = seq_len
                analysis['threshold_candidates'].append({
                    'seq_len': seq_len,
                    'avg_loss': avg_loss,
                    'loss_increase': loss_increase,
                })
        prev_avg = avg_loss
    
    # Calculate averages for short vs long sequences
    short_losses = []
    long_losses = []
    threshold = metrics.d_model if metrics.d_model > 0 else 512
    
    for seq_len, losses in metrics.loss_by_seq_len.items():
        if seq_len < threshold:
            short_losses.extend(losses)
        else:
            long_losses.extend(losses)
    
    if short_losses:
        analysis['short_seq_avg_loss'] = statistics.mean(short_losses)
    if long_losses:
        analysis['long_seq_avg_loss'] = statistics.mean(long_losses)
    
    return analysis


def diagnose(metrics: TrainingMetrics, batches: List[BatchInfo]) -> List[str]:
    """Generate diagnostic messages."""
    issues = []
    
    # Check if training started
    if metrics.initial_loss == 0:
        issues.append("❌ No initial loss baseline found - training may not have started properly")
    else:
        expected_random = 10.5  # ln(vocab_size) for ~37k vocab
        if abs(metrics.initial_loss - expected_random) < 1.0:
            issues.append(f"✓ Initial loss {metrics.initial_loss:.4f} matches random baseline")
        else:
            issues.append(f"⚠ Initial loss {metrics.initial_loss:.4f} differs from expected ~{expected_random:.1f}")
    
    # Check embedding gradients (note: may be tied with LM head)
    if metrics.total_grad_checks > 0:
        zero_pct = 100 * metrics.embedding_grads_zero / metrics.total_grad_checks
        if zero_pct > 90:
            # Check if LM head has gradients - if so, weight tying explains zero emb grads
            lm_has_grads = any(b.grad_components.get('lm', 0) > 0.01 for b in batches[:50] if b.grad_components)
            if lm_has_grads:
                issues.append(f"✓ Embedding gradients zero but LM head has grads - weight tying in use (OK)")
            else:
                issues.append(f"❌ CRITICAL: Embedding gradients are zero {zero_pct:.0f}% of time - backprop is broken!")
        elif zero_pct > 50:
            issues.append(f"⚠ Embedding gradients are zero {zero_pct:.0f}% of time - check backprop")
        else:
            issues.append(f"✓ Embedding gradients look healthy ({100-zero_pct:.0f}% non-zero)")
    
    # Check loss vs sequence length
    loss_analysis = analyze_loss_vs_seq_len(metrics)
    
    if loss_analysis['loss_jump_detected']:
        issues.append(f"❌ CRITICAL: Loss explodes at seq_len >= {loss_analysis['critical_length']}")
        issues.append(f"   Short sequences avg loss: {loss_analysis['short_seq_avg_loss']:.2f}")
        issues.append(f"   Long sequences avg loss: {loss_analysis['long_seq_avg_loss']:.2f}")
        
        # Check if related to d_model boundary
        if metrics.d_model > 0:
            if loss_analysis['critical_length'] and loss_analysis['critical_length'] >= metrics.d_model:
                issues.append(f"   ⚠ Critical length {loss_analysis['critical_length']} >= d_model ({metrics.d_model})")
                issues.append(f"   HYPOTHESIS: Position embedding or ALiBi slope issue for pos >= d_model")
    
    # Check skipped batches
    if metrics.total_batches > 0:
        skip_pct = 100 * metrics.skipped_batches / metrics.total_batches
        if skip_pct > 30:
            issues.append(f"❌ {skip_pct:.0f}% of batches skipped due to loss spikes")
        elif skip_pct > 10:
            issues.append(f"⚠ {skip_pct:.0f}% of batches skipped due to loss spikes")
        else:
            issues.append(f"✓ Only {skip_pct:.1f}% of batches skipped")
    
    # Check gradient norms
    if metrics.grad_norms:
        avg_grad = statistics.mean(metrics.grad_norms)
        max_grad = max(metrics.grad_norms)
        if max_grad > 100:
            issues.append(f"❌ Gradient explosion detected: max grad norm = {max_grad:.2f}")
        elif avg_grad < 0.001:
            issues.append(f"❌ Gradient vanishing: avg grad norm = {avg_grad:.6f}")
        else:
            issues.append(f"✓ Gradient norms healthy: avg={avg_grad:.4f}, max={max_grad:.4f}")
    
    return issues


def print_report(metrics: TrainingMetrics, batches: List[BatchInfo], issues: List[str]):
    """Print formatted analysis report."""
    print("\n" + "="*60)
    print("  GRIM TRAINING LOG ANALYSIS")
    print("="*60)
    
    print("\n📊 MODEL CONFIGURATION")
    print(f"   d_model: {metrics.d_model}")
    print(f"   num_layers: {metrics.num_layers}")
    print(f"   vocab_size: {metrics.vocab_size}")
    
    print("\n📈 TRAINING METRICS")
    print(f"   Initial loss: {metrics.initial_loss:.4f}")
    print(f"   Min loss: {metrics.min_loss:.4f}")
    print(f"   Max loss: {metrics.max_loss:.4f}")
    print(f"   Total batches: {metrics.total_batches}")
    print(f"   Skipped batches: {metrics.skipped_batches}")
    
    print("\n📊 LOSS BY SEQUENCE LENGTH BUCKETS")
    buckets = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 1200), (1200, 2000)]
    for low, high in buckets:
        losses = []
        for seq_len, l in metrics.loss_by_seq_len.items():
            if low <= seq_len < high:
                losses.extend(l)
        if losses:
            avg = statistics.mean(losses)
            print(f"   {low:4d}-{high:4d}: avg_loss={avg:.4f} ({len(losses)} samples)")
    
    print("\n🔍 DIAGNOSIS")
    for issue in issues:
        print(f"   {issue}")
    
    print("\n" + "="*60)


def main():
    if len(sys.argv) < 2:
        # Default to most recent log
        log_dir = Path("d:/G.R.I.M/resources/models/GRIM-text/training/logs")
        logs = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if logs:
            log_path = str(logs[0])
            print(f"Using most recent log: {log_path}")
        else:
            print("Usage: python analyze_training_log.py <log_file>")
            sys.exit(1)
    else:
        log_path = sys.argv[1]
    
    print(f"Parsing {log_path}...")
    metrics, batches = parse_log(log_path)
    issues = diagnose(metrics, batches)
    print_report(metrics, batches, issues)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Attention Diagnostics Analyzer for GRIM-text Training

Parses attndiag.txt and detects:
1. -FLT_MAX anomalies in QK scores (broken causal masking or atomicMax bug)
2. Gradient explosion/vanishing patterns
3. Entropy collapse (attention becoming too peaked)
4. Layer-specific issues
5. Step-based anomaly detection

Can also correlate with training logs to find batch/sequence context.

Usage:
    python analyze_attention_diagnostics.py [path_to_attndiag.txt] [--training-log path]
    python analyze_attention_diagnostics.py --auto  # Auto-find latest files
"""

import re
import sys
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math

# Constants
FLT_MAX = 3.402823e+38
FLT_MAX_THRESHOLD = 1e30  # Anything below -1e30 is suspicious

@dataclass
class AttentionDiagEntry:
    """Single attention diagnostic entry"""
    step: int
    layer: int
    prob_min: float
    prob_max: float
    entropy: float
    qk_min: float
    qk_max: float
    grad_q: float
    grad_k: float
    grad_v: float
    
    @property
    def has_flt_max_qk(self) -> bool:
        """Check if QK contains -FLT_MAX (broken causal mask or atomicMax bug)"""
        return self.qk_min < -FLT_MAX_THRESHOLD or self.qk_max < -FLT_MAX_THRESHOLD
    
    @property
    def has_nan(self) -> bool:
        """Check for NaN values"""
        return any(math.isnan(x) for x in [
            self.prob_min, self.prob_max, self.entropy,
            self.qk_min, self.qk_max, self.grad_q, self.grad_k, self.grad_v
        ])
    
    @property
    def has_inf(self) -> bool:
        """Check for Inf values"""
        return any(math.isinf(x) for x in [
            self.prob_min, self.prob_max, self.entropy,
            self.qk_min, self.qk_max, self.grad_q, self.grad_k, self.grad_v
        ])


@dataclass 
class LayerStats:
    """Accumulated statistics per layer"""
    layer: int
    entries: List[AttentionDiagEntry] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.entries)
    
    @property
    def flt_max_count(self) -> int:
        return sum(1 for e in self.entries if e.has_flt_max_qk)
    
    @property
    def first_flt_max_step(self) -> Optional[int]:
        for e in self.entries:
            if e.has_flt_max_qk:
                return e.step
        return None
    
    @property
    def avg_entropy(self) -> float:
        valid = [e.entropy for e in self.entries if not e.has_nan and not e.has_inf]
        return sum(valid) / len(valid) if valid else 0.0
    
    @property
    def avg_qk_range(self) -> float:
        valid = [e.qk_max - e.qk_min for e in self.entries 
                 if not e.has_flt_max_qk and not e.has_nan]
        return sum(valid) / len(valid) if valid else 0.0
    
    @property
    def avg_grad_q(self) -> float:
        valid = [e.grad_q for e in self.entries if not e.has_nan]
        return sum(valid) / len(valid) if valid else 0.0
    
    @property
    def avg_grad_k(self) -> float:
        valid = [e.grad_k for e in self.entries if not e.has_nan]
        return sum(valid) / len(valid) if valid else 0.0
    
    @property
    def avg_grad_v(self) -> float:
        valid = [e.grad_v for e in self.entries if not e.has_nan]
        return sum(valid) / len(valid) if valid else 0.0


@dataclass
class BatchInfo:
    """Information about a training batch"""
    batch_num: int
    seq_ids: List[int]
    seq_lens: List[int]
    loss: Optional[float] = None
    
    @property
    def max_len(self) -> int:
        return max(self.seq_lens) if self.seq_lens else 0
    
    @property
    def min_len(self) -> int:
        return min(self.seq_lens) if self.seq_lens else 0


def parse_attndiag_line(line: str) -> Optional[AttentionDiagEntry]:
    """
    Parse a single attndiag line.
    Format: [AttnDiag] step=X layer=Y: probs=[min,max] entropy=E qk=[min,max] grads=[Q:x K:y V:z]
    """
    pattern = r'\[AttnDiag\] step=(\d+) layer=(\d+): probs=\[([\d.e+-]+),([\d.e+-]+)\] entropy=([\d.e+-]+) qk=\[([\d.e+-]+),([\d.e+-]+)\] grads=\[Q:([\d.e+-]+) K:([\d.e+-]+) V:([\d.e+-]+)\]'
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    try:
        return AttentionDiagEntry(
            step=int(match.group(1)),
            layer=int(match.group(2)),
            prob_min=float(match.group(3)),
            prob_max=float(match.group(4)),
            entropy=float(match.group(5)),
            qk_min=float(match.group(6)),
            qk_max=float(match.group(7)),
            grad_q=float(match.group(8)),
            grad_k=float(match.group(9)),
            grad_v=float(match.group(10))
        )
    except (ValueError, IndexError) as e:
        return None


def parse_attndiag_file(filepath: Path) -> List[AttentionDiagEntry]:
    """Parse entire attndiag file"""
    entries = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if '[AttnDiag]' in line:
                entry = parse_attndiag_line(line)
                if entry:
                    entries.append(entry)
    return entries


def parse_training_log(filepath: Path) -> Dict[int, BatchInfo]:
    """Parse training log to extract batch info"""
    batches = {}
    
    batch_pattern = r'\[GradTrace\] BATCH_INFO batch=(\d+) seqs=\[([\d,]+)\] lens=\[([\d,]+)\]'
    loss_pattern = r'\[GradTrace\] POST-FORWARD.*loss=([\d.]+)'
    
    current_batch = None
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            batch_match = re.search(batch_pattern, line)
            if batch_match:
                batch_num = int(batch_match.group(1))
                seq_ids = [int(x) for x in batch_match.group(2).split(',')]
                seq_lens = [int(x) for x in batch_match.group(3).split(',')]
                batches[batch_num] = BatchInfo(
                    batch_num=batch_num,
                    seq_ids=seq_ids,
                    seq_lens=seq_lens
                )
                current_batch = batch_num
            
            loss_match = re.search(loss_pattern, line)
            if loss_match and current_batch and current_batch in batches:
                batches[current_batch].loss = float(loss_match.group(1))
    
    return batches


def analyze_by_layer(entries: List[AttentionDiagEntry]) -> Dict[int, LayerStats]:
    """Group and analyze entries by layer"""
    layer_data: Dict[int, LayerStats] = {}
    
    for entry in entries:
        if entry.layer not in layer_data:
            layer_data[entry.layer] = LayerStats(layer=entry.layer)
        layer_data[entry.layer].entries.append(entry)
    
    return layer_data


def detect_anomalies(entries: List[AttentionDiagEntry]) -> Dict[str, List]:
    """Detect various anomalies in the diagnostic data"""
    anomalies = {
        'flt_max_qk': [],
        'nan_values': [],
        'inf_values': [],
        'entropy_collapse': [],
        'gradient_explosion': [],
        'gradient_vanishing': [],
    }
    
    prev_entropy = {}
    
    for entry in entries:
        if entry.has_flt_max_qk:
            anomalies['flt_max_qk'].append({
                'step': entry.step,
                'layer': entry.layer,
                'qk_min': entry.qk_min,
                'qk_max': entry.qk_max
            })
        
        if entry.has_nan:
            anomalies['nan_values'].append({'step': entry.step, 'layer': entry.layer})
        
        if entry.has_inf:
            anomalies['inf_values'].append({'step': entry.step, 'layer': entry.layer})
        
        layer_key = entry.layer
        if layer_key in prev_entropy and prev_entropy[layer_key] > 0:
            entropy_drop = (prev_entropy[layer_key] - entry.entropy) / prev_entropy[layer_key]
            if entropy_drop > 0.2:
                anomalies['entropy_collapse'].append({
                    'step': entry.step,
                    'layer': entry.layer,
                    'prev_entropy': prev_entropy[layer_key],
                    'curr_entropy': entry.entropy,
                    'drop_pct': entropy_drop * 100
                })
        prev_entropy[layer_key] = entry.entropy
        
        max_grad = max(entry.grad_q, entry.grad_k, entry.grad_v)
        if max_grad > 1.0:
            anomalies['gradient_explosion'].append({
                'step': entry.step, 'layer': entry.layer,
                'grad_q': entry.grad_q, 'grad_k': entry.grad_k, 'grad_v': entry.grad_v
            })
        
        if max_grad < 1e-8 and max_grad > 0:
            anomalies['gradient_vanishing'].append({
                'step': entry.step, 'layer': entry.layer,
                'grad_q': entry.grad_q, 'grad_k': entry.grad_k, 'grad_v': entry.grad_v
            })
    
    return anomalies


def analyze_flt_max_pattern(entries: List[AttentionDiagEntry]) -> Dict:
    """Deep analysis of -FLT_MAX pattern"""
    by_step = defaultdict(list)
    for entry in entries:
        by_step[entry.step].append(entry)
    
    first_occurrence = {}
    affected_layers = set()
    
    for entry in sorted(entries, key=lambda e: (e.step, e.layer)):
        if entry.has_flt_max_qk:
            affected_layers.add(entry.layer)
            if entry.layer not in first_occurrence:
                first_occurrence[entry.layer] = entry.step
    
    persistence = {}
    for layer in affected_layers:
        layer_entries = [e for e in entries if e.layer == layer]
        layer_entries.sort(key=lambda e: e.step)
        
        first_step = first_occurrence[layer]
        subsequent = [e for e in layer_entries if e.step > first_step]
        flt_max_after = sum(1 for e in subsequent if e.has_flt_max_qk)
        
        persistence[layer] = {
            'first_step': first_step,
            'subsequent_entries': len(subsequent),
            'flt_max_count': flt_max_after,
            'persistence_rate': flt_max_after / len(subsequent) if subsequent else 0
        }
    
    timeline = []
    steps = sorted(by_step.keys())
    for step in steps:
        step_entries = by_step[step]
        has_flt_max = any(e.has_flt_max_qk for e in step_entries)
        affected = [e.layer for e in step_entries if e.has_flt_max_qk]
        if has_flt_max or step <= min(first_occurrence.values(), default=0) + 5:
            timeline.append({
                'step': step,
                'has_flt_max': has_flt_max,
                'affected_layers': affected,
                'qk_ranges': {e.layer: (e.qk_min, e.qk_max) for e in step_entries}
            })
    
    return {
        'affected_layers': sorted(affected_layers),
        'first_occurrence': first_occurrence,
        'persistence': persistence,
        'timeline': timeline[:20]
    }


def correlate_with_training(attn_entries: List[AttentionDiagEntry], 
                            batches: Dict[int, BatchInfo],
                            anomalies: Dict) -> Dict:
    """Correlate attention anomalies with training batch info"""
    correlation = {
        'anomaly_batch_info': [],
        'sequence_length_at_anomaly': [],
        'pattern_detected': None
    }
    
    if anomalies['flt_max_qk']:
        anomaly_steps = set(a['step'] for a in anomalies['flt_max_qk'])
        
        for step in sorted(anomaly_steps)[:10]:
            batch_num = step + 1
            
            if batch_num in batches:
                batch = batches[batch_num]
                correlation['anomaly_batch_info'].append({
                    'step': step,
                    'batch': batch_num,
                    'seq_lens': batch.seq_lens,
                    'max_len': batch.max_len,
                    'loss': batch.loss
                })
                correlation['sequence_length_at_anomaly'].append(batch.max_len)
        
        if correlation['sequence_length_at_anomaly']:
            lens = correlation['sequence_length_at_anomaly']
            avg_len = sum(lens) / len(lens)
            
            all_lens = [b.max_len for b in batches.values()]
            overall_avg = sum(all_lens) / len(all_lens) if all_lens else 0
            
            if avg_len > overall_avg * 1.3:
                correlation['pattern_detected'] = (
                    f"Anomalies correlate with LONG sequences! "
                    f"Anomaly avg: {avg_len:.0f}, Overall avg: {overall_avg:.0f}"
                )
    
    return correlation


def print_summary(entries: List[AttentionDiagEntry], layer_stats: Dict[int, LayerStats], 
                  anomalies: Dict, flt_max_analysis: Dict):
    """Print comprehensive analysis summary"""
    
    print("\n" + "="*80)
    print("ATTENTION DIAGNOSTICS ANALYSIS")
    print("="*80)
    
    print(f"\n📊 OVERVIEW")
    print(f"   Total entries: {len(entries)}")
    print(f"   Steps covered: {min(e.step for e in entries)} - {max(e.step for e in entries)}")
    print(f"   Layers: {sorted(layer_stats.keys())}")
    
    print(f"\n🔴 -FLT_MAX ANOMALY ANALYSIS")
    if flt_max_analysis['affected_layers']:
        print(f"   CRITICAL: -FLT_MAX detected in QK scores!")
        print(f"   Affected layers: {flt_max_analysis['affected_layers']}")
        print(f"\n   First occurrence per layer:")
        for layer, step in sorted(flt_max_analysis['first_occurrence'].items()):
            pers = flt_max_analysis['persistence'][layer]
            print(f"      Layer {layer}: step {step} (persists in {pers['persistence_rate']*100:.1f}% of subsequent)")
        
        print(f"\n   ROOT CAUSE DIAGNOSIS:")
        sample = anomalies['flt_max_qk'][:3] if anomalies['flt_max_qk'] else []
        for a in sample:
            qk_min_normal = abs(a['qk_min']) < FLT_MAX_THRESHOLD
            qk_max_broken = abs(a['qk_max']) > FLT_MAX_THRESHOLD
            qk_min_str = f"{a['qk_min']:.2e}" if qk_min_normal else "-FLT_MAX"
            qk_max_str = f"{a['qk_max']:.2e}" if not qk_max_broken else "-FLT_MAX"
            print(f"      step={a['step']}, layer={a['layer']}: qk=[{qk_min_str}, {qk_max_str}]")
            
            if qk_min_normal and qk_max_broken:
                print(f"         ↳ qk_min normal but qk_max=-FLT_MAX → atomicMax bug (int compare on negative float)")
            elif not qk_min_normal and qk_max_broken:
                print(f"         ↳ BOTH qk_min and qk_max=-FLT_MAX → All QK scores masked (causal mask bug)")
    else:
        print(f"   ✅ No -FLT_MAX anomalies detected")
    
    print(f"\n📈 PER-LAYER STATISTICS")
    print(f"   {'Layer':<6} {'Entries':<8} {'FLT_MAX':<10} {'Avg Entropy':<12} {'Avg QK Range':<12} {'Grads (Q/K/V)'}")
    print(f"   {'-'*6} {'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*24}")
    for layer in sorted(layer_stats.keys()):
        stats = layer_stats[layer]
        flt_max_pct = (stats.flt_max_count / stats.count * 100) if stats.count > 0 else 0
        flt_str = f"{stats.flt_max_count} ({flt_max_pct:>3.0f}%)"
        print(f"   {layer:<6} {stats.count:<8} {flt_str:<10} "
              f"{stats.avg_entropy:<12.2f} {stats.avg_qk_range:<12.4f} "
              f"{stats.avg_grad_q:.1e}/{stats.avg_grad_k:.1e}/{stats.avg_grad_v:.1e}")
    
    print(f"\n⚠️  OTHER ANOMALIES")
    for anomaly_type, items in anomalies.items():
        if anomaly_type == 'flt_max_qk':
            continue
        count = len(items)
        if count > 0:
            print(f"   {anomaly_type}: {count} occurrences")
        else:
            print(f"   {anomaly_type}: ✅ None")
    
    print(f"\n🔬 GRADIENT FLOW BY LAYER")
    grad_q_by_layer = [(l, layer_stats[l].avg_grad_q) for l in sorted(layer_stats.keys(), reverse=True)]
    print(f"   Layer gradient progression (Q): ", end="")
    print(" → ".join(f"L{l}:{g:.1e}" for l, g in grad_q_by_layer))


def print_training_correlation(correlation: Dict, batches: Dict[int, BatchInfo]):
    """Print training log correlation analysis"""
    if not batches:
        return
    
    print(f"\n📋 TRAINING LOG CORRELATION")
    
    all_lens = [b.max_len for b in batches.values()]
    print(f"   Total batches: {len(batches)}")
    print(f"   Sequence length range: {min(all_lens)} - {max(all_lens)}")
    print(f"   Average sequence length: {sum(all_lens)/len(all_lens):.0f}")
    
    if correlation['anomaly_batch_info']:
        print(f"\n   Batches at anomaly steps:")
        for info in correlation['anomaly_batch_info'][:5]:
            loss_str = f"loss={info['loss']:.4f}" if info['loss'] else ""
            print(f"      step={info['step']}, batch={info['batch']}: "
                  f"lens={info['seq_lens']} (max={info['max_len']}) {loss_str}")
        
        if correlation['pattern_detected']:
            print(f"\n   🔍 PATTERN: {correlation['pattern_detected']}")


def print_recommendations(entries: List[AttentionDiagEntry], layer_stats: Dict[int, LayerStats],
                          anomalies: Dict, flt_max_analysis: Dict):
    """Print actionable recommendations"""
    
    print(f"\n💡 RECOMMENDATIONS")
    recommendations = []
    
    if flt_max_analysis['affected_layers']:
        sample = anomalies['flt_max_qk'][:1] if anomalies['flt_max_qk'] else []
        if sample:
            qk_min_normal = abs(sample[0]['qk_min']) < FLT_MAX_THRESHOLD
            qk_max_broken = abs(sample[0]['qk_max']) > FLT_MAX_THRESHOLD
            if qk_min_normal and qk_max_broken:
                recommendations.append(
                    "FIX REQUIRED: atomicMax(reinterpret_cast<int*>) is broken for negative floats! "
                    "IEEE 754 floats don't compare correctly as two's complement ints. "
                    "Solution: Use CAS-loop atomicMaxFloat/atomicMinFloat instead."
                )
    
    avg_entropies = [layer_stats[l].avg_entropy for l in layer_stats if layer_stats[l].count > 0]
    if avg_entropies:
        min_entropy = min(avg_entropies)
        if min_entropy < 50:
            recommendations.append(f"Low entropy ({min_entropy:.1f}) - attention too peaked")
        elif min_entropy > 90:
            recommendations.append(f"High entropy ({min_entropy:.1f}) - attention too flat, not focusing")
    
    grad_qs = [(l, layer_stats[l].avg_grad_q) for l in sorted(layer_stats.keys())]
    if grad_qs and grad_qs[0][1] > 0 and grad_qs[-1][1] > 0:
        ratio = grad_qs[0][1] / grad_qs[-1][1]
        if ratio > 100:
            recommendations.append(f"Gradient vanishing: L0 grad {ratio:.0f}x larger than L{len(grad_qs)-1}")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        print(f"   No specific recommendations - training looks healthy!")


def export_to_json(entries: List[AttentionDiagEntry], layer_stats: Dict[int, LayerStats],
                   anomalies: Dict, flt_max_analysis: Dict, output_path: Path):
    """Export analysis to JSON"""
    data = {
        'summary': {
            'total_entries': len(entries),
            'step_range': [min(e.step for e in entries), max(e.step for e in entries)],
            'layers': sorted(layer_stats.keys())
        },
        'flt_max_analysis': {
            'affected_layers': flt_max_analysis['affected_layers'],
            'first_occurrence': flt_max_analysis['first_occurrence'],
            'persistence': flt_max_analysis['persistence']
        },
        'layer_stats': {
            layer: {
                'count': stats.count,
                'flt_max_count': stats.flt_max_count,
                'avg_entropy': stats.avg_entropy,
                'avg_qk_range': stats.avg_qk_range,
                'avg_grad_q': stats.avg_grad_q,
                'avg_grad_k': stats.avg_grad_k,
                'avg_grad_v': stats.avg_grad_v
            }
            for layer, stats in layer_stats.items()
        },
        'anomaly_counts': {k: len(v) for k, v in anomalies.items()},
        'anomaly_samples': {k: v[:10] for k, v in anomalies.items()}
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"\n📁 Exported detailed analysis to: {output_path}")


def find_latest_files(base_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the latest attndiag.txt and training log"""
    logs_dir = base_dir / "training" / "logs"
    
    attndiag = logs_dir / "attndiag.txt"
    if not attndiag.exists():
        attndiag = None
    
    training_logs = list(logs_dir.glob("training_*.log"))
    training_log = max(training_logs, key=lambda p: p.stat().st_mtime) if training_logs else None
    
    return attndiag, training_log


def main():
    parser = argparse.ArgumentParser(description='Analyze GRIM-text attention diagnostics')
    parser.add_argument('attndiag', nargs='?', help='Path to attndiag.txt')
    parser.add_argument('--training-log', '-t', help='Path to training log for correlation')
    parser.add_argument('--auto', '-a', action='store_true', help='Auto-find latest files')
    
    args = parser.parse_args()
    
    # Determine file paths
    base_dir = Path("d:/G.R.I.M/resources/models/GRIM-text")
    
    if args.auto:
        attndiag_path, training_log_path = find_latest_files(base_dir)
        if not attndiag_path:
            print("Error: Could not find attndiag.txt")
            sys.exit(1)
    else:
        attndiag_path = Path(args.attndiag) if args.attndiag else base_dir / "training/logs/attndiag.txt"
        training_log_path = Path(args.training_log) if args.training_log else None
    
    if not attndiag_path.exists():
        print(f"Error: File not found: {attndiag_path}")
        sys.exit(1)
    
    print(f"Analyzing: {attndiag_path}")
    if training_log_path:
        print(f"Training log: {training_log_path}")
    
    # Parse entries
    entries = parse_attndiag_file(attndiag_path)
    if not entries:
        print("No [AttnDiag] entries found!")
        sys.exit(1)
    
    # Parse training log if available
    batches = {}
    if training_log_path and training_log_path.exists():
        batches = parse_training_log(training_log_path)
    
    # Analyze
    layer_stats = analyze_by_layer(entries)
    anomalies = detect_anomalies(entries)
    flt_max_analysis = analyze_flt_max_pattern(entries)
    
    # Print summary
    print_summary(entries, layer_stats, anomalies, flt_max_analysis)
    
    # Training log correlation
    if batches:
        correlation = correlate_with_training(entries, batches, anomalies)
        print_training_correlation(correlation, batches)
    
    # Recommendations
    print_recommendations(entries, layer_stats, anomalies, flt_max_analysis)
    
    # Export to JSON
    json_path = attndiag_path.with_suffix('.analysis.json')
    export_to_json(entries, layer_stats, anomalies, flt_max_analysis, json_path)
    
    # Exit code
    if flt_max_analysis['affected_layers']:
        print(f"\n❌ CRITICAL ISSUES DETECTED")
        return 1
    else:
        print(f"\n✅ No critical issues detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())

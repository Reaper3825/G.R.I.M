#!/usr/bin/env python3
"""
GRIM Training Log Analyzer
Automatically finds the latest training log and extracts diagnostic patterns.
Provides actionable insights for training health.
"""

import os
import re
import sys
import glob
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics
import math

# ANSI colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def parse_kv_pairs(line: str) -> Dict[str, float]:
    """Parse key=value pairs from a log line using string splitting."""
    result = {}
    # Remove timestamp and prefix if present
    if ']' in line:
        # Find the last ] that ends the prefix
        idx = line.rfind(']')
        if idx != -1:
            line = line[idx+1:]
    
    # Split by whitespace
    parts = line.strip().split()
    for part in parts:
        if '=' in part:
            key, _, value = part.partition('=')
            # Handle array notation like row_max[0]
            key = key.strip()
            value = value.strip()
            try:
                result[key] = float(value)
            except ValueError:
                pass  # Skip non-numeric values
    return result

def parse_flash_bwd_line(line: str) -> Optional[Dict]:
    """Parse [FlashBwd] line for attention stats."""
    if '[FlashBwd]' not in line:
        return None
    return parse_kv_pairs(line)

def parse_flash_kv_line(line: str) -> Optional[Dict]:
    """Parse [FlashBwd-KV0] line for attention weight stats."""
    if '[FlashBwd-KV' not in line:
        return None
    return parse_kv_pairs(line)

def parse_flash_host_line(line: str) -> Optional[Dict]:
    """Parse [FlashAttn Host] line for sequence info."""
    if '[FlashAttn Host]' not in line:
        return None
    return parse_kv_pairs(line)

def find_latest_log(log_dir: str) -> Optional[str]:
    """Find the most recent training log file."""
    pattern = os.path.join(log_dir, "training_*.log")
    logs = glob.glob(pattern)
    if not logs:
        return None
    # Sort by modification time, newest first
    return max(logs, key=os.path.getmtime)

def parse_timestamp(line: str) -> Optional[datetime]:
    """Extract timestamp from log line."""
    match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S')
    return None

def analyze_log(log_path: str) -> Dict:
    """Parse training log and extract key metrics."""
    
    results = {
        'config': {},
        'loss_history': [],
        'lr_history': [],
        'gradient_stats': [],
        'validation': [],
        'warnings': [],
        'errors': [],
        'dynamic_lr_events': defaultdict(int),
        'grad_checks': defaultdict(list),
        'layer_gradients': defaultdict(list),
        'batch_times': [],
        'total_batches': 0,
        'current_batch': 0,
        'epoch': 1,
        'start_time': None,
        'last_time': None,
        'attention_stats': defaultdict(list),
        'ffn_stats': defaultdict(list),
        'layernorm_stats': defaultdict(list),
        'grad_norm_history': [],
        'clipping_events': 0,
        'soft_restarts': 0,
        'checkpoint_saves': [],
        'gpu_oom_events': 0,
        'batch_size_history': [],
        'sequence_length_stats': [],
        'longest_sequence_length': 0,
        'longest_sequence_batch': None,
        'longest_sequence_batch_size': None,
        'gradguard_skips': 0,
        'gradguard_max_grad': 0.0,
        # Attention collapse detection
        'flash_bwd_stats': [],       # (batch, row_max, row_sum, dQ_sum)
        'flash_p00_history': [],     # P00 values (attention weight at position 0)
        'flash_score00_history': [], # score00 values (pre-softmax logits)
        'flash_dq_history': [],      # dQ gradient magnitude over time
        'flash_seq_lens': [],        # sequence lengths seen
    }
    
    # Patterns to match
    patterns = {
        'step': re.compile(r'\[Step (\d+)\] loss=([\d.]+) lr=([\d.]+)'),
        'batch': re.compile(r'\[Batch (\d+)/(\d+)\] size=(\d+) max_len=(\d+)'),
        'grad_norm': re.compile(r'\[GradNorm\] value=([\d.]+) per_token=([\d.]+) mode=(\w+)'),
        'grad_check': re.compile(r'\[GradCheck\] (\w+): rms=([\d.e+-]+), max_abs=([\d.e+-]+)'),
        'dynamic_lr': re.compile(r'\[DynamicLR\] .*reason=(\w+)'),
        'forced_floor': re.compile(r'\[DynamicLR\] forced_floor lr=([\d.]+) reason=(\w+)'),
        'validation': re.compile(r'\[ValMicro\] step=(\d+).*loss=([\d.]+) ppl=([\d.]+)'),
        'config_lr': re.compile(r'Learning rate: ([\d.]+)'),
        'config_epochs': re.compile(r'Epochs: (\d+)'),
        'config_batch': re.compile(r'Batch size: (\d+)'),
        'config_vocab': re.compile(r'vocab_size=(\d+)'),
        'config_model': re.compile(r'd_model=(\d+), d_ff=(\d+)'),
        'warmup': re.compile(r'Warmup steps: (\d+)'),
        'train_seqs': re.compile(r'Train sequences: (\d+)'),
        'grad_spike_detail': re.compile(r'grad=([\d.]+).*token_adj=([\d.]+)'),
        'diag': re.compile(r'\[Diag\] batch=(\d+).*loss=([\d.]+) preclip_grad=([\d.e+-]+) preclip_norm=([\d.e+-]+)'),
        'soft_restart': re.compile(r'\[SoftRestart\]'),
        'checkpoint': re.compile(r'\[Checkpoint\] Saved.*step=(\d+)'),
        'gpu_oom': re.compile(r'out of memory|CUDA out of memory|OOM', re.IGNORECASE),
        'clipping': re.compile(r'cap_abs=([\d.]+) cap_norm=([\d.]+)'),
        'nan_inf': re.compile(r'\b(nan|inf|-inf)\b', re.IGNORECASE),
        'error': re.compile(r'\[ERROR\]|\[FATAL\]|Exception|Error:', re.IGNORECASE),
        'attention_qkv': re.compile(r'attn_qkv_weight_grads: rms=([\d.e+-]+)'),
        'ffn_grads': re.compile(r'ffn_w1_grads: rms=([\d.e+-]+)'),
        'layernorm_grads': re.compile(r'(ln\d+|rms\d+)_gamma_grads: rms=([\d.e+-]+)'),
        'gradguard_skip': re.compile(r'\[GradGuard\] skip optimizer step.*preclip_norm=([\d.e+-]+).*per_token=([\d.e+-]+)'),
    }
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Get timestamp
            ts = parse_timestamp(line)
            if ts:
                if results['start_time'] is None:
                    results['start_time'] = ts
                results['last_time'] = ts
            
            # Check for NaN/Inf
            if patterns['nan_inf'].search(line):
                results['warnings'].append(f"NaN/Inf detected: {line[:100]}")
            
            # Check for errors
            if patterns['error'].search(line):
                results['errors'].append(line[:150])
            
            # Step/loss/lr
            match = patterns['step'].search(line)
            if match:
                step, loss, lr = int(match.group(1)), float(match.group(2)), float(match.group(3))
                results['loss_history'].append((step, loss))
                results['lr_history'].append((step, lr))
                results['current_batch'] = step
                continue
            
            # Batch info
            match = patterns['batch'].search(line)
            if match:
                batch_idx = int(match.group(1))
                results['current_batch'] = batch_idx
                results['total_batches'] = int(match.group(2))

                batch_size = int(match.group(3))
                max_len = int(match.group(4))
                results['batch_size_history'].append(batch_size)
                results['sequence_length_stats'].append(max_len)

                if max_len > results['longest_sequence_length']:
                    results['longest_sequence_length'] = max_len
                    results['longest_sequence_batch'] = batch_idx
                    results['longest_sequence_batch_size'] = batch_size
                continue
            
            # Gradient check
            match = patterns['grad_check'].search(line)
            if match:
                name, rms, max_abs = match.group(1), float(match.group(2)), float(match.group(3))
                results['grad_checks'][name].append((rms, max_abs))
                # Track layer gradients
                layer_match = re.match(r'layer(\d+)_', name)
                if layer_match:
                    layer = int(layer_match.group(1))
                    results['layer_gradients'][layer].append(rms)
                # Track attention stats
                if 'attn' in name or 'qkv' in name:
                    results['attention_stats'][name].append(rms)
                # Track FFN stats
                if 'ffn' in name or 'w1' in name or 'w2' in name:
                    results['ffn_stats'][name].append(rms)
                # Track LayerNorm stats
                if 'ln' in name or 'rms' in name or 'gamma' in name:
                    results['layernorm_stats'][name].append(rms)
                continue
            
            # Dynamic LR events
            match = patterns['forced_floor'].search(line)
            if match:
                reason = match.group(2)
                results['dynamic_lr_events'][f'forced_floor_{reason}'] += 1
                continue
            
            match = patterns['dynamic_lr'].search(line)
            if match:
                reason = match.group(1)
                results['dynamic_lr_events'][reason] += 1
                continue
            
            # Validation
            match = patterns['validation'].search(line)
            if match:
                step, loss, ppl = int(match.group(1)), float(match.group(2)), float(match.group(3))
                results['validation'].append({'step': step, 'loss': loss, 'ppl': ppl})
                continue
            
            # Config extraction
            for key in ['config_lr', 'config_epochs', 'config_batch', 'config_vocab', 'warmup', 'train_seqs']:
                match = patterns[key].search(line)
                if match:
                    results['config'][key] = match.groups()
                    break
            
            match = patterns['config_model'].search(line)
            if match:
                results['config']['d_model'] = int(match.group(1))
                results['config']['d_ff'] = int(match.group(2))
            
            # Gradient norm
            match = patterns['grad_norm'].search(line)
            if match:
                grad_val, per_token = float(match.group(1)), float(match.group(2))
                results['grad_norm_history'].append((results['current_batch'], grad_val, per_token))
                continue
            
            # Pre-clip gradient diagnostics
            match = patterns['diag'].search(line)
            if match:
                batch = int(match.group(1))
                loss = float(match.group(2))
                preclip_grad = float(match.group(3))
                preclip_norm = float(match.group(4)) if len(match.groups()) >= 4 else 0
                results['gradient_stats'].append({
                    'batch': batch,
                    'loss': loss,
                    'preclip_grad': preclip_grad,
                    'preclip_norm': preclip_norm
                })
                continue
            
            # Soft restart detection
            if patterns['soft_restart'].search(line):
                results['soft_restarts'] += 1
                continue
            
            # Checkpoint saves
            match = patterns['checkpoint'].search(line)
            if match:
                step = int(match.group(1))
                results['checkpoint_saves'].append(step)
                continue
            
            # GPU OOM detection
            if patterns['gpu_oom'].search(line):
                results['gpu_oom_events'] += 1
                results['errors'].append(f"GPU OOM at line: {line[:100]}")
                continue
            
            # Clipping events
            match = patterns['clipping'].search(line)
            if match:
                cap_abs, cap_norm = float(match.group(1)), float(match.group(2))
                if cap_abs > 0 or cap_norm > 0:
                    results['clipping_events'] += 1
                continue
            
            # GradGuard skip detection
            match = patterns['gradguard_skip'].search(line)
            if match:
                preclip_norm = float(match.group(1))
                per_token = float(match.group(2))
                results['gradguard_skips'] += 1
                results['gradguard_max_grad'] = max(results['gradguard_max_grad'], preclip_norm)
                continue
            
            # Flash Attention backward stats (string parsing, no regex)
            if '[FlashBwd]' in line and 'row_max' in line:
                stats = parse_flash_bwd_line(line)
                if stats:
                    results['flash_bwd_stats'].append(stats)
                    if 'dQ_global' in stats:
                        results['flash_dq_history'].append(stats['dQ_global'])
                continue
            
            # Flash Attention KV block stats (P00 = attention weight)
            if '[FlashBwd-KV0]' in line:
                stats = parse_flash_kv_line(line)
                if stats:
                    if 'P00' in stats:
                        results['flash_p00_history'].append(stats['P00'])
                    if 'score00' in stats:
                        results['flash_score00_history'].append(stats['score00'])
                    if 'dQ_sum' in stats:
                        results['flash_dq_history'].append(stats['dQ_sum'])
                continue
            
            # Flash Attention Host line (sequence length)
            if '[FlashAttn Host]' in line and 'seq=' in line:
                stats = parse_flash_host_line(line)
                if stats and 'seq' in stats:
                    results['flash_seq_lens'].append(stats['seq'])
                continue
    
    return results

def calculate_statistics(values: List[float]) -> Dict:
    """Calculate basic statistics for a list of values."""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
    return {
        'mean': statistics.mean(values),
        'std': statistics.stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }

def detect_patterns(results: Dict) -> Tuple[List[str], List[str]]:
    """Detect common training issues and patterns."""
    issues = []
    recommendations = []
    
    loss_history = results['loss_history']
    lr_history = results['lr_history']
    
    # 1. Loss stuck/increasing
    if len(loss_history) >= 10:
        recent_losses = [l[1] for l in loss_history[-10:]]
        early_losses = [l[1] for l in loss_history[:min(10, len(loss_history))]]
        
        avg_recent = statistics.mean(recent_losses)
        avg_early = statistics.mean(early_losses)
        loss_std = statistics.stdev(recent_losses) if len(recent_losses) > 1 else 0
        
        if avg_recent >= avg_early * 0.98:  # Loss not decreasing
            issues.append(f"⚠️  LOSS STAGNANT: Recent avg {avg_recent:.4f} ≥ Early avg {avg_early:.4f}")
            
        if loss_std < 0.01:  # Loss variance too low
            issues.append(f"⚠️  LOSS FLAT: Std dev only {loss_std:.6f} - model may not be learning")
    
    # 2. Learning rate issues (informational only - LR is driven by other metrics we already track)
    # NOTE: LR changes themselves do NOT affect health score (use ℹ️ emoji)
    #       Only the underlying problems (grad spikes, etc.) affect health (use ⚠️/🚨)
    if len(lr_history) >= 5:
        lrs = [l[1] for l in lr_history]
        lr_min, lr_max = min(lrs), max(lrs)
        
        if lr_max / (lr_min + 1e-10) > 10:
            issues.append(f"ℹ️  LR VOLATILE: Range {lr_min:.2e} to {lr_max:.2e} (10x+ swing)")
        
        # Check if LR changed significantly (informational - dynamic LR is working as intended)
        if lrs[-1] < lrs[0] * 0.3:
            issues.append(f"ℹ️  LR ADJUSTED: Started at {lrs[0]:.2e}, now at {lrs[-1]:.2e} (dynamic LR responding to metrics)")
    
    # 3. Dynamic LR events
    events = results['dynamic_lr_events']
    # NOTE: Grad spikes are the underlying problem (⚠️), LR floor is just the response
    grad_spikes = events.get('forced_floor_grad_spike', 0) + events.get('grad_spike', 0)
    if grad_spikes > 20:
        issues.append(f"⚠️  GRAD SPIKES: {grad_spikes} gradient spike events (LR floor auto-triggered)")
        recommendations.append("→ Consider: Increase gradient clip threshold or adjust spike detection sensitivity")
    
    warmup_events = events.get('warmup', 0)
    if warmup_events > 0:
        issues.append(f"ℹ️  WARMUP: {warmup_events} steps were in warmup phase")
    
    # 4. Gradient health
    grad_checks = results['grad_checks']
    
    # Check LM head gradients
    lm_grads = grad_checks.get('lm_head_weight_grads', [])
    if lm_grads:
        rms_values = [g[0] for g in lm_grads]
        avg_rms = statistics.mean(rms_values)
        if avg_rms < 0.01:
            issues.append(f"⚠️  VANISHING GRADS: LM head grads avg RMS = {avg_rms:.6f}")
        elif avg_rms > 5.0:
            issues.append(f"⚠️  EXPLODING GRADS: LM head grads avg RMS = {avg_rms:.4f}")
    
    # Check encoder output gradients
    enc_grads = grad_checks.get('grad_encoder_out', [])
    if enc_grads:
        rms_values = [g[0] for g in enc_grads]
        avg_rms = statistics.mean(rms_values)
        if avg_rms < 1e-6:
            issues.append(f"⚠️  VANISHING ENCODER GRADS: avg RMS = {avg_rms:.2e}")
    
    # 5. Layer gradient flow
    layer_grads = results['layer_gradients']
    if layer_grads:
        layer_avgs = {}
        for layer, grads in layer_grads.items():
            if grads:
                layer_avgs[layer] = statistics.mean(grads)
        
        if layer_avgs:
            max_layer = max(layer_avgs.keys())
            min_layer = min(layer_avgs.keys())
            
            top_avg = layer_avgs.get(max_layer, 0)
            bottom_avg = layer_avgs.get(min_layer, 0)
            
            if top_avg > 0 and bottom_avg / (top_avg + 1e-10) < 0.001:
                issues.append(f"⚠️  GRADIENT VANISHING: Layer {max_layer} avg={top_avg:.2e}, Layer {min_layer} avg={bottom_avg:.2e}")
            elif bottom_avg > 0 and top_avg / (bottom_avg + 1e-10) > 1000:
                issues.append(f"⚠️  GRADIENT EXPLOSION: Layer {min_layer} avg={bottom_avg:.2e}, Layer {max_layer} avg={top_avg:.2e}")
    
    # 6. Validation loss vs training loss (check for overfitting trend, not absolute gap)
    # Only meaningful after 50% training progress - early validation trends are noisy
    progress_pct = (results['current_batch'] / max(results['total_batches'], 1)) * 100
    validation = results['validation']
    if progress_pct > 50 and len(validation) >= 3:  # Need at least 3 validation points to see a trend
        val_losses = [v['loss'] for v in validation]
        
        # Check if validation loss is increasing while training loss decreases
        if len(val_losses) >= 3:
            recent_val_trend = val_losses[-1] - val_losses[-3]  # Last 3 validations
            
            # Get corresponding training losses
            val_steps = [v['step'] for v in validation[-3:]]
            train_at_val = []
            for step in val_steps:
                # Find closest training loss to validation step
                closest = min(loss_history, key=lambda x: abs(x[0] - step), default=None)
                if closest:
                    train_at_val.append(closest[1])
            
            if len(train_at_val) >= 3:
                recent_train_trend = train_at_val[-1] - train_at_val[0]
                
                # Overfitting: val loss increasing while train loss decreasing
                if recent_val_trend > 0.1 and recent_train_trend < -0.1:
                    issues.append(f"⚠️  OVERFITTING TREND: Val loss rising (+{recent_val_trend:.3f}) while train loss falling ({recent_train_trend:.3f})")
                    recommendations.append("→ Consider: Add regularization, reduce model capacity, or use early stopping")
    
    # 7. NaN/Inf detection
    if results['warnings']:
        issues.append(f"🚨 NaN/Inf DETECTED: {len(results['warnings'])} instances")
    
    # 8. Training speed
    if results['start_time'] and results['last_time'] and results['current_batch'] > 0:
        duration = (results['last_time'] - results['start_time']).total_seconds()
        batches = results['current_batch']
        if duration > 0:
            batch_per_sec = batches / duration
            eta_remaining = (results['total_batches'] - batches) / batch_per_sec if batch_per_sec > 0 else 0
            
            if batch_per_sec < 0.1:
                issues.append(f"ℹ️  SLOW TRAINING: {batch_per_sec:.3f} batches/sec")
    
    # 9. Soft restarts
    if results['soft_restarts'] > 5:
        issues.append(f"⚠️  FREQUENT SOFT RESTARTS: {results['soft_restarts']} restarts detected")
        recommendations.append("→ Consider: Adjust soft_restart threshold or investigate loss spikes")
    
    # 10. GPU OOM events
    if results['gpu_oom_events'] > 0:
        issues.append(f"🚨 GPU OUT OF MEMORY: {results['gpu_oom_events']} OOM events")
        recommendations.append("→ Action: Reduce batch_size or max_seq_len immediately")
    
    # 11. Gradient clipping frequency
    if results['clipping_events'] > results['current_batch'] * 0.8:
        issues.append(f"⚠️  EXCESSIVE CLIPPING: {results['clipping_events']}/{results['current_batch']} batches clipped")
        recommendations.append("→ Consider: Increase gradient_clip threshold or reduce learning rate")
    
    # 11b. GradGuard skips (catastrophic gradients)
    if results['gradguard_skips'] > 0:
        skip_pct = (results['gradguard_skips'] / max(results['current_batch'], 1)) * 100
        issues.append(f"🚨 GRADGUARD SKIPS: {results['gradguard_skips']} batches skipped ({skip_pct:.1f}%) - max spike {results['gradguard_max_grad']:.0f}")
        recommendations.append("→ CRITICAL: Check training data for corruption, extreme sequences, or tokenization issues")
        if skip_pct > 5:
            recommendations.append("→ URGENT: >5% skip rate indicates serious data quality problems")
    
    # 12. Checkpoint health
    if results['checkpoint_saves']:
        issues.append(f"✅ CHECKPOINTS: {len(results['checkpoint_saves'])} saves (last: step {results['checkpoint_saves'][-1]})")
    
    # 13. Gradient norm stability
    if results['grad_norm_history']:
        recent_norms = [g[1] for g in results['grad_norm_history'][-20:]]
        if recent_norms:
            avg_norm = statistics.mean(recent_norms)
            if avg_norm > 100:
                issues.append(f"⚠️  HIGH GRADIENT NORMS: avg={avg_norm:.1f} (check for instability)")
    
    # 14. Batch size consistency
    if results['batch_size_history']:
        unique_sizes = set(results['batch_size_history'])
        if len(unique_sizes) > 3:
            issues.append(f"ℹ️  DYNAMIC BATCHING: {len(unique_sizes)} different batch sizes")
    
    # 15. Sequence length distribution
    if results['sequence_length_stats']:
        seq_lens = results['sequence_length_stats']
        avg_len = statistics.mean(seq_lens)
        max_len = max(seq_lens)
        issues.append(f"ℹ️  SEQUENCE STATS: avg={avg_len:.0f}, max={max_len}")

        # Longest sequence sanity check: flag large outliers (no hard-coded max)
        if len(seq_lens) >= 20 and avg_len > 0 and max_len > avg_len * 1.8:
            issues.append(f"⚠️  LONG SEQ OUTLIER: longest {max_len} is >1.8x avg ({avg_len:.0f})")
    
    # 16. QKV variance (attention health)
    if results['attention_stats']:
        # Collect all attention gradient values
        all_attn_grads = []
        for key, values in results['attention_stats'].items():
            if 'qkv' in key.lower() or 'attn' in key.lower():
                all_attn_grads.extend(values)
        
        if len(all_attn_grads) > 10:  # Need sufficient samples
            attn_mean = statistics.mean(all_attn_grads)
            attn_std = statistics.stdev(all_attn_grads)
            attn_variance = attn_std / (attn_mean + 1e-10)  # Coefficient of variation
            
            # Compare attention variance to FFN variance for context
            ffn_variance = None
            if results['ffn_stats']:
                all_ffn_grads = [v for vals in results['ffn_stats'].values() for v in vals]
                if len(all_ffn_grads) > 10:
                    ffn_mean = statistics.mean(all_ffn_grads)
                    ffn_std = statistics.stdev(all_ffn_grads)
                    ffn_variance = ffn_std / (ffn_mean + 1e-10)
            
            # Flag if attention variance is significantly higher than FFN (attention-specific instability)
            if ffn_variance and attn_variance > ffn_variance * 5:  # Increased threshold - attention naturally varies more
                issues.append(f"⚠️  HIGH QKV VARIANCE: {attn_variance:.2f} (vs FFN: {ffn_variance:.2f})")
                recommendations.append("→ Consider: Attention gradients more unstable than FFN - check attention gradient clipping")
            # Or if variance is extremely high in absolute terms (increased from 10 to 200)
            elif attn_variance > 200.0:
                issues.append(f"⚠️  HIGH QKV VARIANCE: {attn_variance:.2f} (attention unstable)")
                recommendations.append("→ Consider: Check attention gradient clipping or reduce learning rate")
            # Or extremely low (saturated)
            elif attn_variance < 0.001:
                issues.append(f"⚠️  LOW QKV VARIANCE: {attn_variance:.4f} (attention may be saturated)")
    
    # 17. Attention collapse detection
    if results['flash_p00_history']:
        p00_vals = results['flash_p00_history']
        # P00 = 1.0 means attention is one-hot (all mass on position 0)
        one_hot_count = sum(1 for p in p00_vals if p > 0.99)
        one_hot_pct = (one_hot_count / len(p00_vals)) * 100
        
        if one_hot_pct > 50:
            issues.append(f"🚨 ATTENTION COLLAPSE: {one_hot_pct:.0f}% of samples have P00>0.99 (one-hot)")
            recommendations.append("→ CRITICAL: Attention is collapsing to single token - check softmax temperature or QKV scaling")
        elif one_hot_pct > 20:
            issues.append(f"⚠️  ATTENTION SATURATION: {one_hot_pct:.0f}% of samples have P00>0.99")
            recommendations.append("→ Consider: Increase softmax temperature or add attention dropout")
    
    # 18. Pre-softmax score explosion (causes saturation)
    if results['flash_score00_history']:
        scores = results['flash_score00_history']
        avg_score = statistics.mean(scores)
        max_score = max(abs(s) for s in scores)
        
        # Scores > 10 will cause softmax saturation (exp(10) dominates)
        if max_score > 15:
            issues.append(f"⚠️  ATTENTION SCORE EXPLOSION: max |score|={max_score:.1f} (>15 causes saturation)")
            recommendations.append("→ Consider: Check QKV scaling (should be 1/sqrt(head_dim)) or reduce LR")
        elif avg_score > 5:
            issues.append(f"ℹ️  HIGH ATTENTION SCORES: avg={avg_score:.2f} (saturation risk)")
    
    # 19. dQ gradient trend (vanishing = attention not learning)
    if len(results['flash_dq_history']) >= 10:
        dq_vals = results['flash_dq_history']
        early_dq = statistics.mean(dq_vals[:len(dq_vals)//3])
        late_dq = statistics.mean(dq_vals[-len(dq_vals)//3:])
        
        if late_dq < early_dq * 0.1 and late_dq < 0.01:
            issues.append(f"🚨 dQ VANISHING: dropped from {early_dq:.4f} to {late_dq:.4f} (attention not learning)")
            recommendations.append("→ CRITICAL: Attention gradients vanishing - possible collapse or saturation")
        elif late_dq < early_dq * 0.3:
            issues.append(f"⚠️  dQ DECLINING: dropped from {early_dq:.4f} to {late_dq:.4f}")
    
    return issues, recommendations

def print_report(results: Dict, log_path: str, save_analysis: bool = True):
    """Print a formatted diagnostic report."""
    
    analysis_lines = []  # Collect all output for saving
    
    def print_and_save(text: str = ""):
        """Print to console and save to analysis log."""
        print(text)
        analysis_lines.append(text)
    
    print_and_save(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print_and_save(f"{Colors.BOLD}{Colors.CYAN}  GRIM Training Log Analysis Report{Colors.RESET}")
    print_and_save(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
    
    # Log info
    print_and_save(f"{Colors.BOLD}Log File:{Colors.RESET} {os.path.basename(log_path)}")
    if results['start_time']:
        print_and_save(f"{Colors.BOLD}Start Time:{Colors.RESET} {results['start_time']}")
    if results['last_time']:
        print_and_save(f"{Colors.BOLD}Last Update:{Colors.RESET} {results['last_time']}")
        if results['start_time']:
            duration = results['last_time'] - results['start_time']
            print_and_save(f"{Colors.BOLD}Duration:{Colors.RESET} {duration}")
    
    print_and_save(f"\n{Colors.BOLD}Progress:{Colors.RESET} Batch {results['current_batch']}/{results['total_batches']}")
    if results['total_batches'] > 0:
        pct = results['current_batch'] / results['total_batches'] * 100
        print_and_save(f"{Colors.BOLD}Completion:{Colors.RESET} {pct:.1f}%")
    
    # Config summary
    print_and_save(f"\n{Colors.BOLD}{Colors.BLUE}--- Configuration ---{Colors.RESET}")
    config = results['config']
    if 'd_model' in config:
        print_and_save(f"  Model: d_model={config['d_model']}, d_ff={config['d_ff']}")
    if 'config_lr' in config:
        print_and_save(f"  Learning Rate: {config['config_lr'][0]}")
    if 'warmup' in config:
        print_and_save(f"  Warmup Steps: {config['warmup'][0]}")
    if 'train_seqs' in config:
        print_and_save(f"  Training Sequences: {config['train_seqs'][0]}")
    
    # Loss summary
    print_and_save(f"\n{Colors.BOLD}{Colors.GREEN}--- Loss Summary ---{Colors.RESET}")
    loss_history = results['loss_history']
    if loss_history:
        losses = [l[1] for l in loss_history]
        print_and_save(f"  Initial Loss: {losses[0]:.4f}")
        print_and_save(f"  Current Loss: {losses[-1]:.4f}")
        print_and_save(f"  Lowest Loss:  {min(losses):.4f}")
        print_and_save(f"  Change:       {losses[-1] - losses[0]:+.4f} ({(losses[-1]/losses[0] - 1)*100:+.1f}%)")
        
        # Loss trend (last 10 steps)
        if len(losses) >= 10:
            recent = losses[-10:]
            trend = recent[-1] - recent[0]
            trend_word = "↓ decreasing" if trend < -0.01 else "↑ increasing" if trend > 0.01 else "→ flat"
            trend_color = Colors.GREEN if trend < -0.01 else Colors.RED if trend > 0.01 else Colors.YELLOW
            print_and_save(f"  Recent Trend: {trend_color}{trend_word} ({trend:+.4f}){Colors.RESET}")
    
    # Learning rate summary
    print_and_save(f"\n{Colors.BOLD}{Colors.MAGENTA}--- Learning Rate ---{Colors.RESET}")
    lr_history = results['lr_history']
    if lr_history:
        lrs = [l[1] for l in lr_history]
        print_and_save(f"  Initial LR: {lrs[0]:.2e}")
        print_and_save(f"  Current LR: {lrs[-1]:.2e}")
        print_and_save(f"  Min LR:     {min(lrs):.2e}")
        print_and_save(f"  Max LR:     {max(lrs):.2e}")
        
        # LR stability
        if lrs[-1] < lrs[0] * 0.5:
            print_and_save(f"  {Colors.RED}⚠ LR dropped by {(1 - lrs[-1]/lrs[0])*100:.0f}% from initial!{Colors.RESET}")
    
    # Dynamic LR events
    events = results['dynamic_lr_events']
    if events:
        print_and_save(f"\n{Colors.BOLD}{Colors.YELLOW}--- Dynamic LR Events ---{Colors.RESET}")
        for event, count in sorted(events.items(), key=lambda x: -x[1]):
            color = Colors.RED if 'spike' in event or 'floor' in event else Colors.WHITE
            print_and_save(f"  {color}{event}: {count}{Colors.RESET}")
    
    # Gradient statistics
    print_and_save(f"\n{Colors.BOLD}{Colors.CYAN}--- Gradient Health ---{Colors.RESET}")
    key_grads = ['lm_head_weight_grads', 'grad_encoder_out', 'grad_logits']
    for key in key_grads:
        if key in results['grad_checks'] and results['grad_checks'][key]:
            grads = results['grad_checks'][key]
            rms_vals = [g[0] for g in grads]
            stats = calculate_statistics(rms_vals)
            health = "✓ healthy" if 1e-6 < stats['mean'] < 10 else "⚠ check!"
            color = Colors.GREEN if "healthy" in health else Colors.RED
            print_and_save(f"  {key}:")
            print_and_save(f"    RMS mean={stats['mean']:.2e}, std={stats['std']:.2e}, range=[{stats['min']:.2e}, {stats['max']:.2e}] {color}{health}{Colors.RESET}")
    
    # Layer gradient flow
    layer_grads = results['layer_gradients']
    if layer_grads:
        print_and_save(f"\n{Colors.BOLD}  Layer Gradient Flow (avg RMS):{Colors.RESET}")
        layers = sorted(layer_grads.keys())
        
        # Calculate all averages first to find max for scaling
        layer_avgs = {}
        for layer in layers:
            grads = layer_grads[layer]
            if grads:
                layer_avgs[layer] = statistics.mean(grads)
        
        if layer_avgs:
            max_avg = max(layer_avgs.values())
            
            for layer in layers:
                if layer in layer_avgs:
                    avg = layer_avgs[layer]
                    # Scale bar relative to max gradient (40 chars max, min 1 char for visibility)
                    bar_len = max(1, int((avg / max_avg) * 40)) if max_avg > 0 else 1
                    bar = '█' * bar_len
                    print_and_save(f"    Layer {layer:2d}: {avg:.2e} {Colors.BLUE}{bar}{Colors.RESET}")
    
    # Gradient Norm History
    if results['grad_norm_history']:
        print_and_save(f"\n{Colors.BOLD}{Colors.CYAN}--- Gradient Norms ---{Colors.RESET}")
        recent_norms = results['grad_norm_history'][-10:]
        if recent_norms:
            avg_grad = statistics.mean([g[1] for g in recent_norms])
            avg_per_token = statistics.mean([g[2] for g in recent_norms])
            print_and_save(f"  Recent Avg Norm: {avg_grad:.2f}")
            print_and_save(f"  Recent Avg Per-Token: {avg_per_token:.4f}")
            if results['clipping_events'] > 0:
                clip_pct = (results['clipping_events'] / max(results['current_batch'], 1)) * 100
                clip_color = Colors.RED if clip_pct > 80 else Colors.YELLOW if clip_pct > 50 else Colors.GREEN
                print_and_save(f"  Clipping Events: {clip_color}{results['clipping_events']} ({clip_pct:.1f}%){Colors.RESET}")
    
    # GradGuard Skips
    if results['gradguard_skips'] > 0:
        skip_pct = (results['gradguard_skips'] / max(results['current_batch'], 1)) * 100
        print_and_save(f"  {Colors.RED}⚠ GradGuard Skips: {results['gradguard_skips']} ({skip_pct:.1f}%){Colors.RESET}")
        print_and_save(f"  {Colors.RED}  Max Spike: {results['gradguard_max_grad']:.0f} (catastrophic gradient){Colors.RESET}")
    
    # Training Stability
    print_and_save(f"\n{Colors.BOLD}{Colors.BLUE}--- Training Stability ---{Colors.RESET}")
    if results['soft_restarts'] > 0:
        print_and_save(f"  Soft Restarts: {results['soft_restarts']}")
    if results['checkpoint_saves']:
        print_and_save(f"  Checkpoints Saved: {len(results['checkpoint_saves'])}")
        print_and_save(f"  Last Checkpoint: Step {results['checkpoint_saves'][-1]}")
    if results['gpu_oom_events'] > 0:
        print_and_save(f"  {Colors.RED}⚠ GPU OOM Events: {results['gpu_oom_events']}{Colors.RESET}")
    
    # Batch Statistics
    if results['batch_size_history']:
        print_and_save(f"\n{Colors.BOLD}{Colors.MAGENTA}--- Batch Statistics ---{Colors.RESET}")
        batch_stats = calculate_statistics(results['batch_size_history'])
        print_and_save(f"  Batch Size: mean={batch_stats['mean']:.1f}, range=[{batch_stats['min']:.0f}, {batch_stats['max']:.0f}]")
    
    if results['sequence_length_stats']:
        seq_stats = calculate_statistics(results['sequence_length_stats'])
        print_and_save(f"  Sequence Length: mean={seq_stats['mean']:.0f}, range=[{seq_stats['min']:.0f}, {seq_stats['max']:.0f}]")

    # Longest encountered sequence
    if results.get('longest_sequence_length', 0) > 0:
        batch_info = ""
        if results.get('longest_sequence_batch') is not None:
            batch_info = f" (batch {results['longest_sequence_batch']}"
            if results.get('longest_sequence_batch_size') is not None:
                batch_info += f", size={results['longest_sequence_batch_size']}"
            batch_info += ")"
        print_and_save(f"  Longest Encountered: {results['longest_sequence_length']}{batch_info}")
    
    # Component Health (Attention, FFN, LayerNorm)
    if results['attention_stats'] or results['ffn_stats'] or results['layernorm_stats']:
        print_and_save(f"\n{Colors.BOLD}{Colors.GREEN}--- Component Health ---{Colors.RESET}")
        
        if results['attention_stats']:
            attn_vals = [v for vals in results['attention_stats'].values() for v in vals]
            if attn_vals:
                attn_avg = statistics.mean(attn_vals)
                # Attention can have higher gradients than other components
                health = "✓ healthy" if 1e-6 < attn_avg < 500 else "⚠ check!"
                color = Colors.GREEN if "healthy" in health else Colors.YELLOW
                print_and_save(f"  Attention: avg_rms={attn_avg:.2e} {color}{health}{Colors.RESET}")
        
        if results['ffn_stats']:
            ffn_vals = [v for vals in results['ffn_stats'].values() for v in vals]
            if ffn_vals:
                ffn_avg = statistics.mean(ffn_vals)
                health = "✓ healthy" if 1e-6 < ffn_avg < 200 else "⚠ check!"
                color = Colors.GREEN if "healthy" in health else Colors.YELLOW
                print_and_save(f"  FeedForward: avg_rms={ffn_avg:.2e} {color}{health}{Colors.RESET}")
        
        if results['layernorm_stats']:
            ln_vals = [v for vals in results['layernorm_stats'].values() for v in vals]
            if ln_vals:
                ln_avg = statistics.mean(ln_vals)
                health = "✓ healthy" if 1e-6 < ln_avg < 200 else "⚠ check!"
                color = Colors.GREEN if "healthy" in health else Colors.YELLOW
                print_and_save(f"  LayerNorm: avg_rms={ln_avg:.2e} {color}{health}{Colors.RESET}")
    
    # Attention Health (collapse detection)
    if results['flash_p00_history'] or results['flash_score00_history'] or results['flash_dq_history']:
        print_and_save(f"\n{Colors.BOLD}{Colors.MAGENTA}--- Attention Health ---{Colors.RESET}")
        
        if results['flash_p00_history']:
            p00_vals = results['flash_p00_history']
            avg_p00 = statistics.mean(p00_vals)
            one_hot_count = sum(1 for p in p00_vals if p > 0.99)
            one_hot_pct = (one_hot_count / len(p00_vals)) * 100
            
            # Ideal: P00 should be distributed, not always 1.0
            health = "🚨 COLLAPSED" if one_hot_pct > 50 else "⚠️ saturating" if one_hot_pct > 20 else "✓ healthy"
            color = Colors.RED if "COLLAPSED" in health else Colors.YELLOW if "saturating" in health else Colors.GREEN
            print_and_save(f"  Attention Weights (P00): avg={avg_p00:.4f}, one-hot={one_hot_pct:.0f}% {color}{health}{Colors.RESET}")
        
        if results['flash_score00_history']:
            scores = results['flash_score00_history']
            avg_score = statistics.mean(scores)
            max_score = max(abs(s) for s in scores)
            
            # Pre-softmax scores > 10 cause saturation
            health = "⚠️ explosive" if max_score > 15 else "✓ bounded" if max_score < 10 else "ℹ️ elevated"
            color = Colors.RED if "explosive" in health else Colors.GREEN if "bounded" in health else Colors.YELLOW
            print_and_save(f"  Pre-softmax Scores: avg={avg_score:.2f}, max={max_score:.2f} {color}{health}{Colors.RESET}")
        
        if results['flash_dq_history']:
            dq_vals = results['flash_dq_history']
            recent_dq = dq_vals[-min(10, len(dq_vals)):]
            avg_dq = statistics.mean(recent_dq)
            
            # Track dQ trend
            if len(dq_vals) >= 10:
                early_dq = statistics.mean(dq_vals[:len(dq_vals)//3])
                late_dq = statistics.mean(dq_vals[-len(dq_vals)//3:])
                trend = late_dq - early_dq
                trend_pct = (trend / (early_dq + 1e-10)) * 100
                trend_str = f"↓{abs(trend_pct):.0f}%" if trend < 0 else f"↑{trend_pct:.0f}%"
                
                health = "🚨 vanishing" if late_dq < 0.001 else "⚠️ declining" if trend_pct < -50 else "✓ stable"
                color = Colors.RED if "vanishing" in health else Colors.YELLOW if "declining" in health else Colors.GREEN
                print_and_save(f"  dQ Gradient: recent_avg={avg_dq:.4f}, trend={trend_str} {color}{health}{Colors.RESET}")
            else:
                print_and_save(f"  dQ Gradient: recent_avg={avg_dq:.4f}")
        
        # Entropy estimate (if we have P00 data)
        if results['flash_seq_lens'] and results['flash_p00_history']:
            avg_seq = statistics.mean(results['flash_seq_lens'])
            max_entropy = math.log(avg_seq) if avg_seq > 1 else 1.0
            # Very rough entropy estimate: if P00 is always 1.0, entropy ~= 0
            # If P00 varies, entropy is higher
            p00_std = statistics.stdev(results['flash_p00_history']) if len(results['flash_p00_history']) > 1 else 0
            print_and_save(f"  Entropy Context: max_entropy(seq={avg_seq:.0f})={max_entropy:.2f}, P00_variance={p00_std:.4f}")
    
    # Training Speed ETA
    if results['start_time'] and results['last_time'] and results['current_batch'] > 0:
        duration = (results['last_time'] - results['start_time']).total_seconds()
        batches = results['current_batch']
        if duration > 0:
            batch_per_sec = batches / duration
            remaining_batches = results['total_batches'] - batches
            eta_seconds = remaining_batches / batch_per_sec if batch_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            print_and_save(f"\n{Colors.BOLD}{Colors.CYAN}--- Training Speed ---{Colors.RESET}")
            print_and_save(f"  Speed: {batch_per_sec:.2f} batches/sec")
            if eta_hours > 1:
                print_and_save(f"  ETA: {eta_hours:.1f} hours ({eta_minutes:.0f} min)")
            else:
                print_and_save(f"  ETA: {eta_minutes:.1f} minutes")
    
    # Validation
    if results['validation']:
        print_and_save(f"\n{Colors.BOLD}{Colors.GREEN}--- Validation ---{Colors.RESET}")
        for val in results['validation'][-5:]:  # Last 5 validations
            print_and_save(f"  Step {val['step']}: loss={val['loss']:.4f}, perplexity={val['ppl']:.2f}")
    
    # Issues and recommendations
    issues, recommendations = detect_patterns(results)
    
    if issues:
        print_and_save(f"\n{Colors.BOLD}{Colors.RED}{'='*70}{Colors.RESET}")
        print_and_save(f"{Colors.BOLD}{Colors.RED}  DETECTED ISSUES{Colors.RESET}")
        print_and_save(f"{Colors.BOLD}{Colors.RED}{'='*70}{Colors.RESET}")
        for issue in issues:
            print_and_save(f"  {issue}")
    
    if recommendations:
        print_and_save(f"\n{Colors.BOLD}{Colors.YELLOW}--- Recommendations ---{Colors.RESET}")
        for rec in recommendations:
            print_and_save(f"  {rec}")
    
    # Overall health score
    print_and_save(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    # NOTE: Health score based on training problems (⚠️ = -15, 🚨 = -30)
    #       LR adjustments (ℹ️) do NOT penalize health - they're responses to problems, not problems themselves
    health_score = 100
    health_score -= len([i for i in issues if '⚠️' in i]) * 15
    health_score -= len([i for i in issues if '🚨' in i]) * 30
    health_score = max(0, health_score)
    
    if health_score >= 80:
        health_color = Colors.GREEN
        health_emoji = "✅"
    elif health_score >= 50:
        health_color = Colors.YELLOW
        health_emoji = "⚠️"
    else:
        health_color = Colors.RED
        health_emoji = "❌"
    
    print_and_save(f"{Colors.BOLD}  Training Health Score: {health_color}{health_score}/100 {health_emoji}{Colors.RESET}")
    print_and_save(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
    
    # Save analysis to file
    if save_analysis:
        # Save in same directory as training log with similar naming pattern
        log_dir = os.path.dirname(log_path)
        log_basename = os.path.basename(log_path)
        
        # Extract timestamp from training log name if it has the pattern training_YYYYMMDD_HHMMSS.log
        timestamp_match = re.search(r'training_(\d{8}_\d{6})\.log', log_basename)
        if timestamp_match:
            # Use the same timestamp as the training log
            timestamp_str = timestamp_match.group(1)
            analysis_filename = f"analysis_{timestamp_str}.log"
        else:
            # Fallback: generate new timestamp
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_filename = f"analysis_{timestamp_str}.log"
        
        analysis_log_path = os.path.join(log_dir, analysis_filename)
        
        try:
            with open(analysis_log_path, 'w', encoding='utf-8') as f:
                # Write timestamp header
                f.write(f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source log: {log_basename}\n")
                f.write(f"{'='*70}\n\n")
                
                # Strip ANSI color codes for the saved file
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                for line in analysis_lines:
                    clean_line = ansi_escape.sub('', line)
                    f.write(clean_line + '\n')
            print_and_save(f"{Colors.GREEN}✓ Analysis saved to: {analysis_filename}{Colors.RESET}\n")
        except Exception as e:
            print_and_save(f"{Colors.RED}⚠ Failed to save analysis: {e}{Colors.RESET}\n")
    
    return health_score

def watch_mode(log_path: str, interval: int = 5):
    """Continuously watch and update analysis."""
    import time
    print(f"Watching {log_path} (Ctrl+C to stop)...")
    print(f"Updates will refresh display only (no file saves during watch)\n")
    last_size = 0
    
    while True:
        try:
            current_size = os.path.getsize(log_path)
            if current_size != last_size:
                os.system('cls' if os.name == 'nt' else 'clear')
                results = analyze_log(log_path)
                print_report(results, log_path, save_analysis=False)  # Don't save during watch
                last_size = current_size
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped watching.")
            # Save final analysis on exit
            print("Saving final analysis...")
            results = analyze_log(log_path)
            print_report(results, log_path, save_analysis=True)
            break

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GRIM Training Log Analyzer')
    parser.add_argument('--log', '-l', type=str, help='Specific log file to analyze')
    parser.add_argument('--dir', '-d', type=str, 
                        default=r'D:\G.R.I.M\resources\models\GRIM-text\training\logs',
                        help='Log directory (default: GRIM training logs)')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch mode - continuously update')
    parser.add_argument('--interval', '-i', type=int, default=5, help='Watch interval in seconds')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    
    args = parser.parse_args()
    
    # Find log file
    if args.log:
        log_path = args.log
    else:
        log_path = find_latest_log(args.dir)
    
    if not log_path or not os.path.exists(log_path):
        print(f"{Colors.RED}Error: No log file found!{Colors.RESET}")
        print(f"Searched in: {args.dir}")
        sys.exit(1)
    
    print(f"{Colors.CYAN}Analyzing: {log_path}{Colors.RESET}\n")
    
    if args.watch:
        watch_mode(log_path, args.interval)
    else:
        results = analyze_log(log_path)
        
        if args.json:
            import json
            # Convert for JSON serialization
            output = {
                'config': results['config'],
                'loss_history': results['loss_history'][-20:],  # Last 20
                'lr_history': results['lr_history'][-20:],
                'current_batch': results['current_batch'],
                'total_batches': results['total_batches'],
                'dynamic_lr_events': dict(results['dynamic_lr_events']),
                'validation': results['validation'],
                'issues': detect_patterns(results)[0],
            }
            print(json.dumps(output, indent=2))
        else:
            print_report(results, log_path)

if __name__ == '__main__':
    main()


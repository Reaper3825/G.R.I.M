#!/usr/bin/env python3
"""Quick gradient norm analysis script"""
import re
import sys

log_path = sys.argv[1] if len(sys.argv) > 1 else r'D:\G.R.I.M\resources\models\GRIM-text\training\logs\training_17656777702397256.log'

diag_pattern = re.compile(r'\[Diag\] batch=(\d+).*loss=([\d.]+) preclip_grad=([\d.]+) preclip_norm=([\d.]+).*seqs=\[(.*?)\]')
guard_pattern = re.compile(r'\[GradGuard\] skip.*preclip_norm=([\d.]+).*batch=(\d+).*tokens=(\d+)')

print('=' * 80)
print('GRADIENT NORM ANALYSIS')
print('=' * 80)

diag_data = []
guard_data = []

with open(log_path, 'r') as f:
    for line in f:
        m = diag_pattern.search(line)
        if m:
            batch, loss, grad, norm, seqs = m.groups()
            # Parse seqs to get total tokens
            seq_parts = seqs.split(',')
            total_tokens = sum(int(s.split(':')[1]) for s in seq_parts)
            diag_data.append({
                'batch': int(batch),
                'loss': float(loss),
                'grad': float(grad),
                'norm': float(norm),
                'tokens': total_tokens,
                'per_token': float(norm) / total_tokens if total_tokens > 0 else 0
            })
        
        m = guard_pattern.search(line)
        if m:
            norm, batch, tokens = m.groups()
            guard_data.append({
                'batch': int(batch),
                'norm': float(norm),
                'tokens': int(tokens),
                'per_token': float(norm) / int(tokens)
            })

print(f'\nTotal batches with [Diag]: {len(diag_data)}')
print(f'Total GradGuard skips: {len(guard_data)}')

if diag_data:
    norms = [d['norm'] for d in diag_data]
    per_tokens = [d['per_token'] for d in diag_data]
    print(f'\n--- Pre-clip Gradient Norms (from [Diag]) ---')
    print(f'  Min: {min(norms):.1f}')
    print(f'  Max: {max(norms):.1f}')
    print(f'  Mean: {sum(norms)/len(norms):.1f}')
    print(f'\n--- Per-Token Norms ---')
    print(f'  Min: {min(per_tokens):.2f}')
    print(f'  Max: {max(per_tokens):.2f}')
    print(f'  Mean: {sum(per_tokens)/len(per_tokens):.2f}')

# Identify problematic batches
print(f'\n--- HIGH GRADIENT BATCHES (norm > 1000) ---')
high_norm = [d for d in diag_data if d['norm'] > 1000]
for d in sorted(high_norm, key=lambda x: -x['norm'])[:10]:
    print(f"  Batch {d['batch']:3d}: norm={d['norm']:10.1f} tokens={d['tokens']:4d} per_token={d['per_token']:7.2f} loss={d['loss']:.4f}")

if guard_data:
    print(f'\n--- CATASTROPHIC GRADIENTS (GradGuard skipped) ---')
    for d in sorted(guard_data, key=lambda x: -x['norm']):
        print(f"  Batch {d['batch']:3d}: norm={d['norm']:10.1f} tokens={d['tokens']:4d} per_token={d['per_token']:7.2f}")

# Per-token distribution
print(f'\n--- PER-TOKEN NORM DISTRIBUTION ---')
all_per_token = [d['per_token'] for d in diag_data] + [d['per_token'] for d in guard_data]
buckets = [0, 10, 50, 100, 250, 500, 1000, float('inf')]
for i in range(len(buckets)-1):
    count = sum(1 for pt in all_per_token if buckets[i] <= pt < buckets[i+1])
    pct = count / len(all_per_token) * 100 if all_per_token else 0
    label = f'{buckets[i]}-{buckets[i+1]}' if buckets[i+1] != float('inf') else f'>{buckets[i]}'
    bar = '█' * int(pct / 2)
    print(f'  {label:>10}: {count:3d} ({pct:5.1f}%) {bar}')

# Correlation with sequence length
print(f'\n--- CORRELATION: TOKENS vs NORM ---')
if len(diag_data) > 5:
    tokens_list = [d['tokens'] for d in diag_data]
    norms_list = [d['norm'] for d in diag_data]
    
    # Simple correlation
    mean_t = sum(tokens_list) / len(tokens_list)
    mean_n = sum(norms_list) / len(norms_list)
    
    cov = sum((t - mean_t) * (n - mean_n) for t, n in zip(tokens_list, norms_list)) / len(tokens_list)
    std_t = (sum((t - mean_t)**2 for t in tokens_list) / len(tokens_list)) ** 0.5
    std_n = (sum((n - mean_n)**2 for n in norms_list) / len(norms_list)) ** 0.5
    
    corr = cov / (std_t * std_n) if std_t > 0 and std_n > 0 else 0
    print(f'  Pearson correlation: {corr:.3f}')
    if corr > 0.5:
        print('  → Strong positive correlation: longer sequences → higher norms')
    elif corr < -0.5:
        print('  → Strong negative correlation: shorter sequences → higher norms')
    else:
        print('  → Weak correlation: gradient issues not sequence-length dependent')

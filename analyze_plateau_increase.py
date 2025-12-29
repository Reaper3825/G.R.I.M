"""
Analyze loss increase after minimum point to quantify plateau behavior.
Finds minimum loss and measures increase factor/rate in subsequent batches.
"""

import re
import sys
from pathlib import Path
import numpy as np


def parse_training_log(log_path):
    """Parse training log and extract batch number and loss."""
    batch_losses = []
    
    # Try multiple patterns
    patterns = [
        r'\[Batch (\d+)/\d+\]\s+loss=([\d.]+)',  # Standard format
        r'\[GradTrace\] POST-BACKWARD batch=(\d+) loss=([\d.]+)',  # GradTrace format
    ]
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    batch_num = int(match.group(1))
                    loss = float(match.group(2))
                    batch_losses.append((batch_num, loss))
                    break  # Found match, don't try other patterns
    
    return batch_losses


def analyze_plateau_increase(batch_losses, window_size=10):
    """
    Find minimum loss and analyze increase behavior in subsequent batches.
    
    Args:
        batch_losses: List of (batch_num, loss) tuples
        window_size: Number of batches after minimum to analyze
    
    Returns:
        dict with analysis results
    """
    if not batch_losses:
        return None
    
    # Find minimum loss
    min_idx = min(range(len(batch_losses)), key=lambda i: batch_losses[i][1])
    min_batch, min_loss = batch_losses[min_idx]
    
    print(f"{'='*70}")
    print(f"LOSS PLATEAU INCREASE ANALYSIS")
    print(f"{'='*70}\n")
    
    print(f"Minimum Loss Point:")
    print(f"  Batch: {min_batch}")
    print(f"  Loss: {min_loss:.6f}")
    print(f"  Position: {min_idx}/{len(batch_losses)} ({min_idx/len(batch_losses)*100:.1f}%)")
    
    # Get subsequent batches
    subsequent_window = batch_losses[min_idx:min_idx + window_size + 1]
    
    if len(subsequent_window) < 2:
        print("\n⚠️ Not enough data after minimum to analyze increase")
        return None
    
    print(f"\nSubsequent {len(subsequent_window)-1} Batches:")
    print(f"{'Batch':<10} {'Loss':<12} {'Delta':<12} {'% Increase':<12}")
    print(f"{'-'*50}")
    
    deltas = []
    pct_increases = []
    
    for i, (batch, loss) in enumerate(subsequent_window):
        if i == 0:
            print(f"{batch:<10} {loss:<12.6f} {'(min)':<12} {'--':<12}")
        else:
            delta = loss - min_loss
            pct_increase = (delta / min_loss) * 100
            deltas.append(delta)
            pct_increases.append(pct_increase)
            print(f"{batch:<10} {loss:<12.6f} {delta:+12.6f} {pct_increase:+11.3f}%")
    
    # Calculate increase metrics
    if deltas:
        print(f"\n{'='*70}")
        print(f"INCREASE METRICS")
        print(f"{'='*70}\n")
        
        # Linear increase rate (loss units per batch)
        batches_elapsed = np.arange(1, len(deltas) + 1)
        linear_rate = np.polyfit(batches_elapsed, deltas, 1)[0]
        
        # Exponential increase factor (if applicable)
        # Model: loss(t) = min_loss * (1 + r)^t
        # Solve for r: r = (loss(t) / min_loss)^(1/t) - 1
        exp_factors = []
        for i, (batch, loss) in enumerate(subsequent_window[1:], 1):
            if loss > min_loss:
                factor = (loss / min_loss) ** (1.0 / i) - 1
                exp_factors.append(factor)
        
        avg_exp_factor = np.mean(exp_factors) if exp_factors else 0.0
        
        # Plateau detection: count batches with <0.5% increase
        plateau_threshold = 0.005  # 0.5%
        plateau_count = sum(1 for pct in pct_increases if abs(pct/100) < plateau_threshold)
        
        print(f"Linear Increase Rate:")
        print(f"  {linear_rate:.6f} loss units per batch")
        print(f"  ({linear_rate * 100:.4f} loss units per 100 batches)")
        
        print(f"\nExponential Growth Factor:")
        print(f"  Average: {avg_exp_factor:.6f} per batch")
        print(f"  ({avg_exp_factor * 100:.4f}% per batch)")
        
        print(f"\nPlateau Detection:")
        print(f"  Batches with <0.5% change: {plateau_count}/{len(pct_increases)}")
        print(f"  Plateau ratio: {plateau_count/len(pct_increases)*100:.1f}%")
        
        # Time to X% increase
        print(f"\nProjected Time to Significant Increase:")
        if linear_rate > 0:
            for pct_target in [1, 5, 10]:
                target_delta = min_loss * (pct_target / 100)
                batches_to_target = target_delta / linear_rate
                print(f"  {pct_target}% increase: {batches_to_target:.1f} batches")
        else:
            print(f"  No increase detected (rate ≤ 0)")
        
        # Reverse engineering: What's causing the increase?
        print(f"\n{'='*70}")
        print(f"REVERSE ENGINEERING: INCREASE CAUSE")
        print(f"{'='*70}\n")
        
        if plateau_count / len(pct_increases) > 0.8:
            print("✓ STABLE PLATEAU:")
            print("  Loss remains within 0.5% of minimum")
            print("  → Model has converged to local minimum")
            print("  → No significant gradient information remaining")
            print("  → Increase factor ≈ 0 (true plateau)")
        
        elif linear_rate > 0.001:
            print("⚠️ SLOW DIVERGENCE:")
            print(f"  Loss increasing at {linear_rate:.6f} per batch")
            print("  → Possible causes:")
            print("    1. Learning rate too high (overshooting minimum)")
            print("    2. Gradient noise dominating signal")
            print("    3. Optimizer momentum causing oscillation")
            print(f"  → Increase factor: {linear_rate:.6f} loss/batch")
        
        elif linear_rate < -0.001:
            print("✓ CONTINUED IMPROVEMENT:")
            print(f"  Loss still decreasing at {abs(linear_rate):.6f} per batch")
            print("  → Model has not plateaued yet")
            print(f"  → Decrease factor: {abs(linear_rate):.6f} loss/batch")
        
        else:
            print("✓ MARGINAL FLUCTUATION:")
            print(f"  Loss fluctuating by ±{np.std(deltas):.6f}")
            print("  → Model oscillating around minimum")
            print("  → Increase factor ≈ 0 (noise only)")
        
        return {
            'min_batch': min_batch,
            'min_loss': min_loss,
            'linear_rate': linear_rate,
            'exp_factor': avg_exp_factor,
            'plateau_ratio': plateau_count / len(pct_increases),
            'subsequent_losses': [loss for _, loss in subsequent_window[1:]],
            'deltas': deltas,
            'pct_increases': pct_increases
        }
    
    return None


def main():
    # Get log file from command line or use most recent
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Find most recent training log
        log_dir = Path("resources/models/GRIM-text/training/logs")
        logs = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not logs:
            print("No training logs found!")
            return
        
        log_path = logs[0]
        print(f"Using most recent log: {log_path.name}\n")
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    # Parse log
    batch_losses = parse_training_log(log_path)
    
    if not batch_losses:
        print("No batch loss data found in log!")
        return
    
    print(f"Loaded {len(batch_losses)} batch loss entries\n")
    
    # Analyze plateau increase with configurable window
    window_size = 50 if len(batch_losses) > 100 else min(20, len(batch_losses) - 1)
    result = analyze_plateau_increase(batch_losses, window_size=window_size)
    
    if result:
        # Additional analysis: Compare to overall training
        overall_losses = [loss for _, loss in batch_losses]
        initial_loss = overall_losses[0]
        final_loss = overall_losses[-1]
        total_improvement = initial_loss - result['min_loss']
        post_min_drift = final_loss - result['min_loss']
        
        print(f"\n{'='*70}")
        print(f"OVERALL TRAINING SUMMARY")
        print(f"{'='*70}\n")
        print(f"Initial loss: {initial_loss:.6f}")
        print(f"Minimum loss: {result['min_loss']:.6f}")
        print(f"Final loss: {final_loss:.6f}")
        print(f"\nTotal improvement: {total_improvement:.6f} ({total_improvement/initial_loss*100:.2f}%)")
        print(f"Post-minimum drift: {post_min_drift:+.6f} ({post_min_drift/result['min_loss']*100:+.2f}%)")
        
        if post_min_drift > 0.01:
            print(f"\n⚠️ Loss INCREASED by {post_min_drift:.4f} after minimum")
            print(f"   → Training continued {len(batch_losses) - result['min_batch']} batches past optimal point")
        elif abs(post_min_drift) < 0.01:
            print(f"\n✓ Loss STABLE after minimum (±{abs(post_min_drift):.4f})")
        else:
            print(f"\n✓ Loss IMPROVED by {abs(post_min_drift):.4f} after initial minimum")


if __name__ == '__main__':
    main()

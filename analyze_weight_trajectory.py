"""
Analyze weight trajectory during training to debug plateau behavior.
Extracts PRE-OPTIMIZER and POST-OPTIMIZER weight samples from training logs.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def parse_weight_samples(log_path):
    """Extract PRE and POST optimizer weight samples from log."""
    pre_weights = []  # (batch, [w0, w1, w2, w3, w4])
    post_weights = []
    
    # Pattern: [GradTrace] PRE-OPTIMIZER batch=X ... lm_w[0:5]=[a,b,c,d,e]
    pre_pattern = r'\[GradTrace\] PRE-OPTIMIZER batch=(\d+).*lm_w\[0:5\]=\[([-\d.,]+)\]'
    post_pattern = r'\[GradTrace\] POST-OPTIMIZER batch=(\d+).*lm_w\[0:5\]=\[([-\d.,]+)\]'
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            pre_match = re.search(pre_pattern, line)
            if pre_match:
                batch = int(pre_match.group(1))
                weights_str = pre_match.group(2)
                weights = [float(w) for w in weights_str.split(',')]
                pre_weights.append((batch, weights))
            
            post_match = re.search(post_pattern, line)
            if post_match:
                batch = int(post_match.group(1))
                weights_str = post_match.group(2)
                weights = [float(w) for w in weights_str.split(',')]
                post_weights.append((batch, weights))
    
    return pre_weights, post_weights


def analyze_weight_trajectory(pre_weights, post_weights, min_loss_batch=None):
    """Analyze weight evolution to detect plateau causes."""
    print(f"{'='*70}")
    print(f"WEIGHT TRAJECTORY ANALYSIS")
    print(f"{'='*70}\n")
    
    if not pre_weights:
        print("No weight samples found!")
        return
    
    print(f"Collected {len(pre_weights)} weight samples")
    print(f"Batch range: {pre_weights[0][0]} - {pre_weights[-1][0]}")
    
    # Convert to numpy arrays
    batches = np.array([b for b, _ in pre_weights])
    pre_w = np.array([w for _, w in pre_weights])  # Shape: (num_samples, 5)
    post_w = np.array([w for _, w in post_weights])
    
    # Compute deltas (weight changes per optimizer step)
    deltas = post_w - pre_w
    
    # Overall trajectory metrics
    print(f"\n{'='*70}")
    print(f"OVERALL TRAJECTORY")
    print(f"{'='*70}\n")
    
    for i in range(5):
        initial = pre_w[0, i]
        final = post_w[-1, i]
        total_change = final - initial
        pct_change = (total_change / abs(initial)) * 100 if abs(initial) > 1e-6 else 0
        
        print(f"Weight[{i}]:")
        print(f"  Initial:  {initial:+.6f}")
        print(f"  Final:    {final:+.6f}")
        print(f"  Change:   {total_change:+.6f} ({pct_change:+.2f}%)")
    
    # Analyze deltas (per-step changes)
    print(f"\n{'='*70}")
    print(f"PER-STEP WEIGHT CHANGES")
    print(f"{'='*70}\n")
    
    for i in range(5):
        mean_delta = np.mean(deltas[:, i])
        std_delta = np.std(deltas[:, i])
        max_delta = np.max(np.abs(deltas[:, i]))
        
        print(f"Weight[{i}] deltas:")
        print(f"  Mean:   {mean_delta:+.6e}")
        print(f"  Std:    {std_delta:.6e}")
        print(f"  Max:    {max_delta:.6e}")
        print(f"  SNR:    {abs(mean_delta)/std_delta if std_delta > 0 else 0:.4f}")
    
    # Detect oscillation patterns
    print(f"\n{'='*70}")
    print(f"OSCILLATION DETECTION")
    print(f"{'='*70}\n")
    
    for i in range(5):
        # Count sign changes in deltas
        sign_changes = np.sum(np.diff(np.sign(deltas[:, i])) != 0)
        oscillation_ratio = sign_changes / len(deltas)
        
        # Check for periodic oscillation
        autocorr = np.correlate(deltas[:, i], deltas[:, i], mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first non-trivial peak
        peak_idx = None
        for j in range(2, min(20, len(autocorr))):
            if autocorr[j] > 0.5:  # Strong correlation
                peak_idx = j
                break
        
        print(f"Weight[{i}]:")
        print(f"  Sign changes: {sign_changes}/{len(deltas)} ({oscillation_ratio*100:.1f}%)")
        if peak_idx:
            print(f"  ⚠️ PERIODIC OSCILLATION detected (period ~{peak_idx} steps)")
        elif oscillation_ratio > 0.7:
            print(f"  ⚠️ HIGH OSCILLATION (learning rate may be too high)")
        elif oscillation_ratio < 0.3:
            print(f"  ✓ Consistent direction")
        else:
            print(f"  ○ Mixed behavior")
    
    # Analyze pre/post minimum loss (if provided)
    if min_loss_batch:
        min_idx = np.searchsorted(batches, min_loss_batch)
        
        print(f"\n{'='*70}")
        print(f"PRE/POST MINIMUM LOSS COMPARISON")
        print(f"{'='*70}\n")
        print(f"Minimum loss at batch: {min_loss_batch}")
        print(f"Sample index: {min_idx}/{len(batches)}")
        
        if min_idx > 0 and min_idx < len(batches):
            pre_min_deltas = deltas[:min_idx]
            post_min_deltas = deltas[min_idx:]
            
            print(f"\nBefore minimum (batches 1-{min_loss_batch}):")
            for i in range(5):
                mean_pre = np.mean(pre_min_deltas[:, i])
                std_pre = np.std(pre_min_deltas[:, i])
                print(f"  Weight[{i}]: mean={mean_pre:+.6e} std={std_pre:.6e}")
            
            print(f"\nAfter minimum (batches {min_loss_batch}-{batches[-1]}):")
            for i in range(5):
                mean_post = np.mean(post_min_deltas[:, i])
                std_post = np.std(post_min_deltas[:, i])
                print(f"  Weight[{i}]: mean={mean_post:+.6e} std={std_post:.6e}")
            
            # Check if behavior changed
            print(f"\nBehavior Change Analysis:")
            for i in range(5):
                mean_pre = np.mean(pre_min_deltas[:, i])
                mean_post = np.mean(post_min_deltas[:, i])
                std_pre = np.std(pre_min_deltas[:, i])
                std_post = np.std(post_min_deltas[:, i])
                
                # Check if mean flipped sign
                if np.sign(mean_pre) != np.sign(mean_post):
                    print(f"  Weight[{i}]: ⚠️ DIRECTION REVERSAL")
                # Check if magnitude increased
                elif abs(mean_post) > abs(mean_pre) * 1.5:
                    print(f"  Weight[{i}]: ⚠️ MAGNITUDE INCREASED ({abs(mean_post)/abs(mean_pre):.2f}x)")
                # Check if noise increased
                elif std_post > std_pre * 1.5:
                    print(f"  Weight[{i}]: ⚠️ NOISE INCREASED ({std_post/std_pre:.2f}x)")
                else:
                    print(f"  Weight[{i}]: ✓ Consistent behavior")
    
    # Verdict
    print(f"\n{'='*70}")
    print(f"DIAGNOSIS")
    print(f"{'='*70}\n")
    
    # Calculate overall metrics
    overall_oscillation = np.mean([np.sum(np.diff(np.sign(deltas[:, i])) != 0) / len(deltas) for i in range(5)])
    overall_mean_delta = np.mean(np.abs(np.mean(deltas, axis=0)))
    overall_std_delta = np.mean(np.std(deltas, axis=0))
    
    if overall_oscillation > 0.7:
        print("⚠️ HIGH OSCILLATION DETECTED")
        print("  → Learning rate likely TOO HIGH")
        print("  → Weights bouncing around optimal point")
        print("  → Recommendation: Reduce learning rate by 2-5x")
    
    elif overall_std_delta > overall_mean_delta * 3:
        print("⚠️ HIGH NOISE-TO-SIGNAL RATIO")
        print("  → Gradient noise dominates useful signal")
        print("  → Possible causes:")
        print("    1. Batch size too small (high variance)")
        print("    2. Data quality issues (noisy gradients)")
        print("    3. SIMILARITY_GROUPED batching (repetitive patterns)")
        print("  → Recommendation: Increase batch size or change batching strategy")
    
    elif overall_mean_delta < 1e-6:
        print("⚠️ WEIGHT UPDATES NEARLY ZERO")
        print("  → Gradients have collapsed or vanished")
        print("  → Possible causes:")
        print("    1. Learning rate too small")
        print("    2. Gradient clipping too aggressive")
        print("    3. Encoder gradients vanishing")
        print("  → Recommendation: Check gradient component logs")
    
    else:
        print("✓ WEIGHT UPDATES APPEAR NORMAL")
        print(f"  Mean delta: {overall_mean_delta:.6e}")
        print(f"  Std delta: {overall_std_delta:.6e}")
        print(f"  Oscillation: {overall_oscillation*100:.1f}%")
        print("  → Plateau likely NOT caused by optimizer/LR issues")
        print("  → Investigate batch composition (SIMILARITY_GROUPED)")
    
    return batches, pre_w, post_w, deltas


def plot_weight_trajectory(batches, pre_w, post_w, deltas, min_loss_batch=None):
    """Create visualization of weight trajectory."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Weight values over time
    ax1 = axes[0]
    for i in range(5):
        ax1.plot(batches, pre_w[:, i], label=f'Weight[{i}]', alpha=0.7)
    if min_loss_batch:
        ax1.axvline(min_loss_batch, color='red', linestyle='--', alpha=0.5, label='Min Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Weight Value')
    ax1.set_title('LM Head Weight Trajectory (First 5 Weights)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight deltas (changes per step)
    ax2 = axes[1]
    for i in range(5):
        ax2.plot(batches, deltas[:, i], label=f'Δ Weight[{i}]', alpha=0.7)
    if min_loss_batch:
        ax2.axvline(min_loss_batch, color='red', linestyle='--', alpha=0.5, label='Min Loss')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Weight Change (Δ)')
    ax2.set_title('Per-Step Weight Changes (POST - PRE optimizer)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling mean of absolute deltas
    ax3 = axes[2]
    window = min(10, len(batches) // 10)
    for i in range(5):
        abs_deltas = np.abs(deltas[:, i])
        rolling_mean = np.convolve(abs_deltas, np.ones(window)/window, mode='valid')
        ax3.plot(batches[window-1:], rolling_mean, label=f'|Δ Weight[{i}]|', alpha=0.7)
    if min_loss_batch:
        ax3.axvline(min_loss_batch, color='red', linestyle='--', alpha=0.5, label='Min Loss')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Rolling Mean |Δ| (window=10)')
    ax3.set_title('Weight Update Magnitude Over Time')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('weight_trajectory_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved: weight_trajectory_analysis.png")


def main():
    # Get log file
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
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
    
    # Parse weights
    pre_weights, post_weights = parse_weight_samples(log_path)
    
    # Get minimum loss batch (if available from analyze_plateau_increase.py)
    min_loss_batch = None
    try:
        import json
        # Check if user ran analyze_plateau_increase.py
        # (we could parse the log again, but let's just use a reasonable estimate)
        # For now, just analyze the full trajectory
        pass
    except:
        pass
    
    # Analyze
    batches, pre_w, post_w, deltas = analyze_weight_trajectory(pre_weights, post_weights, min_loss_batch)
    
    # Plot
    if len(batches) > 0:
        plot_weight_trajectory(batches, pre_w, post_w, deltas, min_loss_batch)


if __name__ == '__main__':
    main()

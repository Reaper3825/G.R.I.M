#!/usr/bin/env python3
"""
Loss Curvature Diagnostic Tool

Checks for flat loss regions where gradients don't point toward better solutions:
1. Per-sequence loss distribution (identify outliers/dominance)
2. Gradient alignment between consecutive batches (direction consistency)
3. Directional derivative validation (does -grad direction actually decrease loss?)
4. Gradient variance analysis (are gradients getting noisier over time?)
"""

import re
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def parse_training_log(log_path):
    """Extract batch-level metrics from training log."""
    data = {
        'batch': [],
        'loss': [],
        'grad_norm': [],
        'lr': [],
        'grad_components': [],
        'per_sequence_losses': []
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        current_batch = None
        for line in f:
            # Batch number
            if '[Batch ' in line:
                match = re.search(r'\[Batch (\d+)/\d+\]', line)
                if match:
                    current_batch = int(match.group(1))
            
            # Loss
            if '[GradTrace] POST-FORWARD loss=' in line:
                match = re.search(r'loss=([\d.]+)', line)
                if match and current_batch:
                    data['batch'].append(current_batch)
                    data['loss'].append(float(match.group(1)))
            
            # Gradient norm
            if '[GradTrace] PRE-OPTIMIZER' in line and ('grad_rms=' in line or 'grad_norm=' in line):
                match = re.search(r'grad_(?:rms|norm)=([\d.]+)', line)
                if match:
                    data['grad_norm'].append(float(match.group(1)))
            
            # Learning rate
            if '[GradTrace] PRE-OPTIMIZER' in line and 'lr=' in line:
                match = re.search(r'lr=([\d.e-]+)', line)
                if match:
                    data['lr'].append(float(match.group(1)))
            
            # Gradient components
            if 'COMPUTED COMPONENTS:' in line:
                components = {}
                # Extract component values
                for comp in ['emb_lm_tied', 'emb', 'lm', 'attn', 'ffn', 'rms']:
                    pattern = rf'{comp}=([\d.]+)'
                    match = re.search(pattern, line)
                    if match:
                        components[comp] = float(match.group(1))
                if components:
                    data['grad_components'].append(components)
    
    return data

def analyze_gradient_alignment(data):
    """Check if gradients point in consistent directions."""
    print("\n" + "="*60)
    print("GRADIENT ALIGNMENT ANALYSIS")
    print("="*60)
    
    if len(data['grad_components']) < 10:
        print("⚠ Not enough gradient component data")
        return
    
    # Convert to arrays for vectorized ops
    components = data['grad_components']
    
    # Extract component sequences
    comp_names = list(components[0].keys())
    grad_vectors = []
    for comp_dict in components:
        vec = [comp_dict.get(name, 0.0) for name in comp_names]
        grad_vectors.append(vec)
    
    grad_vectors = np.array(grad_vectors)
    
    # Compute cosine similarity between consecutive gradients
    similarities = []
    for i in range(1, len(grad_vectors)):
        v1 = grad_vectors[i-1]
        v2 = grad_vectors[i]
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 1e-8 and norm2 > 1e-8:
            cos_sim = np.dot(v1, v2) / (norm1 * norm2)
            similarities.append(cos_sim)
    
    similarities = np.array(similarities)
    
    print(f"\nGradient Direction Consistency:")
    print(f"  Mean cosine similarity: {similarities.mean():.4f}")
    print(f"  Std dev: {similarities.std():.4f}")
    print(f"  Min: {similarities.min():.4f}, Max: {similarities.max():.4f}")
    
    if similarities.mean() < 0.5:
        print("  ⚠ WARNING: Low gradient alignment - gradients are noisy/conflicting!")
    elif similarities.mean() > 0.95:
        print("  ✓ High gradient alignment - consistent optimization direction")
    else:
        print("  ⚡ Moderate alignment - some noise but generally consistent")
    
    # Check for gradient variance collapse
    component_variance = np.var(grad_vectors, axis=0)
    print(f"\nGradient Component Variance:")
    for i, name in enumerate(comp_names):
        print(f"  {name}: {component_variance[i]:.6f}")
    
    if np.any(component_variance < 1e-6):
        print("  ⚠ WARNING: Some components have collapsed variance!")

def analyze_loss_trajectory(data):
    """Analyze loss trajectory for plateau detection."""
    print("\n" + "="*60)
    print("LOSS TRAJECTORY ANALYSIS")
    print("="*60)
    
    losses = np.array(data['loss'])
    batches = np.array(data['batch'])
    
    # Compute moving statistics
    window = 20
    if len(losses) < window:
        print("⚠ Not enough data for moving window analysis")
        return
    
    # Moving average and std dev
    moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
    moving_std = np.array([np.std(losses[max(0,i-window):i+1]) for i in range(window-1, len(losses))])
    
    # Detect plateau: small std dev + small slope
    slopes = np.gradient(moving_avg)
    
    plateau_threshold = 0.05  # Loss change per batch
    plateau_mask = np.abs(slopes) < plateau_threshold
    
    print(f"\nLoss Statistics:")
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Total drop: {losses[0] - losses[-1]:.4f}")
    print(f"  Mean abs slope: {np.abs(slopes).mean():.6f}")
    
    plateau_batches = np.sum(plateau_mask)
    plateau_pct = 100 * plateau_batches / len(plateau_mask)
    print(f"\nPlateau Detection:")
    print(f"  Batches in plateau: {plateau_batches}/{len(plateau_mask)} ({plateau_pct:.1f}%)")
    
    if plateau_pct > 70:
        print("  ⚠ SEVERE PLATEAU DETECTED!")
    elif plateau_pct > 40:
        print("  ⚡ Moderate plateau - loss is stagnating")
    else:
        print("  ✓ Loss is improving steadily")
    
    # Find where plateau starts
    if plateau_pct > 40:
        plateau_start = np.argmax(plateau_mask)
        print(f"  Plateau starts around batch: {batches[plateau_start + window]}")
        print(f"  Loss at plateau start: {losses[plateau_start + window]:.4f}")

def analyze_gradient_noise(data):
    """Analyze gradient norm variance to detect noise."""
    print("\n" + "="*60)
    print("GRADIENT NOISE ANALYSIS")
    print("="*60)
    
    grad_norms = np.array(data['grad_norm'])
    
    if len(grad_norms) < 10:
        print("⚠ Not enough gradient norm data")
        return
    
    # Compute coefficient of variation (std/mean)
    mean_norm = grad_norms.mean()
    std_norm = grad_norms.std()
    cv = std_norm / mean_norm if mean_norm > 0 else 0
    
    print(f"\nGradient Norm Statistics:")
    print(f"  Mean: {mean_norm:.2f}")
    print(f"  Std dev: {std_norm:.2f}")
    print(f"  Coefficient of variation: {cv:.4f}")
    
    if cv > 0.5:
        print("  ⚠ HIGH VARIANCE: Gradients are very noisy!")
        print("     → Suggests batch composition or loss curvature issues")
    elif cv > 0.3:
        print("  ⚡ Moderate variance: Some noise present")
    else:
        print("  ✓ Low variance: Stable gradient magnitudes")
    
    # Check for gradient norm decay
    window = 50
    if len(grad_norms) > window:
        early_mean = grad_norms[:window].mean()
        late_mean = grad_norms[-window:].mean()
        decay_pct = 100 * (1 - late_mean / early_mean) if early_mean > 0 else 0
        
        print(f"\nGradient Magnitude Decay:")
        print(f"  Early batches (0-{window}): {early_mean:.2f}")
        print(f"  Late batches (-{window}:): {late_mean:.2f}")
        print(f"  Decay: {decay_pct:.1f}%")
        
        if decay_pct > 70:
            print("  ⚠ SEVERE GRADIENT DECAY: Model is vanishing!")
        elif decay_pct > 40:
            print("  ⚡ Significant decay: Gradients getting smaller")

def plot_curvature_diagnostics(data):
    """Generate diagnostic plots."""
    print("\n" + "="*60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss trajectory with moving average
    ax = axes[0, 0]
    batches = data['batch']
    losses = data['loss']
    ax.plot(batches, losses, 'b-', alpha=0.3, label='Raw loss')
    
    window = 20
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(batches[window-1:], moving_avg, 'r-', linewidth=2, label='Moving avg')
    
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient norm over time
    ax = axes[0, 1]
    if data['grad_norm']:
        ax.plot(batches[:len(data['grad_norm'])], data['grad_norm'], 'g-', linewidth=1.5)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Magnitude')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Loss vs Gradient Norm (curvature proxy)
    ax = axes[1, 0]
    if data['grad_norm'] and len(data['grad_norm']) == len(losses):
        ax.scatter(data['grad_norm'], losses, alpha=0.5, s=10)
        ax.set_xlabel('Gradient Norm')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Gradient Magnitude')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Learning rate over time
    ax = axes[1, 1]
    if data['lr']:
        ax.plot(batches[:len(data['lr'])], data['lr'], 'orange', linewidth=1.5)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('LR Schedule')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = Path('loss_curvature_diagnostics.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")
    
    plt.close()

def main():
    log_path = Path('resources/models/GRIM-text/training/logs')
    
    # Use specific log provided by user
    latest_log = log_path / 'training_17663877358178526.log'
    
    if not latest_log.exists():
        # Fallback to most recent
        logs = sorted(log_path.glob('training_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            print("❌ No training logs found!")
            return
        latest_log = logs[0]
    print(f"📊 Analyzing: {latest_log.name}")
    print(f"   Modified: {latest_log.stat().st_mtime}")
    
    # Parse log
    data = parse_training_log(latest_log)
    
    if not data['loss']:
        print("❌ No training data found in log!")
        return
    
    print(f"\n✓ Parsed {len(data['loss'])} batches")
    
    # Run diagnostics
    analyze_loss_trajectory(data)
    analyze_gradient_noise(data)
    analyze_gradient_alignment(data)
    
    # Generate plots
    try:
        plot_curvature_diagnostics(data)
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
Based on the diagnostics above:

1. If gradient alignment is LOW (<0.5):
   → Batch composition is creating conflicting gradients
   → Try: Larger batch size, different batching strategy
   
2. If gradient variance is HIGH (CV >0.5):
   → Loss curvature is irregular (some batches much harder)
   → Try: Different loss function, data filtering
   
3. If gradient decay is SEVERE (>70%):
   → Model is converging to flat region
   → Try: Higher LR, different optimizer (SGD with momentum)
   
4. If plateau starts early (<100 batches):
   → Model capacity or data quality issue
   → Try: More layers, larger d_model, clean training data
""")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
GRIM Training Log Gradient Histogram Viewer
============================================
Modular visualization tool for analyzing gradient statistics from training logs.

Usage:
    python visualize_gradients.py <log_file> [--metric rms|max_abs] [--layer N] [--step step_name]
    
Example:
    python visualize_gradients.py logs/training.log --metric rms
    python visualize_gradients.py logs/training.log --layer 11 --step grad_ffn_input
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# ============================================================================
# STEP 1: Data Structures
# ============================================================================

@dataclass
class GradientSample:
    """Single gradient measurement from a log line."""
    layer: Optional[int]       # None for non-layer metrics (e.g., lm_head)
    step_name: str             # e.g., "grad_ffn_input (after LN2)"
    rms: float
    max_abs: float
    range_min: float
    range_max: float
    is_explosion: bool = False


@dataclass 
class LayerGradients:
    """All gradient samples for a single layer."""
    layer_id: int
    samples: List[GradientSample] = field(default_factory=list)
    
    def get_by_step(self, step_pattern: str) -> List[GradientSample]:
        """Filter samples by step name pattern."""
        return [s for s in self.samples if step_pattern.lower() in s.step_name.lower()]


@dataclass
class GradientLog:
    """Parsed gradient log with all layers and global metrics."""
    global_samples: List[GradientSample] = field(default_factory=list)
    layer_samples: Dict[int, LayerGradients] = field(default_factory=dict)
    explosions: List[GradientSample] = field(default_factory=list)
    
    def get_metric_by_layer(self, step_pattern: str, metric: str = 'rms') -> Dict[int, float]:
        """Extract a specific metric across all layers for a given step."""
        result = {}
        for layer_id, layer_data in sorted(self.layer_samples.items()):
            samples = layer_data.get_by_step(step_pattern)
            if samples:
                val = getattr(samples[0], metric, None)
                if val is not None:
                    result[layer_id] = val
        return result
    
    def get_all_steps(self) -> List[str]:
        """Get unique step names across all layers."""
        steps = set()
        for layer_data in self.layer_samples.values():
            for sample in layer_data.samples:
                steps.add(sample.step_name)
        for sample in self.global_samples:
            steps.add(sample.step_name)
        return sorted(steps)


# ============================================================================
# STEP 2: Log Parser
# ============================================================================

class GradientLogParser:
    """Parses GRIM training logs for gradient statistics."""
    
    # Pattern for: [GradCheck] layer11_grad_ffn_input (after LN2): rms=0.0345, max_abs=0.116, range=[-0.116,0.078]
    GRAD_PATTERN = re.compile(
        r'\[GradCheck\]\s+'
        r'(?:layer(\d+)_)?'                           # Optional layer number
        r'(.+?):\s+'                                  # Step name
        r'rms=([0-9.e+-]+),\s*'                       # RMS value
        r'max_abs=([0-9.e+-]+),\s*'                   # Max absolute value
        r'range=\[([0-9.e+-]+),([0-9.e+-]+)\]'       # Range [min, max]
        r'(?:\s*❌\s*EXPLOSION!)?'                    # Optional explosion marker
    )
    
    @classmethod
    def parse_file(cls, filepath: str) -> GradientLog:
        """Parse a training log file and extract gradient statistics."""
        log = GradientLog()
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                sample = cls._parse_line(line)
                if sample:
                    if sample.layer is not None:
                        if sample.layer not in log.layer_samples:
                            log.layer_samples[sample.layer] = LayerGradients(sample.layer)
                        log.layer_samples[sample.layer].samples.append(sample)
                    else:
                        log.global_samples.append(sample)
                    
                    if sample.is_explosion:
                        log.explosions.append(sample)
        
        return log
    
    @classmethod
    def _parse_line(cls, line: str) -> Optional[GradientSample]:
        """Parse a single log line for gradient info."""
        match = cls.GRAD_PATTERN.search(line)
        if not match:
            return None
        
        layer_str, step_name, rms, max_abs, range_min, range_max = match.groups()
        
        return GradientSample(
            layer=int(layer_str) if layer_str else None,
            step_name=step_name.strip(),
            rms=float(rms),
            max_abs=float(max_abs),
            range_min=float(range_min),
            range_max=float(range_max),
            is_explosion='EXPLOSION' in line
        )


# ============================================================================
# STEP 3: Histogram Renderer (ASCII - no dependencies)
# ============================================================================

class ASCIIHistogram:
    """Simple ASCII histogram renderer with no external dependencies."""
    
    def __init__(self, width: int = 60, height: int = 20):
        self.width = width
        self.height = height
    
    def render_horizontal(self, data: Dict[int, float], title: str = "", 
                          label: str = "Layer", value_label: str = "Value") -> str:
        """Render horizontal bar chart."""
        if not data:
            return "No data to display."
        
        lines = []
        if title:
            lines.append(f"\n{'='*self.width}")
            lines.append(f"  {title}")
            lines.append(f"{'='*self.width}\n")
        
        max_val = max(abs(v) for v in data.values()) if data else 1
        max_key_len = max(len(str(k)) for k in data.keys())
        bar_width = self.width - max_key_len - 20  # Space for label and value
        
        for key, value in sorted(data.items(), reverse=True):  # Layer 11 -> 0
            bar_len = int((abs(value) / max_val) * bar_width) if max_val > 0 else 0
            bar = '█' * bar_len
            
            # Mark explosions in red (ANSI)
            if abs(value) > 100:
                bar = f"\033[91m{bar}\033[0m"  # Red
            elif abs(value) > 10:
                bar = f"\033[93m{bar}\033[0m"  # Yellow
            
            lines.append(f"  {label} {key:>{max_key_len}}: {bar} {value:.4g}")
        
        return '\n'.join(lines)
    
    def render_comparison(self, datasets: Dict[str, Dict[int, float]], 
                          title: str = "") -> str:
        """Render multiple metrics side by side."""
        if not datasets:
            return "No data to display."
        
        lines = []
        if title:
            lines.append(f"\n{'='*80}")
            lines.append(f"  {title}")
            lines.append(f"{'='*80}\n")
        
        # Get all layers
        all_layers = set()
        for data in datasets.values():
            all_layers.update(data.keys())
        
        # Header
        header = "  Layer |"
        for name in datasets.keys():
            header += f" {name:>12} |"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        
        # Data rows
        for layer in sorted(all_layers, reverse=True):
            row = f"  {layer:>5} |"
            for data in datasets.values():
                val = data.get(layer, 0)
                # Color code based on magnitude
                if abs(val) > 100:
                    row += f" \033[91m{val:>12.4g}\033[0m |"
                elif abs(val) > 10:
                    row += f" \033[93m{val:>12.4g}\033[0m |"
                else:
                    row += f" {val:>12.4g} |"
            lines.append(row)
        
        return '\n'.join(lines)


# ============================================================================
# STEP 4: Matplotlib Histogram (optional, for rich visuals)
# ============================================================================

class MatplotlibHistogram:
    """Rich histogram renderer using matplotlib."""
    
    def __init__(self):
        self._plt = None
        self._np = None
    
    def _ensure_imports(self):
        """Lazy import matplotlib and numpy."""
        if self._plt is None:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                self._plt = plt
                self._np = np
            except ImportError:
                raise ImportError("matplotlib and numpy required. Install with: pip install matplotlib numpy")
    
    def plot_layer_gradient_flow(self, log: GradientLog, step_pattern: str = "grad_ffn_input",
                                  metric: str = 'rms', save_path: Optional[str] = None):
        """Plot gradient magnitude across layers (gradient flow visualization)."""
        self._ensure_imports()
        plt, np = self._plt, self._np
        
        data = log.get_metric_by_layer(step_pattern, metric)
        if not data:
            print(f"No data found for step pattern: {step_pattern}")
            return
        
        layers = sorted(data.keys(), reverse=True)
        values = [data[l] for l in layers]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color bars by magnitude
        colors = []
        for v in values:
            if abs(v) > 100:
                colors.append('#ff4444')  # Red - explosion
            elif abs(v) > 10:
                colors.append('#ffaa00')  # Orange - warning
            elif abs(v) > 1:
                colors.append('#ffff00')  # Yellow - elevated
            else:
                colors.append('#44ff44')  # Green - healthy
        
        bars = ax.barh([f"Layer {l}" for l in layers], values, color=colors, edgecolor='black')
        
        ax.set_xlabel(f'{metric.upper()} Value')
        ax.set_title(f'Gradient Flow: {step_pattern}\n(Red=Explosion, Orange=Warning, Yellow=Elevated, Green=Healthy)')
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label='Healthy threshold')
        ax.axvline(x=10, color='orange', linestyle='--', alpha=0.5, label='Warning threshold')
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Explosion threshold')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.2e}', va='center', fontsize=8)
        
        ax.set_xscale('symlog', linthresh=0.1)
        ax.legend(loc='lower right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        else:
            plt.show()
    
    def plot_gradient_heatmap(self, log: GradientLog, metric: str = 'rms',
                               save_path: Optional[str] = None):
        """Plot heatmap of all gradients across layers and steps."""
        self._ensure_imports()
        plt, np = self._plt, self._np
        
        # Collect all step names and layers
        steps = []
        step_keywords = ['grad_ffn_input', 'grad_ffn_hidden', 'grad_Q', 'grad_K', 'grad_V', 
                        'grad_before_Wo', 'grad_after_Wo', 'ffn_w1_grads', 'attn_qkv_weight']
        
        for kw in step_keywords:
            for sample in log.get_all_steps():
                if kw in sample and sample not in steps:
                    steps.append(sample)
                    break
        
        if not steps:
            steps = log.get_all_steps()[:10]  # Fallback to first 10
        
        layers = sorted(log.layer_samples.keys(), reverse=True)
        
        # Build data matrix
        data = np.zeros((len(layers), len(steps)))
        for i, layer in enumerate(layers):
            for j, step in enumerate(steps):
                samples = log.layer_samples[layer].get_by_step(step)
                if samples:
                    data[i, j] = getattr(samples[0], metric, 0)
        
        # Apply log scale for better visualization
        data_log = np.sign(data) * np.log10(np.abs(data) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        im = ax.imshow(data_log, cmap='RdYlGn_r', aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([s[:20] + '...' if len(s) > 20 else s for s in steps], 
                          rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"Layer {l}" for l in layers])
        
        ax.set_title(f'Gradient Heatmap ({metric.upper()}) - Log Scale')
        plt.colorbar(im, ax=ax, label=f'log10({metric})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        else:
            plt.show()
    
    def plot_explosion_timeline(self, log: GradientLog, save_path: Optional[str] = None):
        """Plot gradient magnitude progression showing where explosion occurs."""
        self._ensure_imports()
        plt, np = self._plt, self._np
        
        # Get grad_ffn_input progression (key indicator)
        data = log.get_metric_by_layer('grad_ffn_input', 'rms')
        
        if not data:
            print("No grad_ffn_input data found")
            return
        
        layers = sorted(data.keys(), reverse=True)
        values = [data[l] for l in layers]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Linear scale
        ax1.plot(layers, values, 'b-o', markersize=8, linewidth=2)
        ax1.fill_between(layers, values, alpha=0.3)
        ax1.set_ylabel('RMS Gradient')
        ax1.set_xlabel('Layer (backward direction →)')
        ax1.set_title('Gradient Magnitude Progression (Linear Scale)')
        ax1.axhline(y=100, color='red', linestyle='--', label='Explosion threshold')
        ax1.legend()
        ax1.invert_xaxis()
        ax1.grid(True, alpha=0.3)
        
        # Log scale
        ax2.semilogy(layers, values, 'r-o', markersize=8, linewidth=2)
        ax2.fill_between(layers, values, alpha=0.3)
        ax2.set_ylabel('RMS Gradient (log)')
        ax2.set_xlabel('Layer (backward direction →)')
        ax2.set_title('Gradient Magnitude Progression (Log Scale)')
        ax2.axhline(y=100, color='red', linestyle='--', label='Explosion threshold')
        ax2.axhline(y=1, color='green', linestyle='--', label='Healthy')
        ax2.legend()
        ax2.invert_xaxis()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        else:
            plt.show()


# ============================================================================
# STEP 5: Main CLI Interface
# ============================================================================

def print_summary(log: GradientLog):
    """Print a summary of the gradient log."""
    print("\n" + "="*60)
    print("  GRADIENT LOG SUMMARY")
    print("="*60)
    
    print(f"\n  Layers found: {sorted(log.layer_samples.keys())}")
    print(f"  Global metrics: {len(log.global_samples)}")
    print(f"  Explosions detected: {len(log.explosions)}")
    
    if log.explosions:
        print("\n  ⚠️  EXPLOSION LOCATIONS:")
        for exp in log.explosions:
            layer_str = f"Layer {exp.layer}" if exp.layer is not None else "Global"
            print(f"      - {layer_str}: {exp.step_name} (rms={exp.rms:.4g})")
    
    print(f"\n  Available steps: {len(log.get_all_steps())}")
    for step in log.get_all_steps()[:10]:
        print(f"      - {step}")
    if len(log.get_all_steps()) > 10:
        print(f"      ... and {len(log.get_all_steps()) - 10} more")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GRIM Training Log Gradient Visualizer')
    parser.add_argument('logfile', help='Path to training log file')
    parser.add_argument('--metric', choices=['rms', 'max_abs'], default='rms',
                       help='Metric to visualize (default: rms)')
    parser.add_argument('--step', default='grad_ffn_input',
                       help='Step pattern to filter (default: grad_ffn_input)')
    parser.add_argument('--ascii', action='store_true',
                       help='Use ASCII output instead of matplotlib')
    parser.add_argument('--save', help='Save plot to file instead of displaying')
    parser.add_argument('--heatmap', action='store_true',
                       help='Show heatmap of all gradients')
    parser.add_argument('--timeline', action='store_true',
                       help='Show explosion timeline')
    parser.add_argument('--compare', nargs='+',
                       help='Compare multiple step patterns')
    
    args = parser.parse_args()
    
    # Parse log file
    print(f"Parsing: {args.logfile}")
    log = GradientLogParser.parse_file(args.logfile)
    
    # Print summary
    print_summary(log)
    
    if args.ascii:
        # ASCII visualization
        renderer = ASCIIHistogram(width=70)
        
        if args.compare:
            datasets = {}
            for step in args.compare:
                datasets[step[:15]] = log.get_metric_by_layer(step, args.metric)
            print(renderer.render_comparison(datasets, f"Comparing: {', '.join(args.compare)}"))
        else:
            data = log.get_metric_by_layer(args.step, args.metric)
            print(renderer.render_horizontal(data, 
                                            title=f"{args.step} ({args.metric.upper()})",
                                            value_label=args.metric.upper()))
    else:
        # Matplotlib visualization
        try:
            renderer = MatplotlibHistogram()
            
            if args.heatmap:
                renderer.plot_gradient_heatmap(log, args.metric, args.save)
            elif args.timeline:
                renderer.plot_explosion_timeline(log, args.save)
            else:
                renderer.plot_layer_gradient_flow(log, args.step, args.metric, args.save)
                
        except ImportError as e:
            print(f"\n⚠️  {e}")
            print("Falling back to ASCII mode...\n")
            renderer = ASCIIHistogram(width=70)
            data = log.get_metric_by_layer(args.step, args.metric)
            print(renderer.render_horizontal(data, 
                                            title=f"{args.step} ({args.metric.upper()})"))


if __name__ == '__main__':
    main()

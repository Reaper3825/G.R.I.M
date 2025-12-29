#!/usr/bin/env python3
"""
Litmus Test for GRIM-text Training Gradients
Monitors critical gradient flow metrics in real-time:
1. layerX_ln2_gamma_grads RMS (layernorm 2 gamma gradients)
2. layerX_ln2_gamma range (layernorm 2 gamma weights)
3. layerX_grad_ffn_input RMS (FFN input gradients after LN2)
"""

import re
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

class GradientLitmusTest:
    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.patterns = {
            'ln2_gamma_grads': re.compile(r'\[GradCheck\] layer(\d+)_ln2_gamma_grads: rms=([\d.e+-]+), max_abs=([\d.e+-]+), range=\[([-\d.e+-]+),([-\d.e+-]+)\]'),
            'ln2_gamma': re.compile(r'\[GradCheck\] layer(\d+)_ln2_gamma: rms=([\d.e+-]+), max_abs=([\d.e+-]+), range=\[([-\d.e+-]+),([-\d.e+-]+)\]'),
            'grad_ffn_input': re.compile(r'\[GradCheck\] layer(\d+)_grad_ffn_input \(after LN2\): rms=([\d.e+-]+), max_abs=([\d.e+-]+)'),
            'step': re.compile(r'\[Step (\d+)\]')
        }
        
        # Thresholds for health checks
        self.thresholds = {
            'ln2_gamma_grads_rms': {'min': 1e-8, 'max': 10.0, 'warn_max': 1.0},
            'ln2_gamma_range': {'min': 0.0, 'max': 10.0, 'warn_max': 5.0},
            'grad_ffn_input_rms': {'min': 1e-8, 'max': 100.0, 'warn_max': 10.0}
        }
        
        # Storage for metrics per layer
        self.metrics = defaultdict(lambda: {
            'ln2_gamma_grads_rms': [],
            'ln2_gamma_range': [],
            'grad_ffn_input_rms': []
        })
        
        self.current_step = 0
        self.num_layers = 12  # GRIM-text default
    
    def parse_line(self, line: str) -> None:
        """Parse a log line and extract gradient metrics."""
        # Track current step
        step_match = self.patterns['step'].search(line)
        if step_match:
            self.current_step = int(step_match.group(1))
        
        # Extract ln2_gamma_grads
        match = self.patterns['ln2_gamma_grads'].search(line)
        if match:
            layer = int(match.group(1))
            rms = float(match.group(2))
            self.metrics[layer]['ln2_gamma_grads_rms'].append((self.current_step, rms))
        
        # Extract ln2_gamma (weights) range
        match = self.patterns['ln2_gamma'].search(line)
        if match:
            layer = int(match.group(1))
            min_val = float(match.group(4))
            max_val = float(match.group(5))
            range_val = max_val - min_val
            self.metrics[layer]['ln2_gamma_range'].append((self.current_step, range_val, min_val, max_val))
        
        # Extract grad_ffn_input
        match = self.patterns['grad_ffn_input'].search(line)
        if match:
            layer = int(match.group(1))
            rms = float(match.group(2))
            self.metrics[layer]['grad_ffn_input_rms'].append((self.current_step, rms))
    
    def check_health(self, metric_name: str, value: float) -> Tuple[str, str]:
        """
        Check if a metric value is healthy.
        Returns (status, color) where status is 'GOOD', 'WARN', or 'BAD'.
        """
        thresh = self.thresholds.get(metric_name, {})
        min_val = thresh.get('min', 0)
        max_val = thresh.get('max', float('inf'))
        warn_max = thresh.get('warn_max', max_val)
        
        if value < min_val or value > max_val:
            return 'BAD', RED
        elif value > warn_max:
            return 'WARN', YELLOW
        else:
            return 'GOOD', GREEN
    
    def format_value(self, value: float, metric_name: str) -> str:
        """Format a value with color based on health."""
        status, color = self.check_health(metric_name, value)
        if abs(value) < 1e-3 or abs(value) > 1e3:
            formatted = f"{value:.3e}"
        else:
            formatted = f"{value:.6f}"
        return f"{color}{formatted}{RESET}"
    
    def print_summary(self) -> None:
        """Print a summary of the current gradient statistics."""
        print(f"\n{BOLD}{CYAN}{'='*80}{RESET}")
        print(f"{BOLD}{CYAN}GRADIENT LITMUS TEST - Step {self.current_step}{RESET}")
        print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")
        
        for layer in sorted(self.metrics.keys()):
            layer_data = self.metrics[layer]
            
            print(f"{BOLD}Layer {layer:2d}:{RESET}")
            
            # ln2_gamma_grads RMS
            if layer_data['ln2_gamma_grads_rms']:
                recent_values = [v for s, v in layer_data['ln2_gamma_grads_rms'][-5:]]
                avg = statistics.mean(recent_values)
                latest = recent_values[-1]
                print(f"  ln2_gamma_grads RMS: {self.format_value(latest, 'ln2_gamma_grads_rms')} (avg: {avg:.3e})")
            
            # ln2_gamma range
            if layer_data['ln2_gamma_range']:
                recent = layer_data['ln2_gamma_range'][-1]
                step, range_val, min_val, max_val = recent
                print(f"  ln2_gamma range:     {self.format_value(range_val, 'ln2_gamma_range')} (min: {min_val:.4f}, max: {max_val:.4f})")
            
            # grad_ffn_input RMS
            if layer_data['grad_ffn_input_rms']:
                recent_values = [v for s, v in layer_data['grad_ffn_input_rms'][-5:]]
                avg = statistics.mean(recent_values)
                latest = recent_values[-1]
                print(f"  grad_ffn_input RMS:  {self.format_value(latest, 'grad_ffn_input_rms')} (avg: {avg:.3e})")
            
            print()
        
        self.print_health_warnings()
    
    def print_health_warnings(self) -> None:
        """Print any health warnings based on gradient statistics."""
        warnings = []
        errors = []
        
        for layer in sorted(self.metrics.keys()):
            layer_data = self.metrics[layer]
            
            # Check ln2_gamma_grads RMS
            if layer_data['ln2_gamma_grads_rms']:
                latest = layer_data['ln2_gamma_grads_rms'][-1][1]
                status, _ = self.check_health('ln2_gamma_grads_rms', latest)
                if status == 'BAD':
                    errors.append(f"Layer {layer}: ln2_gamma_grads RMS = {latest:.3e} (OUT OF BOUNDS)")
                elif status == 'WARN':
                    warnings.append(f"Layer {layer}: ln2_gamma_grads RMS = {latest:.3e} (HIGH)")
            
            # Check ln2_gamma range
            if layer_data['ln2_gamma_range']:
                range_val = layer_data['ln2_gamma_range'][-1][1]
                status, _ = self.check_health('ln2_gamma_range', range_val)
                if status == 'BAD':
                    errors.append(f"Layer {layer}: ln2_gamma range = {range_val:.3e} (OUT OF BOUNDS)")
                elif status == 'WARN':
                    warnings.append(f"Layer {layer}: ln2_gamma range = {range_val:.3e} (HIGH)")
            
            # Check grad_ffn_input RMS
            if layer_data['grad_ffn_input_rms']:
                latest = layer_data['grad_ffn_input_rms'][-1][1]
                status, _ = self.check_health('grad_ffn_input_rms', latest)
                if status == 'BAD':
                    errors.append(f"Layer {layer}: grad_ffn_input RMS = {latest:.3e} (OUT OF BOUNDS)")
                elif status == 'WARN':
                    warnings.append(f"Layer {layer}: grad_ffn_input RMS = {latest:.3e} (HIGH)")
        
        if errors:
            print(f"{BOLD}{RED}ERRORS:{RESET}")
            for err in errors:
                print(f"  ❌ {err}")
            print()
        
        if warnings:
            print(f"{BOLD}{YELLOW}WARNINGS:{RESET}")
            for warn in warnings:
                print(f"  ⚠️  {warn}")
            print()
        
        if not errors and not warnings:
            print(f"{BOLD}{GREEN}✓ All gradient metrics within healthy ranges{RESET}\n")
    
    def analyze_trends(self) -> None:
        """Analyze trends in gradient metrics over time."""
        print(f"{BOLD}{CYAN}TREND ANALYSIS:{RESET}\n")
        
        for layer in sorted(self.metrics.keys()):
            layer_data = self.metrics[layer]
            
            print(f"{BOLD}Layer {layer:2d}:{RESET}")
            
            # Analyze ln2_gamma_grads RMS trend
            if len(layer_data['ln2_gamma_grads_rms']) >= 10:
                recent = [v for s, v in layer_data['ln2_gamma_grads_rms'][-10:]]
                early = [v for s, v in layer_data['ln2_gamma_grads_rms'][:10]]
                recent_avg = statistics.mean(recent)
                early_avg = statistics.mean(early)
                change_pct = ((recent_avg - early_avg) / (early_avg + 1e-10)) * 100
                
                trend_arrow = "↑" if change_pct > 10 else "↓" if change_pct < -10 else "→"
                trend_color = RED if abs(change_pct) > 50 else YELLOW if abs(change_pct) > 20 else GREEN
                
                print(f"  ln2_gamma_grads RMS: {trend_color}{trend_arrow} {change_pct:+.1f}%{RESET} (early: {early_avg:.3e}, recent: {recent_avg:.3e})")
            
            # Analyze grad_ffn_input RMS trend
            if len(layer_data['grad_ffn_input_rms']) >= 10:
                recent = [v for s, v in layer_data['grad_ffn_input_rms'][-10:]]
                early = [v for s, v in layer_data['grad_ffn_input_rms'][:10]]
                recent_avg = statistics.mean(recent)
                early_avg = statistics.mean(early)
                change_pct = ((recent_avg - early_avg) / (early_avg + 1e-10)) * 100
                
                trend_arrow = "↑" if change_pct > 10 else "↓" if change_pct < -10 else "→"
                trend_color = RED if abs(change_pct) > 50 else YELLOW if abs(change_pct) > 20 else GREEN
                
                print(f"  grad_ffn_input RMS:  {trend_color}{trend_arrow} {change_pct:+.1f}%{RESET} (early: {early_avg:.3e}, recent: {recent_avg:.3e})")
            
            print()
    
    def run_live_monitoring(self, update_interval: float = 2.0) -> None:
        """Monitor the log file in real-time."""
        print(f"{BOLD}{CYAN}Starting live gradient monitoring...{RESET}")
        print(f"Watching: {self.log_path}\n")
        
        if not self.log_path.exists():
            print(f"{RED}Error: Log file not found: {self.log_path}{RESET}")
            sys.exit(1)
        
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read existing content
            for line in f:
                self.parse_line(line)
            
            print(f"{GREEN}Initial parse complete. Monitoring for new entries...{RESET}\n")
            self.print_summary()
            
            # Continue monitoring
            last_update = time.time()
            while True:
                line = f.readline()
                if line:
                    self.parse_line(line)
                    
                    # Update display periodically
                    if time.time() - last_update > update_interval:
                        print("\033[2J\033[H")  # Clear screen
                        self.print_summary()
                        last_update = time.time()
                else:
                    time.sleep(0.1)
    
    def run_batch_analysis(self) -> None:
        """Analyze a complete log file."""
        print(f"{BOLD}{CYAN}Analyzing log file: {self.log_path}{RESET}\n")
        
        if not self.log_path.exists():
            print(f"{RED}Error: Log file not found: {self.log_path}{RESET}")
            sys.exit(1)
        
        with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                self.parse_line(line)
        
        self.print_summary()
        self.analyze_trends()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GRIM-text Gradient Litmus Test')
    parser.add_argument('log_file', nargs='?', help='Path to training log file')
    parser.add_argument('--live', action='store_true', help='Monitor log file in real-time')
    parser.add_argument('--interval', type=float, default=2.0, help='Update interval for live monitoring (seconds)')
    
    args = parser.parse_args()
    
    # Find latest log file if not specified
    if not args.log_file:
        log_dir = Path('resources/models/GRIM-text/training/logs')
        if not log_dir.exists():
            print(f"{RED}Error: Log directory not found: {log_dir}{RESET}")
            sys.exit(1)
        
        log_files = sorted(log_dir.glob('training_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            print(f"{RED}Error: No training log files found in {log_dir}{RESET}")
            sys.exit(1)
        
        args.log_file = str(log_files[0])
        print(f"{CYAN}Using latest log file: {args.log_file}{RESET}\n")
    
    litmus = GradientLitmusTest(args.log_file)
    
    if args.live:
        try:
            litmus.run_live_monitoring(args.interval)
        except KeyboardInterrupt:
            print(f"\n{YELLOW}Monitoring stopped by user{RESET}")
    else:
        litmus.run_batch_analysis()


if __name__ == '__main__':
    main()

"""
Diagnostic tool to analyze GRIM training gradient explosions
Analyzes logs to find root cause of numerical instability
"""

import re
import json
from pathlib import Path
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    # Fallback to basic statistics if numpy not available
    class np:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        @staticmethod
        def median(data):
            sorted_data = sorted(data)
            n = len(sorted_data)
            if n == 0:
                return 0
            if n % 2 == 0:
                return (sorted_data[n//2-1] + sorted_data[n//2]) / 2
            return sorted_data[n//2]

class GradientExplosionDiagnostics:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.steps = []
        self.losses = []
        self.grad_norms = []
        self.learning_rates = []
        self.explosions = []
        
    def parse_log(self):
        """Parse training log and extract metrics"""
        print(f"📖 Parsing log: {self.log_file}")
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Parse step info
                step_match = re.search(r'Step (\d+)', line)
                loss_match = re.search(r'Loss: ([\d.e+-]+)', line)
                grad_match = re.search(r'(?:GradNorm.*(?:Norm|RMS):|grad_(?:rms|norm)=)([\d.e+-]+)', line)
                lr_match = re.search(r'LR: ([\d.e+-]+)', line)
                
                if step_match and loss_match:
                    step = int(step_match.group(1))
                    loss = float(loss_match.group(1))
                    
                    self.steps.append(step)
                    self.losses.append(loss)
                    
                    # Check for explosion
                    if 'EXPLOSION' in line:
                        self.explosions.append(step)
                
                if grad_match:
                    grad_norm = float(grad_match.group(1))
                    self.grad_norms.append(grad_norm)
                
                if lr_match:
                    lr = float(lr_match.group(1))
                    self.learning_rates.append(lr)
        
        print(f"✓ Parsed {len(self.steps)} training steps")
        print(f"✓ Found {len(self.explosions)} gradient explosions")
    
    def analyze_explosion_pattern(self):
        """Analyze when and why gradient explosion occurs"""
        print(f"\n{'='*60}")
        print("🔍 GRADIENT EXPLOSION ANALYSIS")
        print(f"{'='*60}\n")
        
        if not self.explosions:
            print("✅ No gradient explosions detected!")
            return
        
        first_explosion = self.explosions[0]
        print(f"💥 First explosion at step: {first_explosion}")
        
        # Find the window before explosion
        explosion_idx = self.steps.index(first_explosion) if first_explosion in self.steps else None
        
        if explosion_idx:
            window_start = max(0, explosion_idx - 10)
            window_end = min(len(self.steps), explosion_idx + 5)
            
            print(f"\n📊 Loss progression (steps {window_start} to {window_end}):")
            for i in range(window_start, window_end):
                marker = "💥" if self.steps[i] in self.explosions else "  "
                print(f"  {marker} Step {self.steps[i]:3d}: Loss = {self.losses[i]:.6f}")
            
            # Analyze loss growth rate
            if explosion_idx > 5:
                recent_losses = self.losses[explosion_idx-5:explosion_idx]
                growth_rates = []
                for i in range(1, len(recent_losses)):
                    if recent_losses[i-1] > 0:
                        growth_rate = recent_losses[i] / recent_losses[i-1]
                        growth_rates.append(growth_rate)
                
                if growth_rates:
                    avg_growth = np.mean(growth_rates)
                    print(f"\n📈 Average loss growth rate (last 5 steps): {avg_growth:.3f}x")
                    
                    if avg_growth > 1.1:
                        print(f"⚠️  CAUSE: Loss is growing exponentially before explosion!")
                        print(f"   Recommendation: Reduce learning rate or add gradient clipping")
    
    def analyze_gradient_behavior(self):
        """Analyze gradient norm progression"""
        print(f"\n{'='*60}")
        print("📉 GRADIENT NORM ANALYSIS")
        print(f"{'='*60}\n")
        
        if not self.grad_norms:
            print("⚠️  No gradient norms found in log")
            return
        
        print(f"Gradient norm statistics:")
        print(f"  Min:    {min(self.grad_norms):.2f}")
        print(f"  Max:    {max(self.grad_norms):.2f}")
        print(f"  Mean:   {np.mean(self.grad_norms):.2f}")
        print(f"  Median: {np.median(self.grad_norms):.2f}")
        
        # Find sudden spikes
        spikes = []
        for i in range(1, len(self.grad_norms)):
            if self.grad_norms[i] > self.grad_norms[i-1] * 2:
                spikes.append((i, self.grad_norms[i-1], self.grad_norms[i]))
        
        if spikes:
            print(f"\n⚡ Found {len(spikes)} gradient spikes (2x jumps):")
            for idx, before, after in spikes[:5]:
                print(f"  Step ~{idx}: {before:.2f} → {after:.2f} ({after/before:.1f}x)")
    
    def diagnose_root_cause(self):
        """Provide root cause analysis and recommendations"""
        print(f"\n{'='*60}")
        print("🔬 ROOT CAUSE DIAGNOSIS")
        print(f"{'='*60}\n")
        
        issues_found = []
        
        # Check 1: High initial loss
        if self.losses and self.losses[0] > 10:
            issues_found.append({
                'issue': 'High initial loss',
                'value': self.losses[0],
                'severity': 'HIGH',
                'cause': 'Poor weight initialization or logits scale',
                'fix': 'Check Xavier initialization, verify logit computation'
            })
        
        # Check 2: Loss growing consistently
        if len(self.losses) > 10:
            first_10_avg = np.mean(self.losses[:10])
            next_10_avg = np.mean(self.losses[10:20]) if len(self.losses) > 20 else np.mean(self.losses[10:])
            
            if next_10_avg > first_10_avg * 1.2:
                issues_found.append({
                    'issue': 'Loss growing (not decreasing)',
                    'value': f"{first_10_avg:.2f} → {next_10_avg:.2f}",
                    'severity': 'CRITICAL',
                    'cause': 'Learning rate too high, incorrect gradients, or numerical instability',
                    'fix': 'Reduce LR to 1e-5, check backward pass, add loss scaling'
                })
        
        # Check 3: Gradient explosion pattern
        if self.explosions and len(self.explosions) > 0:
            explosion_step = self.explosions[0]
            if explosion_step < 100:
                issues_found.append({
                    'issue': 'Early gradient explosion',
                    'value': f'Step {explosion_step}',
                    'severity': 'CRITICAL',
                    'cause': 'Fundamental numerical instability in forward/backward pass',
                    'fix': 'Add gradient value clamping BEFORE norm computation, verify softmax numerics'
                })
        
        # Check 4: Gradient norm trend
        if len(self.grad_norms) > 20:
            early_norms = self.grad_norms[:10]
            later_norms = self.grad_norms[10:20]
            
            early_avg = np.mean(early_norms)
            later_avg = np.mean(later_norms)
            
            if later_avg > early_avg * 2:
                issues_found.append({
                    'issue': 'Gradient norms increasing',
                    'value': f"{early_avg:.1f} → {later_avg:.1f}",
                    'severity': 'HIGH',
                    'cause': 'Weights becoming unstable, possible feedback loop',
                    'fix': 'Tighter gradient clipping (< 1.0), weight normalization, or gradient checkpointing'
                })
        
        # Report findings
        if not issues_found:
            print("✅ No obvious issues detected in available data")
            return
        
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. 🔴 {issue['severity']}: {issue['issue']}")
            print(f"   Value: {issue['value']}")
            print(f"   Likely cause: {issue['cause']}")
            print(f"   Recommended fix: {issue['fix']}")
            print()
    
    def suggest_fixes(self):
        """Provide actionable fixes"""
        print(f"\n{'='*60}")
        print("💡 RECOMMENDED FIXES (Priority Order)")
        print(f"{'='*60}\n")
        
        fixes = [
            {
                'priority': 1,
                'fix': 'Reduce learning rate to 1e-5 (currently 1e-4)',
                'reason': 'Current LR is 10x too high for your model size',
                'code': 'float learning_rate = 0.00001f;  // Was 0.0001f'
            },
            {
                'priority': 2,
                'fix': 'Add loss scaling to prevent underflow in gradients',
                'reason': 'Large vocabulary (1330) causes small softmax values',
                'code': '''// In computeLossBatch:
float loss_scale = 10.0f;
float scaled_loss = loss * loss_scale;
// ... backward pass ...
model->scaleGradients(1.0f / loss_scale);  // Unscale after backward'''
            },
            {
                'priority': 3,
                'fix': 'Tighten gradient clipping to 1.0 (currently 1000.0)',
                'reason': 'Current clip norm is way too permissive',
                'code': 'float grad_clip_norm = 1.0f;  // Was 1000.0f'
            },
            {
                'priority': 4,
                'fix': 'Add gradient value clamping BEFORE norm computation',
                'reason': 'Individual gradient values may be inf/nan before clipping',
                'code': '''// In train_gpu.cu, before computeGradNorm():
model->clampGradients(-10.0f, 10.0f);  // Clamp individual values
float grad_norm = model->computeGradNorm();'''
            },
            {
                'priority': 5,
                'fix': 'Add logit scaling in forward pass',
                'reason': 'Prevents extreme softmax values',
                'code': '''// After computing logits:
float logit_scale = 1.0f / sqrtf(cfg.d_model);
scale_logits_kernel<<<grid, block>>>(logits, logit_scale);'''
            }
        ]
        
        for fix in fixes:
            print(f"Priority {fix['priority']}: {fix['fix']}")
            print(f"  Why: {fix['reason']}")
            print(f"  Code:")
            for line in fix['code'].split('\n'):
                print(f"    {line}")
            print()

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║  GRIM Gradient Explosion Diagnostics                     ║
║  Root cause analysis for training instability            ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Find most recent log
    log_dir = Path("resources/models/GRIM-text/training/logs")
    log_files = sorted(log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not log_files:
        print("❌ No training logs found!")
        return
    
    log_file = log_files[0]
    print(f"📄 Analyzing: {log_file.name}\n")
    
    diagnostics = GradientExplosionDiagnostics(log_file)
    diagnostics.parse_log()
    diagnostics.analyze_explosion_pattern()
    diagnostics.analyze_gradient_behavior()
    diagnostics.diagnose_root_cause()
    diagnostics.suggest_fixes()
    
    print(f"\n{'='*60}")
    print("✅ Diagnostic complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

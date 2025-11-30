"""
GRIM Training Evaluation Framework
Comprehensive evaluation system for tracking training stability and gradient health
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class GradientHealthEvaluator:
    """Evaluates gradient stability and detects explosions"""
    
    def __init__(self, explosion_threshold=100.0, warning_threshold=50.0):
        self.explosion_threshold = explosion_threshold
        self.warning_threshold = warning_threshold
    
    def __call__(self, *, gradient_norm: float, step: int, **kwargs) -> Dict[str, Any]:
        """Evaluate gradient health"""
        if gradient_norm > self.explosion_threshold:
            status = "EXPLOSION"
            severity = "critical"
            recommendation = "Skip batch, check backward pass for numerical instability"
        elif gradient_norm > self.warning_threshold:
            status = "WARNING"
            severity = "high"
            recommendation = "Monitor closely, may need tighter clipping"
        else:
            status = "HEALTHY"
            severity = "normal"
            recommendation = "Continue training"
        
        return {
            "gradient_health_status": status,
            "gradient_norm": gradient_norm,
            "severity": severity,
            "recommendation": recommendation,
            "step": step
        }


class LossConvergenceEvaluator:
    """Evaluates if loss is converging or diverging"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.loss_history = []
    
    def __call__(self, *, loss: float, step: int, **kwargs) -> Dict[str, Any]:
        """Evaluate loss convergence"""
        self.loss_history.append(loss)
        
        if len(self.loss_history) < self.window_size:
            return {
                "convergence_status": "INSUFFICIENT_DATA",
                "loss": loss,
                "trend": "unknown",
                "step": step
            }
        
        # Keep only recent history
        self.loss_history = self.loss_history[-self.window_size:]
        
        # Calculate trend
        first_half = sum(self.loss_history[:self.window_size//2]) / (self.window_size//2)
        second_half = sum(self.loss_history[self.window_size//2:]) / (self.window_size - self.window_size//2)
        
        if second_half < first_half * 0.95:
            status = "CONVERGING"
            trend = "improving"
        elif second_half > first_half * 1.05:
            status = "DIVERGING"
            trend = "worsening"
        else:
            status = "STABLE"
            trend = "flat"
        
        variance = sum((l - sum(self.loss_history)/len(self.loss_history))**2 for l in self.loss_history) / len(self.loss_history)
        
        return {
            "convergence_status": status,
            "loss": loss,
            "trend": trend,
            "loss_variance": variance,
            "step": step
        }


class NumericalStabilityEvaluator:
    """Checks for inf/nan and extreme values"""
    
    def __init__(self):
        pass
    
    def __call__(self, *, loss: float, gradient_norm: float, **kwargs) -> Dict[str, Any]:
        """Evaluate numerical stability"""
        issues = []
        
        # Check for inf/nan
        if not (float('-inf') < loss < float('inf')):
            issues.append("loss_inf_or_nan")
        if not (float('-inf') < gradient_norm < float('inf')):
            issues.append("gradient_norm_inf_or_nan")
        
        # Check for extreme values
        if loss > 1000:
            issues.append("loss_extremely_high")
        if gradient_norm > 1000:
            issues.append("gradient_norm_extremely_high")
        
        is_stable = len(issues) == 0
        
        return {
            "is_numerically_stable": is_stable,
            "issues": issues if issues else ["none"],
            "loss": loss,
            "gradient_norm": gradient_norm
        }


class TrainingProgressEvaluator:
    """Evaluates overall training progress and health"""
    
    def __init__(self):
        self.total_steps = 0
        self.successful_steps = 0
        self.skipped_steps = 0
        self.explosion_count = 0
    
    def __call__(self, *, step: int, gradient_health_status: str, **kwargs) -> Dict[str, Any]:
        """Evaluate training progress"""
        self.total_steps = step
        
        if gradient_health_status == "EXPLOSION":
            self.explosion_count += 1
            self.skipped_steps += 1
        elif gradient_health_status == "HEALTHY":
            self.successful_steps += 1
        
        success_rate = self.successful_steps / max(step, 1)
        
        if success_rate > 0.9:
            progress_status = "EXCELLENT"
        elif success_rate > 0.7:
            progress_status = "GOOD"
        elif success_rate > 0.5:
            progress_status = "FAIR"
        else:
            progress_status = "POOR"
        
        return {
            "training_progress_status": progress_status,
            "total_steps": self.total_steps,
            "successful_steps": self.successful_steps,
            "skipped_steps": self.skipped_steps,
            "explosion_count": self.explosion_count,
            "success_rate": success_rate
        }


class TrainingDataExtractor:
    """Extracts training metrics from log files and converts to JSONL"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.data = []
    
    def extract_metrics(self) -> List[Dict[str, Any]]:
        """Extract training metrics from log file"""
        print(f"📖 Extracting metrics from: {self.log_file}")
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        current_step = None
        current_loss = None
        current_grad_norm = None
        
        for line in lines:
            # Parse step and loss together (they're on the same line)
            step_loss_match = re.search(r'Step (\d+) \| Loss: ([\d.e+-]+)', line)
            if step_loss_match:
                current_step = int(step_loss_match.group(1))
                current_loss = float(step_loss_match.group(2))
                
                # If we have grad norm from previous line, save it
                if current_grad_norm is not None and current_step is not None:
                    self.data.append({
                        "step": current_step,
                        "loss": current_loss,
                        "gradient_norm": current_grad_norm
                    })
                continue
            
            # Parse gradient norm (appears before Step line)
            grad_match = re.search(r'\[GradNorm\].*Norm: ([\d.e+-]+)', line)
            if grad_match:
                current_grad_norm = float(grad_match.group(1))
                continue
            
            # Check for SKIPPING explosions
            if 'SKIPPING' in line and 'EXPLOSION' in line:
                explosion_match = re.search(r'EXPLOSION DETECTED: ([\d.e+-]+)', line)
                if explosion_match and current_step is not None:
                    # Record the explosion
                    self.data.append({
                        "step": current_step + 1,  # Next step would have been this
                        "loss": current_loss if current_loss else 0.0,
                        "gradient_norm": float(explosion_match.group(1))
                    })
        
        print(f"✓ Extracted {len(self.data)} training steps")
        return self.data
    
    def save_to_jsonl(self, output_file: str):
        """Save extracted data to JSONL format"""
        output_path = Path(output_file)
        
        with open(output_path, 'w') as f:
            for record in self.data:
                f.write(json.dumps(record) + '\n')
        
        print(f"✓ Saved to {output_path}")


def run_training_evaluation(data_file: str, output_dir: str = "evaluation_results"):
    """
    Run comprehensive training evaluation
    
    Args:
        data_file: Path to JSONL file with training data
        output_dir: Directory to save evaluation results
    """
    print("""
╔══════════════════════════════════════════════════════════╗
║  GRIM Training Evaluation Framework                      ║
║  Comprehensive training health assessment                ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Create evaluators
    gradient_health_eval = GradientHealthEvaluator(
        explosion_threshold=100.0,
        warning_threshold=50.0
    )
    
    loss_convergence_eval = LossConvergenceEvaluator(window_size=10)
    numerical_stability_eval = NumericalStabilityEvaluator()
    training_progress_eval = TrainingProgressEvaluator()
    
    # Load data
    print(f"\n📂 Loading data from: {data_file}")
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"✓ Loaded {len(data)} training steps\n")
    
    # Run evaluations
    results = []
    
    print("🔍 Running evaluations...")
    for record in data:
        # Run each evaluator
        grad_health = gradient_health_eval(
            gradient_norm=record['gradient_norm'],
            step=record['step']
        )
        
        loss_conv = loss_convergence_eval(
            loss=record['loss'],
            step=record['step']
        )
        
        num_stability = numerical_stability_eval(
            loss=record['loss'],
            gradient_norm=record['gradient_norm']
        )
        
        progress = training_progress_eval(
            step=record['step'],
            gradient_health_status=grad_health['gradient_health_status']
        )
        
        # Combine results
        result = {
            **record,
            **grad_health,
            **loss_conv,
            **num_stability,
            **progress
        }
        
        results.append(result)
    
    print(f"✓ Completed {len(results)} evaluations\n")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"evaluation_results_{timestamp}.jsonl"
    
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"💾 Results saved to: {results_file}\n")
    
    # Generate summary report
    print("="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    
    explosions = sum(1 for r in results if r['gradient_health_status'] == 'EXPLOSION')
    warnings = sum(1 for r in results if r['gradient_health_status'] == 'WARNING')
    healthy = sum(1 for r in results if r['gradient_health_status'] == 'HEALTHY')
    
    print(f"\n🎯 Gradient Health:")
    print(f"  ✅ Healthy: {healthy}")
    print(f"  ⚠️  Warnings: {warnings}")
    print(f"  💥 Explosions: {explosions}")
    
    converging = sum(1 for r in results if r.get('convergence_status') == 'CONVERGING')
    diverging = sum(1 for r in results if r.get('convergence_status') == 'DIVERGING')
    stable = sum(1 for r in results if r.get('convergence_status') == 'STABLE')
    
    print(f"\n📈 Loss Convergence:")
    print(f"  ⬇️  Converging: {converging}")
    print(f"  ⬆️  Diverging: {diverging}")
    print(f"  ➡️  Stable: {stable}")
    
    numerically_stable = sum(1 for r in results if r['is_numerically_stable'])
    print(f"\n🔢 Numerical Stability:")
    print(f"  Stable steps: {numerically_stable}/{len(results)}")
    
    if results:
        final_progress = results[-1]
        print(f"\n📊 Training Progress:")
        print(f"  Status: {final_progress['training_progress_status']}")
        print(f"  Success Rate: {final_progress['success_rate']:.1%}")
        print(f"  Total Steps: {final_progress['total_steps']}")
        print(f"  Skipped Steps: {final_progress['skipped_steps']}")
    
    print(f"\n{'='*60}\n")
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if explosions > len(results) * 0.1:
        print("  🔴 HIGH: Frequent gradient explosions detected!")
        print("     → Reduce learning rate by 10x")
        print("     → Implement gradient value clamping (±5.0)")
        print("     → Check backward pass for numerical errors")
    elif explosions > 0:
        print("  🟡 MEDIUM: Occasional gradient explosions")
        print("     → Tighten gradient clipping threshold")
        print("     → Monitor backward pass gradients")
    else:
        print("  🟢 GOOD: No gradient explosions detected")
    
    if diverging > converging and len(results) > 20:
        print("  🔴 CRITICAL: Loss is diverging!")
        print("     → Learning rate is too high")
        print("     → Check for bugs in forward/backward pass")
    
    print(f"\n{'='*60}\n")
    
    return results


def main():
    """Main entry point"""
    import sys
    
    # Check for test output file first
    test_output = Path("test_fixed_gradients.txt")
    if test_output.exists():
        log_file = test_output
        print(f"📄 Using test output: {log_file.name}\n")
    else:
        # Find most recent log file
        log_dir = Path("resources/models/GRIM-text/training/logs")
        log_files = sorted(log_dir.glob("training_*.log"), 
                          key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not log_files:
            print("❌ No training logs found!")
            return
        
        log_file = log_files[0]
        print(f"📄 Using log file: {log_file.name}\n")
    
    # Extract metrics and convert to JSONL
    extractor = TrainingDataExtractor(log_file)
    extractor.extract_metrics()
    
    data_file = "training_metrics.jsonl"
    extractor.save_to_jsonl(data_file)
    
    # Run evaluation
    run_training_evaluation(data_file)


if __name__ == "__main__":
    main()

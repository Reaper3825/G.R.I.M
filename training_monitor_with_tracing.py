"""
GRIM Training Monitor with OpenTelemetry Tracing
Monitors training progress and traces gradient explosion issues
"""

import subprocess
import json
import time
import re
from pathlib import Path
from datetime import datetime
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Set up OpenTelemetry tracing
resource = Resource.create({"service.name": "grim-training-monitor"})
provider = TracerProvider(resource=resource)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)

class TrainingMonitor:
    def __init__(self, log_dir="resources/models/GRIM-text/training/logs"):
        self.log_dir = Path(log_dir)
        self.current_log = None
        self.gradient_explosion_detected = False
        
    def parse_log_line(self, line):
        """Parse training log lines for metrics"""
        data = {}
        
        # Parse step number
        step_match = re.search(r'Step (\d+)', line)
        if step_match:
            data['step'] = int(step_match.group(1))
        
        # Parse loss
        loss_match = re.search(r'Loss: ([\d.e+-]+)', line)
        if loss_match:
            data['loss'] = float(loss_match.group(1))
        
        # Parse gradient norm
        grad_match = re.search(r'GradNorm.*(?:Norm|RMS): ([\d.e+-]+)', line)
        if grad_match:
            data['grad_norm'] = float(grad_match.group(1))
        
        # Parse learning rate
        lr_match = re.search(r'LR: ([\d.e+-]+)', line)
        if lr_match:
            data['learning_rate'] = float(lr_match.group(1))
        
        # Detect gradient explosion
        if 'EXPLOSION DETECTED' in line:
            data['gradient_explosion'] = True
        
        return data if data else None
    
    def monitor_training(self, duration_minutes=60):
        """Monitor training with tracing"""
        with tracer.start_as_current_span("training_session") as session_span:
            session_span.set_attribute("session.start_time", datetime.now().isoformat())
            
            start_time = time.time()
            last_step = -1
            explosion_count = 0
            stable_steps = 0
            
            print(f"🔍 Starting training monitor with tracing...")
            print(f"📊 Traces will be sent to http://localhost:4318")
            print(f"⏱️  Monitoring for {duration_minutes} minutes")
            
            while time.time() - start_time < duration_minutes * 60:
                # Find the most recent log file
                log_files = sorted(self.log_dir.glob("training_*.log"), key=lambda p: p.stat().st_mtime)
                if not log_files:
                    time.sleep(1)
                    continue
                
                current_log = log_files[-1]
                if current_log != self.current_log:
                    self.current_log = current_log
                    print(f"📝 Monitoring log: {current_log.name}")
                
                # Read and parse log
                try:
                    with open(current_log, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                    
                    for line in lines[-100:]:  # Check last 100 lines
                        data = self.parse_log_line(line)
                        if not data:
                            continue
                        
                        step = data.get('step', -1)
                        if step <= last_step:
                            continue
                        
                        last_step = step
                        
                        # Create span for each training step
                        with tracer.start_as_current_span(f"training_step_{step}") as step_span:
                            step_span.set_attribute("step", step)
                            
                            if 'loss' in data:
                                loss = data['loss']
                                step_span.set_attribute("loss", loss)
                                step_span.set_attribute("loss_is_finite", float('inf') > loss > -float('inf'))
                                
                                # Detect numerical issues
                                if loss > 1e6:
                                    step_span.set_attribute("issue.type", "catastrophic_loss")
                                    step_span.set_attribute("issue.severity", "critical")
                                    print(f"🚨 CRITICAL: Catastrophic loss at step {step}: {loss:.2e}")
                                elif loss > 100:
                                    step_span.set_attribute("issue.type", "high_loss")
                                    step_span.set_attribute("issue.severity", "warning")
                                    print(f"⚠️  WARNING: High loss at step {step}: {loss:.2f}")
                                else:
                                    stable_steps += 1
                            
                            if 'grad_norm' in data:
                                grad_norm = data['grad_norm']
                                step_span.set_attribute("gradient_norm", grad_norm)
                                
                                # Analyze gradient behavior
                                if grad_norm > 10000:
                                    step_span.set_attribute("gradient.state", "exploding")
                                    step_span.set_attribute("issue.type", "gradient_explosion")
                                    print(f"💥 GRADIENT EXPLOSION at step {step}: {grad_norm:.2e}")
                                elif grad_norm > 1000:
                                    step_span.set_attribute("gradient.state", "very_high")
                                    print(f"⚠️  Very high gradient at step {step}: {grad_norm:.2f}")
                                elif grad_norm < 1e-6:
                                    step_span.set_attribute("gradient.state", "vanishing")
                                    step_span.set_attribute("issue.type", "gradient_vanishing")
                                    print(f"📉 Gradient vanishing at step {step}: {grad_norm:.2e}")
                                else:
                                    step_span.set_attribute("gradient.state", "normal")
                            
                            if 'learning_rate' in data:
                                step_span.set_attribute("learning_rate", data['learning_rate'])
                            
                            if data.get('gradient_explosion'):
                                explosion_count += 1
                                step_span.set_attribute("emergency_scaling_triggered", True)
                                
                                # Create incident span
                                with tracer.start_as_current_span("gradient_explosion_incident") as incident_span:
                                    incident_span.set_attribute("step", step)
                                    incident_span.set_attribute("explosion_count", explosion_count)
                                    incident_span.set_attribute("time", datetime.now().isoformat())
                                    
                                    if explosion_count == 1:
                                        print(f"\n{'='*60}")
                                        print(f"🔥 FIRST GRADIENT EXPLOSION DETECTED AT STEP {step}")
                                        print(f"{'='*60}")
                                    
                                    self.gradient_explosion_detected = True
                
                except Exception as e:
                    print(f"❌ Error reading log: {e}")
                
                time.sleep(0.5)  # Poll every 500ms
            
            # Session summary
            session_span.set_attribute("session.end_time", datetime.now().isoformat())
            session_span.set_attribute("session.total_steps", last_step)
            session_span.set_attribute("session.explosion_count", explosion_count)
            session_span.set_attribute("session.stable_steps", stable_steps)
            session_span.set_attribute("session.gradient_explosion_detected", self.gradient_explosion_detected)
            
            print(f"\n{'='*60}")
            print(f"📊 Training Monitor Summary")
            print(f"{'='*60}")
            print(f"Total steps: {last_step}")
            print(f"Stable steps: {stable_steps}")
            print(f"Gradient explosions: {explosion_count}")
            print(f"Explosion detected: {self.gradient_explosion_detected}")

def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║  GRIM Training Monitor with OpenTelemetry Tracing        ║
║  Monitors training and traces gradient explosions        ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    print("⚠️  IMPORTANT: Make sure AI Toolkit trace collector is running!")
    print("   Open VS Code Command Palette and run: 'AI Toolkit: Open Tracing'")
    print()
    
    monitor = TrainingMonitor()
    
    try:
        monitor.monitor_training(duration_minutes=30)
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    finally:
        # Flush traces
        trace.get_tracer_provider().force_flush()
        print("✅ Traces flushed to collector")

if __name__ == "__main__":
    main()

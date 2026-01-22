#!/usr/bin/env python3
"""
Test Hardcoded Hidden States Diagnostic (Issue #42)

This script runs training with different hardcoded hidden state patterns to isolate
whether mode collapse originates from the encoder or LM head/gradient system.

Usage:
    python test_hardcoded_states.py --pattern random_centered --batches 10
    python test_hardcoded_states.py --pattern orthogonal_w277 --batches 20
    python test_hardcoded_states.py --disable  # Turn off diagnostic
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

CONFIG_PATH = Path("ai_config.json")
TRAIN_EXE = Path("resources/models/GRIM-text/training/TrainingLoop/build/Release/train_gpu.exe")

PATTERNS = {
    "random_centered": "Random normal with mean=0 (tests Issue #37 fix)",
    "orthogonal_w277": "Orthogonal to W[277] (should give logit[277]≈0)",
    "aligned_w277": "Aligned with W[277] (should give high logit[277])",
    "constant_uniform": "Constant [1/√768] (tests Issue #40 row sum bias)",
    "zero_mean_sine": "Sine wave with zero mean (tests centering robustness)"
}

def load_config():
    """Load ai_config.json"""
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def save_config(config):
    """Save ai_config.json"""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

def enable_hardcoded_pattern(pattern: str, log_every_n: int = 1):
    """Enable hardcoded hidden states with specified pattern"""
    config = load_config()
    
    # Navigate to training.config.hardcoded_hidden_states
    train_config = config.get("training", {})
    train_subconfig = train_config.get("config", {})
    
    hardcoded_config = {
        "enabled": True,
        "pattern": pattern,
        "log_every_n_batches": log_every_n,
        "patterns": {
            "random_centered": "Random normal with mean=0 (tests Issue #37 fix)",
            "orthogonal_w277": "Orthogonal to W[277] (should give logit[277]≈0)",
            "aligned_w277": "Aligned with W[277] (should give high logit[277])",
            "constant_uniform": "Constant [1/√768] (tests Issue #40 row sum bias)",
            "zero_mean_sine": "Sine wave with zero mean (tests centering robustness)"
        }
    }
    
    train_subconfig["hardcoded_hidden_states"] = hardcoded_config
    train_config["config"] = train_subconfig
    config["training"] = train_config
    
    save_config(config)
    print(f"✅ Enabled hardcoded pattern: {pattern}")
    print(f"   Description: {PATTERNS.get(pattern, 'Unknown')}")
    print(f"   Logging every {log_every_n} batch(es)")

def disable_hardcoded():
    """Disable hardcoded hidden states diagnostic"""
    config = load_config()
    
    train_config = config.get("training", {})
    train_subconfig = train_config.get("config", {})
    
    if "hardcoded_hidden_states" in train_subconfig:
        train_subconfig["hardcoded_hidden_states"]["enabled"] = False
        train_config["config"] = train_subconfig
        config["training"] = train_config
        save_config(config)
        print("✅ Disabled hardcoded hidden states (using real encoder output)")

def show_status():
    """Show current hardcoded states configuration"""
    config = load_config()
    
    train_config = config.get("training", {})
    train_subconfig = train_config.get("config", {})
    hardcoded_config = train_subconfig.get("hardcoded_hidden_states", {})
    
    enabled = hardcoded_config.get("enabled", False)
    pattern = hardcoded_config.get("pattern", "N/A")
    log_every = hardcoded_config.get("log_every_n_batches", 1)
    
    print("\n" + "="*80)
    print("HARDCODED HIDDEN STATES DIAGNOSTIC STATUS")
    print("="*80)
    
    if enabled:
        print(f"Status:     ENABLED")
        print(f"Pattern:    {pattern}")
        print(f"Log Every:  {log_every} batch(es)")
        print(f"\nDescription: {PATTERNS.get(pattern, 'Unknown pattern')}")
        
        print("\n" + "-"*80)
        print("EXPECTED BEHAVIOR:")
        print("-"*80)
        
        if pattern == "random_centered":
            print("• Hidden state mean should be exactly 0.0")
            print("• Logits should be roughly uniform (no strong prediction)")
            print("• If collapse still occurs → gradient/optimizer bug")
        elif pattern == "orthogonal_w277":
            print("• cosine(h, W[277]) should be ~0.0 (orthogonal)")
            print("• logit[277] should be ~0.0 (dot product = 0)")
            print("• If logit[277] is high → LM head computation bug")
        elif pattern == "aligned_w277":
            print("• cosine(h, W[277]) should be ~1.0 (aligned)")
            print("• logit[277] should be HIGHEST (strong SPACE prediction)")
            print("• This simulates encoder outputting W[277]-aligned states")
        elif pattern == "constant_uniform":
            print("• logit[v] = sum(W[v,:]) / sqrt(d_model)")
            print("• If logit[277] is systematically higher → Issue #40 row sum bias")
        elif pattern == "zero_mean_sine":
            print("• Mean should be ~0.0 (sine wave is symmetric)")
            print("• Tests centering robustness with structured data")
            
    else:
        print("Status:     DISABLED (using real encoder output)")
    
    print("="*80 + "\n")

def run_training(max_batches: int = None):
    """Run training executable"""
    if not TRAIN_EXE.exists():
        print(f"❌ Training executable not found: {TRAIN_EXE}")
        print("   Build it with:")
        print("   cd resources/models/GRIM-text/training/TrainingLoop")
        print("   cmake --build build --config Release --target train_gpu")
        return 1
    
    print(f"\n🚀 Starting training (max batches: {max_batches or 'unlimited'})...")
    print("="*80)
    
    cmd = [str(TRAIN_EXE)]
    if max_batches:
        # Note: train_gpu.exe doesn't have --max-batches flag yet
        # This would need to be added to the C++ code
        print("   (Note: max_batches not yet implemented in train_gpu.exe)")
    
    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        return result.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 130

def main():
    parser = argparse.ArgumentParser(
        description="Test hardcoded hidden states diagnostic for Issue #42",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Patterns:
  random_centered    - Random normal with mean=0 (tests Issue #37 fix)
  orthogonal_w277    - Orthogonal to W[277] (should give logit[277]≈0)
  aligned_w277       - Aligned with W[277] (should give high logit[277])
  constant_uniform   - Constant [1/√768] (tests Issue #40 row sum bias)
  zero_mean_sine     - Sine wave with zero mean (tests centering robustness)

Examples:
  # Test if centering fix works
  python test_hardcoded_states.py --pattern random_centered --run --batches 50
  
  # Test if LM head computes logits correctly
  python test_hardcoded_states.py --pattern orthogonal_w277 --run --batches 20
  
  # Simulate encoder alignment (should cause collapse if gradient bug exists)
  python test_hardcoded_states.py --pattern aligned_w277 --run --batches 30
  
  # Disable and return to normal training
  python test_hardcoded_states.py --disable
        """
    )
    
    parser.add_argument("--pattern", choices=list(PATTERNS.keys()),
                        help="Hardcoded pattern to use")
    parser.add_argument("--disable", action="store_true",
                        help="Disable hardcoded states (use real encoder)")
    parser.add_argument("--status", action="store_true",
                        help="Show current configuration")
    parser.add_argument("--run", action="store_true",
                        help="Run training after setting pattern")
    parser.add_argument("--batches", type=int,
                        help="Max batches to train (requires --run)")
    parser.add_argument("--log-every", type=int, default=1,
                        help="Log diagnostics every N batches (default: 1)")
    
    args = parser.parse_args()
    
    # Show status if requested or no other action
    if args.status or (not args.pattern and not args.disable):
        show_status()
        return 0
    
    # Disable if requested
    if args.disable:
        disable_hardcoded()
        show_status()
        return 0
    
    # Enable pattern
    if args.pattern:
        enable_hardcoded_pattern(args.pattern, args.log_every)
        show_status()
        
        # Run training if requested
        if args.run:
            return run_training(args.batches)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

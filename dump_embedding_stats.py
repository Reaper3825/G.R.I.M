#!/usr/bin/env python3
"""
Dump embedding statistics to diagnose asymmetric range issue.
The ForwardDiag shows min=-3.11 max=1.21 which is highly asymmetric.
"""

import re
import sys
from pathlib import Path

def analyze_training_run():
    """Analyze training_run.txt for embedding statistics."""
    run_file = Path("resources/models/GRIM-text/training/logs/training_run.txt")
    if not run_file.exists():
        print(f"File not found: {run_file}")
        return
    
    content = run_file.read_text()
    
    # Find all ForwardDiag embedding entries
    pattern = r'\[ForwardDiag\] embeddings.*?min=([-\d.]+) max=([-\d.]+)'
    matches = re.findall(pattern, content)
    
    if not matches:
        print("No ForwardDiag embedding entries found")
        return
    
    print(f"Found {len(matches)} ForwardDiag embedding entries:")
    mins = [float(m[0]) for m in matches]
    maxs = [float(m[1]) for m in matches]
    
    print(f"  Min values: min={min(mins):.4f}, max={max(mins):.4f}, avg={sum(mins)/len(mins):.4f}")
    print(f"  Max values: min={min(maxs):.4f}, max={max(maxs):.4f}, avg={sum(maxs)/len(maxs):.4f}")
    print(f"  Asymmetry ratios (|min|/|max|): ", end="")
    ratios = [abs(mi)/abs(ma) if abs(ma) > 0.001 else 0 for mi, ma in zip(mins, maxs)]
    print(f"avg={sum(ratios)/len(ratios):.2f}")
    
    # Check if asymmetry is consistent
    print("\nFirst 10 entries:")
    for i, (mi, ma) in enumerate(matches[:10]):
        ratio = abs(float(mi))/abs(float(ma)) if abs(float(ma)) > 0.001 else 0
        print(f"  {i+1}: min={mi}, max={ma}, ratio={ratio:.2f}")

if __name__ == "__main__":
    analyze_training_run()

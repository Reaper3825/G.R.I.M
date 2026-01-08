#!/usr/bin/env python3
"""
Diagnose EOS and atom token issues in training log samples.
"""

import re
from pathlib import Path

log_file = Path("d:/G.R.I.M/resources/models/GRIM-text/training/logs/training_17674012547596604.log")

print("Analyzing training samples for EOS and atom issues...\n")

# Find all [Sample] lines
with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

samples = []
for i, line in enumerate(lines):
    if "[Sample]" in line and "step=" in line:
        # Get the sample text (next line usually)
        if i + 1 < len(lines):
            sample_text = lines[i+1].strip()
            if sample_text.startswith("[Sample]"):
                samples.append((line.strip(), sample_text))

print(f"Found {len(samples)} samples\n")

# Analyze for issues
eos_mid_sentence = []
atom_without_value = []

for step_line, sample_line in samples:
    # Extract step number
    step_match = re.search(r'step=(\d+)', step_line)
    step = step_match.group(1) if step_match else "?"
    
    # Check for EOS mid-sentence (not at the very end)
    if '</s>' in sample_line:
        # Find position of </s>
        eos_pos = sample_line.find('</s>')
        # Check if there's text after it (excluding whitespace and log markers)
        after_eos = sample_line[eos_pos+4:].strip()
        if after_eos and not after_eos.startswith('['):
            eos_mid_sentence.append((step, sample_line))
    
    # Check for atom tokens that appear as literal text
    atom_patterns = ['<ATOM>', '<PATH>', '<ID>', '<INT>', '<FLOAT>', '<URL>', '<EMAIL>', '<DATE>', '<TIME>']
    for pattern in atom_patterns:
        if pattern in sample_line:
            atom_without_value.append((step, pattern, sample_line))

print("=" * 80)
print("EOS TOKENS APPEARING MID-SENTENCE:")
print("=" * 80)
if eos_mid_sentence:
    for step, text in eos_mid_sentence[:10]:  # Show first 10
        print(f"\nStep {step}:")
        print(f"  {text[:150]}...")
else:
    print("  None found ✓")

print("\n" + "=" * 80)
print("ATOM TOKENS WITHOUT NUMERIC VALUES:")
print("=" * 80)
if atom_without_value:
    # Group by atom type
    by_type = {}
    for step, pattern, text in atom_without_value:
        if pattern not in by_type:
            by_type[pattern] = []
        by_type[pattern].append((step, text))
    
    for pattern in sorted(by_type.keys()):
        occurrences = by_type[pattern]
        print(f"\n{pattern}: {len(occurrences)} occurrences")
        for step, text in occurrences[:3]:  # Show first 3 of each type
            print(f"  Step {step}: {text[:120]}...")
else:
    print("  None found ✓")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"Total samples analyzed: {len(samples)}")
print(f"EOS mid-sentence: {len(eos_mid_sentence)}")
print(f"Atom tokens as literals: {len(atom_without_value)}")

if eos_mid_sentence or atom_without_value:
    print("\n⚠️  ISSUES DETECTED - Model is generating special tokens!")
else:
    print("\n✓ No issues detected")

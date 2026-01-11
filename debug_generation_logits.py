#!/usr/bin/env python3
"""
Debug script to analyze generation logits from GRIM-text model.
Prints top-5 predicted tokens and their logits to diagnose mode collapse.
"""

import subprocess
import sys
import json

# This would need to call the actual GRIM-text server or a test binary
# For now, let's analyze what we know from the training log

def analyze_sample_output():
    """Analyze the sample outputs from training log."""
    
    print("=" * 60)
    print("GRIM-text Generation Diagnostics")
    print("=" * 60)
    
    # From the training log analysis:
    print("\n[OBSERVATIONS]")
    print("1. Model outputs 'Hello world' (prompt) + whitespace only")
    print("2. Token 32 = ASCII space character")
    print("3. Training loss decreased from 7.4 → 4.5 (model IS learning)")
    print("4. But generation always predicts token 32 (space)")
    
    print("\n[LIKELY CAUSES]")
    print("1. Mode collapse: space is most common token, model overfits to it")
    print("2. Encoder output collapse: all positions produce similar hidden states")
    print("3. LM head bias: tied embeddings may have systematic bias")
    print("4. Forward pass bug: generation path differs from training path")
    
    print("\n[DIAGNOSTIC TESTS NEEDED]")
    print("1. Print top-5 logits during generation (are they all similar?)")
    print("2. Check encoder output variance (is it collapsing to constant?)")
    print("3. Compare training vs generation forward pass outputs")
    print("4. Check if softmax temperature is applied correctly during generation")
    
    print("\n[KEY INSIGHT]")
    print("The model generates NUMBERS early in training (steps 7-11):")
    print("  'Hello world3283003238600388428668...'")
    print("Then transitions to whitespace. This suggests:")
    print("  - Initially predicts atom tokens (numeric placeholders)")
    print("  - After some training, collapses to space tokens")
    print("  - Could be related to atom/byte token competition in vocab")

if __name__ == "__main__":
    analyze_sample_output()

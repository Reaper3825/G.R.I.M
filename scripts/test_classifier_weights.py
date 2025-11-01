#!/usr/bin/env python3
"""
Test script to verify the modular classifier weights system.
Prints the contents of the FlatBuffer weights file.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ai'))

try:
    import flatbuffers
    from ClassifierWeights import ClassifierConfig
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

def main():
    fb_path = os.path.join(os.path.dirname(__file__), '..', 'resources', 'classifier_weights.fb')
    
    if not os.path.exists(fb_path):
        print(f"ERROR: {fb_path} not found")
        sys.exit(1)
    
    with open(fb_path, 'rb') as f:
        buf = bytearray(f.read())
    
    config = ClassifierConfig.ClassifierConfig.GetRootAs(buf, 0)
    
    print(f"FlatBuffer Classifier Weights")
    print(f"=" * 60)
    print(f"Version: {config.Version()}")
    print(f"Priority: {config.Priority()}")
    print(f"Merge Strategy: {config.MergeStrategy().decode() if config.MergeStrategy() else 'None'}")
    print(f"Categories: {config.CategoriesLength()}")
    print()
    
    for i in range(config.CategoriesLength()):
        cat = config.Categories(i)
        print(f"\nCategory: {cat.Category().decode()}")
        print(f"  Default Weight: {cat.DefaultWeight()}")
        print(f"  Weights: {cat.WeightsLength()} entries")
        
        # Show first 10 weights
        print(f"  Sample weights:")
        for j in range(min(10, cat.WeightsLength())):
            entry = cat.Weights(j)
            token = entry.Token().decode()
            weight = entry.Weight()
            print(f"    {token}: {weight}")
        
        if cat.WeightsLength() > 10:
            print(f"    ... and {cat.WeightsLength() - 10} more")

if __name__ == '__main__':
    main()

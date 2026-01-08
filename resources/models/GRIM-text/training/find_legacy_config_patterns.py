#!/usr/bin/env python3
"""
Scan GRIM-text codebase for legacy manual JSON config parsing patterns.

Identifies code that should use centralized TrainingHyperparameters instead.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# Patterns to detect
LEGACY_PATTERNS = {
    'manual_contains_check': r'\.contains\(["\'](?:dynamic_lr|soft_restart|auto_stop|micro_validation|guess_feedback|cache_limits|scratch_blocks|stability_overrides)["\']',
    'is_boolean_check': r'\.is_boolean\(\)',
    'is_object_check': r'\.is_object\(\)',
    'direct_config_value': r'config\[["\']training["\']\]\[["\']config["\']\]',
    'hyperparams_contains': r'hyperparams\.contains\(',
    'json_value_extraction': r'\.value\(["\'](?:enabled|min|max|threshold|interval|steps|patience)["\']',
}

def scan_file(filepath):
    """Scan a single file for legacy patterns."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        return None
    
    findings = defaultdict(list)
    lines = content.split('\n')
    
    for pattern_name, pattern in LEGACY_PATTERNS.items():
        for match in re.finditer(pattern, content, re.MULTILINE):
            line_num = content[:match.start()].count('\n') + 1
            line_content = lines[line_num - 1].strip()
            findings[pattern_name].append({
                'line': line_num,
                'content': line_content[:100]  # First 100 chars
            })
    
    return findings if findings else None

def scan_directory(root_dir, extensions=None):
    """Recursively scan directory for legacy config patterns."""
    if extensions is None:
        extensions = {'.cpp', '.cu', '.hpp', '.h'}
    
    results = {}
    root_path = Path(root_dir)
    
    for filepath in root_path.rglob('*'):
        if filepath.suffix in extensions and filepath.is_file():
            # Skip build directories and external deps
            if any(part in str(filepath) for part in ['build', 'external', 'vcpkg_installed', '.git']):
                continue
            
            findings = scan_file(filepath)
            if findings:
                results[str(filepath.relative_to(root_path))] = findings
    
    return results

def print_report(results):
    """Print formatted report of findings."""
    if not results:
        print("✓ No legacy config patterns found!")
        return
    
    print("=" * 80)
    print("Legacy Configuration Pattern Report")
    print("=" * 80)
    print()
    
    # Summary
    total_files = len(results)
    total_issues = sum(len(findings) for file_findings in results.values() 
                       for findings in file_findings.values())
    
    print(f"Found {total_issues} legacy patterns across {total_files} files")
    print()
    
    # Pattern breakdown
    pattern_counts = defaultdict(int)
    for file_findings in results.values():
        for pattern_name, findings in file_findings.items():
            pattern_counts[pattern_name] += len(findings)
    
    print("Pattern Breakdown:")
    for pattern_name, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f"  {pattern_name}: {count}")
    print()
    
    # Detailed findings
    print("Detailed Findings:")
    print("-" * 80)
    
    for filepath, file_findings in sorted(results.items()):
        print(f"\n{filepath}")
        print("  " + "─" * 76)
        
        for pattern_name, findings in file_findings.items():
            print(f"  [{pattern_name}] - {len(findings)} occurrences")
            for finding in findings[:3]:  # Show first 3 per pattern
                print(f"    Line {finding['line']}: {finding['content']}")
            if len(findings) > 3:
                print(f"    ... and {len(findings) - 3} more")
        print()

def generate_refactor_plan(results):
    """Generate actionable refactor plan."""
    print("=" * 80)
    print("Recommended Refactoring Plan")
    print("=" * 80)
    print()
    
    # Identify train_gpu.cu separately
    train_gpu_files = [f for f in results.keys() if 'train_gpu.cu' in f]
    other_files = [f for f in results.keys() if 'train_gpu.cu' not in f]
    
    if train_gpu_files:
        print("HIGH PRIORITY - Main Training File:")
        for filepath in train_gpu_files:
            total = sum(len(findings) for findings in results[filepath].values())
            print(f"  • {filepath}: {total} patterns")
            print(f"    → Refactor to use helper functions for config loading")
            print(f"    → Extract repetitive if-blocks into loadBoolOrObject()")
        print()
    
    if other_files:
        print("MEDIUM PRIORITY - Other Files:")
        for filepath in sorted(other_files, key=lambda f: sum(len(findings) for findings in results[f].values()), reverse=True):
            total = sum(len(findings) for findings in results[filepath].values())
            print(f"  • {filepath}: {total} patterns")
            
            # Check if it's a header or implementation
            if filepath.endswith(('.hpp', '.h')):
                print(f"    → Consider adding TrainingHyperparameters parameter")
            else:
                print(f"    → Use GRIM::Config::loadAiConfigSnapshot() instead of manual parsing")
        print()
    
    print("Steps:")
    print("  1. Create config loading helper functions in HyperParameters_GPU.hpp")
    print("  2. Refactor train_gpu.cu to use helpers (reduce ~400 lines to ~50)")
    print("  3. Update other files to use centralized TrainingHyperparameters")
    print("  4. Remove manual JSON parsing where possible")
    print()

if __name__ == '__main__':
    import sys
    
    # Default to GRIM-text directory
    root_dir = Path(__file__).parent.parent
    
    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    
    print(f"Scanning: {root_dir.absolute()}")
    print()
    
    results = scan_directory(root_dir)
    print_report(results)
    generate_refactor_plan(results)
    
    # Exit code: 0 if clean, 1 if issues found
    sys.exit(0 if not results else 1)

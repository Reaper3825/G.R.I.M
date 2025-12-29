#!/usr/bin/env python3
"""
Find all gradient zeroing operations in GRIM-text codebase.
Non-regex based approach using string searching.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple

class GradientZeroFinder:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.results = []
        
    def search_file(self, filepath: Path) -> List[Dict]:
        """Search a single file for gradient zeroing operations."""
        findings = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, start=1):
                line_lower = line.lower()
                
                # Search patterns (case-insensitive)
                zero_patterns = [
                    'cudamemset',
                    'cudamemsetasync',
                    'zero_grad',
                    'zerograd',
                    'gradient_grads',
                    '_grads',
                ]
                
                # Check if line contains gradient zeroing
                for pattern in zero_patterns:
                    if pattern in line_lower:
                        # Additional context: check if it's zeroing (0 value)
                        if ', 0,' in line or ', 0 ' in line or '= 0' in line or 'zero' in line_lower:
                            findings.append({
                                'file': str(filepath.relative_to(self.root_dir)),
                                'line_num': line_num,
                                'line': line.strip(),
                                'pattern': pattern
                            })
                            break
                        
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
        return findings
    
    def search_directory(self, extensions: List[str] = None):
        """Recursively search directory for gradient zeroing operations."""
        if extensions is None:
            extensions = ['.cu', '.cpp', '.hpp', '.h', '.cuh']
            
        target_dirs = [
            'resources/models/GRIM-text',
        ]
        
        for target_dir in target_dirs:
            search_path = self.root_dir / target_dir
            if not search_path.exists():
                print(f"Warning: {search_path} does not exist")
                continue
                
            print(f"\nSearching in: {search_path}")
            
            for ext in extensions:
                for filepath in search_path.rglob(f'*{ext}'):
                    findings = self.search_file(filepath)
                    self.results.extend(findings)
                    
    def print_results(self):
        """Print all findings organized by file."""
        if not self.results:
            print("\n✗ No gradient zeroing operations found!")
            return
            
        print(f"\n{'='*80}")
        print(f"FOUND {len(self.results)} GRADIENT ZEROING OPERATIONS")
        print(f"{'='*80}\n")
        
        # Group by file
        by_file = {}
        for result in self.results:
            file = result['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(result)
            
        # Print grouped results
        for file, findings in sorted(by_file.items()):
            print(f"\n📁 {file}")
            print(f"   {len(findings)} occurrences\n")
            
            for finding in findings:
                print(f"   Line {finding['line_num']:5d}: {finding['line']}")
                
        # Summary by pattern
        print(f"\n{'='*80}")
        print("SUMMARY BY PATTERN")
        print(f"{'='*80}\n")
        
        pattern_counts = {}
        for result in self.results:
            pattern = result['pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"   {pattern:20s}: {count:3d} occurrences")
            
    def save_results(self, output_file: str = "gradient_zero_locations.txt"):
        """Save results to a text file."""
        output_path = self.root_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"GRADIENT ZEROING OPERATIONS - GRIM-text Codebase\n")
            f.write("="*80 + "\n\n")
            
            # Group by file
            by_file = {}
            for result in self.results:
                file = result['file']
                if file not in by_file:
                    by_file[file] = []
                by_file[file].append(result)
                
            for file, findings in sorted(by_file.items()):
                f.write(f"\n{'─'*80}\n")
                f.write(f"FILE: {file}\n")
                f.write(f"{'─'*80}\n")
                
                for finding in findings:
                    f.write(f"\nLine {finding['line_num']:5d}:\n")
                    f.write(f"  {finding['line']}\n")
                    
        print(f"\n✓ Results saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find all gradient zeroing operations in GRIM-text codebase"
    )
    parser.add_argument(
        '--root',
        default='.',
        help='Root directory of GRIM project (default: current directory)'
    )
    parser.add_argument(
        '--output',
        default='gradient_zero_locations.txt',
        help='Output file name (default: gradient_zero_locations.txt)'
    )
    
    args = parser.parse_args()
    
    finder = GradientZeroFinder(args.root)
    
    print("="*80)
    print("GRIM-text Gradient Zeroing Locator")
    print("="*80)
    
    finder.search_directory()
    finder.print_results()
    finder.save_results(args.output)
    
    print(f"\n{'='*80}")
    print(f"Search complete! Total occurrences: {len(finder.results)}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

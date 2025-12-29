#!/usr/bin/env python3
"""
Analyze training data to find common <ATOM:X> patterns that should be added to vocabulary.
This prevents gradient spikes from rare atom placeholder tokens.
"""

import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

def extract_atom_patterns(text: str) -> List[str]:
    """Extract all <ATOM:X> patterns and their common contexts."""
    patterns = []
    
    # Match standalone atoms
    atoms = re.findall(r'<ATOM:\d+>', text)
    patterns.extend(atoms)
    
    # Match common atom contexts (mathematical notation)
    contexts = [
        r'\(<ATOM:\d+>',      # (<ATOM:1>
        r'<ATOM:\d+>\)',      # <ATOM:1>)
        r'= <ATOM:\d+>',      # = <ATOM:1>
        r'<ATOM:\d+> =',      # <ATOM:1> =
        r'± <ATOM:\d+>',      # ± <ATOM:1>
        r'× <ATOM:\d+>',      # × <ATOM:1>
        r'<ATOM:\d+>,',       # <ATOM:1>,
        r', <ATOM:\d+>',      # , <ATOM:1>
        r'- <ATOM:\d+>',      # - <ATOM:1>
        r'\+ <ATOM:\d+>',     # + <ATOM:1>
    ]
    
    for pattern in contexts:
        matches = re.findall(pattern, text)
        patterns.extend(matches)
    
    return patterns

def analyze_training_data(data_path: Path, max_files: int = 1000) -> Tuple[Counter, int]:
    """Analyze training data to count atom pattern frequencies."""
    atom_counter = Counter()
    total_sequences = 0
    
    # Check if it's a directory or single file
    if data_path.is_dir():
        files = list(data_path.glob('*.txt'))[:max_files]
    else:
        files = [data_path]
    
    print(f"Analyzing {len(files)} file(s)...")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                patterns = extract_atom_patterns(text)
                atom_counter.update(patterns)
                total_sequences += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
    
    return atom_counter, total_sequences

def generate_special_tokens(atom_counter: Counter, min_freq: int = 10, top_n: int = 100) -> List[str]:
    """Generate list of special tokens to add to vocabulary."""
    special_tokens = []
    
    # Add most common atom numbers
    atom_numbers = Counter()
    for pattern, count in atom_counter.items():
        match = re.search(r'<ATOM:(\d+)>', pattern)
        if match:
            atom_numbers[match.group(1)] += count
    
    # Add top N most common atom numbers as standalone tokens
    for atom_num, count in atom_numbers.most_common(top_n):
        if count >= min_freq:
            special_tokens.append(f'<ATOM:{atom_num}>')
    
    # Add common contextual patterns if frequent enough
    for pattern, count in atom_counter.most_common(200):
        if count >= min_freq and pattern not in special_tokens:
            # Only add patterns with context (not standalone atoms)
            if not re.match(r'^<ATOM:\d+>$', pattern):
                special_tokens.append(pattern)
    
    return special_tokens

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze atom patterns in training data')
    parser.add_argument('data_path', help='Path to training data file or directory')
    parser.add_argument('--output', '-o', default='atom_special_tokens.txt', 
                        help='Output file for special tokens list')
    parser.add_argument('--min-freq', type=int, default=10,
                        help='Minimum frequency for inclusion')
    parser.add_argument('--top-n', type=int, default=100,
                        help='Number of top atom numbers to include')
    parser.add_argument('--max-files', type=int, default=1000,
                        help='Maximum number of files to analyze')
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: {data_path} does not exist", file=sys.stderr)
        return 1
    
    # Analyze data
    atom_counter, total_sequences = analyze_training_data(data_path, args.max_files)
    
    print(f"\nAnalyzed {total_sequences} sequences")
    print(f"Found {len(atom_counter)} unique atom patterns")
    print(f"\nTop 20 most common patterns:")
    for pattern, count in atom_counter.most_common(20):
        print(f"  {pattern:20s} : {count:6d}")
    
    # Generate special tokens
    special_tokens = generate_special_tokens(atom_counter, args.min_freq, args.top_n)
    
    print(f"\n{len(special_tokens)} special tokens to add to vocabulary")
    
    # Write to file
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in special_tokens:
            f.write(f"{token}\n")
    
    print(f"\nWrote special tokens to: {output_path}")
    
    # Show stats
    total_atom_occurrences = sum(atom_counter.values())
    covered_occurrences = sum(count for pattern, count in atom_counter.items() 
                              if pattern in special_tokens)
    coverage = (covered_occurrences / total_atom_occurrences * 100) if total_atom_occurrences > 0 else 0
    
    print(f"\nCoverage: {coverage:.1f}% of atom occurrences")
    print(f"Vocabulary size increase: +{len(special_tokens)} tokens")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

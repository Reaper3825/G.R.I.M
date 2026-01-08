#!/usr/bin/env python3
"""
Sample random sequences from training data and classify them.
Helps audit data quality and verify classifier accuracy.

Usage:
    python sample_training_data.py --count 20
    python sample_training_data.py --output samples.txt --count 50
"""

import sys
import struct
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_grmt_sequences(data_path: str):
    """Load sequences from .grmt FlatBuffer format"""
    sequences = []
    
    with open(data_path, 'rb') as f:
        # Read FlatBuffer data (simplified - assumes structure from DataLoader.cu)
        # Magic header check
        magic = f.read(4)
        if magic != b'GRMT':
            print(f"Warning: Invalid magic header, trying to parse anyway...")
            f.seek(0)
        
        # Try to read as raw token sequences
        while True:
            # Read sequence length (4 bytes)
            len_bytes = f.read(4)
            if not len_bytes or len(len_bytes) < 4:
                break
            
            seq_len = struct.unpack('I', len_bytes)[0]
            if seq_len == 0 or seq_len > 10000:  # Sanity check
                continue
            
            # Read tokens (4 bytes each)
            tokens = []
            for _ in range(seq_len):
                token_bytes = f.read(4)
                if len(token_bytes) < 4:
                    break
                token = struct.unpack('i', token_bytes)[0]
                tokens.append(token)
            
            if len(tokens) == seq_len:
                sequences.append(tokens)
    
    return sequences


def classify_text(text: str) -> tuple[str, dict]:
    """
    Classify sequence using same logic as train_gpu.cu
    Returns (class_name, score_dict)
    """
    if not text:
        return ("mixed_junk", {})
    
    text_lower = text.lower()
    
    scores = {
        'boilerplate': 0,
        'documentation': 0,
        'prose': 0,
        'code': 0,
        'junk': 0
    }
    
    # Boilerplate patterns
    boilerplate_patterns = [
        "find us on", "follow us", "search this site", "submit search",
        "copyright", "all rights reserved", "privacy policy", "terms of",
        "contact us", "sign up", "log in", "sign in", "subscribe",
        "newsletter", "share on", "tweet", "facebook", "instagram",
        "skip to content", "skip to main", "navigation", "breadcrumb",
        "menu", "footer", "header", "sidebar", "retrieved from",
        "category :", "categories:", "tags:", "related posts",
        "previous post", "next post", "read more", "click here",
        "android", "ios", "app store", "play store", "download"
    ]
    for pat in boilerplate_patterns:
        if pat in text_lower:
            scores['boilerplate'] += 2
    
    # Documentation patterns
    doc_patterns = [
        "parameters", "returns", "example", "usage:", "syntax:",
        "arguments", "description:", "note:", "warning:", "deprecated",
        "see also", "api", "reference", "documentation", "method",
        "function", "class", "module", "import", "export",
        "configuration", "settings", "options", "version"
    ]
    for pat in doc_patterns:
        if pat in text_lower:
            scores['documentation'] += 2
    
    # Prose indicators (punctuation density)
    sentence_ends = text.count('.') + text.count('!') + text.count('?')
    commas = text.count(',')
    punct_ratio = (sentence_ends + commas) / max(len(text), 1)
    if 0.02 < punct_ratio < 0.08:
        scores['prose'] += 5
    if sentence_ends > 3:
        scores['prose'] += 3
    
    # Code patterns
    code_patterns = [
        "def ", "class ", "function ", "return ", "if (", "for (",
        "while (", "switch (", "case ", "import ", "from ",
        "#include", "using namespace", "public:", "private:",
        "const ", "let ", "var ", "async ", "await ",
        "();", ");", "};", "= {", "=> {", "->", "::",
        "null", "nullptr", "undefined", "true", "false"
    ]
    for pat in code_patterns:
        if pat in text:
            scores['code'] += 2
    
    # Special chars for code
    braces = text.count('{') + text.count('}')
    brackets = text.count('[') + text.count(']')
    semicolons = text.count(';')
    if braces > 4:
        scores['code'] += 3
    if semicolons > 5:
        scores['code'] += 3
    
    # Junk patterns
    junk_patterns = [
        "}} ", "{{ ", " ... ", "...", " <url>", "<url> ",
        " ip }}", "work to do", "peer-to-peer university"
    ]
    for pat in junk_patterns:
        if pat in text_lower:
            scores['junk'] += 3
    
    # Repetitive words (junk indicator)
    words = [w for w in text_lower.split() if len(w) >= 3]
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    repetitive = sum(1 for count in word_counts.values() if count > 5)
    if repetitive > 3:
        scores['junk'] += 5
    
    # Line length variance
    newlines = text.count('\n')
    avg_line_len = len(text) / max(newlines + 1, 1)
    if avg_line_len < 20.0 or avg_line_len > 500.0:
        scores['junk'] += 2
    
    # Determine winner
    max_score = max(scores.values())
    if max_score == 0 or scores['junk'] >= max_score:
        return ("mixed_junk", scores)
    
    winner = max(scores, key=scores.get)
    if winner == 'junk':
        return ("mixed_junk", scores)
    if winner == 'boilerplate':
        return ("boilerplate/nav", scores)
    if winner == 'documentation':
        return ("documentation", scores)
    if winner == 'prose':
        return ("prose", scores)
    if winner == 'code':
        return ("code", scores)
    
    return ("mixed_junk", scores)


def main():
    parser = argparse.ArgumentParser(description='Sample and classify training data')
    parser.add_argument('--data', default='D:/G.R.I.M/resources/models/GRIM-text/training/data/training_data.grmt',
                        help='Path to training data (.grmt file)')
    parser.add_argument('--vocab', default='D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.bin',
                        help='Path to vocab file')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of random samples to inspect')
    parser.add_argument('--output', default='training_data_samples.txt',
                        help='Output file for samples')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
    
    print(f"Loading tokenizer from {args.vocab}...")
    try:
        tokenizer = GrimTokenizer(args.vocab, max_length=8192)
        print(f"✓ Loaded tokenizer with {tokenizer.vocab_size()} tokens")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Trying alternative method...")
        # Fallback: just show token IDs without decoding
        tokenizer = None
    
    print(f"Loading training data from {args.data}...")
    try:
        sequences = load_grmt_sequences(args.data)
        print(f"✓ Loaded {len(sequences)} sequences")
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)
    
    if len(sequences) == 0:
        print("Error: No sequences found in training data")
        sys.exit(1)
    
    # Sample random sequences
    sample_count = min(args.count, len(sequences))
    sampled_indices = random.sample(range(len(sequences)), sample_count)
    
    # Write samples to file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"Training Data Sample Audit\n")
        f.write(f"{'='*80}\n")
        f.write(f"Total sequences: {len(sequences)}\n")
        f.write(f"Sampled: {sample_count}\n")
        f.write(f"Date: {Path(__file__).stat().st_mtime}\n")
        f.write(f"{'='*80}\n\n")
        
        class_counts = {}
        
        for i, idx in enumerate(sampled_indices, 1):
            tokens = sequences[idx]
            f.write(f"\n{'='*80}\n")
            f.write(f"SAMPLE {i}/{sample_count} - Sequence {idx}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Tokens: {len(tokens)}\n")
            
            if tokenizer:
                try:
                    decoded = tokenizer.decode(tokens)
                    seq_class, scores = classify_text(decoded)
                    
                    class_counts[seq_class] = class_counts.get(seq_class, 0) + 1
                    
                    f.write(f"Classification: [{seq_class}]\n")
                    f.write(f"Scores: {scores}\n")
                    f.write(f"\nDecoded Text (first 1000 chars):\n")
                    f.write(f"{'-'*80}\n")
                    preview = decoded[:1000]
                    f.write(preview)
                    if len(decoded) > 1000:
                        f.write(f"\n... (truncated, total {len(decoded)} chars)")
                    f.write(f"\n{'-'*80}\n")
                except Exception as e:
                    f.write(f"Error decoding: {e}\n")
                    f.write(f"Token IDs: {tokens[:50]}{'...' if len(tokens) > 50 else ''}\n")
            else:
                f.write(f"Token IDs: {tokens[:50]}{'...' if len(tokens) > 50 else ''}\n")
        
        # Summary
        f.write(f"\n\n{'='*80}\n")
        f.write(f"CLASSIFICATION SUMMARY\n")
        f.write(f"{'='*80}\n")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            pct = (count / sample_count) * 100
            f.write(f"  {cls:20s}: {count:3d} ({pct:5.1f}%)\n")
    
    print(f"\n✓ Wrote {sample_count} samples to {args.output}")
    print(f"\nClassification breakdown:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = (count / sample_count) * 100
        print(f"  {cls:20s}: {count:3d} ({pct:5.1f}%)")


if __name__ == '__main__':
    main()

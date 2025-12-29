#!/usr/bin/env python3
"""
Data Diversity Analyzer

Checks training data for:
1. Repetitive patterns (n-gram overlap)
2. Sequence length distribution
3. Token frequency imbalance (Zipf's law deviation)
4. HTML/boilerplate artifacts
"""

import re
import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def analyze_grmt_data(grmt_path, vocab_path):
    """Analyze GRIM training data for quality issues."""
    
    print("="*60)
    print("TRAINING DATA DIVERSITY ANALYSIS")
    print("="*60)
    
    # Load vocab
    print("\nLoading vocabulary...")
    with open(vocab_path, 'rb') as f:
        vocab_size = int.from_bytes(f.read(4), 'little')
        vocab = []
        for _ in range(vocab_size):
            token_len = int.from_bytes(f.read(4), 'little')
            token_bytes = f.read(token_len)
            vocab.append(token_bytes.decode('utf-8', errors='replace'))
    
    print(f"✓ Loaded {len(vocab)} tokens")
    
    # Load training sequences
    print("\nLoading training sequences...")
    with open(grmt_path, 'rb') as f:
        magic = f.read(8)
        if magic != b'GRMT\x00\x00\x00\x01':
            print("❌ Invalid GRMT file format")
            return
        
        num_sequences = int.from_bytes(f.read(8), 'little')
        sequences = []
        
        for i in range(min(num_sequences, 1000)):  # Sample first 1000
            seq_len = int.from_bytes(f.read(4), 'little')
            token_ids = []
            for _ in range(seq_len):
                token_id = int.from_bytes(f.read(4), 'little')
                token_ids.append(token_id)
            sequences.append(token_ids)
            
            if i % 200 == 0:
                print(f"  Loaded {i}/{min(num_sequences, 1000)} sequences...", end='\r')
        
        print(f"\n✓ Loaded {len(sequences)} sequences (sample)")
    
    # Analyze sequence lengths
    print("\n" + "="*60)
    print("SEQUENCE LENGTH DISTRIBUTION")
    print("="*60)
    
    lengths = [len(seq) for seq in sequences]
    print(f"\nSequence lengths:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")
    print(f"  Std dev: {np.std(lengths):.1f}")
    
    # Histogram
    bins = [0, 128, 256, 512, 1024, 2048, 4096]
    hist, _ = np.histogram(lengths, bins=bins)
    print(f"\nLength distribution:")
    for i in range(len(bins)-1):
        pct = 100 * hist[i] / len(lengths)
        bar = '█' * int(pct / 2)
        print(f"  {bins[i]:4d}-{bins[i+1]:4d}: {hist[i]:4d} ({pct:5.1f}%) {bar}")
    
    # Analyze token frequency
    print("\n" + "="*60)
    print("TOKEN FREQUENCY ANALYSIS")
    print("="*60)
    
    all_tokens = []
    for seq in sequences:
        all_tokens.extend(seq)
    
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    print(f"\nTotal tokens: {total_tokens}")
    print(f"Unique tokens: {len(token_counts)}")
    print(f"Vocab coverage: {100 * len(token_counts) / len(vocab):.1f}%")
    
    # Top tokens
    print(f"\nTop 20 most frequent tokens:")
    for token_id, count in token_counts.most_common(20):
        token_str = vocab[token_id] if token_id < len(vocab) else f"<UNK:{token_id}>"
        pct = 100 * count / total_tokens
        print(f"  {token_id:5d} {token_str:30s} {count:6d} ({pct:5.2f}%)")
    
    # Check for frequency imbalance (Zipf's law)
    freqs = sorted(token_counts.values(), reverse=True)
    top_10_pct = 100 * sum(freqs[:10]) / total_tokens
    top_100_pct = 100 * sum(freqs[:100]) / total_tokens
    
    print(f"\nFrequency concentration:")
    print(f"  Top 10 tokens: {top_10_pct:.1f}% of all tokens")
    print(f"  Top 100 tokens: {top_100_pct:.1f}% of all tokens")
    
    if top_10_pct > 30:
        print("  ⚠ WARNING: Vocabulary is highly imbalanced!")
    elif top_10_pct > 20:
        print("  ⚡ Moderate imbalance detected")
    else:
        print("  ✓ Reasonable token distribution")
    
    # Analyze n-gram overlap (repetitiveness)
    print("\n" + "="*60)
    print("SEQUENCE DIVERSITY ANALYSIS")
    print("="*60)
    
    print("\nComputing n-gram statistics...")
    trigrams = []
    for seq in sequences[:100]:  # Sample subset
        for i in range(len(seq) - 2):
            trigrams.append(tuple(seq[i:i+3]))
    
    trigram_counts = Counter(trigrams)
    unique_trigrams = len(trigram_counts)
    total_trigrams = len(trigrams)
    
    print(f"\nTrigram statistics:")
    print(f"  Total trigrams: {total_trigrams}")
    print(f"  Unique trigrams: {unique_trigrams}")
    print(f"  Diversity ratio: {unique_trigrams / total_trigrams:.4f}")
    
    if unique_trigrams / total_trigrams < 0.5:
        print("  ⚠ WARNING: High repetition detected!")
        print("     → Sequences contain many repeated patterns")
    elif unique_trigrams / total_trigrams < 0.7:
        print("  ⚡ Moderate repetition")
    else:
        print("  ✓ Good sequence diversity")
    
    # Most repeated trigrams
    print(f"\nMost repeated 3-grams:")
    for trigram, count in trigram_counts.most_common(10):
        tokens_str = [vocab[tid] if tid < len(vocab) else f"<{tid}>" for tid in trigram]
        print(f"  {count:4d}x: {' '.join(tokens_str)}")
    
    # Check for HTML artifacts
    print("\n" + "="*60)
    print("HTML/BOILERPLATE DETECTION")
    print("="*60)
    
    html_indicators = ['<', '>', '&lt;', '&gt;', '&nbsp;', 'div', 'span', 'class', 'href']
    boilerplate_indicators = ['copyright', 'all rights reserved', 'privacy policy', 'terms of service']
    
    html_count = 0
    boilerplate_count = 0
    
    for seq in sequences:
        seq_text = ' '.join([vocab[tid] if tid < len(vocab) else '' for tid in seq]).lower()
        
        if any(indicator in seq_text for indicator in html_indicators):
            html_count += 1
        if any(indicator in seq_text for indicator in boilerplate_indicators):
            boilerplate_count += 1
    
    html_pct = 100 * html_count / len(sequences)
    boilerplate_pct = 100 * boilerplate_count / len(sequences)
    
    print(f"\nArtifact detection (in {len(sequences)} sequences):")
    print(f"  HTML artifacts: {html_count} ({html_pct:.1f}%)")
    print(f"  Boilerplate text: {boilerplate_count} ({boilerplate_pct:.1f}%)")
    
    if html_pct > 10:
        print("  ⚠ WARNING: Significant HTML contamination!")
    elif html_pct > 5:
        print("  ⚡ Some HTML artifacts present")
    else:
        print("  ✓ Clean data (no HTML)")
    
    if boilerplate_pct > 10:
        print("  ⚠ WARNING: High boilerplate content!")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    issues = []
    if top_10_pct > 30:
        issues.append("Severe token frequency imbalance")
    if unique_trigrams / total_trigrams < 0.5:
        issues.append("High sequence repetition (low diversity)")
    if html_pct > 10:
        issues.append("HTML contamination in training data")
    if boilerplate_pct > 10:
        issues.append("Excessive boilerplate text")
    
    if issues:
        print("\n⚠ DATA QUALITY ISSUES DETECTED:")
        for issue in issues:
            print(f"  • {issue}")
        print("\nRECOMMENDATIONS:")
        print("  1. Re-run data collection with stricter filtering")
        print("  2. Remove HTML artifacts in DataLoader")
        print("  3. Deduplicate near-identical sequences")
        print("  4. Balance token frequencies (downsampling common tokens)")
    else:
        print("\n✓ Data quality looks reasonable")
        print("\n⚠ If plateau persists with good data:")
        print("  → The model may have genuinely learned all patterns")
        print("  → Consider more diverse data sources")

if __name__ == '__main__':
    grmt_path = Path('resources/models/GRIM-text/training/data/training_data.grmt')
    vocab_path = Path('resources/models/GRIM-text/training/data/vocab.bin')
    
    if not grmt_path.exists():
        print(f"❌ Training data not found: {grmt_path}")
        exit(1)
    
    if not vocab_path.exists():
        print(f"❌ Vocab not found: {vocab_path}")
        exit(1)
    
    analyze_grmt_data(grmt_path, vocab_path)

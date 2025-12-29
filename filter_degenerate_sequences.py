#!/usr/bin/env python3
"""
Filter out degenerate sequences from training data.
Filters:
1. High bigram repetition (same bigram appearing >12% of time)
2. High fragment token ratio (>15% of tokens are known fragments)
"""
import struct
from collections import Counter
from pathlib import Path

INPUT_PATH = Path(r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data_original.grmt')
OUTPUT_PATH = Path(r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data_filtered.grmt')
BIGRAM_THRESHOLD = 0.12  # Filter if any bigram appears >12% of the time
FRAGMENT_THRESHOLD = 0.15  # Filter if >15% of tokens are fragment tokens

# Known fragment tokens that indicate corrupted/garbled text
FRAGMENT_TOKENS = {257, 258, 516, 620, 686, 544}  # ect, A, ion of, etc.

def is_degenerate(tokens, threshold=BIGRAM_THRESHOLD):
    """Check if sequence has degenerate repetitive patterns"""
    if len(tokens) < 20:
        return False
    
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    bigram_counts = Counter(bigrams)
    top_count = bigram_counts.most_common(1)[0][1]
    ratio = top_count / len(bigrams)
    
    return ratio > threshold

def has_high_fragment_ratio(tokens, threshold=FRAGMENT_THRESHOLD):
    """Check if sequence has too many fragment tokens (corrupted data)"""
    fragment_count = sum(1 for t in tokens if t in FRAGMENT_TOKENS)
    return fragment_count / len(tokens) > threshold

def main():
    # Read input file
    with open(INPUT_PATH, 'rb') as f:
        magic = f.read(4)
        version = struct.unpack('<I', f.read(4))[0]
        num_sequences = struct.unpack('<I', f.read(4))[0]
        vocab_size = struct.unpack('<I', f.read(4))[0]
        
        print(f"Reading {num_sequences} sequences from {INPUT_PATH}")
        print(f"Magic: {magic}, Version: {version}, Vocab: {vocab_size}")
        
        # Read all sequences
        sequences = []
        degenerate_count = 0
        fragment_count = 0
        
        for seq_idx in range(num_sequences):
            seq_len = struct.unpack('<I', f.read(4))[0]
            tokens = struct.unpack(f'<{seq_len}i', f.read(seq_len * 4))
            
            if is_degenerate(tokens):
                degenerate_count += 1
            elif has_high_fragment_ratio(tokens):
                fragment_count += 1
            else:
                sequences.append(tokens)
    
    total_filtered = degenerate_count + fragment_count
    print(f"\nFiltered {total_filtered} sequences:")
    print(f"  - {degenerate_count} degenerate (high bigram repetition)")
    print(f"  - {fragment_count} corrupted (high fragment token ratio)")
    print(f"Remaining: {len(sequences)} sequences")
    
    # Write output file
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(magic)
        f.write(struct.pack('<I', version))
        f.write(struct.pack('<I', len(sequences)))
        f.write(struct.pack('<I', vocab_size))
        
        for tokens in sequences:
            f.write(struct.pack('<I', len(tokens)))
            f.write(struct.pack(f'<{len(tokens)}i', *tokens))
    
    print(f"\nWritten filtered data to {OUTPUT_PATH}")
    print(f"\nTo use: copy to training_data.grmt")

if __name__ == '__main__':
    main()

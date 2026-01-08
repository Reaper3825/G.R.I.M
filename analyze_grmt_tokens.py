#!/usr/bin/env python3
"""Analyze GRMT token distribution to understand why training plateaus."""
import struct
from collections import Counter
from pathlib import Path

def analyze_grmt(grmt_path: str, max_sequences: int = 1000):
    """Analyze token distribution in GRMT file."""
    with open(grmt_path, 'rb') as f:
        magic, version, num_seq, vocab_size = struct.unpack('<IIII', f.read(16))
        print(f'Header: magic={hex(magic)}, version={version}, num_seq={num_seq}, vocab_size={vocab_size}')
        
        all_tokens = []
        seq_lengths = []
        
        sequences_to_read = min(num_seq, max_sequences)
        for i in range(sequences_to_read):
            seq_len = struct.unpack('<I', f.read(4))[0]
            seq_lengths.append(seq_len)
            tokens = struct.unpack(f'<{seq_len}I', f.read(seq_len * 4))
            all_tokens.extend(tokens)
            
            # Skip: numeric_values (float*seq_len), numeric_mask (uint8*seq_len), 
            #       text_features (uint16*16*seq_len), text_mask (uint8*seq_len)
            f.read(seq_len * 4)       # numeric_values
            f.read(seq_len)           # numeric_mask
            f.read(seq_len * 16 * 2)  # text_features
            f.read(seq_len)           # text_mask
        
        token_counts = Counter(all_tokens)
        
        print(f'\n=== Sequence Statistics ({sequences_to_read} sequences) ===')
        print(f'Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}')
        print(f'Total tokens: {len(all_tokens):,}')
        print(f'Unique tokens: {len(token_counts):,} ({len(token_counts)/vocab_size*100:.1f}% of vocab)')
        
        print(f'\n=== Top 30 Most Common Tokens ===')
        for tok, count in token_counts.most_common(30):
            pct = count / len(all_tokens) * 100
            print(f'  Token {tok:5d}: {count:6d} ({pct:5.2f}%)')
        
        print(f'\n=== Token Range ===')
        print(f'Min token ID: {min(all_tokens)}')
        print(f'Max token ID: {max(all_tokens)}')
        
        # Token type distribution
        byte_tokens = [t for t in all_tokens if t < 256]
        atom_tokens = [t for t in all_tokens if 256 <= t < 512]
        unigram_tokens = [t for t in all_tokens if t >= 512]
        
        print(f'\n=== Token Type Distribution ===')
        print(f'Byte (0-255):     {len(byte_tokens):7,} ({len(byte_tokens)/len(all_tokens)*100:5.1f}%)')
        print(f'Atom (256-511):   {len(atom_tokens):7,} ({len(atom_tokens)/len(all_tokens)*100:5.1f}%)')
        print(f'Unigram (512+):   {len(unigram_tokens):7,} ({len(unigram_tokens)/len(all_tokens)*100:5.1f}%)')
        
        # Frequency distribution
        singletons = sum(1 for c in token_counts.values() if c == 1)
        rare = sum(1 for c in token_counts.values() if c <= 5)
        common = sum(1 for c in token_counts.values() if c >= 100)
        
        print(f'\n=== Token Frequency Distribution ===')
        print(f'Tokens appearing only once:  {singletons:5d} ({singletons/len(token_counts)*100:.1f}% of unique)')
        print(f'Tokens appearing <= 5 times: {rare:5d} ({rare/len(token_counts)*100:.1f}% of unique)')
        print(f'Tokens appearing >= 100:     {common:5d} ({common/len(token_counts)*100:.1f}% of unique)')
        
        # Coverage: what % of total tokens come from top N tokens?
        print(f'\n=== Coverage Analysis ===')
        sorted_counts = sorted(token_counts.values(), reverse=True)
        cumsum = 0
        for threshold in [50, 100, 500, 1000, 5000]:
            cumsum = sum(sorted_counts[:threshold])
            print(f'Top {threshold:4d} tokens cover: {cumsum/len(all_tokens)*100:5.1f}% of all token occurrences')
        
        # Check for suspicious patterns
        print(f'\n=== Suspicious Patterns ===')
        # Check if there's a dominant token
        top_token, top_count = token_counts.most_common(1)[0]
        if top_count / len(all_tokens) > 0.1:
            print(f'WARNING: Token {top_token} appears {top_count} times ({top_count/len(all_tokens)*100:.1f}% of all tokens)')
        
        # Check for many unused vocab entries
        unused_vocab = vocab_size - len(token_counts)
        if unused_vocab / vocab_size > 0.5:
            print(f'WARNING: {unused_vocab:,} vocab entries never appear ({unused_vocab/vocab_size*100:.1f}% of vocab)')

if __name__ == '__main__':
    grmt_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data.grmt'
    analyze_grmt(grmt_path)

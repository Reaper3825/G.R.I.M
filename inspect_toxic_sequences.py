#!/usr/bin/env python3
"""Inspect sequences that cause gradient explosions"""
import struct

data_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data.grmt'
bad_seqs = [1118, 4251, 7528, 6359]  # Loss=90 sequences

with open(data_path, 'rb') as f:
    magic = f.read(4)
    print(f'Magic bytes: {magic}')
    
    version = struct.unpack('<I', f.read(4))[0]
    num_sequences = struct.unpack('<I', f.read(4))[0]
    vocab_size = struct.unpack('<I', f.read(4))[0]  # Read vocab_size from header
    
    print(f'Version: {version}, Total sequences: {num_sequences}, Vocab size: {vocab_size}')
    print(f'\n{"="*70}')
    print(f'INSPECTING TOXIC SEQUENCES (cause gradient explosions)')
    print(f'{"="*70}\n')
    
    for seq_idx in range(min(num_sequences, max(bad_seqs) + 1)):
        seq_len = struct.unpack('<I', f.read(4))[0]
        tokens = struct.unpack(f'<{seq_len}i', f.read(seq_len * 4))
        
        if seq_idx in bad_seqs:
            print(f'Sequence {seq_idx}: length={seq_len}')
            print(f'  First 15 tokens: {list(tokens[:15])}')
            print(f'  Last 15 tokens:  {list(tokens[-15:])}')
            print(f'  Token range: [{min(tokens)}, {max(tokens)}]')
            
            # Check for anomalies
            issues = []
            if any(t < 0 for t in tokens):
                issues.append('NEGATIVE tokens')
                neg_tokens = [t for t in tokens if t < 0]
                print(f'    ❌ Found negative tokens: {neg_tokens[:10]}')
            
            if any(t > vocab_size for t in tokens):
                issues.append(f'OUT-OF-VOCAB (vocab_size={vocab_size})')
                oov_tokens = [(i, t) for i, t in enumerate(tokens) if t > vocab_size]
                print(f'    ❌ Found {len(oov_tokens)} out-of-vocab tokens')
                print(f'       Examples: {oov_tokens[:5]}')
            
            if seq_len > 2048:
                issues.append(f'TOO LONG (max=2048, got={seq_len})')
            
            if seq_len < 10:
                issues.append('TOO SHORT (<10 tokens)')
            
            # Check for repeated tokens
            if len(set(tokens)) < len(tokens) * 0.3:
                issues.append('HIGH REPETITION')
                from collections import Counter
                most_common = Counter(tokens).most_common(3)
                print(f'    ⚠️  Most repeated tokens: {most_common}')
            
            if issues:
                print(f'  🔥 ISSUES: {" | ".join(issues)}')
            else:
                print(f'  ✓ No obvious data issues found')
            
            print()

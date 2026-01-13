# Diagnose GRMT file structure and token distribution
# Determine ACTUAL BOS/EOS token IDs from vocab and constants
import struct
import os
from collections import Counter

# Token layout constants (from Unigram.hpp)
BYTE_TOKEN_OFFSET = 0
ATOM_TOKEN_OFFSET = 256
kAtomTypeCount = 25  # AtomType::ATOM_ACTIVE_COUNT from enum (0-24)
UNIGRAM_VOCAB_OFFSET = ATOM_TOKEN_OFFSET + kAtomTypeCount  # 256 + 25 = 281

# Special token indices in unigram vocab (from Unigram.hpp defaults)
UNK_IDX = 0  # <unk>
PAD_IDX = 1  # <pad>  
BOS_IDX = 2  # <s>
EOS_IDX = 3  # </s>

# Calculate actual global token IDs
ACTUAL_BOS_ID = UNIGRAM_VOCAB_OFFSET + BOS_IDX  # 281 + 2 = 283
ACTUAL_EOS_ID = UNIGRAM_VOCAB_OFFSET + EOS_IDX  # 281 + 3 = 284

print(f'=== Token Layout ===')
print(f'BYTE_TOKEN_OFFSET = {BYTE_TOKEN_OFFSET}')
print(f'ATOM_TOKEN_OFFSET = {ATOM_TOKEN_OFFSET}')
print(f'kAtomTypeCount = {kAtomTypeCount}')
print(f'UNIGRAM_VOCAB_OFFSET = {UNIGRAM_VOCAB_OFFSET}')
print(f'')
print(f'Calculated BOS ID = {ACTUAL_BOS_ID} (UNIGRAM_VOCAB_OFFSET + {BOS_IDX})')
print(f'Calculated EOS ID = {ACTUAL_EOS_ID} (UNIGRAM_VOCAB_OFFSET + {EOS_IDX})')
print('')

# Read vocab.txt to verify special tokens
vocab_path = 'D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.txt'
with open(vocab_path, 'r', encoding='utf-8') as vf:
    vocab_lines = vf.readlines()
    print(f'vocab.txt has {len(vocab_lines)} lines')
    print(f'First 5 lines (should be special tokens):')
    for i, line in enumerate(vocab_lines[:5]):
        token = line.split('\t')[0] if '\t' in line else line.split()[0]
        global_id = UNIGRAM_VOCAB_OFFSET + i
        print(f'  Line {i} = "{token}" → Global ID = {global_id}')
    print('')

grmt_path = 'D:/G.R.I.M/resources/models/GRIM-text/training/data/training_data.grmt'
file_size = os.path.getsize(grmt_path)
print(f'=== GRMT File Analysis ===')
print(f'File size: {file_size / 1024 / 1024:.2f} MB')

with open(grmt_path, 'rb') as f:
    # Read header
    magic = f.read(4)
    print(f'Magic: {magic.decode("ascii")}')
    
    version = struct.unpack('<I', f.read(4))[0]
    print(f'Version: {version}')
    
    num_sequences = struct.unpack('<I', f.read(4))[0]
    print(f'Num sequences: {num_sequences}')
    
    vocab_size = struct.unpack('<I', f.read(4))[0]
    print(f'Vocab size in GRMT: {vocab_size}')
    
    # Calculate what EOS *should* be based on GRMT vocab size
    implied_unigram_vocab = vocab_size - UNIGRAM_VOCAB_OFFSET
    print(f'Implied unigram vocab = {vocab_size} - {UNIGRAM_VOCAB_OFFSET} = {implied_unigram_vocab}')
    print('')
    
    total_tokens = 0
    total_valid_targets = 0
    
    # Track token frequency
    token_counter = Counter()
    target_counter = Counter()
    masked_target_count = 0
    
    print('Reading sequences...')
    for seq_idx in range(min(200, num_sequences)):
        seq_len = struct.unpack('<I', f.read(4))[0]
        total_tokens += seq_len
        
        # Read token_ids
        token_ids_bytes = f.read(seq_len * 4)
        token_ids = struct.unpack(f'<{seq_len}I', token_ids_bytes)
        
        # Read targets (GRMT v5+)
        targets_bytes = f.read(seq_len * 4)
        targets = struct.unpack(f'<{seq_len}i', targets_bytes)  # signed int for -1 masking
        
        # Read numeric values (float32)
        numeric_values = f.read(seq_len * 4)
        
        # Read numeric mask (uint8)
        numeric_mask = f.read(seq_len)
        
        # Read text features (uint16, kTextFeatureDim=16)
        text_features = f.read(seq_len * 16 * 2)
        
        # Read text feature mask (uint8)  
        text_feature_mask = f.read(seq_len)
        
        # Count tokens and targets
        for t in token_ids:
            token_counter[t] += 1
        for t in targets:
            if t >= 0:
                target_counter[t] += 1
                total_valid_targets += 1
            else:
                masked_target_count += 1
        
        if seq_idx < 3:
            print(f'\n  Seq {seq_idx}: len={seq_len}')
            print(f'    Token[0..9]: {token_ids[:10]}')
            print(f'    Target[0..9]: {targets[:10]}')
            eos_in_tokens = sum(1 for t in token_ids if t == ACTUAL_EOS_ID)
            eos_in_targets = sum(1 for t in targets if t == ACTUAL_EOS_ID)
            bos_in_tokens = sum(1 for t in token_ids if t == ACTUAL_BOS_ID)
            print(f'    BOS({ACTUAL_BOS_ID}) in tokens: {bos_in_tokens}')
            print(f'    EOS({ACTUAL_EOS_ID}) in tokens: {eos_in_tokens}')
            print(f'    EOS({ACTUAL_EOS_ID}) in targets: {eos_in_targets}')
            # Also check for token 277 (old/wrong ID)
            old_eos_in_tokens = sum(1 for t in token_ids if t == 277)
            old_eos_in_targets = sum(1 for t in targets if t == 277)
            if old_eos_in_tokens > 0 or old_eos_in_targets > 0:
                print(f'    *** OLD_ID(277) in tokens: {old_eos_in_tokens}, in targets: {old_eos_in_targets}')
    
    print(f'\n=== SUMMARY (first 200 sequences) ===')
    print(f'Total tokens: {total_tokens}')
    print(f'Total valid targets: {total_valid_targets}')
    print(f'Total masked targets (-1): {masked_target_count}')
    
    # Check actual EOS/BOS presence
    actual_bos_count = token_counter.get(ACTUAL_BOS_ID, 0)
    actual_eos_count = token_counter.get(ACTUAL_EOS_ID, 0)
    actual_eos_targets = target_counter.get(ACTUAL_EOS_ID, 0)
    old_eos_count = token_counter.get(277, 0)
    old_eos_targets = target_counter.get(277, 0)
    
    print(f'\nActual BOS ({ACTUAL_BOS_ID}) in token_ids: {actual_bos_count}')
    print(f'Actual EOS ({ACTUAL_EOS_ID}) in token_ids: {actual_eos_count}')
    print(f'Actual EOS ({ACTUAL_EOS_ID}) in targets: {actual_eos_targets}')
    print(f'')
    print(f'Old ID 277 in token_ids: {old_eos_count}')
    print(f'Old ID 277 in targets: {old_eos_targets}')
    
    # Most common tokens
    print(f'\n=== Top 20 Most Common Tokens ===')
    for tid, count in token_counter.most_common(20):
        pct = count / total_tokens * 100
        if tid < ATOM_TOKEN_OFFSET:
            desc = f'BYTE({tid:02x}={chr(tid) if 32 <= tid < 127 else "ctrl"})'
        elif tid < UNIGRAM_VOCAB_OFFSET:
            desc = f'ATOM({tid - ATOM_TOKEN_OFFSET})'
        else:
            unigram_idx = tid - UNIGRAM_VOCAB_OFFSET
            if unigram_idx < len(vocab_lines):
                token_str = vocab_lines[unigram_idx].split('\t')[0] if '\t' in vocab_lines[unigram_idx] else vocab_lines[unigram_idx].split()[0]
                desc = f'"{token_str}"'
            else:
                desc = f'UNIGRAM({unigram_idx})'
        print(f'  Token {tid:5d}: {count:6d} ({pct:5.2f}%) - {desc}')
    
    # Most common targets (excluding masked)
    print(f'\n=== Top 10 Most Common Targets ===')
    for tid, count in target_counter.most_common(10):
        pct = count / total_valid_targets * 100
        if tid < ATOM_TOKEN_OFFSET:
            desc = f'BYTE({tid:02x}={chr(tid) if 32 <= tid < 127 else "ctrl"})'
        elif tid < UNIGRAM_VOCAB_OFFSET:
            desc = f'ATOM({tid - ATOM_TOKEN_OFFSET})'
        else:
            unigram_idx = tid - UNIGRAM_VOCAB_OFFSET
            if unigram_idx < len(vocab_lines):
                token_str = vocab_lines[unigram_idx].split('\t')[0] if '\t' in vocab_lines[unigram_idx] else vocab_lines[unigram_idx].split()[0]
                desc = f'"{token_str}"'
            else:
                desc = f'UNIGRAM({unigram_idx})'
        print(f'  Token {tid:5d}: {count:6d} ({pct:5.2f}%) - {desc}')

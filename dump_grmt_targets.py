#!/usr/bin/env python3
"""
Dump GRMT v6 file targets to verify position 0 masking.

GRMT v6 format (per sequence):
- uint32 seq_len
- int[seq_len] token_ids
- int[seq_len] targets
- float[seq_len] numeric_values
- uint8[seq_len] numeric_mask
- uint16[seq_len * 16] text_features (16 dims per token)
- uint8[seq_len] text_feature_mask
- uint16[seq_len] byte_lengths
"""

import struct
import sys
from pathlib import Path

def read_grmt_v6(filepath: str, max_sequences: int = 10):
    """Read GRMT v6 file and dump first N sequences."""
    
    TEXT_FEATURE_DIM = 16  # kTextFeatureDim
    
    with open(filepath, 'rb') as f:
        # Header
        magic = struct.unpack('I', f.read(4))[0]
        version = struct.unpack('I', f.read(4))[0]
        num_sequences = struct.unpack('I', f.read(4))[0]
        vocab_size = struct.unpack('I', f.read(4))[0]
        
        print(f"=== GRMT Header ===")
        print(f"Magic: 0x{magic:08X} ({'GRMT' if magic == 0x474D5254 else 'INVALID'})")
        print(f"Version: {version}")
        print(f"Num sequences: {num_sequences}")
        print(f"Vocab size: {vocab_size}")
        print()
        
        if magic != 0x474D5254:
            print("ERROR: Invalid magic number!")
            return
        
        if version != 6:
            print(f"ERROR: Expected version 6, got {version}")
            return
        
        # Read sequences
        for seq_idx in range(min(num_sequences, max_sequences)):
            seq_len = struct.unpack('I', f.read(4))[0]
            
            # Read token_ids
            token_ids = list(struct.unpack(f'{seq_len}i', f.read(seq_len * 4)))
            
            # Read targets
            targets = list(struct.unpack(f'{seq_len}i', f.read(seq_len * 4)))
            
            # Read numeric_values
            f.read(seq_len * 4)  # Skip
            
            # Read numeric_mask
            f.read(seq_len)  # Skip
            
            # Read text_features (16 dims * seq_len)
            f.read(seq_len * TEXT_FEATURE_DIM * 2)  # Skip
            
            # Read text_feature_mask
            f.read(seq_len)  # Skip
            
            # Read byte_lengths
            f.read(seq_len * 2)  # Skip
            
            # Dump sequence
            print(f"=== Sequence {seq_idx} (len={seq_len}) ===")
            print(f"First 10 token_ids: {token_ids[:10]}")
            print(f"First 10 targets:   {targets[:10]}")
            print()
            
            # Check position 0 masking
            if targets[0] != -1:
                print(f"  ⚠️  WARNING: targets[0] = {targets[0]} (should be -1!)")
            else:
                print(f"  ✓ targets[0] = -1 (correctly masked)")
            
            # Count masked vs unmasked
            masked_count = sum(1 for t in targets if t == -1)
            print(f"  Masked positions: {masked_count}/{seq_len} ({100*masked_count/seq_len:.1f}%)")
            
            # Verify next-token prediction: targets[i] should equal token_ids[i+1]
            mismatches = 0
            for i in range(seq_len - 1):
                if targets[i] != -1 and targets[i] != token_ids[i + 1]:
                    mismatches += 1
                    if mismatches <= 3:
                        print(f"  ⚠️  Mismatch at pos {i}: targets[{i}]={targets[i]} != token_ids[{i+1}]={token_ids[i+1]}")
            if mismatches > 3:
                print(f"  ... and {mismatches - 3} more mismatches")
            elif mismatches == 0:
                print(f"  ✓ All unmasked targets correctly predict next token")
            
            print()

if __name__ == "__main__":
    grmt_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\training_data.grmt"
    
    if len(sys.argv) > 1:
        grmt_path = sys.argv[1]
    
    max_seq = 10
    if len(sys.argv) > 2:
        max_seq = int(sys.argv[2])
    
    print(f"Reading: {grmt_path}")
    print(f"Max sequences: {max_seq}")
    print()
    
    if not Path(grmt_path).exists():
        print(f"ERROR: File not found: {grmt_path}")
        sys.exit(1)
    
    read_grmt_v6(grmt_path, max_seq)

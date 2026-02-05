#!/usr/bin/env python3
"""
Inspect GRMT file to debug BOS token and target masking at positions 0, 1.

Issue: User observed [GRAD_NONTARGET_EQUATION] token_pos=1 vocab_id=20000 target=560
       but expected position 1 to be BOS with masked target.

This script:
1. Reads GRMT v6 file format
2. Shows first 5 tokens and targets for first N sequences
3. Identifies whether BOS (275) is at position 0 or 1
4. Checks if targets[-1] masking is correct
"""

import struct
import sys
from pathlib import Path

# Token ID constants from copilot-instructions.md
BOS_TOKEN_ID = 275
EOS_TOKEN_ID = 276
PAD_TOKEN_ID = 274

# Text feature dimension from tokenizer
TEXT_FEATURE_DIM = 16  # kTextFeatureDim (was 5 - WRONG!)

def read_grmt_v6(path: Path, max_sequences: int = 10):
    """Read GRMT v6 format and show first N sequences."""
    
    with open(path, 'rb') as f:
        # Header: 16 bytes
        header = f.read(16)
        magic, version, num_sequences, vocab_size = struct.unpack('<IIII', header)
        
        print(f"=== GRMT File Header ===")
        print(f"Magic: 0x{magic:08X} ({'OK' if magic == 0x474D5254 else 'INVALID'})")
        print(f"Version: {version} ({'OK' if version == 6 else 'UNSUPPORTED'})")
        print(f"Sequences: {num_sequences}")
        print(f"Vocab size: {vocab_size}")
        print()
        
        if version != 6:
            print(f"ERROR: Only GRMT v6 supported, got v{version}")
            return
        
        print(f"=== First {min(max_sequences, num_sequences)} Sequences ===")
        print()
        
        bos_at_pos0_count = 0
        bos_at_pos1_count = 0
        target_masked_at_0_count = 0
        target_masked_at_1_count = 0
        
        for seq_idx in range(min(max_sequences, num_sequences)):
            # Read sequence length
            seq_len_bytes = f.read(4)
            seq_len = struct.unpack('<I', seq_len_bytes)[0]
            
            # Read token_ids (int array)
            token_ids = list(struct.unpack(f'<{seq_len}i', f.read(seq_len * 4)))
            
            # Read targets (int array)
            targets = list(struct.unpack(f'<{seq_len}i', f.read(seq_len * 4)))
            
            # Read and skip numeric_values, numeric_mask, text_features, text_mask, byte_lengths
            if seq_len > 0:
                f.read(seq_len * 4)  # numeric_values (float)
                f.read(seq_len * 1)  # numeric_mask (uint8)
                f.read(seq_len * TEXT_FEATURE_DIM * 2)  # text_features (uint16)
                f.read(seq_len * 1)  # text_mask (uint8)
                f.read(seq_len * 2)  # byte_lengths (uint16)
            
            # Analysis
            has_bos_at_0 = token_ids[0] == BOS_TOKEN_ID if seq_len > 0 else False
            has_bos_at_1 = token_ids[1] == BOS_TOKEN_ID if seq_len > 1 else False
            target_masked_at_0 = targets[0] == -1 if seq_len > 0 else False
            target_masked_at_1 = targets[1] == -1 if seq_len > 1 else False
            
            if has_bos_at_0:
                bos_at_pos0_count += 1
            if has_bos_at_1:
                bos_at_pos1_count += 1
            if target_masked_at_0:
                target_masked_at_0_count += 1
            if target_masked_at_1:
                target_masked_at_1_count += 1
            
            # Print details for first sequences
            if seq_idx < 5:
                print(f"--- Sequence {seq_idx} (len={seq_len}) ---")
                for pos in range(min(5, seq_len)):
                    token_id = token_ids[pos]
                    target = targets[pos]
                    
                    # Decode special tokens
                    token_name = ""
                    if token_id == BOS_TOKEN_ID:
                        token_name = " <BOS>"
                    elif token_id == EOS_TOKEN_ID:
                        token_name = " <EOS>"
                    elif token_id == PAD_TOKEN_ID:
                        token_name = " <PAD>"
                    
                    target_name = ""
                    if target == -1:
                        target_name = " [MASKED]"
                    elif target == BOS_TOKEN_ID:
                        target_name = " <BOS>"
                    elif target == EOS_TOKEN_ID:
                        target_name = " <EOS>"
                    
                    print(f"  pos={pos}: token_id={token_id:5d}{token_name:8s} target={target:5d}{target_name}")
                
                # Show last 2 positions too
                if seq_len > 7:
                    print(f"  ...")
                for pos in range(max(5, seq_len - 2), seq_len):
                    token_id = token_ids[pos]
                    target = targets[pos]
                    
                    token_name = ""
                    if token_id == BOS_TOKEN_ID:
                        token_name = " <BOS>"
                    elif token_id == EOS_TOKEN_ID:
                        token_name = " <EOS>"
                    elif token_id == PAD_TOKEN_ID:
                        token_name = " <PAD>"
                    
                    target_name = ""
                    if target == -1:
                        target_name = " [MASKED]"
                    elif target == BOS_TOKEN_ID:
                        target_name = " <BOS>"
                    elif target == EOS_TOKEN_ID:
                        target_name = " <EOS>"
                    
                    print(f"  pos={pos}: token_id={token_id:5d}{token_name:8s} target={target:5d}{target_name}")
                print()
        
        # Summary stats
        print(f"=== Statistics (first {min(max_sequences, num_sequences)} sequences) ===")
        print(f"BOS at position 0: {bos_at_pos0_count}/{min(max_sequences, num_sequences)}")
        print(f"BOS at position 1: {bos_at_pos1_count}/{min(max_sequences, num_sequences)}")
        print(f"Target masked (-1) at position 0: {target_masked_at_0_count}/{min(max_sequences, num_sequences)}")
        print(f"Target masked (-1) at position 1: {target_masked_at_1_count}/{min(max_sequences, num_sequences)}")
        
        if bos_at_pos0_count > 0 and target_masked_at_0_count == 0:
            print()
            print("⚠️  WARNING: BOS tokens at position 0 but targets NOT masked!")
            print("   This may indicate GRMT file was generated before target masking fix.")
            print("   Consider regenerating: python DataCollection/main_data_collection.py")
        
        if bos_at_pos1_count > 0 and target_masked_at_1_count == 0:
            print()
            print("⚠️  WARNING: BOS tokens at position 1 but targets NOT masked!")
            print("   This suggests position 1 should also be masked if it's BOS.")


def main():
    # Default GRMT file path
    default_path = Path("d:/G.R.I.M/resources/models/GRIM-text/training/data/training_data.grmt")
    
    if len(sys.argv) > 1:
        grmt_path = Path(sys.argv[1])
    else:
        grmt_path = default_path
    
    if not grmt_path.exists():
        print(f"ERROR: GRMT file not found: {grmt_path}")
        print(f"Usage: python {sys.argv[0]} [grmt_file_path]")
        sys.exit(1)
    
    print(f"Inspecting: {grmt_path}")
    print()
    
    # Read more sequences for statistics
    read_grmt_v6(grmt_path, max_sequences=100)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Filter out sequences longer than max_seq_len from training data.
Creates a new filtered GRMT file.
"""
import struct
import sys

MAX_SEQ_LEN = 900  # Must match your model config

input_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data.grmt'
output_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data_filtered.grmt'

print(f"Filtering sequences longer than {MAX_SEQ_LEN} tokens...")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print()

sequences_read = 0
sequences_kept = 0
sequences_filtered = 0
tokens_removed = 0

with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
    # Read and copy header
    magic = infile.read(4)
    version = struct.unpack('<I', infile.read(4))[0]
    num_sequences = struct.unpack('<I', infile.read(4))[0]
    
    print(f"Input file: GRMT v{version}, {num_sequences} sequences")
    
    # Write header with placeholder count (we'll update it later)
    outfile.write(magic)
    outfile.write(struct.pack('<I', version))
    num_sequences_pos = outfile.tell()
    outfile.write(struct.pack('<I', 0))  # Placeholder
    
    # Process sequences
    for seq_idx in range(num_sequences):
        seq_len = struct.unpack('<I', infile.read(4))[0]
        tokens = infile.read(seq_len * 4)
        
        sequences_read += 1
        
        if seq_len <= MAX_SEQ_LEN:
            # Keep this sequence
            outfile.write(struct.pack('<I', seq_len))
            outfile.write(tokens)
            sequences_kept += 1
        else:
            # Filter out this sequence
            sequences_filtered += 1
            tokens_removed += seq_len
            if sequences_filtered <= 10:
                print(f"  Filtered seq {seq_idx}: {seq_len} tokens (exceeds {MAX_SEQ_LEN})")
    
    # Update sequence count in output file
    outfile.seek(num_sequences_pos)
    outfile.write(struct.pack('<I', sequences_kept))

print()
print("="*70)
print(f"✓ Filtering complete!")
print(f"  Total sequences read: {sequences_read}")
print(f"  Sequences kept:       {sequences_kept} ({100*sequences_kept/sequences_read:.1f}%)")
print(f"  Sequences filtered:   {sequences_filtered} ({100*sequences_filtered/sequences_read:.1f}%)")
print(f"  Tokens removed:       {tokens_removed:,}")
print()
print(f"Output file: {output_path}")
print()
print("Next steps:")
print("  1. Backup your original file if needed")
print("  2. Replace training_data.grmt with training_data_filtered.grmt")
print("  3. Restart training with filtered data")

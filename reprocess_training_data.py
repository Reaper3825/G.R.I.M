#!/usr/bin/env python3
"""
Reprocess existing training_data.grmt:
1. Decode sequences from GRMT
2. Filter out sequences > max_seq_len
3. Re-encode back to GRMT format
"""
import struct
import sys
from pathlib import Path

MAX_SEQ_LEN = 900  # Must match your model config

input_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data.grmt'
output_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data_filtered.grmt'
backup_path = r'D:\G.R.I.M\resources\models\GRIM-text\training\data\training_data_backup.grmt'

print(f"Reprocessing training data with max_seq_len={MAX_SEQ_LEN}")
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print()

# Read all sequences
sequences = []
vocab_size = 0

print("[1/3] Reading existing GRMT file...")
with open(input_path, 'rb') as f:
    magic = f.read(4)
    if magic != b'TRMG':
        print(f"ERROR: Invalid magic bytes: {magic}")
        sys.exit(1)
    
    version = struct.unpack('<I', f.read(4))[0]
    num_sequences = struct.unpack('<I', f.read(4))[0]
    
    # Check if vocab_size is in header (version 1 has it)
    if version == 1:
        vocab_size = struct.unpack('<I', f.read(4))[0]
    
    print(f"  Version: {version}")
    print(f"  Total sequences: {num_sequences}")
    if vocab_size > 0:
        print(f"  Vocab size: {vocab_size}")
    
    # Read all sequences
    for i in range(num_sequences):
        seq_len = struct.unpack('<I', f.read(4))[0]
        tokens = struct.unpack(f'<{seq_len}i', f.read(seq_len * 4))
        sequences.append(list(tokens))
        
        if (i + 1) % 500 == 0:
            print(f"  Read {i + 1}/{num_sequences} sequences...", end='\r')
    
    print(f"  ✓ Read all {num_sequences} sequences" + " " * 30)

# Filter sequences
print(f"\n[2/3] Filtering sequences longer than {MAX_SEQ_LEN} tokens...")
filtered_sequences = []
filtered_count = 0
filtered_tokens = 0

for seq in sequences:
    if len(seq) <= MAX_SEQ_LEN:
        filtered_sequences.append(seq)
    else:
        filtered_count += 1
        filtered_tokens += len(seq)

kept_percentage = 100 * len(filtered_sequences) / len(sequences)
print(f"  Kept: {len(filtered_sequences)} sequences ({kept_percentage:.1f}%)")
print(f"  Filtered: {filtered_count} sequences ({100 - kept_percentage:.1f}%)")
print(f"  Tokens removed: {filtered_tokens:,}")

if filtered_count == 0:
    print("\n  No sequences to filter! All sequences are within limits.")
    sys.exit(0)

# Show some examples of filtered sequences
if filtered_count > 0:
    print(f"\n  Examples of filtered sequences:")
    example_count = 0
    for i, seq in enumerate(sequences):
        if len(seq) > MAX_SEQ_LEN:
            print(f"    Sequence {i}: {len(seq)} tokens (exceeds {MAX_SEQ_LEN})")
            example_count += 1
            if example_count >= 5:
                break

# Write filtered GRMT
print(f"\n[3/3] Writing filtered GRMT file...")
with open(output_path, 'wb') as f:
    magic = b'TRMG'
    version = 1
    num_sequences_out = len(filtered_sequences)
    
    f.write(magic)
    f.write(struct.pack('<I', version))
    f.write(struct.pack('<I', num_sequences_out))
    
    # Write vocab_size if we have it
    if vocab_size > 0:
        f.write(struct.pack('<I', vocab_size))
    
    for seq in filtered_sequences:
        seq_len = len(seq)
        f.write(struct.pack('<I', seq_len))
        f.write(struct.pack(f'<{seq_len}i', *seq))

print(f"  ✓ Wrote {len(filtered_sequences)} sequences to {output_path}")

print("\n" + "="*70)
print("✓ Reprocessing complete!")
print()
print("Next steps:")
print(f"  1. Backup original: mv {input_path} {backup_path}")
print(f"  2. Replace with filtered: mv {output_path} {input_path}")
print(f"  3. Restart training")
print()
print("Or run these commands:")
print(f'  Move-Item "{input_path}" "{backup_path}"')
print(f'  Move-Item "{output_path}" "{input_path}"')

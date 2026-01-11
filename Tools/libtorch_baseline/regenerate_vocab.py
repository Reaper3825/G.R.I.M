#!/usr/bin/env python3
"""
Regenerate vocab.bin with correct sequential token IDs.

The original vocab.bin has corrupted token IDs:
- ID 277 is missing (gap)
- ID 278 is duplicated (both space and 'e')

This script reads vocab.txt and creates a new vocab.bin with correct IDs:
token_id = UNIGRAM_VOCAB_OFFSET + index
where UNIGRAM_VOCAB_OFFSET = 256 + ATOM_VOCAB_SIZE = 256 + 26 = 282
"""

import struct
import os

# Token layout constants - MUST MATCH C++ code in Unigram.hpp!
# kAtomTypeCount = 17 (ATOM_NONE through ATOM_EXPRESSION)
# UNIGRAM_VOCAB_OFFSET = 256 + 17 = 273
BYTE_VOCAB_SIZE = 256          # [0-255] = byte tokens
ATOM_VOCAB_SIZE = 17           # Must match kAtomTypeCount in Unigram.hpp
UNIGRAM_VOCAB_OFFSET = BYTE_VOCAB_SIZE + ATOM_VOCAB_SIZE  # 273

vocab_txt_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.txt"
vocab_bin_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.bin"
backup_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.bin.bak"

# Backup existing vocab.bin
if os.path.exists(vocab_bin_path):
    print(f"Backing up existing vocab.bin to {backup_path}")
    import shutil
    shutil.copy(vocab_bin_path, backup_path)

# Read vocab.txt
pieces = []
print(f"Reading {vocab_txt_path}...")
with open(vocab_txt_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n\r')
        if not line:
            continue
        # Split on last tab (text may contain tabs)
        idx = line.rfind('\t')
        if idx < 0:
            print(f"Warning: skipping invalid line: {line!r}")
            continue
        text = line[:idx]
        score_str = line[idx+1:]
        try:
            score = float(score_str)
        except ValueError:
            print(f"Warning: invalid score '{score_str}' for piece '{text}'")
            continue
        pieces.append((text, score))

print(f"Read {len(pieces)} pieces from vocab.txt")

# Write new vocab.bin
print(f"Writing {vocab_bin_path}...")
with open(vocab_bin_path, 'wb') as f:
    # Magic: KTMG
    f.write(b'KTMG')
    
    # Version: 3 (uint16)
    f.write(struct.pack('<H', 3))
    
    # Checksum: 0 (uint32) - not used
    f.write(struct.pack('<I', 0))
    
    # Config vocab_size: number of unigram pieces (uint32)
    f.write(struct.pack('<I', len(pieces)))
    
    # Max piece length: 64 (uint32)
    f.write(struct.pack('<I', 64))
    
    # Flags: 3 bytes reserved
    f.write(bytes([0, 0, 0]))
    
    # Total vocab size including bytes + atoms (uint32)
    total_vocab = BYTE_VOCAB_SIZE + ATOM_VOCAB_SIZE + len(pieces)
    f.write(struct.pack('<I', total_vocab))
    
    # Write pieces
    for i, (text, score) in enumerate(pieces):
        text_bytes = text.encode('utf-8')
        token_id = UNIGRAM_VOCAB_OFFSET + i  # Sequential ID starting at 282
        
        # Length (uint32)
        f.write(struct.pack('<I', len(text_bytes)))
        # Text bytes
        f.write(text_bytes)
        # Score (float32)
        f.write(struct.pack('<f', score))
        # Token ID (int32)
        f.write(struct.pack('<I', token_id))

print(f"Written {len(pieces)} pieces with token IDs {UNIGRAM_VOCAB_OFFSET} to {UNIGRAM_VOCAB_OFFSET + len(pieces) - 1}")
print(f"Total vocab size: {total_vocab}")

# Verify
print("\nVerifying new vocab.bin:")
with open(vocab_bin_path, 'rb') as f:
    f.read(4)  # magic
    version = struct.unpack('<H', f.read(2))[0]
    f.read(4)  # checksum
    config_vocab = struct.unpack('<I', f.read(4))[0]
    f.read(4)  # max_len
    f.read(3)  # flags
    total_vocab = struct.unpack('<I', f.read(4))[0]
    
    print(f"Version {version}, unigram count {config_vocab}, total vocab {total_vocab}")
    print("First 10 pieces:")
    
    for i in range(10):
        piece_len = struct.unpack('<I', f.read(4))[0]
        text = f.read(piece_len).decode('utf-8')
        score = struct.unpack('<f', f.read(4))[0]
        token_id = struct.unpack('<I', f.read(4))[0]
        expected_id = UNIGRAM_VOCAB_OFFSET + i
        status = "✓" if token_id == expected_id else f"✗ (expected {expected_id})"
        print(f"  {i}: '{text}' score={score:.4f} token_id={token_id} {status}")

print("\n✅ Vocab regeneration complete!")
print(f"   UNIGRAM_VOCAB_OFFSET = {UNIGRAM_VOCAB_OFFSET}")
print(f"   First unigram token ID = {UNIGRAM_VOCAB_OFFSET}")
print(f"   Last unigram token ID = {UNIGRAM_VOCAB_OFFSET + len(pieces) - 1}")

#!/usr/bin/env python3
"""
Inspect vocab.bin file structure to understand what's in it
"""
import struct
import sys

path = "D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.bin"

with open(path, 'rb') as f:
    # Read header
    magic = struct.unpack('I', f.read(4))[0]
    version = struct.unpack('H', f.read(2))[0]
    print(f"Magic: 0x{magic:08x} ({chr(magic&0xFF)}{chr((magic>>8)&0xFF)}{chr((magic>>16)&0xFF)}{chr((magic>>24)&0xFF)})")
    print(f"Version: {version}")
    
    # Version 2+ has checksum
    if version >= 2:
        checksum = struct.unpack('I', f.read(4))[0]
        print(f"Checksum: 0x{checksum:08x}")
    
    # Read vocab_size
    vocab_size = struct.unpack('I', f.read(4))[0]
    print(f"Config vocab_size: {vocab_size}")
    
    # Read max_length
    max_length = struct.unpack('I', f.read(4))[0]
    print(f"Max token length: {max_length}")
    
    # Version 2+ normalization flags
    if version >= 2:
        nfkc = struct.unpack('?', f.read(1))[0]
        lower = struct.unpack('?', f.read(1))[0]
        fallback = struct.unpack('?', f.read(1))[0]
        print(f"NFKC: {nfkc}, Lowercase: {lower}, Fallback: {fallback}")
    
    # Actual vocab size (second occurrence)
    vocab_size_2 = struct.unpack('I', f.read(4))[0]
    print(f"Actual vocab_size (2nd field): {vocab_size_2}")
    
    # Read first few tokens
    print(f"\nFirst 500 tokens:")
    for i in range(min(500, vocab_size_2)):
        length = struct.unpack('I', f.read(4))[0]
        token = f.read(length).decode('utf-8', errors='ignore')
        print(f"  {i}: ({length} bytes) '{token}'")
    
    # Skip to near the end
    f.seek(-100, 2)  # 100 bytes from end
    print(f"\nLast section (last 100 bytes):")
    data = f.read()
    print(f"  {repr(data)}")
    
    print(f"\nFile size: {f.seek(0, 2)} bytes")
    print(f"Expected size for {vocab_size_2} tokens: ~{4*vocab_size_2 + 1000} bytes (rough estimate)")


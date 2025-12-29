#!/usr/bin/env python3
"""
Find token IDs for common garbage patterns to debug tokenizer
"""
import struct

path = "D:/G.R.I.M/resources/models/GRIM-text/training/models/vocab.bin"

tokens_to_find = ["tot", "oot", "stor", "tto", "ooo", "tof", "ofo", "pcq", "sts"]

with open(path, 'rb') as f:
    # Skip header
    f.read(6)  # magic + version
    f.read(4)  # checksum (version 2)
    f.read(4)  # config vocab_size
    f.read(4)  # max_length
    f.read(3)  # flags (version 2)
    
    # Read actual vocab
    vocab_size = struct.unpack('I', f.read(4))[0]
    print(f"Reading {vocab_size} tokens\n")
    
    for i in range(vocab_size):
        length = struct.unpack('I', f.read(4))[0]
        token = f.read(length).decode('utf-8', errors='ignore')
        
        if token in tokens_to_find:
            print(f"Token ID {i}: '{token}'")
        
        # Also look for tokens that end with "k" or start with "q" (for partial pattern match)
        if any(token == t for t in tokens_to_find):
            tokens_to_find.remove(token)
    
    print(f"\nNot found: {tokens_to_find}")
    
    # Print some tokens to understand vocab
    print(f"\nSample tokens from vocab:")
    f.seek(6+4+4+4+3+4)  # Back to start of tokens
    for i in range(min(100, vocab_size)):
        length = struct.unpack('I', f.read(4))[0]
        token = f.read(length).decode('utf-8', errors='ignore')
        if i < 20 or i % 100 == 0:
            print(f"  {i}: '{token}'")


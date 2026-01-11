#!/usr/bin/env python3
"""Decode token 277 from vocab.bin to understand whitespace output"""

import struct

vocab_path = r'D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.bin'

with open(vocab_path, 'rb') as f:
    # Read KTMG magic
    magic = f.read(4)
    print(f'Magic: {magic}')
    
    # Version (2 bytes)
    version = struct.unpack('<H', f.read(2))[0]
    print(f'Version: {version}')
    
    # Checksum (4 bytes) 
    checksum = struct.unpack('<I', f.read(4))[0]
    
    # Config vocab size (4 bytes)
    config_vocab_size = struct.unpack('<I', f.read(4))[0]
    print(f'Config vocab size: {config_vocab_size}')
    
    # Max length (4 bytes)
    max_length = struct.unpack('<I', f.read(4))[0]
    
    # Flags (3 bytes)
    flags = f.read(3)
    
    # Total vocab size (4 bytes)
    total_vocab_size = struct.unpack('<I', f.read(4))[0]
    print(f'Total vocab size: {total_vocab_size}')
    
    # UNIGRAM_VOCAB_OFFSET = 256 (bytes) + 17 (atoms) = 273
    UNIGRAM_VOCAB_OFFSET = 273
    
    print(f'\n=== First 20 Unigram Pieces ===')
    print(f'(Token 277 = UNIGRAM_VOCAB_OFFSET + 4 = unigram index 4)')
    print()
    
    # Read pieces
    for i in range(20):
        length = struct.unpack('<I', f.read(4))[0]
        text = f.read(length)
        score = struct.unpack('<f', f.read(4))[0]
        token_id = struct.unpack('<i', f.read(4))[0]
        
        try:
            text_str = text.decode('utf-8')
        except:
            text_str = repr(text)
        
        # Make whitespace visible
        vis_text = (text_str
            .replace(' ', '\u00b7')  # Middle dot for space
            .replace('\n', '\\n')
            .replace('\t', '\\t')
            .replace('\r', '\\r'))
        
        marker = ' <== TOKEN 277' if token_id == 277 else ''
        print(f'  Index {i:2d} -> Token {token_id:4d}: "{vis_text}" (bytes: {text.hex()}) score={score:.4f}{marker}')

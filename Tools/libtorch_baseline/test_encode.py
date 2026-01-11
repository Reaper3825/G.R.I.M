#!/usr/bin/env python3
"""Test the vocabulary encoding for a prompt."""

import struct
import sys

UNIGRAM_TOKEN_BASE = 512

def load_vocab_v3(path):
    """Load GRIM vocab.bin format v3."""
    pieces = {}  # text -> token_id
    
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'KTMG', f"Bad magic: {magic}"
        
        version = struct.unpack('<H', f.read(2))[0]
        assert version >= 2, f"Bad version: {version}"
        
        checksum = struct.unpack('<I', f.read(4))[0]
        config_vocab = struct.unpack('<I', f.read(4))[0]  # unigram count
        max_len = struct.unpack('<I', f.read(4))[0]       # 4 bytes, not 2!
        flags = f.read(3)                                  # 3 bytes, not 4!
        total_vocab = struct.unpack('<I', f.read(4))[0]
        
        print(f"Vocab v{version}: {total_vocab} total tokens ({config_vocab} unigram pieces)")
        
        # Read pieces - length is 4 bytes!
        num_unigram_pieces = config_vocab  # Use the explicit count, not total - 512
        for i in range(num_unigram_pieces):
            piece_len = struct.unpack('<I', f.read(4))[0]  # 4 bytes, not 2!
            text = f.read(piece_len).decode('utf-8', errors='replace')
            score = struct.unpack('<f', f.read(4))[0]
            
            if version >= 3:
                token_id = struct.unpack('<I', f.read(4))[0]
            else:
                token_id = UNIGRAM_TOKEN_BASE + i
            
            pieces[text] = token_id
    
    return pieces


def encode(text, pieces):
    """Encode text using vocab (longest match), fallback to bytes."""
    tokens = []
    pos = 0
    
    while pos < len(text):
        # Try longest match first (up to 32 chars)
        found = False
        for length in range(min(32, len(text) - pos), 0, -1):
            substr = text[pos:pos+length]
            if substr in pieces:
                tokens.append((substr, pieces[substr], 'VOCAB'))
                pos += length
                found = True
                break
        
        # Fallback to byte
        if not found:
            byte_val = ord(text[pos])
            tokens.append((text[pos], byte_val, 'BYTE'))
            pos += 1
    
    return tokens


def main():
    vocab_path = r"D:\G.R.I.M\resources\models\GRIM-text\training\data\vocab.bin"
    
    print("Loading vocabulary...")
    pieces = load_vocab_v3(vocab_path)
    print(f"Loaded {len(pieces)} pieces")
    
    # Test encoding
    test_prompts = ["Hello world", "The", " the", "a", "Hello", "world"]
    
    for prompt in test_prompts:
        print(f"\n=== Encoding: '{prompt}' ===")
        tokens = encode(prompt, pieces)
        for text, tid, source in tokens:
            print(f"  '{text}' -> {tid} ({source})")
        
        # Check if any tokens are byte fallback
        byte_count = sum(1 for _, _, s in tokens if s == 'BYTE')
        vocab_count = sum(1 for _, _, s in tokens if s == 'VOCAB')
        print(f"  Summary: {vocab_count} vocab, {byte_count} byte-fallback")
    
    # Check if specific pieces exist
    print("\n=== Checking specific pieces ===")
    test_pieces = ["Hello", "hello", "Hello world", "The", " the ", " ", "H", "e", "l", "o"]
    for piece in test_pieces:
        if piece in pieces:
            print(f"  '{piece}' -> {pieces[piece]} (exists)")
        else:
            print(f"  '{piece}' -> NOT FOUND")


if __name__ == '__main__':
    main()

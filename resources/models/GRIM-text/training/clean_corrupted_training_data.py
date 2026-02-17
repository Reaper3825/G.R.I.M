#!/usr/bin/env python3
"""
Clean training data by removing log probability contamination.

This script:
1. Loads corrupted training_data.grmt
2. Decodes each sequence
3. Removes log probability patterns: " -XX.YYYY"
4. Re-tokenizes cleaned text
5. Saves to training_data_cleaned.grmt

Usage:
    python clean_corrupted_training_data.py
    python clean_corrupted_training_data.py --input data/training_data.grmt --output data/training_data_cleaned.grmt
"""

import re
import sys
import struct
import argparse
from pathlib import Path
from typing import List, Tuple
import subprocess


def load_vocab(vocab_path: str) -> dict:
    """Load vocab.txt file"""
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            vocab[idx] = line.rstrip('\n')
    return vocab


def decode_tokens(tokens: List[int], vocab: dict) -> str:
    """Decode token IDs to text"""
    pieces = []
    for tid in tokens:
        if tid in vocab:
            pieces.append(vocab[tid])
        elif 4 <= tid < 260:
            # Byte fallback (token IDs 4-259 = byte values 0-255)
            byte_val = tid - 4
            pieces.append(chr(byte_val) if 32 <= byte_val <= 126 else f"<{byte_val:02X}>")
        else:
            pieces.append(f"<UNK{tid}>")
    
    text = ''.join(pieces)
    text = text.replace('▁', ' ')  # Unigram space marker
    return text


def remove_log_prob_contamination(text: str) -> str:
    """
    Remove log probability patterns from text.
    
    Pattern: " -XX.YYYY" where XX is 1-3 digits and YYYY is 2-6 digits
    Examples: " -11.4139", " -9.557647", " -13.0234"
    """
    # Primary pattern: space followed by negative float (2-6 decimal places)
    text = re.sub(r'\s-\d{1,3}\.\d{2,6}', ' ', text)
    
    # Also remove isolated minus signs left over
    text = re.sub(r'\s-\s', ' ', text)
    
    # Clean up multiple spaces
    text = ' '.join(text.split())
    
    return text


def encode_text_with_tokenizer(text: str, tokenizer_exe: str) -> List[int]:
    """
    Encode text back to token IDs using external tokenizer binary.
    
    Args:
        text: Cleaned text to encode
        tokenizer_exe: Path to tokenizer executable (e.g., vocab_tool.exe)
    
    Returns:
        List of token IDs
    """
    try:
        # Write text to temp file
        temp_input = Path("temp_encode_input.txt")
        temp_output = Path("temp_encode_output.bin")
        
        with open(temp_input, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Call tokenizer: vocab_tool.exe encode temp_input.txt temp_output.bin
        result = subprocess.run(
            [tokenizer_exe, "encode", str(temp_input), str(temp_output)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Read binary token output
        with open(temp_output, 'rb') as f:
            token_data = f.read()
        
        # Parse token IDs (4 bytes each, little-endian)
        num_tokens = len(token_data) // 4
        tokens = []
        for i in range(num_tokens):
            token = struct.unpack('<I', token_data[i*4:(i+1)*4])[0]
            tokens.append(token)
        
        # Clean up temp files
        temp_input.unlink()
        temp_output.unlink()
        
        return tokens
        
    except Exception as e:
        print(f"Error encoding text: {e}")
        return []


def simple_tokenize(text: str, vocab: dict) -> List[int]:
    """
    Simple greedy tokenization fallback (no external tokenizer needed).
    Uses longest-match greedy algorithm.
    """
    # Invert vocab: text -> id
    text_to_id = {v: k for k, v in vocab.items()}
    
    tokens = []
    pos = 0
    
    while pos < len(text):
        # Try longest matches first
        best_match = None
        best_len = 0
        
        # Check all vocab entries for match at current position
        for vocab_text, vocab_id in text_to_id.items():
            if text[pos:pos+len(vocab_text)] == vocab_text:
                if len(vocab_text) > best_len:
                    best_match = vocab_id
                    best_len = len(vocab_text)
        
        if best_match is not None:
            tokens.append(best_match)
            pos += best_len
        else:
            # Byte fallback
            byte_val = ord(text[pos]) if pos < len(text) else 0
            tokens.append(byte_val)
            pos += 1
    
    return tokens


def load_training_sequences(data_path: str) -> List[List[int]]:
    """Load sequences from .grmt file"""
    sequences = []
    
    with open(data_path, 'rb') as f:
        data = f.read()
    
    offset = 0
    while offset < len(data) - 100:
        if offset + 4 > len(data):
            break
        
        seq_len = struct.unpack('<I', data[offset:offset+4])[0]
        
        if 16 <= seq_len <= 512:
            token_bytes_needed = seq_len * 4
            if offset + 4 + token_bytes_needed <= len(data):
                tokens = []
                valid = True
                
                for i in range(seq_len):
                    tok_offset = offset + 4 + i * 4
                    token = struct.unpack('<I', data[tok_offset:tok_offset+4])[0]
                    
                    if token > 100000:
                        valid = False
                        break
                    tokens.append(token)
                
                if valid and len(tokens) == seq_len:
                    sequences.append(tokens)
                    offset += 4 + token_bytes_needed
                    continue
        
        offset += 4
    
    return sequences


def save_cleaned_sequences(sequences: List[List[int]], output_path: str):
    """
    Save cleaned sequences to .grmt format.
    Format: [seq_len: 4 bytes][token_ids: seq_len * 4 bytes]
    """
    with open(output_path, 'wb') as f:
        # Write magic header
        f.write(b'GRMT')
        
        for seq in sequences:
            # Write sequence length
            f.write(struct.pack('<I', len(seq)))
            
            # Write token IDs
            for token in seq:
                f.write(struct.pack('<I', token))
    
    print(f"✓ Saved {len(sequences)} sequences to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Clean log probability contamination from training data'
    )
    parser.add_argument(
        '--input',
        default='D:/G.R.I.M/resources/models/GRIM-text/training/data/training_data.grmt',
        help='Input corrupted .grmt file'
    )
    parser.add_argument(
        '--vocab',
        default='D:/G.R.I.M/resources/models/GRIM-text/training/data/vocab.txt',
        help='Path to vocab.txt'
    )
    parser.add_argument(
        '--output',
        default='D:/G.R.I.M/resources/models/GRIM-text/training/data/training_data_cleaned.grmt',
        help='Output cleaned .grmt file'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=None,
        help='Only process first N sequences (for testing)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimum sequence length after cleaning (default: 50 tokens)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRAINING DATA CLEANING TOOL")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Min length: {args.min_length} tokens")
    print()
    
    # Load vocab
    print("Loading vocabulary...")
    vocab = load_vocab(args.vocab)
    print(f"✓ Loaded {len(vocab)} tokens")
    
    # Load training data
    print("\nLoading training data...")
    sequences = load_training_sequences(args.input)
    print(f"✓ Loaded {len(sequences)} sequences")
    
    if args.sample:
        sequences = sequences[:args.sample]
        print(f"  (Processing first {args.sample} for testing)")
    
    # Clean sequences
    print("\nCleaning sequences...")
    cleaned_sequences = []
    rejected = 0
    
    for i, tokens in enumerate(sequences):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(sequences)} ({(i+1)*100//len(sequences)}%)")
        
        # Decode
        text = decode_tokens(tokens, vocab)
        
        # Clean
        cleaned_text = remove_log_prob_contamination(text)
        
        # Re-tokenize (simple greedy approach)
        cleaned_tokens = simple_tokenize(cleaned_text, vocab)
        
        # Filter short sequences
        if len(cleaned_tokens) >= args.min_length:
            cleaned_sequences.append(cleaned_tokens)
        else:
            rejected += 1
    
    print(f"\n✓ Cleaning complete:")
    print(f"  Original: {len(sequences)} sequences")
    print(f"  Cleaned: {len(cleaned_sequences)} sequences")
    print(f"  Rejected (too short): {rejected} sequences")
    print(f"  Retention rate: {len(cleaned_sequences)*100//len(sequences)}%")
    
    # Save cleaned data
    print(f"\nSaving to {args.output}...")
    save_cleaned_sequences(cleaned_sequences, args.output)
    
    # Show sample
    print("\n" + "=" * 80)
    print("SAMPLE CLEANED SEQUENCE")
    print("=" * 80)
    if cleaned_sequences:
        sample_tokens = cleaned_sequences[0]
        sample_text = decode_tokens(sample_tokens, vocab)
        print(f"Length: {len(sample_tokens)} tokens")
        print(f"\nText preview (first 500 chars):")
        print("-" * 80)
        print(sample_text[:500])
        print("-" * 80)
    
    print("\n✓ Done! Use cleaned data file for training:")
    print(f"  {args.output}")


if __name__ == '__main__':
    main()

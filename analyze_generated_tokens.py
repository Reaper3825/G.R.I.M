#!/usr/bin/env python3
"""
Decode the actual token IDs to see what the model is generating
"""
import subprocess
import time
import requests
import struct

# Start server and make request
server_path = r'resources\models\GRIM-text\training\build\Release\grim_text_server.exe'
vocab_path = r'resources\models\GRIM-text\training\models\vocab.bin'
model_path = r'resources\models\GRIM-text\checkpoints\checkpoint_epoch_1.bin'

proc = subprocess.Popen([server_path, vocab_path, model_path, '11435'], 
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(5)

# Make request
try:
    resp = requests.post('http://127.0.0.1:11435/api/generate',
                        json={'prompt': 'Hello', 'max_tokens': 20, 'temperature': 0.3},
                        timeout=60)
    
    if resp.status_code == 200:
        text = resp.json().get('response', '')
        print(f"Output text: {text}\n")
        
        # Load vocab to decode token IDs
        vocab = []
        with open(vocab_path, 'rb') as f:
            f.read(6)  # magic + version
            f.read(4)  # checksum
            f.read(4)  # config vocab_size
            f.read(4)  # max_length
            f.read(3)  # flags
            vocab_size = struct.unpack('I', f.read(4))[0]
            
            for i in range(vocab_size):
                length = struct.unpack('I', f.read(4))[0]
                token = f.read(length).decode('utf-8', errors='ignore')
                vocab.append(token)
        
        # Try to reconstruct what token IDs would produce this output
        print("Character analysis:")
        print(f"Prompt 'Hello' would be tokens: {[vocab.index(c) if c in vocab else f'?{ord(c)}?' for c in 'Hello']}")
        print(f"\nFirst 40 chars as token IDs (if single-char tokens):")
        
        for i, char in enumerate(text[:40]):
            # Try to find token ID
            if char in vocab:
                token_id = vocab.index(char)
                print(f"  pos {i}: '{char}' = token {token_id}")
            else:
                print(f"  pos {i}: '{char}' = NOT IN VOCAB (ASCII {ord(char)})")
        
        # Specific check: what are tokens 99-120?
        print(f"\nTokens 99-120 in vocab:")
        for i in range(99, min(121, len(vocab))):
            print(f"  Token {i}: '{vocab[i]}'")
            
except Exception as e:
    print(f"Error: {e}")

finally:
    proc.terminate()


#!/usr/bin/env python3
"""
Test model with different temperature settings to see if greedy decoding helps
"""
import subprocess
import time
import requests
import json
import os

server_path = r'resources\models\GRIM-text\training\build_vs_cuda\Release\grim_text_server.exe'
vocab = r'resources\models\GRIM-text\training\models\vocab.bin'
model = r'resources\models\GRIM-text\checkpoints\checkpoint_epoch_5.bin'

for temp in [0.1, 0.5, 0.7, 1.0]:
    print(f"\n{'='*60}")
    print(f"Testing with temperature={temp}")
    print(f"{'='*60}")
    
    # Kill old server if running
    os.system("taskkill /F /FI \"WINDOWTITLE eq grim_text_server*\" >nul 2>&1")
    time.sleep(1)
    
    # Start server
    proc = subprocess.Popen([server_path, vocab, model, '11435'], 
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(5)
    
    try:
        resp = requests.post('http://127.0.0.1:11435/api/generate',
                            json={'prompt': 'Hello world', 'max_tokens': 30, 'temperature': temp},
                            timeout=60)
        
        if resp.status_code == 200:
            text = resp.json().get('response', '')
            print(f"Output: {text}")
            
            # Count non-ASCII characters
            non_ascii = sum(1 for c in text if ord(c) > 127)
            # Count uppercase letters
            upper = sum(1 for c in text if c.isupper())
            # Count digits
            digits = sum(1 for c in text if c.isdigit())
            
            print(f"Non-ASCII: {non_ascii}, Uppercase: {upper}, Digits: {digits}")
            
            # Check coherence
            words = text.split()
            print(f"Words: {words[:10]}")
        else:
            print(f"Error: {resp.status_code}")
    
    except Exception as e:
        print(f"Request failed: {e}")
    
    finally:
        proc.terminate()
        time.sleep(0.5)

print("\n" + "="*60)
print("Summary: If temperature=0.1 produces better output, model is just oversampling")
print("If all temperatures produce garbage, model isn't trained well")
print("="*60)


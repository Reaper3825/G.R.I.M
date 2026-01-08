#!/usr/bin/env python3
"""
Simple test: Ask model to repeat 'hello' five times
"""
import subprocess
import time
import requests

server_path = r'resources\models\GRIM-text\training\build\Release\grim_text_server.exe'
vocab = r'resources/models/GRIM-text/training/data/vocab.bin'
model = r'resources/models/GRIM-text/checkpoints/checkpoint_epoch_1.bin'

proc = subprocess.Popen([server_path, vocab, model, '11435'], 
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(5)

try:
    prompt = "Repeat the word 'hello' five times."
    
    resp = requests.post('http://127.0.0.1:11435/api/generate',
                        json={'prompt': prompt, 'max_tokens': 50, 'temperature': 0.3},
                        timeout=60)
    
    if resp.status_code == 200:
        text = resp.json().get('response', '')
        print(f"Prompt: {prompt}")
        print(f"\nModel output:\n{text}")
        print(f"\nLength: {len(text)} chars")
        
        # Check if it actually repeated hello
        hello_count = text.lower().count('hello')
        print(f"'hello' appears {hello_count} times")

except Exception as e:
    print(f"Error: {e}")

finally:
    proc.terminate()

#!/usr/bin/env python3
"""Quick test to verify token validation fixes"""
import subprocess
import time
import requests
import json
import sys
import os

# Start server
print('[Test] Starting grim_text_server...')
server_path = r'resources\models\GRIM-text\training\build_vs_cuda\Release\grim_text_server.exe'
vocab = r'resources\models\GRIM-text\training\models\vocab.bin'
model = r'resources\models\GRIM-text\checkpoints\checkpoint_epoch_5.bin'

if not os.path.exists(server_path):
    print(f'ERROR: Server not found at {server_path}')
    sys.exit(1)

proc = subprocess.Popen([server_path, vocab, model, '11435'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Wait for server to start
print('[Test] Waiting for server initialization...')
for i in range(60):
    time.sleep(1)
    try:
        resp = requests.get('http://127.0.0.1:11435/', timeout=1)
        print(f'[Test] Server is up (after {i+1}s)')
        break
    except:
        if (i+1) % 10 == 0:
            print(f'[Test] Still waiting... ({i+1}s)')

# Send request
print('[Test] Sending test prompt to /api/generate...')
try:
    print('[Test] Making request with 120s timeout...')
    resp = requests.post('http://127.0.0.1:11435/api/generate', 
                        json={'prompt': 'Hello', 'max_tokens': 20, 'temperature': 0.7}, 
                        timeout=120)
    
    if resp.status_code == 200:
        data = resp.json()
        response_text = data.get('response', '')
        print(f'\n[SUCCESS] Model responded:')
        print(f'  "{response_text}"')
        
        # Check for garbage tokens (rhrhrhcqcq pattern or tokens beyond vocab)
        if 'rhrhrhcq' in response_text.lower() or 'cqoz' in response_text.lower():
            print('\n[WARNING] Found garbage token pattern in response!')
        elif len(response_text.split()) > 100:
            print(f'\n[WARNING] Response suspiciously long: {len(response_text.split())} tokens')
        else:
            print('\n[GOOD] Response looks clean (no garbage tokens detected)')
    else:
        print(f'[ERROR] HTTP {resp.status_code}: {resp.text}')
        
except requests.exceptions.Timeout:
    print('[ERROR] Request timed out (model still loading?)')
except requests.exceptions.ConnectionError:
    print('[ERROR] Could not connect to server on port 11435')
except Exception as e:
    print(f'[ERROR] {e}')

finally:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except:
        proc.kill()
    print('\n[Test] Server stopped')

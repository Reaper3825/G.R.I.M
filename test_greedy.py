#!/usr/bin/env python3
"""
Test with greedy decoding (temperature near 0) to see if it helps
"""
import subprocess
import time
import requests

server_path = r'resources\models\GRIM-text\training\build_vs_cuda\Release\grim_text_server.exe'
vocab = r'resources\models\GRIM-text\training\models\vocab.bin'
model = r'resources\models\GRIM-text\checkpoints\checkpoint_epoch_1.bin'

proc = subprocess.Popen([server_path, vocab, model, '11435'], 
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(5)

try:
    for temp in [0.01, 0.1, 0.3, 0.7]:
        resp = requests.post('http://127.0.0.1:11435/api/generate',
                            json={'prompt': 'Hello world', 'max_tokens': 30, 'temperature': temp},
                            timeout=60)
        
        if resp.status_code == 200:
            text = resp.json().get('response', '')
            print(f"Temperature {temp}: {text[:60]}")
        time.sleep(1)

except Exception as e:
    print(f"Error: {e}")

finally:
    proc.terminate()

print("\nConclusion:")
print("If greedy (temp=0.01) still produces garbage, the logits themselves are wrong")
print("If greedy produces better output, it's a sampling/temperature issue")

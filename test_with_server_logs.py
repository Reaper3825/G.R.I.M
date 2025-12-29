#!/usr/bin/env python3
"""
Test GRIM-text server with full stderr logging to see what vocab_size and logits.size() are
"""
import subprocess
import time
import requests
import json
import os
import sys

def main():
    # Start server with stderr redirected to a log file
    log_file = "server_stderr.log"
    print(f"[Test] Starting server, logging to {log_file}...")
    
    # Kill any existing process on port 11435
    os.system("netstat -ano | findstr :11435 | findstr LISTENING > nul && taskkill /F /PID 11435 2>nul")
    time.sleep(0.5)
    
    # Start server with output redirection
    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            ["D:\\G.R.I.M\\resources\\models\\GRIM-text\\training\\build_vs_cuda\\Release\\grim_text_server.exe"],
            cwd="D:\\G.R.I.M",
            stdout=log,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            text=True
        )
    
    time.sleep(3)
    print("[Test] Server should be starting...")
    
    # Try to make a request
    try:
        url = "http://127.0.0.1:11435/api/generate"
        payload = {
            "prompt": "Hello",
            "max_tokens": 30,
            "temperature": 0.7
        }
        
        print(f"[Test] Sending request to {url}...")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("response", "")
            print(f"\n[SUCCESS] Got response: {text}\n")
        else:
            print(f"[ERROR] Status {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
    
    finally:
        # Stop server
        proc.terminate()
        time.sleep(0.5)
        proc.kill()
        
        # Print server logs
        print("\n" + "="*60)
        print("SERVER LOGS (stderr):")
        print("="*60)
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                # Print lines with vocab, logits, or debug info
                for line in lines:
                    if any(x in line for x in ["vocab", "logits", "DEBUG", "ERROR", "Config"]):
                        print(line.rstrip())
        except Exception as e:
            print(f"Could not read log: {e}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GRIM Model Test Script
Tests the GRIM-text model with various questions via the HTTP server.
Automatically starts and stops the server.
"""

import requests
import time
import sys
import subprocess
import os
import signal
import atexit

# Server configuration
SERVER_URL = "http://localhost:11435"
GRIM_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_EXE = os.path.join(GRIM_ROOT, "resources", "models", "GRIM-text", "training", "build_vs_cuda", "Release", "grim_text_server.exe")
VOCAB_PATH = os.path.join(GRIM_ROOT, "resources", "models", "GRIM-text", "training", "models", "vocab.bin")
MODEL_PATH = os.path.join(GRIM_ROOT, "resources", "models", "GRIM-text", "grim_text.bin")

# Global server process
server_process = None

# Test questions
TEST_QUESTIONS = [
    "Hello, how are you?",
    "What is your name?",
    "What can you help me with?",
    "Tell me a joke.",
    "What is the capital of France?",
    "Explain what machine learning is in simple terms.",
    "Write a short poem about coding.",
    "What is 2 + 2?",
    "How do I make coffee?",
    "What's the weather like today?",
]


def kill_existing_servers():
    """Kill any existing grim_text_server processes."""
    print("Checking for existing server instances...")
    try:
        # Windows-specific: use taskkill
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq grim_text_server.exe"],
            capture_output=True,
            text=True
        )
        if "grim_text_server.exe" in result.stdout:
            print("  Found existing server instance(s), terminating...")
            subprocess.run(
                ["taskkill", "/F", "/IM", "grim_text_server.exe"],
                capture_output=True
            )
            time.sleep(1)
            print("  ✓ Existing servers terminated")
        else:
            print("  No existing servers found")
    except Exception as e:
        print(f"  Warning: Could not check for existing servers: {e}")


def start_server():
    """Start the GRIM server."""
    global server_process
    
    print("\nStarting GRIM server...")
    print(f"  Model: {MODEL_PATH}")
    
    # Get model file size if it exists
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"  Model size: {size_mb:.1f} MB")
    
    # Check if executable exists
    if not os.path.exists(SERVER_EXE):
        print(f"[ERROR] Server executable not found: {SERVER_EXE}")
        return False
    
    # Check if vocab exists
    if not os.path.exists(VOCAB_PATH):
        print(f"[WARNING] Vocab file not found: {VOCAB_PATH}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found: {MODEL_PATH}")
    
    try:
        # Create log file for server output
        log_file = os.path.join(GRIM_ROOT, "server_test_output.log")
        log_handle = open(log_file, 'w')
        
        # Start the server process with output redirected
        server_process = subprocess.Popen(
            [SERVER_EXE],
            cwd=GRIM_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        )
        
        print(f"  Server started (PID: {server_process.pid})")
        print(f"  Logs: {log_file}")
        
        # Wait for server to be ready
        print("  Waiting for server to initialize...")
        for i in range(60):  # Wait up to 60 seconds
            time.sleep(1)
            if check_server():
                print("  ✓ Server is ready!")
                return True
            # Check if process died
            if server_process.poll() is not None:
                log_handle.close()
                print(f"  [ERROR] Server process exited with code {server_process.returncode}")
                # Print last 20 lines of output
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        print(f"  Server output (last 20 lines):")
                        for line in lines[-20:]:
                            print(f"    {line.rstrip()}")
                except:
                    pass
                return False
            if i % 5 == 4:  # Print every 5 seconds
                print(f"  Still waiting... ({i+1}s)")
        
        log_handle.close()
        print("  [ERROR] Server failed to start within 60 seconds")
        # Print last 30 lines to see where it's stuck
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"  Server output (last 30 lines):")
                for line in lines[-30:]:
                    print(f"    {line.rstrip()}")
        except:
            pass
        return False
        
    except Exception as e:
        print(f"  [ERROR] Failed to start server: {e}")
        return False


def stop_server():
    """Stop the GRIM server."""
    global server_process
    
    print("\nStopping GRIM server...")
    
    if server_process is not None:
        try:
            # Try graceful termination first
            if sys.platform == "win32":
                server_process.terminate()
            else:
                server_process.send_signal(signal.SIGTERM)
            
            # Wait a bit for graceful shutdown
            try:
                server_process.wait(timeout=5)
                print(f"  ✓ Server stopped gracefully (PID: {server_process.pid})")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                server_process.kill()
                server_process.wait()
                print(f"  ✓ Server force-killed (PID: {server_process.pid})")
                
        except Exception as e:
            print(f"  Warning: Error stopping server: {e}")
        
        server_process = None
    
    # Also kill any remaining instances
    kill_existing_servers()


def check_server():
    """Check if the GRIM server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False


def ask_question(prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
    """Send a question to the GRIM model and get a response."""
    try:
        payload = {
            "prompt": prompt,
            "max_tokens": min(max_tokens, 64),  # Limit to 64 tokens for faster response
            "temperature": temperature
        }
        response = requests.post(
            f"{SERVER_URL}/api/generate",
            json=payload,
            timeout=60  # Increase timeout to 60 seconds
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", data.get("text", str(data)))
        else:
            return f"[Error: HTTP {response.status_code}] {response.text}"
    except requests.exceptions.Timeout:
        return "[Error: Request timed out after 60s]"
    except requests.exceptions.ConnectionError:
        return "[Error: Could not connect to server]"
    except Exception as e:
        return f"[Error: {str(e)}]"


def main():
    # Register cleanup on exit
    atexit.register(stop_server)
    
    print("=" * 60)
    print("  GRIM-text Model Test")
    print("=" * 60)
    print()
    
    # Kill any existing server instances
    kill_existing_servers()
    
    # Start the server
    if not start_server():
        print("\n[ERROR] Failed to start server. Exiting.")
        sys.exit(1)
    
    print("\n" + "-" * 60)
    
    try:
        # Run through test questions
        for i, question in enumerate(TEST_QUESTIONS, 1):
            print(f"\n[Question {i}/{len(TEST_QUESTIONS)}]")
            print(f"Q: {question}")
            print()
            
            start_time = time.time()
            answer = ask_question(question)
            elapsed = time.time() - start_time
            
            print(f"A: {answer}")
            print(f"\n(Response time: {elapsed:.2f}s)")
            print("-" * 60)
            
            # Small delay between questions
            time.sleep(0.5)
        
        print("\n" + "=" * 60)
        print("  Test Complete!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    finally:
        # Stop the server
        stop_server()


if __name__ == "__main__":
    main()

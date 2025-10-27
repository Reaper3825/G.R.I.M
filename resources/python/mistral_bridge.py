#!/usr/bin/env python3
"""
Persistent Mistral bridge for intent classification.
Loads model once, stays alive, processes JSON commands via stdin.
"""

import sys
import json
import subprocess
from typing import Optional

def log(msg: str):
    """Log to stderr (never stdout - that's for JSON protocol)"""
    print(f"[Mistral Bridge] {msg}", file=sys.stderr, flush=True)

def send_json(obj: dict):
    """Send JSON response to stdout"""
    print(json.dumps(obj), flush=True)

def call_ollama(prompt: str, model: str = "mistral:latest") -> Optional[str]:
    """Call Ollama API for intent classification"""
    try:
        # Use ollama run with JSON output
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            log(f"Ollama error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        log("Ollama timeout (10s)")
        return None
    except Exception as e:
        log(f"Ollama call failed: {e}")
        return None

def classify_intent(text: str) -> dict:
    """Classify text as command or banter using Mistral"""
    
    prompt = f"""You are GRIM's intent classifier.
Decide if this message is a COMMAND (actionable instruction) or BANTER (casual conversation).
Respond ONLY with valid JSON: {{"intent":"command"}} or {{"intent":"banter"}}.

Examples:
- "open notepad" ? {{"intent":"command"}}
- "hey there" ? {{"intent":"banter"}}
- "close the window" ? {{"intent":"command"}}
- "thanks!" ? {{"intent":"banter"}}

Message: "{text}"
Response:"""

    response = call_ollama(prompt)
    
    if not response:
        return {"intent": "unknown", "error": "ollama_failed"}
    
    # Extract JSON from response
    try:
        # Find JSON in response (Mistral might include extra text)
        start = response.find('{')
        end = response.find('}', start)
        
        if start == -1 or end == -1:
            log(f"No JSON found in response: {response}")
            return {"intent": "unknown", "error": "invalid_response"}
        
        json_str = response[start:end+1]
        result = json.loads(json_str)
        
        intent = result.get("intent", "unknown").lower()
        
        if intent not in ["command", "banter"]:
            log(f"Invalid intent value: {intent}")
            return {"intent": "unknown", "error": "invalid_intent"}
        
        return {"intent": intent}
        
    except json.JSONDecodeError as e:
        log(f"JSON parse error: {e}")
        return {"intent": "unknown", "error": "json_parse_failed"}

def persistent_loop():
    """Main loop - read commands from stdin, process, write to stdout"""
    
    log("Starting persistent mode")
    
    # Check if Ollama is available
    try:
        subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        log("Ollama available")
        send_json({"status": "ready"})
    except Exception as e:
        log(f"Ollama not available: {e}")
        send_json({"status": "error", "message": "ollama_not_available"})
        return
    
    # Process commands
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                log(f"Invalid JSON from GRIM: {line} ({e})")
                send_json({"status": "error", "message": "invalid_json"})
                continue
            
            cmd = req.get("command")
            
            if cmd == "exit":
                send_json({"status": "bye"})
                break
                
            elif cmd == "classify":
                text = req.get("text", "")
                if not text:
                    send_json({"status": "error", "message": "missing_text"})
                    continue
                
                log(f"Classifying: {text}")
                result = classify_intent(text)
                send_json({"status": "ok", "result": result})
                
            else:
                send_json({"status": "error", "message": f"unknown_command: {cmd}"})
    
    except (KeyboardInterrupt, EOFError):
        log("Graceful shutdown (CTRL+C or pipe closed)")
        send_json({"status": "bye"})

if __name__ == "__main__":
    persistent_loop()

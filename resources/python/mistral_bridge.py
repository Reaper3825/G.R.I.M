#!/usr/bin/env python3
"""
Persistent LLM bridge for intent classification.
Loads model once, stays alive, processes JSON commands via stdin.
Uses configured LLM via Ollama for fast intent detection.
"""

import sys
import json
import subprocess
from typing import Optional
from pathlib import Path

# Load AI config to get model name
def load_ai_config():
    """Load ai_config.json to get configured model"""
    try:
        config_path = Path(__file__).parent.parent.parent / "ai_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return config.get("default_model", "llama3.1:8b")
    except Exception as e:
        log(f"Failed to load ai_config.json: {e}")
    return "llama3.1:8b"  # Fallback default

# Global model name from config
MODEL_NAME = load_ai_config()

def log(msg: str):
    """Log to stderr (never stdout - that's for JSON protocol)"""
    print(f"[LLM Bridge] {msg}", file=sys.stderr, flush=True)

def send_json(obj: dict):
    """Send JSON response to stdout"""
    print(json.dumps(obj), flush=True)

def call_ollama(prompt: str, model: str = None) -> Optional[str]:
    """Call Ollama API for intent classification"""
    if model is None:
        model = MODEL_NAME
    
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
    """Classify text as command or banter using Llama"""
    
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
        # Find JSON in response (model might include extra text)
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
    
    log(f"Starting persistent mode with model: {MODEL_NAME}")
    
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

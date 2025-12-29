#!/usr/bin/env python3
"""
GRIM-text HTTP Server - Python Edition
Ollama-compatible API using ctypes to call C++ model
"""

import ctypes
import json
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Global model handle (will be initialized on startup)
model_lib = None
g_model = None
g_tokenizer = None

def initialize_model():
    """Load the GRIM model DLL and initialize"""
    global model_lib, g_model, g_tokenizer
    
    # Load the model library (adjust path as needed)
    dll_path = Path("resources/models/GRIM-text/training/build/Release/grim_language_model_gpu_impl.dll")
    if not dll_path.exists():
        print(f"ERROR: Model DLL not found at {dll_path}")
        return False
    
    try:
        model_lib = ctypes.CDLL(str(dll_path))
        print(f"✓ Loaded model library: {dll_path}")
        
        # TODO: Define ctypes function signatures for model initialization
        # For now, we'll use subprocess to call the existing server
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

@app.route('/api/tags', methods=['GET'])
def get_tags():
    """List available models"""
    return jsonify({
        "models": [
            {
                "name": "grim-text",
                "modified_at": datetime.now().isoformat() + "Z",
                "size": 332200000,
                "digest": "sha256:grim"
            }
        ]
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate text completion"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 256)
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        print(f"[GRIM-text] Generating response for: {prompt[:50]}...")
        
        # TODO: Call C++ model via ctypes
        # For now, return placeholder
        response_text = "[Python server placeholder - C++ integration pending]"
        
        return jsonify({
            "model": "grim-text",
            "created_at": datetime.now().isoformat() + "Z",
            "response": response_text,
            "done": True
        })
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat completion endpoint"""
    try:
        data = request.json
        messages = data.get('messages', [])
        
        # Convert chat messages to prompt
        prompt = ""
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role == 'user':
                prompt += f"User: {content}\n"
            elif role == 'assistant':
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant: "
        
        # Generate response
        response_text = "[Python server placeholder]"
        
        return jsonify({
            "model": "grim-text",
            "created_at": datetime.now().isoformat() + "Z",
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "done": True
        })
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("="*40)
    print("  GRIM-text HTTP Server (Python)")
    print("  Ollama-compatible API")
    print("="*40)
    
    # Initialize model
    if not initialize_model():
        print("Failed to initialize model")
        sys.exit(1)
    
    print("[GRIM-text] Starting server on http://127.0.0.1:11435")
    print("[GRIM-text] API endpoints:")
    print("  - GET  /api/tags")
    print("  - POST /api/generate")
    print("  - POST /api/chat")
    print("[GRIM-text] Press Ctrl+C to stop")
    
    # Run Flask server
    app.run(host='127.0.0.1', port=11435, debug=False)

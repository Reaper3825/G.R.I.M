"""
Test to understand the prompt echo + garbage pattern
"""
import requests
import json

def test_generation(prompt, max_tokens=50, temp=0.3):
    response = requests.post(
        'http://localhost:11435/api/generate',
        json={
            'model': 'grim',
            'prompt': prompt,
            'stream': False,
            'options': {
                'num_predict': max_tokens,
                'temperature': temp
            }
        },
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        return data.get('response', '')
    return None

# Test with different prompt types
test_cases = [
    ("Hello", 20),
    ("The", 20),
    ("Repeat: hello", 20),
    ("Say hello", 20),
    ("Q: What is 2+2? A:", 30),
]

print("Testing generation patterns:\n")
for prompt, max_tok in test_cases:
    result = test_generation(prompt, max_tok, temp=0.1)  # Low temp for consistency
    if result:
        # Remove the prompt from response to see what was generated
        if result.startswith(prompt.lower().replace(' ', '')):
            generated_only = result[len(prompt.replace(' ', '')):]
        else:
            generated_only = result
        
        print(f"Prompt: {repr(prompt):25s}")
        print(f"  Full output: {repr(result[:80])}")
        print(f"  Generated:   {repr(generated_only[:80])}")
        print()

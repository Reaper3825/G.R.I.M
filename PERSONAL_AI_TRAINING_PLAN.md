# GRIM Personal AI Companion - Training Strategy
**Your JARVIS-Style AI Assistant**

---

## 🎯 What GRIM Needs to Learn

### Core Capabilities (JARVIS-Style):
1. **Command Execution** - "Open notepad", "Search for X", "Set timer"
2. **Conversational Assistant** - Natural dialogue, context awareness
3. **System Control** - Manage windows, apps, files, settings
4. **Information Retrieval** - Answer questions, look up data
5. **Proactive Suggestions** - "You usually check email now", "Battery low"
6. **Personality** - Brief, direct, helpful (like your config shows)

---

## 📊 Data Sources for YOUR Personal GRIM

### 1. **Your Existing GRIM Usage Data** ✅ (BEST SOURCE)
**What to mine:**
- Your Ollama conversation logs (if any)
- Your command history from `memory.json`
- Your voice transcripts (Whisper outputs)
- Your NLP rule successes (learned patterns)
- Your error corrections and feedback

**Why this is GOLD:**
- It's YOUR language, YOUR commands, YOUR patterns
- Model will learn exactly how YOU talk to it
- No generic internet garbage

**Where to find it:**
```
D:\G.R.I.M\memory.json              - Your conversation history
D:\G.R.I.M\grim.log                 - All interactions logged
D:\G.R.I.M\logs\*.log               - Build logs (not useful)
~/.ollama/logs (if exists)          - Ollama conversation history
```

---

### 2. **Synthetic Assistant Data** 🤖 (QUICK START)
**Create training examples that match YOUR use case:**

```json
// Example conversations YOU would have with GRIM:
{
  "conversations": [
    {
      "user": "GRIM, open Visual Studio Code",
      "assistant": "Opening Visual Studio Code.",
      "action": "open_app vscode"
    },
    {
      "user": "What's my CPU usage?",
      "assistant": "CPU usage is at 34%. System running normally.",
      "action": "system_info cpu"
    },
    {
      "user": "Remind me to check the build in 10 minutes",
      "assistant": "Timer set for 10 minutes. I'll remind you about the build.",
      "action": "set_timer 600 'check the build'"
    },
    {
      "user": "Is the training model ready?",
      "assistant": "Checking... Yes, test_training_with_flatbuffer.exe completed successfully. All tests passed.",
      "action": "check_file status"
    }
  ]
}
```

**Generate 1000+ examples like this:**
- Commands you actually use
- Questions you actually ask
- System interactions you need
- Your preferred response style (brief, direct)

---

### 3. **Curated External Data** 🌐 (SUPPLEMENT)
**ONLY scrape data relevant to YOU:**

#### Tech Documentation (Your Stack):
- C++ reference (cppreference.com)
- CMake docs
- CUDA programming guides
- Windows API docs
- Git/GitHub help

#### Your Interests (Based on GRIM's purpose):
- AI/ML papers (arXiv - relevant to your work)
- System programming forums
- Voice assistant development
- Home automation/IoT

**What to AVOID:**
- Generic chat data (Reddit random threads)
- Social media noise
- Off-topic content
- Anything not related to your assistant use case

---

## 🛠️ Training Data Collection Pipeline

### Phase 1: Mine Existing GRIM Data (Week 1, Days 1-2)

**Step 1: Extract Your Conversation History**
```python
# Script to extract from memory.json and logs
import json
import re
from pathlib import Path

def extract_grim_conversations():
    data = []
    
    # Parse memory.json
    memory = json.load(open('D:/G.R.I.M/memory.json'))
    if 'last_input' in memory and 'last_reply' in memory:
        data.append({
            'user': memory['last_input'],
            'assistant': memory['last_reply']
        })
    
    # Parse grim.log for conversations
    log_pattern = r'\[AI\].*user:\s*"([^"]+)".*reply:\s*"([^"]+)"'
    with open('D:/G.R.I.M/grim.log', 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(log_pattern, line)
            if match:
                data.append({
                    'user': match.group(1),
                    'assistant': match.group(2)
                })
    
    return data
```

**Step 2: Extract Command Patterns**
```python
def extract_nlp_patterns():
    # Your NLP rules show what commands work
    rules = json.load(open('D:/G.R.I.M/nlp_rules.json'))
    
    training_data = []
    for rule in rules:
        # Convert regex patterns to example sentences
        # E.g., "open (.+)" → "open notepad", "open chrome", etc.
        examples = generate_examples_from_pattern(rule['pattern'])
        
        for example in examples:
            training_data.append({
                'user': example,
                'assistant': f"Executing: {rule['intent']}",
                'action': rule['intent']
            })
    
    return training_data
```

---

### Phase 2: Generate Synthetic Assistant Data (Week 1, Days 3-4)

**Create a template generator:**

```python
class GRIMDataGenerator:
    def __init__(self):
        # Your actual apps from aliases
        self.apps = ["notepad", "chrome", "vscode", "discord", "steam"]
        self.commands = ["open", "close", "search", "list", "show"]
        self.system_queries = ["cpu", "memory", "disk", "network", "processes"]
        
    def generate_command_examples(self, count=500):
        examples = []
        
        # App commands
        for app in self.apps:
            examples.append({
                'user': f"open {app}",
                'assistant': f"Opening {app}.",
                'action': f"open_app {app}"
            })
            examples.append({
                'user': f"launch {app}",
                'assistant': f"Launching {app}.",
                'action': f"open_app {app}"
            })
        
        # System info
        for metric in self.system_queries:
            examples.append({
                'user': f"what's my {metric} usage?",
                'assistant': f"Checking {metric}...",
                'action': f"system_info {metric}"
            })
        
        # Timers/reminders
        for minutes in [5, 10, 15, 30, 60]:
            examples.append({
                'user': f"remind me in {minutes} minutes",
                'assistant': f"Timer set for {minutes} minutes.",
                'action': f"set_timer {minutes*60}"
            })
        
        return examples
    
    def generate_conversation_examples(self, count=500):
        # Based on YOUR personality config: brief, direct
        examples = [
            {
                'user': "how are you?",
                'assistant': "Operating normally. Ready to assist."
            },
            {
                'user': "thanks",
                'assistant': "You're welcome."
            },
            {
                'user': "good job",
                'assistant': "Acknowledged. Standing by."
            }
        ]
        
        return examples
```

---

### Phase 3: Web Collection (Week 1, Days 5-7)

**Configure `source_data.json` for YOUR interests:**

```json
{
  "sources": [
    {
      "type": "github",
      "url": "https://api.github.com/repos/microsoft/vcpkg/readme",
      "keywords": ["cmake", "build", "dependencies"],
      "max_entries": 100
    },
    {
      "type": "docs",
      "url": "https://en.cppreference.com/w/cpp",
      "keywords": ["c++17", "std", "algorithm"],
      "max_entries": 200
    },
    {
      "type": "tech_docs",
      "url": "https://docs.nvidia.com/cuda/",
      "keywords": ["cuda", "gpu", "kernel"],
      "max_entries": 150
    }
  ],
  "filters": {
    "min_length": 100,
    "max_length": 5000,
    "exclude_patterns": ["advertisement", "cookie policy", "subscribe"],
    "quality_threshold": 0.8
  }
}
```

**Run the collector:**
```bash
cd D:/G.R.I.M/resources/models/GRIM-text/training
./build/Release/collect_data.exe source_data.json
```

---

## 📈 Training Timeline (Realistic for Personal AI)

### Week 1: Data Collection
- **Day 1-2:** Mine existing GRIM logs → ~100-500 examples
- **Day 3-4:** Generate synthetic assistant data → ~1000 examples
- **Day 5-7:** Web collection + verification → ~500-1000 examples
- **Total:** ~2000-3000 high-quality examples

### Week 2: Training & Integration
- **Day 8-9:** Train tokenizer on your data (BPE vocab ~10k-20k)
- **Day 10-12:** Train transformer model (3-5 epochs, GPU)
- **Day 13:** Validate model quality, test responses
- **Day 14:** Integrate into GRIM, test end-to-end

---

## 🎯 Model Configuration (Optimized for Personal Assistant)

```json
{
  "model": {
    "vocab_size": 20000,        // Smaller vocab (personal use)
    "d_model": 512,             // Smaller model (faster inference)
    "num_layers": 6,            // Fewer layers (faster, still capable)
    "num_heads": 8,             // Reduced heads
    "d_ff": 2048,               // Proportional
    "max_seq_len": 1024,        // Shorter context (assistants don't need long history)
    "use_gpu": true             // Your 3080 Ti
  },
  "training": {
    "learning_rate": 5e-5,
    "batch_size": 16,
    "num_epochs": 5,
    "warmup_steps": 100,
    "gradient_clip": 1.0
  },
  "generation": {
    "temperature": 0.7,         // Slightly deterministic
    "top_p": 0.9,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "max_new_tokens": 128       // Short responses (JARVIS style)
  }
}
```

**Why smaller model?**
- Faster responses (<50ms on GPU)
- Less VRAM (leaves room for vision models)
- Easier to fine-tune with limited data
- Personal assistant doesn't need GPT-4 level reasoning

---

## 🚀 Quick Start Commands

### 1. Generate Synthetic Data
```python
# Create generate_assistant_data.py
python scripts/generate_assistant_data.py --count 1000 --output data/synthetic_assistant.jsonl
```

### 2. Run Data Pipeline
```bash
cd resources/models/GRIM-text/training
./build/Release/collect_data.exe config_personal.json
```

### 3. Train Tokenizer
```bash
./build/Release/train_tokenizer.exe --data data/combined.txt --vocab-size 20000 --output tokenizer.bin
```

### 4. Train Model
```bash
./build/Release/train_model.exe --config config_personal.json --data data/training.grmt --output checkpoints/grim_personal.bin
```

### 5. Integrate & Test
```bash
# Update ai_config.json
{
  "backend": "grim_native",
  "grim_native": {
    "model_path": "checkpoints/grim_personal.bin",
    "tokenizer_path": "tokenizer.bin"
  }
}
```

---

## 💡 Expected Results

**Response Time:** <50ms on GPU (vs 500-2000ms with Ollama)  
**Quality:** Should handle 90%+ of your daily commands/questions  
**Personality:** Brief, direct, helpful (matches your config)  
**Offline:** 100% local, no internet required  
**Privacy:** All your data stays on your machine  

**Example Interaction:**
```
You: "GRIM, what's my system status?"
GRIM: "CPU 28%, RAM 45%, GPU 12%. All systems nominal."

You: "Open my development environment"
GRIM: "Launching Visual Studio Code and Chrome DevTools."

You: "Set a timer for the build"
GRIM: "Build typically takes 3 minutes. Timer set."
```

---

## 📝 Next Immediate Steps

**Want me to:**
1. ✅ Create the Python script to mine your GRIM logs?
2. ✅ Create the synthetic data generator for assistant examples?
3. ✅ Set up the web collector config for your tech stack?
4. ✅ All of the above?

Just say "generate the scripts" and I'll create everything you need! 🚀

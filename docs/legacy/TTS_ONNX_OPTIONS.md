# TTS ONNX Integration Options for GRIM

## 📊 Current Situation

**XTTS v2 (Current System):**
- ✅ Excellent voice quality
- ✅ Multilingual support
- ✅ Voice cloning capability
- ❌ Slow startup (120s model load)
- ❌ Heavy memory (2GB+)
- ❌ Slow inference (~500ms per sentence)
- ❌ **Not easily exportable to ONNX** (tightly coupled architecture)

## 🔍 Why XTTS v2 ONNX Export is Difficult

1. **Autoregressive GPT** - Generates tokens iteratively (hard to optimize)
2. **Conditional Inputs** - Requires reference audio embeddings
3. **Multiple Components** - GPT → Latent Codes → HiFiGAN (not independent)
4. **Dynamic Shapes** - Variable-length text/audio sequences

**Attempted Export Results:**
- ❌ Full pipeline: Architecture mismatch errors
- ❌ HiFiGAN only: Expects latent codes from GPT, not mel spectrograms
- ⚠️ Would need custom ONNX operators for full compatibility

## ✅ Recommended Solutions

### **Option 1: Piper TTS (BEST for Speed)**

**Pros:**
- ✅ **Already fully ONNX-optimized**
- ✅ **10-100x faster than XTTS v2**
- ✅ **Tiny models** (5-50MB vs 2GB)
- ✅ **Fast loading** (< 1 second)
- ✅ **Native C++ library available**
- ✅ **No Python runtime needed**
- ✅ Many high-quality voices available

**Cons:**
- ❌ No voice cloning
- ❌ English-focused (some multilingual)
- ❌ Slightly robotic compared to XTTS v2

**Implementation:**
```bash
# Download Piper
vcpkg install piper-tts:x64-windows

# Or use pre-built binaries
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_windows_amd64.zip

# Get voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

**C++ Integration:**
```cpp
#include <piper.hpp>

piper::PiperConfig config;
config.eSpeakDataPath = "D:/G.R.I.M/resources/espeak-ng-data";

piper::Voice voice;
piper::loadVoice(config, "en_US-amy-medium.onnx", voice);

std::vector<int16_t> audio;
piper::textToAudio("Hello from GRIM!", voice, audio, nullptr);
// audio is ready to play - takes ~10-50ms
```

---

### **Option 2: VITS Models + ONNX**

**Pros:**
- ✅ Simpler architecture (easy ONNX export)
- ✅ Good quality (better than Piper, close to XTTS v2)
- ✅ 5-20x faster than XTTS v2
- ✅ Multilingual support

**Cons:**
- ❌ Larger models than Piper (200-500MB)
- ❌ No voice cloning
- ❌ Requires custom ONNX integration

**Implementation:**
```python
# Export VITS to ONNX
from TTS.api import TTS
import torch

tts = TTS("tts_models/en/ljspeech/vits")
# VITS has simpler architecture, exports cleanly
```

---

### **Option 3: Hybrid Approach (RECOMMENDED)**

**Use both systems:**

```cpp
// Fast path: Piper for general responses
if (text.length() < 200 && !needsVoiceCloning) {
    return piperTTS.synthesize(text);  // 10-50ms
}

// Quality path: XTTS v2 for important/long content
else {
    return xttsV2.synthesize(text);  // 500ms but better quality
}
```

**Benefits:**
- ✅ 90% of responses use fast Piper
- ✅ Important content uses high-quality XTTS v2
- ✅ Best of both worlds

---

### **Option 4: Keep XTTS v2, Optimize Python Bridge**

**Instead of ONNX, optimize current system:**

1. **Pre-generate common phrases**
   ```python
   # Cache 100 most common responses
   responses = ["Hello", "How can I help?", "Goodbye", ...]
   for text in responses:
       audio = tts.tts(text)
       cache.store(text, audio)
   ```

2. **Persistent Python process** (already done ✅)
   - Model stays loaded
   - No startup cost per request

3. **Batch processing**
   ```python
   # Process multiple sentences at once
   texts = ["Sentence 1", "Sentence 2", "Sentence 3"]
   audios = tts.tts_batch(texts)  # Faster than sequential
   ```

4. **GPU acceleration** (already enabled ✅)
   - Using CUDA
   - 3-5x faster than CPU

---

## 📊 Performance Comparison

| Method | Startup | Inference (1s audio) | Memory | Quality | Voice Clone |
|--------|---------|---------------------|--------|---------|-------------|
| **XTTS v2 (current)** | 120s | 500ms | 2GB | ⭐⭐⭐⭐⭐ | ✅ |
| **Piper ONNX** | 0.5s | 10ms | 50MB | ⭐⭐⭐⭐ | ❌ |
| **VITS ONNX** | 3s | 50ms | 500MB | ⭐⭐⭐⭐⭐ | ❌ |
| **Hybrid (Both)** | 121s | 10ms (fast) / 500ms (quality) | 2GB | ⭐⭐⭐⭐⭐ | ✅ |
| **XTTS v2 + Cache** | 120s | 1ms (cached) / 500ms (new) | 2.5GB | ⭐⭐⭐⭐⭐ | ✅ |

---

## 🚀 Recommended Implementation Plan

### Phase 1: Add Piper TTS (1-2 days)
1. Install Piper: `vcpkg install onnxruntime:x64-windows`
2. Download Piper C++ library or use ONNX Runtime directly
3. Create `voice/tts_piper.cpp` wrapper
4. Add fast path in `voice_speak.cpp`

### Phase 2: Hybrid System (1 day)
1. Detect short/simple responses → use Piper
2. Keep XTTS v2 for complex/important content
3. Add configuration option to choose default

### Phase 3: Optimize XTTS v2 (1 day)
1. Expand TTS cache from current 100 to 1000 phrases
2. Add batch processing for multiple sentences
3. Profile and optimize Python bridge communication

---

## 💾 Storage Requirements

**Current (XTTS v2 only):**
- Model: 1.8GB
- Cache: 100MB (100 phrases)
- **Total: ~2GB**

**With Piper Added:**
- XTTS v2: 1.8GB
- Piper model: 50MB
- Cache (expanded): 500MB
- **Total: ~2.4GB**

---

## 🎯 Recommendation

**Start with Hybrid Approach:**
1. Add Piper TTS for fast responses (this weekend)
2. Keep XTTS v2 for quality when needed
3. Expand cache to cover common phrases
4. Gives you **10-50ms response time** for 90% of interactions
5. Fall back to XTTS v2's quality when it matters

**Implementation Priority:**
1. ✅ Piper TTS integration (biggest speed improvement)
2. ✅ Hybrid routing logic
3. ✅ Expanded caching
4. ⏳ VITS ONNX (if Piper quality insufficient)
5. ❌ XTTS v2 ONNX (not practical due to architecture)

---

## 📦 Next Steps

Run these commands to get started with Piper:

```powershell
# Install ONNX Runtime
cd D:\G.R.I.M
vcpkg install onnxruntime:x64-windows

# Download Piper voice
mkdir resources\models\piper
cd resources\models\piper
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# Test with piper binary
curl -L -O https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_windows_amd64.zip
unzip piper_windows_amd64.zip
./piper.exe --model en_US-amy-medium.onnx --output_file test.wav < test.txt
```

Would you like me to proceed with implementing Piper TTS integration?

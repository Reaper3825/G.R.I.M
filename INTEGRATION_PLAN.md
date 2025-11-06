# GRIM Native LLM Backend Integration Plan

**Date:** November 5, 2025  
**Scope:** Replace external LLM backends (Ollama/Mistral/GPT) with trained GRIM-text transformer model  
**Out of Scope:** Whisper STT, Coqui TTS, Vision AI, YOLO - these remain unchanged

---

## 📋 Current Architecture

### What Uses External LLMs (TO BE REPLACED):
1. **`callAIAsync()`** - Main async text generation function
   - Current: Calls Ollama/LocalAI/OpenAI APIs
   - Used for: Conversation, command interpretation, question answering
   
2. **`ai_interpret()`** - Intent classification with LLM
   - Current: Asks LLM to classify user input as command/conversation/question
   - Returns: JSON with intent + suggested command
   
3. **`ai_process()`** - Blocking conversation handler
   - Current: Synchronous wrapper around callAIAsync
   - Used for: Direct user questions/conversation

4. **`ai_process_stream()`** - Streaming text generation
   - Current: Streams responses from external LLM
   - Used for: Long-form responses

### What Stays UNCHANGED:
- ✅ Whisper (voice recognition) - `voice/voice.cpp`
- ✅ Coqui TTS (voice synthesis) - `voice/voice_speak.cpp`  
- ✅ Vision AI (ONNX models) - `perception/vision_ai.cpp`
- ✅ YOLO (object detection) - `vision/`
- ✅ FastClassifier (intent classification) - `ai/fast_classifier.cpp`
- ✅ NLP rules (pattern matching) - `nlp/nlp.cpp`

---

## 🎯 GRIM-Text Model Capabilities

### ✅ What Your Model HAS:
```cpp
// From resources/models/GRIM-text/
1. LanguageModel          - Full transformer with encoder layers
2. GrimTokenizer          - BPE tokenizer with NFKC normalization
3. TextGenerator          - Multiple generation strategies
4. GPU acceleration       - CUDA kernels for inference
5. Weight serialization   - FlatBuffer binary format
6. Context management     - Handles conversation history
```

### ✅ Model Features:
- **Vocab Size:** 50,000 tokens (configurable)
- **Model Dimensions:** 768d (configurable)
- **Max Sequence:** 8,192 tokens (configurable)
- **Generation Modes:** Greedy, beam search, nucleus sampling, top-k
- **Performance:** ~741 tokens/sec (CPU), much faster on GPU
- **Format:** Binary weights + FlatBuffer metadata

---

## 🔄 Integration Points

### 1. Configuration (`ai_config.json`)

**Add new backend option:**
```json
{
  "backend": "grim_native",  // NEW: Add alongside "ollama", "localai", "openai"
  
  "grim_native": {
    "model_path": "resources/models/GRIM-text/grim_model.bin",
    "tokenizer_path": "resources/models/GRIM-text/tokenizer.bin",
    "use_gpu": true,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "strategy": "nucleus"  // greedy, beam, nucleus, topk
  }
}
```

### 2. Code Changes (`ai/ai.cpp`)

**Modify `callAIAsync()` to support native backend:**
```cpp
std::future<std::string> callAIAsync(const std::string& prompt) {
    return std::async(std::launch::async, [prompt]() -> std::string {
        std::string backend = resolveBackendURL();
        
        // NEW: Handle native backend
        if (backend == "grim_native") {
            auto* nativeBackend = GRIM::getGRIMBackend();
            if (!nativeBackend || !nativeBackend->isReady()) {
                LOG_ERROR("AI", "GRIM native backend not initialized");
                return "[AI] Native backend unavailable";
            }
            
            // Generate response with history
            return nativeBackend->generateWithHistory(
                g_conversationHistory,
                prompt
            );
        }
        
        // Existing ollama/localai/openai code remains...
    });
}
```

**Modify `resolveBackendURL()` to detect native:**
```cpp
std::string resolveBackendURL() {
    std::string backend = aiConfig.value("backend", "auto");
    
    if (backend == "auto") {
        // Check if native model is available
        if (GRIM::getGRIMBackend() && GRIM::getGRIMBackend()->isReady()) {
            return "grim_native";
        }
        
        // Existing fallback logic...
    }
    
    return backend;
}
```

### 3. Initialization (`bootstrap/bootstrap.cpp`)

**Initialize native backend at startup:**
```cpp
// After other initializations in runBootstrapChecks()

// Initialize GRIM native backend if enabled
if (aiConfig.value("backend", "ollama") == "grim_native") {
    LOG_PHASE("Initializing GRIM native LLM backend", true);
    
    if (!GRIM::initGRIMBackend(aiConfig)) {
        LOG_ERROR("Bootstrap", "Failed to initialize GRIM native backend");
        LOG_PHASE("GRIM native backend initialization", false);
    } else {
        LOG_PHASE("GRIM native backend initialized", true);
    }
}
```

### 4. CMake Integration

**Add GRIM-text sources to main build:**
```cmake
# In cmake/Sources.cmake

# Add GRIM-text model sources
set(GRIM_MODEL_SOURCES
    "${CMAKE_SOURCE_DIR}/resources/models/GRIM-text/grim_text_embedding.cpp"
    # Other .cpp files if needed
)

# Add to GRIM_SOURCES
set(GRIM_SOURCES
    # ... existing sources ...
    ${GRIM_MODEL_SOURCES}
)

# Add model headers to include path
target_include_directories(GRIM PRIVATE
    ${CMAKE_SOURCE_DIR}/resources/models/GRIM-text
)
```

---

## 📦 Files to Create/Modify

### NEW FILES (Already created by me):
1. ✅ `ai/grim_backend.hpp` - Native backend wrapper interface
2. ✅ `ai/grim_backend.cpp` - Implementation with thread safety

### MODIFY FILES:
1. **`ai/ai.cpp`**
   - Add `#include "ai/grim_backend.hpp"`
   - Modify `callAIAsync()` to handle "grim_native" case
   - Modify `resolveBackendURL()` to auto-detect native
   - Add native backend to `ai_interpret()` and `ai_process_stream()`

2. **`ai_config.json`**
   - Add `"grim_native"` configuration section
   - Update valid backend options

3. **`bootstrap/bootstrap.cpp`**
   - Add native backend initialization
   - Load model weights at startup

4. **`cmake/Sources.cmake`**
   - Add GRIM-text model sources
   - Add include directories

5. **`cmake/Dependencies.cmake`** (if needed)
   - Link FlatBuffers library (already in vcpkg)

---

## 🔧 Implementation Steps

### Phase 1: Basic Integration (Minimal Working Version)
1. ✅ Create `grim_backend.hpp/cpp` wrapper (DONE)
2. ⬜ Add to CMake build system
3. ⬜ Update `ai_config.json` with native config
4. ⬜ Modify `ai.cpp` to support native backend
5. ⬜ Initialize at bootstrap
6. ⬜ Test with simple prompts

### Phase 2: Feature Parity
7. ⬜ Add streaming support (`generateStream()`)
8. ⬜ Implement conversation history management
9. ⬜ Add temperature/top_p/top_k parameter controls
10. ⬜ Add model warmup for faster first response

### Phase 3: Optimization
11. ⬜ Enable GPU acceleration (CUDA)
12. ⬜ Optimize context window management
13. ⬜ Add model caching/persistence
14. ⬜ Performance benchmarking vs Ollama

### Phase 4: Production Polish
15. ⬜ Error handling and fallback to external LLM
16. ⬜ Model update/reload without restart
17. ⬜ Statistics and monitoring
18. ⬜ Documentation and examples

---

## 🎪 Usage Examples

### After Integration:

```json
// ai_config.json
{
  "backend": "grim_native",  // Use local trained model
  // OR
  "backend": "auto",         // Auto-select (prefers native if available)
  // OR  
  "backend": "ollama"        // Fallback to external
}
```

### User Experience:
```
User: "What's the weather like?"
GRIM: [Uses native model] "I don't have access to weather data..."

User: "Open notepad"
GRIM: [Intent classification with native model] → Executes command

User: "Tell me a joke"
GRIM: [Generates response with native model] "Why did the..."
```

---

## 📊 Expected Benefits

### Performance:
- **Latency:** ~50ms (native GPU) vs ~500-2000ms (Ollama first call)
- **Throughput:** 741 tokens/sec (CPU) → 3000+ tokens/sec (GPU)
- **No network:** Zero HTTP overhead

### Privacy:
- **100% local:** No data sent to external servers
- **Offline capable:** Works without internet
- **Data sovereignty:** All inference happens on your machine

### Cost:
- **Free:** No API costs
- **Predictable:** No rate limits or quotas
- **Scalable:** Only limited by your hardware

### Control:
- **Customizable:** Train on your own data
- **Deterministic:** Same input → same output (with same seed)
- **Versionable:** Lock model version, update on your schedule

---

## ⚠️ Potential Challenges

### 1. Model Quality
- **Issue:** GRIM-text needs training on conversational data
- **Solution:** Use training pipeline to train on dialogue datasets
- **Fallback:** Keep Ollama as backup for complex queries

### 2. Model Size
- **Issue:** 768d x 12 layers = ~300MB weights
- **Solution:** FlatBuffer compressed format, memory mapping
- **Fallback:** Quantize to INT8 for 4x size reduction

### 3. First Load Time
- **Issue:** Loading model weights takes time
- **Solution:** Warmup at startup, keep in memory
- **Fallback:** Lazy load on first use with spinner

### 4. GPU Memory
- **Issue:** Your 3080 Ti has 12GB VRAM (already used by vision models?)
- **Solution:** Monitor VRAM usage, offload to CPU if needed
- **Fallback:** CPU-only mode with SIMD optimizations

---

## 🧪 Testing Strategy

### Unit Tests:
```cpp
// Test native backend initialization
TEST(GRIMBackend, Initialize)
TEST(GRIMBackend, LoadModel)
TEST(GRIMBackend, Generate)

// Test integration with ai.cpp
TEST(AI, NativeBackendSelection)
TEST(AI, NativeGeneration)
TEST(AI, NativeStreaming)
```

### Integration Tests:
```cpp
// Test end-to-end flow
TEST(System, NativeConversation)
TEST(System, NativeIntentClassification)
TEST(System, NativeFallback)
```

### Benchmarks:
```cpp
// Compare performance
BENCHMARK(Native_vs_Ollama_Latency)
BENCHMARK(Native_vs_Ollama_Throughput)
BENCHMARK(Native_GPU_vs_CPU)
```

---

## 📝 Next Steps

**What I need to know from you:**

1. **Do you have trained weights?**
   - Where is `grim_model.bin`?
   - Where is `tokenizer.bin`?
   - Or do we need to train first?

2. **Training data ready?**
   - Do you have `.grmt` FlatBuffer files with conversational data?
   - Or use the test data for POC?

3. **Deployment priority?**
   - Start with Phase 1 (basic working version)?
   - Or focus on training first, then integrate?

4. **Performance targets?**
   - What response time is acceptable? (<100ms, <500ms?)
   - GPU mandatory or CPU acceptable?

**Ready to implement?** Just say "go" and I'll start with Phase 1! 🚀

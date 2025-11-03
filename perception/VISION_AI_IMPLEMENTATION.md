# Vision AI Implementation with Llama 3.2 Vision

## Overview
This document describes the implementation of vision AI capabilities using llama3.2-vision and the migration from Mistral to Llama 3.1 for the main AI backend.

## Changes Summary

### 1. Vision AI Implementation (perception/vision_ai.cpp)

#### Base64 Encoding ✅
- Already implemented in `encodeImageToBase64()` function
- Converts OpenCV Mat images to PNG format
- Encodes to base64 string for Ollama API

#### Ollama HTTP Client ✅
- Added CPR library for HTTP requests
- Implemented `analyzeWithOllama()` with full functionality:
  - Default model: **llama3.2-vision:11b** (quantized Q4_K_M by Ollama)
  - JSON request building with image data
  - HTTP POST to `localhost:11434/api/generate`
  - 2-minute timeout for vision models
  - Automatic context type classification (coding, browsing, gaming, etc.)
  - Structured response parsing

#### Vision Analysis Integration ✅
- Updated `analyzeWithVisionAI()` in perception_context.cpp
- Context-aware prompts based on scene type:
  - IDE/Code: Focused on programming language and development
  - WebBrowser: Describes website content and purpose
  - Terminal: Analyzes commands and output
  - Gaming/Video: Describes media content
  - General: Comprehensive screen description
- Temperature: 0.3 (consistent descriptions)
- Max tokens: 300 (reasonable length)

### 2. AI Backend Migration (Mistral → Llama 3.1)

#### ai/ai.cpp ✅
Updated default_model in 3 locations:
- Line 135: `callAIAsync()` - Main AI call function
- Line 315: `ai_interpret()` - Intent interpretation
- Line 458: `ai_process_stream()` - Streaming responses

Changed from `"mistral"` to `"llama3.1:8b"`

#### ai/lm_intent.cpp ✅
- Updated log messages from "Mistral bridge" to "LLM intent bridge"
- Updated comments to be model-agnostic
- Bridge now uses configured Ollama model

#### resources/python/mistral_bridge.py ✅
- Updated header comments to "LLM bridge"
- Changed default model: `llama3.1:8b`
- Updated function docstrings to reference "Llama" instead of "Mistral"
- Log messages now say "[LLM Bridge]"

## Model Configuration

### Main AI Backend
- **Model**: llama3.1:8b
- **Quantization**: Q4_K_M (default Ollama quantization)
- **Use Cases**: 
  - General conversation
  - Intent classification
  - Command interpretation
  - Response generation

### Vision AI
- **Model**: llama3.2-vision:11b
- **Quantization**: Q4_K_M (default)
- **Use Cases**:
  - Screen understanding
  - Visual context analysis
  - Multi-monitor content description
  - Application/activity detection

### Alternative Quantization Options
Available through Ollama model variants:
- **Q8_0**: Higher quality, slower (e.g., `llama3.1:8b-q8_0`)
- **Q4_0**: Faster, lower memory (e.g., `llama3.1:8b-q4_0`)
- **Q4_K_M**: Balanced (default, recommended)

## Offline Operation

All models run locally via Ollama:
- **Ollama API**: `http://localhost:11434`
- **No internet required** for inference
- Models pulled once, cached locally
- GGUF format with llama.cpp backend

## Installation Requirements

### Pull Required Models
```bash
# Main AI backend
ollama pull llama3.1:8b

# Vision AI
ollama pull llama3.2-vision:11b
```

### Optional: Different Quantizations
```bash
# Higher quality (more VRAM)
ollama pull llama3.1:8b-q8_0
ollama pull llama3.2-vision:11b-q8_0

# Faster (less VRAM)
ollama pull llama3.1:8b-q4_0
```

## Usage

### Vision AI Initialization
```cpp
// Initialize vision AI (automatic in perception system)
GRIM::Perception::initVisionAI(VisionAIBackend::Ollama_LLaVA);

// Check if available
if (g_visionAI->isAvailable()) {
    // Vision AI ready
}
```

### Automatic Integration
Vision AI is automatically called during:
- Multi-monitor context capture
- Screen content analysis
- Vision questions ("what's on my screen?")
- Continuous perception (background thread)

### Manual Vision Analysis
```cpp
VisionAnalysisRequest request;
request.image = screenshot;
request.prompt = "Describe this screen";
request.backend = VisionAIBackend::Ollama_LLaVA;
request.temperature = 0.3f;
request.maxTokens = 300;

VisionAnalysisResult result = g_visionAI->analyzeImage(request);
if (result.success) {
    std::string description = result.description;
    std::string contextType = result.contextType;
    float confidence = result.confidence;
}
```

## Performance Characteristics

### Llama 3.1:8b (Q4_K_M)
- **Speed**: Fast (~10-30 tokens/sec on consumer GPU)
- **Memory**: ~5-6 GB VRAM
- **Quality**: Excellent for most tasks
- **Use Case**: General AI, intent classification

### Llama 3.2 Vision:11b (Q4_K_M)
- **Speed**: Moderate (~5-15 tokens/sec with images)
- **Memory**: ~7-8 GB VRAM
- **Quality**: Good vision understanding
- **Use Case**: Screen analysis, visual questions
- **Latency**: 2-5 seconds per image analysis

## Testing Checklist

- [ ] Pull llama3.1:8b model via Ollama
- [ ] Pull llama3.2-vision:11b model via Ollama
- [ ] Test main AI responses (verify Llama vs Mistral)
- [ ] Test vision questions ("what's on my screen?")
- [ ] Test multi-monitor vision ("what's on monitor 2?")
- [ ] Verify offline operation (disconnect internet)
- [ ] Test continuous capture with vision AI
- [ ] Check performance with different quants (optional)

## Architecture Flow

```
User Question → Question Handler
    ↓
Vision Detection? → Yes → PerceptionContextManager
    ↓                           ↓
    ↓                    captureScreen()
    ↓                           ↓
    ↓                    performOCREnhanced() (3 strategies)
    ↓                           ↓
    ↓                    analyzeWithVisionAI()
    ↓                           ↓
    ↓                    VisionAIManager::analyzeImage()
    ↓                           ↓
    ↓                    analyzeWithOllama()
    ↓                           ↓
    ↓                    HTTP POST to localhost:11434
    ↓                           ↓
    ↓                    Llama 3.2 Vision Processing
    ↓                           ↓
    ↓                    Parse & Return Description
    ↓                           ↓
    └────────────────── Combine OCR + Vision → Response
```

## File Modifications

### New Includes
- `perception/vision_ai.cpp`: Added `<cpr/cpr.h>` and `<nlohmann/json.hpp>`
- `perception/perception_context.cpp`: Added `vision_ai.hpp`

### Modified Functions
1. **VisionAIManager::analyzeWithOllama()** - Full implementation
2. **PerceptionContextManager::analyzeWithVisionAI()** - Active vision AI calls
3. **ai/ai.cpp** - Three default_model updates
4. **ai/lm_intent.cpp** - Generic logging
5. **mistral_bridge.py** - Llama 3.1 as default

## Future Enhancements

### Potential Improvements
1. **Custom quantization**: Allow users to specify quant level in config
2. **Model auto-detection**: Check which models Ollama has available
3. **Fallback models**: Try different vision models if primary unavailable
4. **GPU selection**: For multi-GPU systems
5. **Response caching**: Cache vision responses for identical screens
6. **Streaming vision**: Stream vision AI responses for faster perceived latency

### Optional Vision Models
- `llava:13b` - Larger, more detailed
- `llava-phi3` - Faster, smaller
- `bakllava` - Alternative implementation

## Notes

- Vision AI is **optional** - system works without it
- Falls back to OCR-only if vision AI unavailable
- All processing happens **locally** via Ollama
- No API keys or internet required
- Quantized models balance speed and quality
- GGUF format optimized for CPU/GPU inference

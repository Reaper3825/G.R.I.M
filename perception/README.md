# G.R.I.M Perception & Digital Context Layer

## Overview

The G.R.I.M perception system provides advanced digital context awareness, allowing the AI to "see" and understand what's on your screen. This enables natural interactions like "what's on my screen?", "read this text", and contextual awareness of your current activity.

## Architecture

The production digital acquisition spine is documented in
[`digital/README.md`](digital/README.md). It publishes immutable capture attempts
to `DigitalFrameBus`, runs frame-coherent OCR plus optional host-native UI
automation, and projects both into live reasoning context. The older
`PerceptionContextManager` remains as an on-demand compatibility layer.

### Components

1. **Core Perception** (`perception/perception.cpp`)
   - Screen capture (Windows GDI+)
   - OCR via Tesseract
   - Object detection (YOLO/color-based)
   - Basic visual analysis

2. **Context Manager** (`perception/perception_context.hpp/cpp`)
   - Unified visual context representation
   - Intelligent caching (2-second validity window)
   - Change detection to avoid redundant analysis
   - Window/application tracking
   - Scene classification
   - Visual characteristics analysis

3. **Vision AI** (`perception/vision_ai.hpp/cpp`)
   - Multimodal vision-language models
   - Support for LLaVA, Phi-3 Vision, GPT-4 Vision
   - Semantic understanding of screen content
   - Natural language Q&A about visual content

### Visual Context Structure

```cpp
struct VisualContext {
    // Screen metadata
    int screenWidth, screenHeight;
    chrono::time_point captureTime;
    
    // Active window tracking
    string activeWindowTitle;
    string activeProcessName;
    int activeWindowX, activeWindowY, activeWindowWidth, activeWindowHeight;
    
    // OCR results
    string screenText;
    float ocrConfidence;
    
    // Object detection
    vector<DetectedObject> detectedObjects;
    
    // Vision AI analysis
    string aiDescription;
    float aiConfidence;
    
    // Scene classification
    SceneType sceneType; // Desktop, WebBrowser, IDE_Code, Terminal, etc.
    
    // Visual characteristics
    float brightnessMean;
    float contrastScore;
    float textDensity;
    bool isDarkTheme;
    
    // Change detection
    float changeScore; // 0.0 = no change, 1.0 = completely different
};
```

## Features

### 1. **Intelligent Context Caching**
- Captures screen context only when needed
- 2-second cache validity by default
- Change detection (5% threshold) triggers re-analysis
- Prevents redundant processing for static screens

### 2. **Multi-Layer Analysis**

#### Layer 1: Screen Capture
- Full screen or region capture
- OpenCV Mat format for processing
- Windows GDI+ backend

#### Layer 2: Window Awareness
- Tracks active window title and position
- Identifies running process
- Window bounds and focus state

#### Layer 3: Visual Characteristics
- Brightness analysis (dark/light theme detection)
- Contrast scoring
- Text density estimation
- Edge detection for UI elements

#### Layer 4: Scene Classification
Automatically identifies:
- 🌐 Web Browser
- 💻 IDE/Code Editor  
- 🖥️ Terminal/Console
- 📄 Document Editor
- 💬 Chat/Messaging
- 🎮 Game
- 🖼️ Image/Video
- 🖱️ Desktop

#### Layer 5: OCR (Tesseract)
- Full-screen text extraction
- Confidence scoring
- Preprocessing (grayscale, thresholding)
- Structured text output

#### Layer 6: Object Detection
- YOLO-based detection (when available)
- Fallback to color-based detection
- Bounding box coordinates
- Confidence scores

#### Layer 7: Vision AI (Future)
- Semantic understanding via LLaVA/Phi-Vision
- Natural language descriptions
- Context-aware Q&A
- Activity recognition

### 3. **Smart Question Routing**

The question handler now routes vision queries to the context manager:

```cpp
// Detects vision-related questions
"what's on my screen"
"read the text"
"what am I looking at"
"describe my screen"
"what's in this window"
```

Enhanced routing:
1. Check if question is vision-related
2. Call `answerVisionQuestionWithContext(question)`
3. Use cached context if recent enough
4. Fallback to legacy OCR/detection if needed

## Usage Examples

### From User Perspective

**Q:** "What's on my screen?"

**A:** 
```
I can see your screen:

You're viewing: IDE/Code Editor
Active window: "Visual Studio Code - G.R.I.M"
Theme: Dark mode

Visible text:
#include "perception_context.hpp"
namespace GRIM {
namespace Perception {
...
```

**Q:** "Read the text"

**A:**
```
Screen text:
struct VisualContext {
    int screenWidth = 0;
    int screenHeight = 0;
    ...
}
```

### From Code Perspective

```cpp
// Get current visual context
VisualContext ctx = GRIM::Perception::getCurrentVisualContext();

if (ctx.isValid) {
    LOG_DEBUG("Main", "Scene type: " + ctx.sceneTypeToString());
    LOG_DEBUG("Main", "Active window: " + ctx.activeWindowTitle);
    
    if (ctx.hasText) {
        LOG_DEBUG("Main", "OCR text length: " + to_string(ctx.screenText.length()));
    }
}

// Answer vision questions
string answer = GRIM::Perception::answerVisionQuestionWithContext("what am I looking at?");
```

## Configuration

### Enable/Disable Features

```cpp
auto* manager = GRIM::Perception::g_contextManager.get();

// Disable expensive features if needed
manager->setFeatureEnabled("ocr", true);              // Tesseract OCR
manager->setFeatureEnabled("object_detection", true); // YOLO/color detection
manager->setFeatureEnabled("vision_ai", false);       // Vision-language models (not impl yet)
manager->setFeatureEnabled("window_tracking", true);  // Active window tracking
```

### Cache Settings

Edit `PerceptionContextManager::Impl`:
```cpp
std::chrono::milliseconds cacheValidDuration{2000}; // 2 seconds
float changeThreshold = 0.05f; // 5% change triggers refresh
```

## Dependencies

### Required
- OpenCV 4.x (screen capture, image processing)
- Tesseract OCR (text extraction)
- Windows GDI+ (screen capture backend)

### Optional
- YOLO model files (advanced object detection)
  - Place in: `D:/G.R.I.M/resources/models/yolo/`
  - Files: `yolov3.cfg`, `yolov3.weights`, `coco.names`

### Future (Vision AI)
- Ollama (for local LLaVA/Phi-Vision models)
- OpenAI API (for GPT-4 Vision)
- GitHub Models API

## Installation & Setup

### 1. Tesseract OCR

**Windows:**
```powershell
# Install via vcpkg (recommended)
vcpkg install tesseract:x64-windows

# Or download installer
# https://github.com/UB-Mannheim/tesseract/wiki

# Ensure tessdata is available
# G.R.I.M expects: D:/G.R.I.M/resources/tessdata/eng.traineddata
```

**Directory Structure:**
```
D:/G.R.I.M/
├── resources/
│   ├── tessdata/
│   │   └── eng.traineddata
│   └── models/
│       └── yolo/  (optional)
│           ├── yolov3.cfg
│           ├── yolov3.weights
│           └── coco.names
```

### 2. Build Configuration

Ensure CMake finds dependencies:
```cmake
find_package(OpenCV REQUIRED)
find_package(Tesseract REQUIRED)

target_link_libraries(GRIM
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_dnn
    tesseract
    leptonica
)
```

## Performance Considerations

### Memory Usage
- Cached screenshot: ~6MB for 1920x1080 (BGR, 3 channels)
- Context object: <1KB without screenshot
- OCR engine: ~100MB (Tesseract model)
- YOLO model: ~250MB (if loaded)

### CPU Usage
- Screen capture: <5ms
- OCR (full screen): 200-500ms
- YOLO detection: 500-2000ms (CPU) or 50-200ms (GPU)
- Change detection: <10ms
- Context caching reduces average overhead to <1ms for repeated queries

### Optimization Tips
1. **Caching**: Keep cache duration at 2 seconds for responsive UX
2. **Lazy loading**: YOLO loads on-demand
3. **Screenshot release**: Don't keep raw pixels in memory unless needed
4. **Selective features**: Disable OCR/detection if not needed
5. **Region capture**: Capture specific windows instead of full screen

## Future Enhancements

### Phase 1: Vision AI Integration (In Progress)
- [ ] HTTP client for Ollama API
- [ ] LLaVA model integration
- [ ] Phi-3 Vision support
- [ ] Semantic scene understanding
- [ ] Activity recognition

### Phase 2: Memory Integration
- [ ] Store visual context in memory system
- [ ] Temporal awareness (remember what was on screen)
- [ ] Visual memory search
- [ ] Context-based proactive suggestions

### Phase 3: Advanced Features
- [ ] Multi-monitor support
- [ ] Window-specific capture
- [ ] UI element detection (buttons, forms, etc.)
- [ ] Screenshot annotation
- [ ] Visual diff (highlight changes)
- [ ] Accessibility tree integration

### Phase 4: Proactive Context
- [ ] Continuous monitoring mode
- [ ] Activity-based triggers
- [ ] Contextual suggestions
- [ ] Visual anomaly detection

## Troubleshooting

### "OCR engine not initialized"
**Cause:** Tesseract data files not found  
**Fix:** 
```powershell
# Ensure tessdata exists
mkdir D:\G.R.I.M\resources\tessdata
# Copy eng.traineddata from Tesseract installation
cp "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata" D:\G.R.I.M\resources\tessdata\
```

### "Screen capture failed"
**Cause:** Windows permissions or GDI+ not initialized  
**Fix:**
- Run G.R.I.M with admin privileges
- Check `Perception::init()` is called in main.cpp

### Poor OCR results
**Cause:** Low contrast, unusual fonts, or image quality  
**Fix:**
- Increase screen resolution
- Use higher contrast themes
- Check Tesseract PSM mode (currently PSM_AUTO)

### High CPU usage
**Cause:** Frequent re-captures without caching  
**Fix:**
- Increase cache duration
- Disable YOLO if not needed
- Use change detection threshold

## API Reference

### PerceptionContextManager

```cpp
// Get current context (uses cache if valid)
VisualContext getCurrentContext(bool forceRefresh = false);

// Full capture and analysis
VisualContext captureAndAnalyze(bool includeAI = true, bool saveScreenshot = false);

// Answer vision questions
string answerVisionQuestion(const string& question);

// Feature control
void setFeatureEnabled(const string& feature, bool enabled);

// Status check
PerceptionStatus getStatus();
```

### VisualContext

```cpp
// Human-readable summary
string toSummary() const;

// Detailed string representation
string toDetailedString() const;
```

### Global Functions

```cpp
// Initialize context manager (called from main)
void GRIM::Perception::initContextManager();

// Get current context globally
VisualContext getCurrentVisualContext(bool forceRefresh = false);

// Answer vision questions globally
string answerVisionQuestionWithContext(const string& question);
```

## Logging & Debugging

Enable debug logging:
```cpp
// See perception activity
LOG_DEBUG("PerceptionContext", "...");

// Example output:
[DEBUG][PerceptionContext] Capturing fresh context
[DEBUG][PerceptionContext] Active window: "Visual Studio Code"
[DEBUG][PerceptionContext] OCR extracted 1247 characters
[DEBUG][PerceptionContext] Visual characteristics - Brightness: 0.23, Contrast: 0.45
[DEBUG][PerceptionContext] Context analysis complete: Screen: 1920x1080 | Scene: IDE/Code Editor...
```

### Physical Perception Telemetry

The physical perception pipeline now carries timing telemetry in the same
snapshots that move through the buses, so UI/debug consumers can profile
without parsing logs:

- `PhysicalFrameMetadata` reports frame-conditioning stage timings plus
    `frame_bus_publish_copy_ms` and per-consumer `frame_bus_pull_copy_ms`.
    After the shared-packet frame bus refactor, pull timing measures mutex +
    shallow `cv::Mat` header handoff overhead, not per-consumer pixel copies.
- `PhysicalPerceptionPrimitiveResults::telemetry` reports Stage-2 tick time,
    frame-pull time, per-operator wall time, cache-hit count, fresh inference
    count, and no-signal forced refreshes.
- `PhysicalSpatialGroundingResults` reports Stage-3 pull/depth/grounder/publish
    timing.
- `PhysicalWorldStateSnapshot` reports Stage-4 pull/build/publish timing.

Operator envelopes still carry model-specific timings such as
`last_inference_ms`, SAM encoder/decoder timing, tracker route time, and class
policy apply time. Use the loop telemetry for integration overhead and the
operator envelope timings for model hot spots.

## Contributing

Physical-camera files use the ownership taxonomy in
[`physical/README.md`](physical/README.md). Update that file whenever adding,
moving, or splitting a `perception/physical/Physical*` file.

When extending perception:
1. Add new analysis to `PerceptionContextManager::captureAndAnalyze()`
2. Update `VisualContext` struct with new fields
3. Implement feature in separate function for modularity
4. Add enable/disable flag via `setFeatureEnabled()`
5. Update `toDetailedString()` for output
6. Document in this README

## License

Part of the G.R.I.M (General Responsive Intelligence Matrix) project.

#include "perception_context.hpp"
#include "perception.hpp"
#include "multi_monitor.hpp" // ✅ Multi-monitor support
#include "vision_ai.hpp"      // ✅ Vision AI integration
#include "logger.hpp"         // ✅ For logging functions
#include "memory/memory_storage.hpp" // ✅ For memory integration
#include "core/input/InputController.hpp" // ✅ Input control integration
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <ctime>
#include <regex> // ✅ For monitor number parsing
#include <thread> // ✅ For continuous capture thread
#include <mutex>  // ✅ For thread-safe context access
#include <chrono> // ✅ For timing

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include <opencv2/opencv.hpp>

// External reference to global memory storage
extern GRIM::MemoryStorage g_memoryStorage;

namespace GRIM {
namespace Perception {

std::unique_ptr<PerceptionContextManager> g_contextManager = nullptr;

// Convert scene type to string
static std::string sceneTypeToString(VisualContext::SceneType type) {
    switch (type) {
        case VisualContext::SceneType::Desktop: return "Desktop";
        case VisualContext::SceneType::WebBrowser: return "Web Browser";
        case VisualContext::SceneType::IDE_Code: return "IDE/Code Editor";
        case VisualContext::SceneType::Terminal: return "Terminal/Console";
        case VisualContext::SceneType::Document: return "Document";
        case VisualContext::SceneType::Image_Video: return "Image/Video";
        case VisualContext::SceneType::Game: return "Game";
        case VisualContext::SceneType::Chat_Messaging: return "Chat/Messaging";
        default: return "Unknown";
    }
}

// Helper function to clean special symbols from text for TTS
static std::string cleanTextForTTS(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.length());
    
    for (char c : text) {
        // Keep alphanumeric, spaces, basic punctuation, and newlines
        if (std::isalnum(static_cast<unsigned char>(c)) || 
            c == ' ' || c == '.' || c == ',' || c == '?' || c == '!' || 
            c == ':' || c == ';' || c == '\n' || c == '\r' || c == '-' || 
            c == '\'' || c == '"' || c == '(' || c == ')') {
            cleaned += c;
        }
    }
    
    return cleaned;
}

std::string VisualContext::toSummary() const {
    if (!isValid) {
        return "Visual context unavailable: " + errorMessage;
    }
    
    std::ostringstream ss;
    ss << "Screen: " << screenWidth << "x" << screenHeight << " | ";
    ss << "Scene: " << sceneTypeToString(sceneType) << " | ";
    
    if (!activeWindowTitle.empty()) {
        ss << "Window: \"" << activeWindowTitle << "\" | ";
    }
    
    if (hasText && !screenText.empty()) {
        std::string preview = screenText.substr(0, std::min<size_t>(50, screenText.length()));
        ss << "Text: \"" << preview << (screenText.length() > 50 ? "...\"" : "\"") << " | ";
    }
    
    if (!detectedObjects.empty()) {
        ss << "Objects: " << detectedObjects.size() << " | ";
    }
    
    if (hasAIAnalysis && !aiDescription.empty()) {
        std::string preview = aiDescription.substr(0, std::min<size_t>(60, aiDescription.length()));
        ss << "AI: \"" << preview << (aiDescription.length() > 60 ? "...\"" : "\"");
    }
    
    return ss.str();
}

std::string VisualContext::toDetailedString() const {
    if (!isValid) {
        return "Visual context unavailable: " + errorMessage;
    }
    
    std::ostringstream ss;
    
    // Timestamp
    auto timeT = std::chrono::system_clock::to_time_t(captureTime);
    ss << "=== Visual Context Snapshot ===\n";
    ss << "Captured: " << std::put_time(std::localtime(&timeT), "%Y-%m-%d %H:%M:%S") << "\n\n";
    
    // Screen info
    ss << "Screen: " << screenWidth << "x" << screenHeight << "\n";
    
    // Multi-monitor info
    if (isMultiMonitor) {
        ss << "Monitor: " << (monitorIndex + 1) << " of " << totalMonitors << "\n";
    }
    
    ss << "Scene Type: " << sceneTypeToString(sceneType) << "\n";
    ss << "Theme: " << (isDarkTheme ? "Dark" : "Light") << "\n";
    ss << "Brightness: " << (int)(brightnessMean * 100) << "% | ";
    ss << "Contrast: " << (int)(contrastScore * 100) << "% | ";
    ss << "Text Density: " << (int)(textDensity * 100) << "%\n\n";
    
    // Active window
    if (!activeWindowTitle.empty()) {
        ss << "Active Window:\n";
        ss << "  Title: \"" << activeWindowTitle << "\"\n";
        ss << "  Process: " << activeProcessName << "\n";
        ss << "  Position: (" << activeWindowX << ", " << activeWindowY << ") ";
        ss << activeWindowWidth << "x" << activeWindowHeight << "\n\n";
    }
    
    // OCR results
    if (hasText) {
        ss << "Screen Text (OCR confidence: " << (int)(ocrConfidence * 100) << "%):\n";
        ss << "---\n" << screenText << "\n---\n\n";
    }
    
    // Objects
    if (!detectedObjects.empty()) {
        ss << "Detected Objects (" << detectedObjects.size() << "):\n";
        for (const auto& obj : detectedObjects) {
            ss << "  - " << obj.label << " (" << (int)(obj.confidence * 100) << "%) ";
            ss << "at [" << obj.x << "," << obj.y << " " << obj.width << "x" << obj.height << "]\n";
        }
        ss << "\n";
    }
    
    // AI analysis
    if (hasAIAnalysis && !aiDescription.empty()) {
        ss << "AI Visual Analysis (confidence: " << (int)(aiConfidence * 100) << "%):\n";
        ss << aiDescription << "\n\n";
    }
    
    // Change detection
    if (changeScore > 0.0f) {
        ss << "Change Score: " << (int)(changeScore * 100) << "% different from previous\n";
    }
    
    return ss.str();
}

// Implementation details
struct PerceptionContextManager::Impl {
    bool initialized = false;
    
    // Cached context (per monitor)
    VisualContext lastContext;
    std::vector<VisualContext> monitorContexts; // One per monitor
    cv::Mat lastScreenshot;
    std::chrono::system_clock::time_point lastCaptureTime;
    
    // Configuration
    bool featureOCR = true;
    bool featureObjectDetection = true;
    bool featureVisionAI = true;
    bool featureWindowTracking = true;
    
    // Cache settings
    std::chrono::milliseconds cacheValidDuration{2000}; // 2 seconds
    float changeThreshold = 0.05f; // 5% change triggers re-analysis
    
    // ✅ Continuous capture
    ContinuousCaptureConfig captureConfig;
    VisualContext latestCapturedContext;
    std::mutex contextMutex;
    int currentFrame = 0;
    std::chrono::steady_clock::time_point lastCaptureTimePoint;
};

PerceptionContextManager::PerceptionContextManager()
    : pImpl(std::make_unique<Impl>()) {
}

PerceptionContextManager::~PerceptionContextManager() {
    shutdown();
}

bool PerceptionContextManager::init() {
    if (pImpl->initialized) {
        return true;
    }
    
    LOG_DEBUG("PerceptionContext", "Initializing context manager");
    
    // Ensure base perception system is initialized
    if (!GRIM::Perception::isAvailable()) {
        GRIM::Perception::init();
    }
    
    // ✅ Initialize multi-monitor support
    GRIM::Perception::initMultiMonitor();
    
    // ✅ Initialize Vision AI system for screen understanding
    // Using ONNX backend for fast local inference (150ms vs 10-30s with Ollama)
    initVisionAI(VisionAIBackend::ONNX_Vision);
    LOG_DEBUG("PerceptionContext", "Vision AI initialized with ONNX backend");
    
    // Initialize monitor context cache
    int monitorCount = GRIM::Perception::getMonitorCount();
    pImpl->monitorContexts.resize(monitorCount);
    
    pImpl->initialized = true;
    LOG_DEBUG("PerceptionContext", "Context manager initialized with " + 
              std::to_string(monitorCount) + " monitors");
    return true;
}

void PerceptionContextManager::shutdown() {
    if (!pImpl->initialized) {
        return;
    }
    
    LOG_DEBUG("PerceptionContext", "Shutting down context manager");
    
    // ✅ Stop continuous capture if running
    stopContinuousCapture();
    
    pImpl->lastScreenshot.release();
    pImpl->initialized = false;
}

VisualContext PerceptionContextManager::getCurrentContext(bool forceRefresh) {
    if (!pImpl->initialized) {
        VisualContext ctx;
        ctx.isValid = false;
        ctx.errorMessage = "Context manager not initialized";
        return ctx;
    }
    
    // Check if cached context is still valid
    auto now = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - pImpl->lastCaptureTime);
    
    if (!forceRefresh && 
        pImpl->lastContext.isValid && 
        elapsed < pImpl->cacheValidDuration) {
        LOG_DEBUG("PerceptionContext", "Returning cached context (age: " + 
                  std::to_string(elapsed.count()) + "ms)");
        return pImpl->lastContext;
    }
    
    // Capture fresh context
    LOG_DEBUG("PerceptionContext", "Capturing fresh context");
    return captureAndAnalyze(pImpl->featureVisionAI, false);
}

VisualContext PerceptionContextManager::getMonitorContext(int monitorIndex, bool forceRefresh) {
    if (!pImpl->initialized) {
        VisualContext ctx;
        ctx.isValid = false;
        ctx.errorMessage = "Context manager not initialized";
        return ctx;
    }
    
    // Validate monitor index
    int monitorCount = GRIM::Perception::getMonitorCount();
    if (monitorIndex < 0 || monitorIndex >= monitorCount) {
        VisualContext ctx;
        ctx.isValid = false;
        ctx.errorMessage = "Invalid monitor index: " + std::to_string(monitorIndex) + 
                           " (have " + std::to_string(monitorCount) + " monitors)";
        return ctx;
    }
    
    // Check if we have cached context for this monitor
    if (!forceRefresh && 
        monitorIndex < static_cast<int>(pImpl->monitorContexts.size()) &&
        pImpl->monitorContexts[monitorIndex].isValid) {
        
        auto now = std::chrono::system_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - pImpl->monitorContexts[monitorIndex].captureTime);
        
        if (elapsed < pImpl->cacheValidDuration) {
            LOG_DEBUG("PerceptionContext", "Returning cached context for monitor " + 
                      std::to_string(monitorIndex) + " (age: " + 
                      std::to_string(elapsed.count()) + "ms)");
            return pImpl->monitorContexts[monitorIndex];
        }
    }
    
    // Capture fresh context for this monitor
    LOG_DEBUG("PerceptionContext", "Capturing fresh context for monitor " + std::to_string(monitorIndex));
    return captureAndAnalyzeMonitor(monitorIndex, pImpl->featureVisionAI, false);
}

VisualContext PerceptionContextManager::captureAndAnalyze(bool includeAI, bool saveScreenshot) {
    // Capture active monitor by default
    return captureAndAnalyzeMonitor(-1, includeAI, saveScreenshot);
}

VisualContext PerceptionContextManager::captureAndAnalyzeMonitor(int monitorIndex, bool includeAI, bool saveScreenshot) {
    VisualContext ctx;
    
    try {
        // Step 1: Capture screen (specific monitor or active)
        ctx = captureScreen(monitorIndex);
        if (!ctx.isValid) {
            return ctx;
        }
        
        // Step 2: Track active window
        if (pImpl->featureWindowTracking) {
            analyzeWindowContext(ctx);
        }
        
        // Step 3: Compute visual characteristics
        computeVisualCharacteristics(ctx);
        
        // Step 4: Classify scene type
        classifyScene(ctx);
        
        // Step 5: OCR if enabled (✅ Use enhanced OCR with preprocessing)
        if (pImpl->featureOCR) {
            performOCREnhanced(ctx);
        }
        
        // Step 6: Object detection if enabled
        if (pImpl->featureObjectDetection) {
            detectObjects(ctx);
        }
        
        // Step 7: Vision AI analysis if enabled
        if (includeAI && pImpl->featureVisionAI) {
            analyzeWithVisionAI(ctx);
        }
        
        // Step 8: Compute change score
        if (!pImpl->lastScreenshot.empty() && !ctx.screenshot.empty()) {
            ctx.changeScore = computeChangeScore(ctx.screenshot, pImpl->lastScreenshot);
        }
        
        // Update cache
        pImpl->lastContext = ctx;
        pImpl->lastCaptureTime = ctx.captureTime;
        
        // ✅ Update monitor-specific cache if applicable
        if (ctx.monitorIndex >= 0 && ctx.monitorIndex < static_cast<int>(pImpl->monitorContexts.size())) {
            pImpl->monitorContexts[ctx.monitorIndex] = ctx;
        }
        
        if (!ctx.screenshot.empty()) {
            ctx.screenshot.copyTo(pImpl->lastScreenshot);
            
            // Clear screenshot from context if not saving
            if (!saveScreenshot) {
                ctx.screenshot.release();
            }
        }
        
        // LOG_DEBUG("PerceptionContext", "Context analysis complete: " + ctx.toSummary());
        
    } catch (const std::exception& e) {
        ctx.isValid = false;
        ctx.errorMessage = std::string("Exception during analysis: ") + e.what();
        LOG_ERROR("PerceptionContext", ctx.errorMessage);
    }
    
    return ctx;
}

VisualContext PerceptionContextManager::captureScreen(int monitorIndex) {
    VisualContext ctx;
    ctx.captureTime = std::chrono::system_clock::now();
    
    // Get monitor count
    int monitorCount = GRIM::Perception::getMonitorCount();
    ctx.totalMonitors = monitorCount;
    ctx.isMultiMonitor = (monitorCount > 1);
    
    cv::Mat screenshot;
    
    if (monitorIndex == -1) {
        // Capture active monitor
        screenshot = GRIM::Perception::captureActiveMonitor();
        ctx.monitorIndex = GRIM::Perception::getActiveMonitorInfo().monitorIndex;
        // LOG_DEBUG("PerceptionContext", "Capturing active monitor (index: " + std::to_string(ctx.monitorIndex) + ")");
    } else if (monitorIndex >= 0 && monitorIndex < monitorCount) {
        // Capture specific monitor
        if (!g_multiMonitor) {
            LOG_ERROR("PerceptionContext", "Multi-monitor manager not initialized");
            ctx.isValid = false;
            ctx.errorMessage = "Multi-monitor manager not initialized";
            return ctx;
        }
        
        LOG_DEBUG("PerceptionContext", "Capturing specific monitor: " + std::to_string(monitorIndex));
        screenshot = g_multiMonitor->captureMonitor(monitorIndex);
        ctx.monitorIndex = monitorIndex;
        
        if (screenshot.empty()) {
            LOG_ERROR("PerceptionContext", "Failed to capture monitor " + std::to_string(monitorIndex));
        }
    } else {
        ctx.isValid = false;
        ctx.errorMessage = "Invalid monitor index";
        return ctx;
    }
    
    if (screenshot.empty()) {
        ctx.isValid = false;
        ctx.errorMessage = "Failed to capture screen";
        return ctx;
    }
    
    ctx.screenshot = screenshot;
    ctx.screenWidth = screenshot.cols;
    ctx.screenHeight = screenshot.rows;
    ctx.isValid = true;
    
    // LOG_DEBUG("PerceptionContext", "Captured monitor " + std::to_string(ctx.monitorIndex) + 
    //           " (" + std::to_string(ctx.screenWidth) + "x" + std::to_string(ctx.screenHeight) + ")");
    
    return ctx;
}

void PerceptionContextManager::analyzeWindowContext(VisualContext& ctx) {
#ifdef _WIN32
    HWND hwnd = GetForegroundWindow();
    if (!hwnd) {
        return;
    }
    
    // Get window title
    char title[256] = {0};
    GetWindowTextA(hwnd, title, sizeof(title));
    ctx.activeWindowTitle = title;
    
    // Get window position
    RECT rect;
    if (GetWindowRect(hwnd, &rect)) {
        ctx.activeWindowX = rect.left;
        ctx.activeWindowY = rect.top;
        ctx.activeWindowWidth = rect.right - rect.left;
        ctx.activeWindowHeight = rect.bottom - rect.top;
    }
    
    // Get process name
    DWORD processId = 0;
    GetWindowThreadProcessId(hwnd, &processId);
    
    if (processId != 0) {
        HANDLE hProcess = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, FALSE, processId);
        if (hProcess) {
            char processName[MAX_PATH] = {0};
            if (GetModuleBaseNameA(hProcess, nullptr, processName, sizeof(processName))) {
                ctx.activeProcessName = processName;
            }
            CloseHandle(hProcess);
        }
    }
    
    // LOG_DEBUG("PerceptionContext", "Active window: \"" + ctx.activeWindowTitle + 
    //           "\" (" + ctx.activeProcessName + ")");
#endif
}

void PerceptionContextManager::performOCR(VisualContext& ctx) {
    if (ctx.screenshot.empty()) {
        return;
    }
    
    // Use existing perception OCR functionality
    // For now, call the basic readText function
    std::string ocrResult = GRIM::Perception::readText();
    
    if (!ocrResult.empty() && ocrResult.find("[Error]") == std::string::npos) {
        // Parse OCR result to extract text and confidence
        ctx.hasText = true;
        
        // Extract confidence if present
        size_t confPos = ocrResult.find("confidence: ");
        if (confPos != std::string::npos) {
            size_t endPos = ocrResult.find("%", confPos);
            if (endPos != std::string::npos) {
                std::string confStr = ocrResult.substr(confPos + 12, endPos - (confPos + 12));
                ctx.ocrConfidence = std::stof(confStr) / 100.0f;
            }
        }
        
        // Extract text content
        size_t textStart = ocrResult.find("---\n");
        size_t textEnd = ocrResult.rfind("\n---");
        if (textStart != std::string::npos && textEnd != std::string::npos) {
            ctx.screenText = ocrResult.substr(textStart + 4, textEnd - (textStart + 4));
        } else {
            ctx.screenText = ocrResult;
        }
        
        LOG_DEBUG("PerceptionContext", "OCR extracted " + 
                  std::to_string(ctx.screenText.length()) + " characters");
    }
}

void PerceptionContextManager::detectObjects(VisualContext& ctx) {
    if (ctx.screenshot.empty()) {
        return;
    }
    
    // Use existing perception object detection
    std::string detectionResult = GRIM::Perception::detectObjects();
    
    if (!detectionResult.empty() && detectionResult.find("[Error]") == std::string::npos) {
        // Parse object detection results
        // This is a simplified parser - you may need to enhance based on actual format
        // LOG_DEBUG("PerceptionContext", "Object detection result: " + detectionResult);
        
        // TODO: Parse detectionResult and populate ctx.detectedObjects
        // For now, just log that detection was attempted
    }
}

void PerceptionContextManager::analyzeWithVisionAI(VisualContext& ctx) {
    if (ctx.screenshot.empty()) {
        return;
    }
    
    // Check if vision AI is available
    if (!g_visionAI || !g_visionAI->isAvailable()) {
        LOG_DEBUG("PerceptionContext", "Vision AI not available, skipping");
        return;
    }
    
    // Build a context-aware prompt based on the scene type
    std::string prompt;
    switch (ctx.sceneType) {
        case VisualContext::SceneType::IDE_Code:
            prompt = "In one brief sentence, describe what code/project the developer is working on.";
            break;
        case VisualContext::SceneType::WebBrowser:
            prompt = "In one brief sentence, what website or web content is being viewed?";
            break;
        case VisualContext::SceneType::Terminal:
            prompt = "In one brief sentence, what is happening in this terminal?";
            break;
        case VisualContext::SceneType::Image_Video:
        case VisualContext::SceneType::Game:
            prompt = "In one brief sentence, describe what's being displayed.";
            break;
        default:
            prompt = "In one brief sentence, describe what the user is doing on this screen.";
            break;
    }
    
    // Call vision AI
    VisionAnalysisRequest request;
    request.image = ctx.screenshot;
    request.prompt = prompt;
    // ✅ Use Ollama vision model for full image-to-text understanding
    // ONNX encoder is available and GPU-accelerated (extracts embeddings in ~300ms)
    // TODO: Integrate ONNX embeddings with Ollama language model for hybrid approach
    request.backend = VisionAIBackend::Ollama_LLaVA; // Full vision-language model
    request.temperature = 0.3f; // Lower temperature for more consistent descriptions
    request.maxTokens = 300; // Reasonable length for screen descriptions
    
    LOG_DEBUG("PerceptionContext", "Requesting vision AI analysis...");
    VisionAnalysisResult result = g_visionAI->analyzeImage(request);
    
    if (result.success) {
        ctx.visionAIDescription = result.description;
        ctx.visionAIConfidence = result.confidence;
        ctx.visionModelUsed = result.modelUsed;
        
        // Update context type if vision AI provided better classification
        if (!result.contextType.empty()) {
            LOG_DEBUG("PerceptionContext", "Vision AI context type: " + result.contextType);
        }
        
        LOG_DEBUG("PerceptionContext", "Vision AI analysis complete: " + 
                  result.description.substr(0, 100) + "...");
    } else {
        LOG_DEBUG("PerceptionContext", "Vision AI analysis failed: " + result.errorMessage);
    }
}

void PerceptionContextManager::classifyScene(VisualContext& ctx) {
    // Use window title and process name to classify scene
    std::string lowerTitle = ctx.activeWindowTitle;
    std::string lowerProcess = ctx.activeProcessName;
    std::transform(lowerTitle.begin(), lowerTitle.end(), lowerTitle.begin(), ::tolower);
    std::transform(lowerProcess.begin(), lowerProcess.end(), lowerProcess.begin(), ::tolower);
    
    // Browser detection
    if (lowerProcess.find("chrome") != std::string::npos ||
        lowerProcess.find("firefox") != std::string::npos ||
        lowerProcess.find("edge") != std::string::npos ||
        lowerProcess.find("brave") != std::string::npos) {
        ctx.sceneType = VisualContext::SceneType::WebBrowser;
    }
    // IDE detection
    else if (lowerProcess.find("code") != std::string::npos ||
             lowerProcess.find("visual studio") != std::string::npos ||
             lowerProcess.find("idea") != std::string::npos ||
             lowerProcess.find("pycharm") != std::string::npos) {
        ctx.sceneType = VisualContext::SceneType::IDE_Code;
    }
    // Terminal detection
    else if (lowerProcess.find("powershell") != std::string::npos ||
             lowerProcess.find("cmd") != std::string::npos ||
             lowerProcess.find("terminal") != std::string::npos ||
             lowerProcess.find("wt.exe") != std::string::npos) {
        ctx.sceneType = VisualContext::SceneType::Terminal;
    }
    // Chat/messaging
    else if (lowerProcess.find("discord") != std::string::npos ||
             lowerProcess.find("slack") != std::string::npos ||
             lowerProcess.find("teams") != std::string::npos) {
        ctx.sceneType = VisualContext::SceneType::Chat_Messaging;
    }
    // Document editors
    else if (lowerProcess.find("word") != std::string::npos ||
             lowerProcess.find("excel") != std::string::npos ||
             lowerProcess.find("notepad") != std::string::npos) {
        ctx.sceneType = VisualContext::SceneType::Document;
    }
    else {
        ctx.sceneType = VisualContext::SceneType::Desktop;
    }
}

void PerceptionContextManager::computeVisualCharacteristics(VisualContext& ctx) {
    if (ctx.screenshot.empty()) {
        return;
    }
    
    try {
        // Convert to grayscale for analysis
        cv::Mat gray;
        cv::cvtColor(ctx.screenshot, gray, cv::COLOR_BGR2GRAY);
        
        // Compute mean brightness
        cv::Scalar meanBrightness = cv::mean(gray);
        ctx.brightnessMean = static_cast<float>(meanBrightness[0]) / 255.0f;
        ctx.isDarkTheme = ctx.brightnessMean < 0.4f;
        
        // Compute contrast (standard deviation)
        cv::Mat meanMat, stddevMat;
        cv::meanStdDev(gray, meanMat, stddevMat);
        ctx.contrastScore = static_cast<float>(stddevMat.at<double>(0, 0)) / 128.0f;
        
        // Estimate text density using edge detection
        cv::Mat edges;
        cv::Canny(gray, edges, 50, 150);
        int edgePixels = cv::countNonZero(edges);
        int totalPixels = edges.rows * edges.cols;
        ctx.textDensity = static_cast<float>(edgePixels) / totalPixels;
        
        // LOG_DEBUG("PerceptionContext", "Visual characteristics - Brightness: " + 
        //           std::to_string(ctx.brightnessMean) + ", Contrast: " + 
        //           std::to_string(ctx.contrastScore) + ", TextDensity: " +
        //           std::to_string(ctx.textDensity));
                  
    } catch (const cv::Exception& e) {
        LOG_ERROR("PerceptionContext", "Failed to compute visual characteristics: " + 
                  std::string(e.what()));
    }
}

float PerceptionContextManager::computeChangeScore(const cv::Mat& current, const cv::Mat& previous) {
    if (current.empty() || previous.empty()) {
        return 1.0f; // Completely different
    }
    
    if (current.size() != previous.size()) {
        return 1.0f; // Different resolutions
    }
    
    try {
        // Resize for faster comparison
        cv::Mat curr_small, prev_small;
        cv::resize(current, curr_small, cv::Size(320, 180));
        cv::resize(previous, prev_small, cv::Size(320, 180));
        
        // Compute absolute difference
        cv::Mat diff;
        cv::absdiff(curr_small, prev_small, diff);
        
        // Calculate mean difference
        cv::Scalar meanDiff = cv::mean(diff);
        double avgDiff = (meanDiff[0] + meanDiff[1] + meanDiff[2]) / 3.0;
        
        // Normalize to 0.0-1.0 range
        float changeScore = static_cast<float>(avgDiff / 255.0);
        
        // LOG_DEBUG("PerceptionContext", "Change score: " + std::to_string(changeScore));
        return changeScore;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("PerceptionContext", "Failed to compute change score: " + 
                  std::string(e.what()));
        return 0.5f; // Unknown change
    }
}

float PerceptionContextManager::getChangeScore() {
    if (!pImpl->lastContext.isValid) {
        return 1.0f;
    }
    
    return pImpl->lastContext.changeScore;
}

void PerceptionContextManager::setFeatureEnabled(const std::string& feature, bool enabled) {
    if (feature == "ocr") {
        pImpl->featureOCR = enabled;
    } else if (feature == "object_detection") {
        pImpl->featureObjectDetection = enabled;
    } else if (feature == "vision_ai") {
        pImpl->featureVisionAI = enabled;
    } else if (feature == "window_tracking") {
        pImpl->featureWindowTracking = enabled;
    }
    
    LOG_DEBUG("PerceptionContext", "Feature '" + feature + "' " + 
              (enabled ? "enabled" : "disabled"));
}

PerceptionContextManager::PerceptionStatus PerceptionContextManager::getStatus() {
    PerceptionStatus status;
    
#ifdef _WIN32
    status.screenCaptureAvailable = true;
    status.windowTrackingAvailable = true;
#else
    status.screenCaptureAvailable = false;
    status.windowTrackingAvailable = false;
#endif
    
    status.ocrAvailable = GRIM::Perception::isAvailable();
    status.ocrEngine = status.ocrAvailable ? "Tesseract" : "None";
    
    status.objectDetectionAvailable = GRIM::Perception::isAvailable();
    status.objectDetectionModel = "Basic/YOLO";
    
    status.visionAIAvailable = false; // Not yet implemented
    status.visionAIModel = "None";
    
    return status;
}

std::string PerceptionContextManager::answerVisionQuestion(const std::string& question) {
    // Parse question for monitor-specific requests
    std::string lowerQ = question;
    std::transform(lowerQ.begin(), lowerQ.end(), lowerQ.begin(), ::tolower);
    
    int requestedMonitor = -1; // -1 = active/default
    
    // Check for "monitor 1", "monitor 2", "screen 2", etc.
    std::regex monitorRegex(R"((?:monitor|screen|display)\s+(\d+))");
    std::smatch match;
    if (std::regex_search(lowerQ, match, monitorRegex)) {
        requestedMonitor = std::stoi(match[1].str()) - 1; // Convert to 0-based index
        LOG_DEBUG("PerceptionContext", "User requested monitor " + std::to_string(requestedMonitor + 1));
    }
    
    // Check for "left monitor", "right monitor", "middle monitor"
    int monitorCount = GRIM::Perception::getMonitorCount();
    if (lowerQ.find("left monitor") != std::string::npos || 
        lowerQ.find("leftmost") != std::string::npos) {
        requestedMonitor = 0; // Assume first is leftmost
    } else if (lowerQ.find("right monitor") != std::string::npos || 
               lowerQ.find("rightmost") != std::string::npos) {
        requestedMonitor = monitorCount - 1; // Last is rightmost
    } else if (lowerQ.find("middle monitor") != std::string::npos || 
               lowerQ.find("center monitor") != std::string::npos) {
        if (monitorCount >= 3) {
            requestedMonitor = 1; // Middle monitor in 3-monitor setup
        }
    }
    
    // Get context for requested monitor
    VisualContext ctx;
    if (requestedMonitor >= 0) {
        LOG_DEBUG("PerceptionContext", "Getting context for monitor " + std::to_string(requestedMonitor) + 
                  " (0-based index, user said monitor " + std::to_string(requestedMonitor + 1) + ")");
        ctx = getMonitorContext(requestedMonitor, false);
    } else {
        LOG_DEBUG("PerceptionContext", "Getting context for active monitor (default)");
        ctx = getCurrentContext(false);
    }
    
    if (!ctx.isValid) {
        return "I cannot see your screen right now. " + ctx.errorMessage;
    }
    
    std::string lowerQClean = question;
    std::transform(lowerQClean.begin(), lowerQClean.end(), lowerQClean.begin(), ::tolower);
    
    std::ostringstream response;
    
    // ✅ NEW: Application location queries ("where is Chrome?", "which monitor has VS Code?")
    if ((lowerQClean.find("where is") != std::string::npos ||
         lowerQClean.find("which monitor") != std::string::npos ||
         lowerQClean.find("what monitor") != std::string::npos ||
         lowerQClean.find("find") != std::string::npos) &&
        (lowerQClean.find("app") != std::string::npos ||
         lowerQClean.find("application") != std::string::npos ||
         lowerQClean.find("chrome") != std::string::npos ||
         lowerQClean.find("firefox") != std::string::npos ||
         lowerQClean.find("code") != std::string::npos ||
         lowerQClean.find("vscode") != std::string::npos ||
         lowerQClean.find("browser") != std::string::npos ||
         lowerQClean.find("explorer") != std::string::npos)) {
        
        // Extract app name from question
        std::string appName;
        
        // Try to extract from "where is X" pattern
        size_t wherePos = lowerQClean.find("where is ");
        if (wherePos != std::string::npos) {
            appName = question.substr(wherePos + 9);
            // Clean up
            size_t questionMark = appName.find('?');
            if (questionMark != std::string::npos) {
                appName = appName.substr(0, questionMark);
            }
        }
        // Try "which monitor has X"
        else if (lowerQClean.find("which monitor has") != std::string::npos) {
            size_t hasPos = lowerQClean.find("has ");
            appName = question.substr(hasPos + 4);
            size_t questionMark = appName.find('?');
            if (questionMark != std::string::npos) {
                appName = appName.substr(0, questionMark);
            }
        }
        // Fallback: look for known app names
        else {
            if (lowerQClean.find("chrome") != std::string::npos) appName = "chrome";
            else if (lowerQClean.find("firefox") != std::string::npos) appName = "firefox";
            else if (lowerQClean.find("vscode") != std::string::npos || 
                     lowerQClean.find("vs code") != std::string::npos ||
                     lowerQClean.find("code") != std::string::npos) appName = "code";
            else if (lowerQClean.find("explorer") != std::string::npos) appName = "explorer";
        }
        
        // Trim whitespace
        appName.erase(0, appName.find_first_not_of(" \t\n\r"));
        appName.erase(appName.find_last_not_of(" \t\n\r") + 1);
        
        if (!appName.empty()) {
            auto locations = findApplication(appName);
            
            if (locations.empty()) {
                return "I don't see " + appName + " on any of your monitors right now.";
            } else if (locations.size() == 1) {
                std::string result = "I found " + locations[0].appName + " on your " + 
                                   locations[0].monitorDescription;
                if (!locations[0].windowTitle.empty() && 
                    locations[0].windowTitle != locations[0].appName) {
                    result += ":\n\"" + locations[0].windowTitle + "\"";
                }
                return result;
            } else {
                std::string result = "I found " + appName + " on multiple monitors:\n";
                for (const auto& loc : locations) {
                    result += "  • " + loc.monitorDescription + ": " + loc.windowTitle + "\n";
                }
                return result;
            }
        }
    }
    
    // ✅ NEW: Overview of all monitors ("what's on all my monitors?")
    if ((lowerQClean.find("all") != std::string::npos || 
         lowerQClean.find("each") != std::string::npos) &&
        (lowerQClean.find("monitor") != std::string::npos || 
         lowerQClean.find("screen") != std::string::npos)) {
        return describeApplicationLocations();
    }
    
    // General "what's on my screen" questions
    if (lowerQClean.find("what") != std::string::npos && 
        (lowerQClean.find("screen") != std::string::npos || lowerQClean.find("see") != std::string::npos ||
         lowerQClean.find("monitor") != std::string::npos)) {
        
        response << "I can see ";
        
        // Monitor information removed from voice output - not needed for most interactions
        // if (ctx.isMultiMonitor && ctx.monitorIndex >= 0) {
        //     response << "monitor " << (ctx.monitorIndex + 1) << " (of " << ctx.totalMonitors << "):\n\n";
        // } else {
            response << "your screen:\n\n";
        // }
        
        // ✅ NEW: Prioritize AI vision description if available
        if (ctx.hasAIAnalysis && !ctx.aiDescription.empty()) {
            response << ctx.aiDescription << "\n\n";
        }
        
        response << "You're viewing: " << sceneTypeToString(ctx.sceneType) << "\n";
        
        if (!ctx.activeWindowTitle.empty()) {
            response << "Active window: \"" << ctx.activeWindowTitle << "\"\n";
        }
        
        // Theme detection removed from voice output - not useful for most interactions
        // if (ctx.isDarkTheme) {
        //     response << "Theme: Dark mode\n";
        // }
        
        // ✅ Only show OCR text if no AI description is available
        if (!ctx.hasAIAnalysis && ctx.hasText && !ctx.screenText.empty()) {
            response << "\nVisible text:\n";
            // Clean symbols from text for better TTS output
            std::string cleanedText = cleanTextForTTS(ctx.screenText);
            // ✅ Shortened preview - only show first 150 characters to keep voice response brief
            std::string preview = cleanedText.substr(0, std::min<size_t>(150, cleanedText.length()));
            response << preview;
            if (cleanedText.length() > 150) {
                response << "...";
            }
        }
        
        if (!ctx.detectedObjects.empty()) {
            response << "\n\nDetected objects:\n";
            for (const auto& obj : ctx.detectedObjects) {
                response << "  - " << obj.label << " (" << (int)(obj.confidence * 100) << "%)\n";
            }
        }
        
        return response.str();
    }
    
    // Text reading questions
    if (lowerQClean.find("read") != std::string::npos || lowerQClean.find("text") != std::string::npos) {
        if (ctx.hasText && !ctx.screenText.empty()) {
            return "Screen text:\n" + ctx.screenText;
        } else {
            return "I don't see any readable text on your screen right now.";
        }
    }
    
    // Object/element questions
    if (lowerQClean.find("object") != std::string::npos || lowerQClean.find("element") != std::string::npos) {
        if (!ctx.detectedObjects.empty()) {
            response << "I can see these objects:\n";
            for (const auto& obj : ctx.detectedObjects) {
                response << "  - " << obj.label << " at position (" 
                         << obj.x << ", " << obj.y << ")\n";
            }
            std::string result = response.str();
            
            // ✅ Store in memory for recall
            storeContextInMemory(ctx, "User asked: " + question);
            
            return result;
        } else {
            return "Object detection didn't identify specific objects on your screen.";
        }
    }
    
    // ✅ Store context in memory before returning (for temporal awareness)
    storeContextInMemory(ctx, "User asked: " + question);
    
    // Fallback to context summary
    return ctx.toDetailedString();
}

// Global initialization functions
void initContextManager() {
    if (!g_contextManager) {
        g_contextManager = std::make_unique<PerceptionContextManager>();
        g_contextManager->init();
    }
}

VisualContext getCurrentVisualContext(bool forceRefresh) {
    if (!g_contextManager) {
        initContextManager();
    }
    return g_contextManager->getCurrentContext(forceRefresh);
}

std::string answerVisionQuestionWithContext(const std::string& question) {
    if (!g_contextManager) {
        initContextManager();
    }
    return g_contextManager->answerVisionQuestion(question);
}

// ✅ NEW: Continuous capture implementation
void PerceptionContextManager::startContinuousCapture(const ContinuousCaptureConfig& config) {
    if (m_captureThreadRunning) {
        LOG_DEBUG("PerceptionContext", "Continuous capture already running");
        return;
    }
    
    pImpl->captureConfig = config;
    pImpl->currentFrame = 0;
    pImpl->lastCaptureTimePoint = std::chrono::steady_clock::now();
    
    m_captureThreadRunning = true;
    m_captureThread = std::make_unique<std::thread>(&PerceptionContextManager::continuousCaptureThread, this);
    
    LOG_DEBUG("PerceptionContext", "Started continuous capture (frame skip: " + 
              std::to_string(config.frameSkip) + ", interval: " + 
              std::to_string(config.captureIntervalMs) + "ms)");
}

void PerceptionContextManager::stopContinuousCapture() {
    if (!m_captureThreadRunning) {
        return;
    }
    
    m_captureThreadRunning = false;
    if (m_captureThread && m_captureThread->joinable()) {
        m_captureThread->join();
    }
    m_captureThread.reset();
    
    LOG_DEBUG("PerceptionContext", "Stopped continuous capture");
}

bool PerceptionContextManager::isContinuousCaptureRunning() const {
    return m_captureThreadRunning;
}

VisualContext PerceptionContextManager::getLatestContext() const {
    std::lock_guard<std::mutex> lock(pImpl->contextMutex);
    return pImpl->latestCapturedContext;
}

void PerceptionContextManager::continuousCaptureThread() {
    LOG_DEBUG("PerceptionContext", "Continuous capture thread started");
    
    while (m_captureThreadRunning) {
        bool shouldCapture = false;
        
        if (pImpl->captureConfig.useFrameSkip) {
            // Frame-based capture
            pImpl->currentFrame++;
            if (pImpl->currentFrame >= pImpl->captureConfig.frameSkip) {
                shouldCapture = true;
                pImpl->currentFrame = 0;
            }
        } else {
            // Time-based capture
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - pImpl->lastCaptureTimePoint).count();
            
            if (elapsed >= pImpl->captureConfig.captureIntervalMs) {
                shouldCapture = true;
                pImpl->lastCaptureTimePoint = now;
            }
        }
        
        if (shouldCapture) {
            try {
                VisualContext newContext;
                
                // ✅ Vision AI is disabled by default for continuous capture (too slow!)
                bool useAI = pImpl->captureConfig.useVisionAI;
                
                if (pImpl->captureConfig.captureAllMonitors) {
                    // Capture all monitors (use primary for now)
                    newContext = captureAndAnalyzeMonitor(0, useAI, false); // No screenshot save
                } else {
                    // Capture active monitor
                    newContext = captureAndAnalyzeMonitor(-1, useAI, false); // No screenshot save
                }
                
                // Check if significant change occurred
                if (!pImpl->latestCapturedContext.isValid || 
                    newContext.changeScore >= pImpl->captureConfig.changeThreshold) {
                    
                    std::lock_guard<std::mutex> lock(pImpl->contextMutex);
                    pImpl->latestCapturedContext = newContext;
                    
                    LOG_DEBUG("PerceptionContext", "Captured new context (change: " + 
                              std::to_string((int)(newContext.changeScore * 100)) + "%)");
                    
                    // ✅ Store significant changes in memory for temporal awareness
                    if (newContext.changeScore >= 0.15f) { // 15% change = significant
                        std::string description = "Significant screen change detected (" + 
                                                std::to_string((int)(newContext.changeScore * 100)) + "% change)";
                        storeContextInMemory(newContext, description);
                    }
                }
            } catch (const std::exception& e) {
                LOG_DEBUG("PerceptionContext", std::string("Continuous capture error: ") + e.what());
            }
        }
        
        // Sleep briefly to avoid CPU spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps check rate
    }
    
    LOG_DEBUG("PerceptionContext", "Continuous capture thread stopped");
}

// ✅ NEW: Enhanced OCR with preprocessing
void PerceptionContextManager::performOCREnhanced(VisualContext& ctx) {
    if (!pImpl->featureOCR || ctx.screenshot.empty()) {
        return;
    }
    
    // Validate screenshot dimensions
    if (ctx.screenshot.cols < 10 || ctx.screenshot.rows < 10) {
        LOG_DEBUG("PerceptionContext", "Screenshot too small for OCR, skipping");
        return;
    }
    
    // LOG_DEBUG("PerceptionContext", "Running enhanced OCR with preprocessing");
    
    // Convert to grayscale
    cv::Mat gray;
    try {
        if (ctx.screenshot.channels() == 3) {
            cv::cvtColor(ctx.screenshot, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = ctx.screenshot.clone();
        }
        
        // Validate grayscale conversion
        if (gray.empty() || gray.cols < 10 || gray.rows < 10) {
            LOG_DEBUG("PerceptionContext", "Invalid grayscale image, skipping OCR");
            return;
        }
    } catch (const cv::Exception& e) {
        LOG_ERROR("PerceptionContext", "Failed to convert to grayscale: " + std::string(e.what()));
        return;
    } catch (const std::exception& e) {
        LOG_ERROR("PerceptionContext", "Exception in grayscale conversion: " + std::string(e.what()));
        return;
    }
    
    // Try multiple preprocessing strategies and pick best result
    std::vector<std::string> candidateTexts;
    std::vector<float> candidateConfidences;
    
    // Strategy 1: Simple binary threshold (fast and reliable)
    try {
        cv::Mat binary;
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        if (!binary.empty() && binary.cols >= 10 && binary.rows >= 10) {
            // ✅ Check if image has valid data
            cv::Scalar meanVal = cv::mean(binary);
            if (meanVal[0] > 1.0 && meanVal[0] < 254.0) { // Not all black or all white
                std::string text = GRIM::Perception::readTextFromImage(binary);
                candidateTexts.push_back(text);
                candidateConfidences.push_back(0.75f);
                // LOG_DEBUG("PerceptionContext", "Strategy 1 (binary) completed");
            } else {
                LOG_DEBUG("PerceptionContext", "Strategy 1: Image is blank or invalid (mean: " + std::to_string(meanVal[0]) + ")");
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("PerceptionContext", "Strategy 1 failed: " + std::string(e.what()));
    }
    
    // Strategy 2: Adaptive threshold (good for varied lighting)
    try {
        cv::Mat adaptiveImg;
        cv::adaptiveThreshold(gray, adaptiveImg, 255, 
                             cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv::THRESH_BINARY, 11, 2);
        
        if (!adaptiveImg.empty() && adaptiveImg.cols >= 10 && adaptiveImg.rows >= 10) {
            // ✅ Check if image has valid data (not blank)
            cv::Scalar meanVal = cv::mean(adaptiveImg);
            if (meanVal[0] > 1.0 && meanVal[0] < 254.0) { // Not all black or all white
                std::string text = GRIM::Perception::readTextFromImage(adaptiveImg);
                candidateTexts.push_back(text);
                candidateConfidences.push_back(0.8f);
                // LOG_DEBUG("PerceptionContext", "Strategy 2 (adaptive) completed");
            } else {
                LOG_DEBUG("PerceptionContext", "Strategy 2: Image is blank or invalid (mean: " + std::to_string(meanVal[0]) + ")");
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("PerceptionContext", "Strategy 2 failed: " + std::string(e.what()));
    }
    
    // Strategy 3: CLAHE only (no denoise - that's what crashes!)
    try {
        cv::Mat enhanced;
        
        // CLAHE (Contrast Limited Adaptive Histogram Equalization)
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(gray, enhanced);
        
        // Validate after CLAHE
        if (!enhanced.empty() && enhanced.cols >= 10 && enhanced.rows >= 10) {
            cv::Scalar meanVal = cv::mean(enhanced);
            if (meanVal[0] > 1.0 && meanVal[0] < 254.0) {
                // Apply simple threshold after CLAHE
                cv::Mat thresholded;
                cv::threshold(enhanced, thresholded, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
                
                if (!thresholded.empty()) {
                    std::string text = GRIM::Perception::readTextFromImage(thresholded);
                    candidateTexts.push_back(text);
                    candidateConfidences.push_back(0.85f);
                    // LOG_DEBUG("PerceptionContext", "Strategy 3 (CLAHE) completed");
                }
            } else {
                LOG_DEBUG("PerceptionContext", "Strategy 3: Image is blank after CLAHE");
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("PerceptionContext", "Strategy 3 failed: " + std::string(e.what()));
    }
    
    // If no strategies succeeded, return
    if (candidateTexts.empty()) {
        LOG_DEBUG("PerceptionContext", "All OCR strategies failed");
        return;
    }
    
    // Pick longest non-empty result (more text usually means better detection)
    size_t bestIdx = 0;
    size_t maxLength = 0;
    for (size_t i = 0; i < candidateTexts.size(); ++i) {
        if (candidateTexts[i].length() > maxLength) {
            maxLength = candidateTexts[i].length();
            bestIdx = i;
        }
    }
    
    ctx.screenText = candidateTexts[bestIdx];
    ctx.ocrConfidence = candidateConfidences[bestIdx];
    ctx.hasText = !ctx.screenText.empty();
    
    // LOG_DEBUG("PerceptionContext", "Enhanced OCR found " + std::to_string(ctx.screenText.length()) + 
    //           " chars (strategy " + std::to_string(bestIdx + 1) + ")");
}

// ✅ NEW: Store visual context in memory for temporal awareness
void PerceptionContextManager::storeContextInMemory(const VisualContext& ctx, const std::string& description) {
    if (!ctx.isValid) {
        LOG_DEBUG("PerceptionContext", "Not storing invalid context in memory");
        return;
    }
    
    try {
        MemoryObject memory;
        memory.source = SourceTag::GrimInternal;
        memory.intent = IntentTag::Inform;        // ✅ Using Inform instead of Observe
        memory.context = ContextTag::Monitor;     // ✅ Using Monitor instead of Environment
        memory.type = TypeTag::Event;
        memory.confidence = 0.8f;
        
        // Build memory content
        std::ostringstream content;
        
        if (!description.empty()) {
            content << description << "\n\n";
        }
        
        // Screen info
        content << "Screen " << ctx.screenWidth << "x" << ctx.screenHeight;
        if (ctx.isMultiMonitor) {
            content << " (Monitor " << (ctx.monitorIndex + 1) << " of " << ctx.totalMonitors << ")";
        }
        content << "\n";
        
        // Window info
        if (!ctx.activeWindowTitle.empty()) {
            content << "Active: " << ctx.activeWindowTitle;
            if (!ctx.activeProcessName.empty()) {
                content << " (" << ctx.activeProcessName << ")";
            }
            content << "\n";
        }
        
        // Scene classification
        content << "Scene: " << sceneTypeToString(ctx.sceneType);
        content << " | Theme: " << (ctx.isDarkTheme ? "Dark" : "Light") << "\n";
        
        // Text preview (first 500 chars)
        if (ctx.hasText && !ctx.screenText.empty()) {
            std::string textPreview = ctx.screenText.substr(0, std::min<size_t>(500, ctx.screenText.length()));
            content << "Text: " << textPreview;
            if (ctx.screenText.length() > 500) {
                content << "... (" << ctx.screenText.length() << " total chars)";
            }
            content << "\n";
        }
        
        // Objects
        if (!ctx.detectedObjects.empty()) {
            content << "Objects: ";
            for (size_t i = 0; i < std::min<size_t>(5, ctx.detectedObjects.size()); ++i) {
                if (i > 0) content << ", ";
                content << ctx.detectedObjects[i].label;
            }
            if (ctx.detectedObjects.size() > 5) {
                content << " (+" << (ctx.detectedObjects.size() - 5) << " more)";
            }
            content << "\n";
        }
        
        memory.raw = content.str();
        memory.normalized = memory.raw;
        
        // Add tags
        memory.tags.push_back("visual-context");
        memory.tags.push_back("screen-capture");
        memory.tags.push_back(sceneTypeToString(ctx.sceneType));
        
        if (!ctx.activeProcessName.empty()) {
            memory.tags.push_back(ctx.activeProcessName);
        }
        
        if (ctx.isMultiMonitor) {
            memory.tags.push_back("monitor-" + std::to_string(ctx.monitorIndex + 1));
        }
        
        // Store in long-term memory
        g_memoryStorage.storeLongTerm(memory);
        
        LOG_DEBUG("PerceptionContext", "Stored visual context in memory: " + memory.id);
        
    } catch (const std::exception& e) {
        LOG_ERROR("PerceptionContext", std::string("Failed to store context in memory: ") + e.what());
    }
}

// ✅ NEW: Find which monitor(s) have a specific application
std::vector<PerceptionContextManager::AppLocation> PerceptionContextManager::findApplication(const std::string& appName) {
    std::vector<AppLocation> locations;
    
    if (!pImpl->initialized) {
        return locations;
    }
    
    std::string lowerAppName = appName;
    std::transform(lowerAppName.begin(), lowerAppName.end(), lowerAppName.begin(), ::tolower);
    
    int monitorCount = GRIM::Perception::getMonitorCount();
    
    // Scan all monitors for the application
    for (int i = 0; i < monitorCount; ++i) {
        try {
            VisualContext ctx = getMonitorContext(i, false);
            
            if (!ctx.isValid) {
                continue;
            }
            
            // Check if this monitor has the app
            std::string lowerProcessName = ctx.activeProcessName;
            std::transform(lowerProcessName.begin(), lowerProcessName.end(), lowerProcessName.begin(), ::tolower);
            
            std::string lowerWindowTitle = ctx.activeWindowTitle;
            std::transform(lowerWindowTitle.begin(), lowerWindowTitle.end(), lowerWindowTitle.begin(), ::tolower);
            
            bool foundOnThisMonitor = false;
            
            // Check process name
            if (lowerProcessName.find(lowerAppName) != std::string::npos) {
                foundOnThisMonitor = true;
            }
            // Check window title
            else if (lowerWindowTitle.find(lowerAppName) != std::string::npos) {
                foundOnThisMonitor = true;
            }
            // Check common app name variations
            else {
                // VS Code variations
                if ((lowerAppName.find("vscode") != std::string::npos || 
                     lowerAppName.find("vs code") != std::string::npos ||
                     lowerAppName.find("code") != std::string::npos) &&
                    lowerProcessName.find("code") != std::string::npos) {
                    foundOnThisMonitor = true;
                }
                // Chrome variations
                else if ((lowerAppName.find("chrome") != std::string::npos ||
                         lowerAppName.find("browser") != std::string::npos) &&
                        lowerProcessName.find("chrome") != std::string::npos) {
                    foundOnThisMonitor = true;
                }
                // Firefox variations
                else if ((lowerAppName.find("firefox") != std::string::npos ||
                         lowerAppName.find("browser") != std::string::npos) &&
                        lowerProcessName.find("firefox") != std::string::npos) {
                    foundOnThisMonitor = true;
                }
            }
            
            if (foundOnThisMonitor) {
                AppLocation location;
                location.appName = ctx.activeProcessName;
                location.windowTitle = ctx.activeWindowTitle;
                location.monitorIndex = i;
                
                // Generate friendly monitor description
                if (monitorCount == 1) {
                    location.monitorDescription = "your monitor";
                } else if (monitorCount == 2) {
                    location.monitorDescription = (i == 0) ? "left monitor" : "right monitor";
                } else if (monitorCount == 3) {
                    if (i == 0) location.monitorDescription = "left monitor";
                    else if (i == 1) location.monitorDescription = "middle monitor";
                    else location.monitorDescription = "right monitor";
                } else {
                    location.monitorDescription = "monitor " + std::to_string(i + 1);
                }
                
                location.isActive = true; // Simplified - could check if it's the foreground window
                
                locations.push_back(location);
                
                LOG_DEBUG("PerceptionContext", "Found " + ctx.activeProcessName + " on monitor " + 
                          std::to_string(i + 1));
            }
            
        } catch (const std::exception& e) {
            LOG_ERROR("PerceptionContext", std::string("Error scanning monitor ") + 
                      std::to_string(i) + ": " + e.what());
        }
    }
    
    return locations;
}

// ✅ NEW: Describe all application locations across monitors
std::string PerceptionContextManager::describeApplicationLocations() {
    if (!pImpl->initialized) {
        return "Perception system not initialized";
    }
    
    int monitorCount = GRIM::Perception::getMonitorCount();
    std::ostringstream result;
    
    result << "Application overview across " << monitorCount << " monitor(s):\n\n";
    
    for (int i = 0; i < monitorCount; ++i) {
        try {
            VisualContext ctx = getMonitorContext(i, false);
            
            if (!ctx.isValid) {
                result << "Monitor " << (i + 1) << ": Unable to capture\n";
                continue;
            }
            
            // Monitor description
            if (monitorCount == 2) {
                result << (i == 0 ? "Left" : "Right") << " monitor";
            } else if (monitorCount == 3) {
                if (i == 0) result << "Left monitor";
                else if (i == 1) result << "Middle monitor";
                else result << "Right monitor";
            } else {
                result << "Monitor " << (i + 1);
            }
            
            result << " (" << ctx.screenWidth << "x" << ctx.screenHeight << "):\n";
            
            // Active window
            if (!ctx.activeWindowTitle.empty()) {
                result << "  • " << ctx.activeWindowTitle;
                if (!ctx.activeProcessName.empty() && 
                    ctx.activeProcessName != ctx.activeWindowTitle) {
                    result << " (" << ctx.activeProcessName << ")";
                }
                result << "\n";
            } else {
                result << "  • No active window\n";
            }
            
            // Scene type
            result << "  • Scene: " << sceneTypeToString(ctx.sceneType);
            if (ctx.isDarkTheme) {
                result << " (Dark theme)";
            }
            result << "\n";
            
            // Text preview if available
            if (ctx.hasText && !ctx.screenText.empty()) {
                std::string preview = ctx.screenText.substr(0, std::min<size_t>(100, ctx.screenText.length()));
                // Clean up preview
                std::replace(preview.begin(), preview.end(), '\n', ' ');
                result << "  • Text preview: " << preview;
                if (ctx.screenText.length() > 100) {
                    result << "...";
                }
                result << "\n";
            }
            
            result << "\n";
            
        } catch (const std::exception& e) {
            result << "Monitor " << (i + 1) << ": Error - " << e.what() << "\n\n";
        }
    }
    
    return result.str();
}

// =========================================================
// INPUT CONTROL IMPLEMENTATION
// =========================================================

// Helper: Convert string button name to enum
static MouseButton stringToMouseButton(const std::string& button) {
    std::string lower = button;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "right") return MouseButton::Right;
    if (lower == "middle") return MouseButton::Middle;
    return MouseButton::Left; // default
}

// Helper: Convert string key name to virtual key code
static WORD stringToVirtualKey(const std::string& key) {
    std::string lower = key;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Common keys
    if (lower == "enter" || lower == "return") return VK_RETURN;
    if (lower == "esc" || lower == "escape") return VK_ESCAPE;
    if (lower == "tab") return VK_TAB;
    if (lower == "space") return VK_SPACE;
    if (lower == "backspace") return VK_BACK;
    if (lower == "delete" || lower == "del") return VK_DELETE;
    if (lower == "home") return VK_HOME;
    if (lower == "end") return VK_END;
    if (lower == "pageup" || lower == "pgup") return VK_PRIOR;
    if (lower == "pagedown" || lower == "pgdn") return VK_NEXT;
    
    // Arrow keys
    if (lower == "left") return VK_LEFT;
    if (lower == "right") return VK_RIGHT;
    if (lower == "up") return VK_UP;
    if (lower == "down") return VK_DOWN;
    
    // Modifier keys
    if (lower == "shift") return VK_SHIFT;
    if (lower == "ctrl" || lower == "control") return VK_CONTROL;
    if (lower == "alt") return VK_MENU;
    if (lower == "win" || lower == "windows") return VK_LWIN;
    
    // Function keys
    if (lower == "f1") return VK_F1;
    if (lower == "f2") return VK_F2;
    if (lower == "f3") return VK_F3;
    if (lower == "f4") return VK_F4;
    if (lower == "f5") return VK_F5;
    if (lower == "f6") return VK_F6;
    if (lower == "f7") return VK_F7;
    if (lower == "f8") return VK_F8;
    if (lower == "f9") return VK_F9;
    if (lower == "f10") return VK_F10;
    if (lower == "f11") return VK_F11;
    if (lower == "f12") return VK_F12;
    
    // Single character keys
    if (key.length() == 1) {
        char c = std::toupper(key[0]);
        if (c >= 'A' && c <= 'Z') return c;
        if (c >= '0' && c <= '9') return c;
    }
    
    return 0; // Unknown key
}

void PerceptionContextManager::moveMouseTo(int x, int y) {
    InputController::moveMouse(x, y);
    logDebug("Input", "Moved mouse to (" + std::to_string(x) + ", " + std::to_string(y) + ")");
}

void PerceptionContextManager::clickMouse(const std::string& button) {
    InputController::click(stringToMouseButton(button));
    logDebug("Input", "Clicked " + button + " mouse button");
}

void PerceptionContextManager::doubleClickMouse(const std::string& button) {
    InputController::doubleClick(stringToMouseButton(button));
    logDebug("Input", "Double-clicked " + button + " mouse button");
}

void PerceptionContextManager::scrollMouse(int delta) {
    InputController::scroll(delta);
    logDebug("Input", "Scrolled mouse by " + std::to_string(delta));
}

void PerceptionContextManager::typeText(const std::string& text, int delayMs) {
    InputController::typeText(text, delayMs);
    logDebug("Input", "Typed text (length: " + std::to_string(text.length()) + ")");
}

void PerceptionContextManager::pressKey(const std::string& key) {
    WORD vk = stringToVirtualKey(key);
    if (vk != 0) {
        InputController::keyEvent(vk, KeyAction::Press);
        logDebug("Input", "Pressed key '" + key + "'");
    } else {
        logError("Input", "Unknown key '" + key + "'");
    }
}

void PerceptionContextManager::releaseKey(const std::string& key) {
    WORD vk = stringToVirtualKey(key);
    if (vk != 0) {
        InputController::keyEvent(vk, KeyAction::Release);
        logDebug("Input", "Released key '" + key + "'");
    } else {
        logError("Input", "Unknown key '" + key + "'");
    }
}

void PerceptionContextManager::tapKey(const std::string& key) {
    WORD vk = stringToVirtualKey(key);
    if (vk != 0) {
        InputController::keyEvent(vk, KeyAction::Press);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        InputController::keyEvent(vk, KeyAction::Release);
        logDebug("Input", "Tapped key '" + key + "'");
    } else {
        logError("Input", "Unknown key '" + key + "'");
    }
}

void PerceptionContextManager::pressKeyCombo(const std::vector<std::string>& keys) {
    std::vector<WORD> vkKeys;
    for (const auto& key : keys) {
        WORD vk = stringToVirtualKey(key);
        if (vk != 0) {
            vkKeys.push_back(vk);
        }
    }
    
    if (!vkKeys.empty()) {
        InputController::combo(vkKeys);
        
        std::string comboStr;
        for (size_t i = 0; i < keys.size(); ++i) {
            comboStr += keys[i];
            if (i < keys.size() - 1) comboStr += "+";
        }
        logDebug("Input", "Pressed key combo '" + comboStr + "'");
    }
}

void PerceptionContextManager::clickAt(int x, int y, const std::string& button) {
    moveMouseTo(x, y);
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small delay for mouse to settle
    clickMouse(button);
}

void PerceptionContextManager::clickOnText(const std::string& text, const std::string& button) {
    // Get current context and find text
    auto ctx = getCurrentContext(true); // Force refresh
    
    if (!ctx.isValid || !ctx.hasText) {
        logError("Input", "Cannot click on text - no OCR data available");
        return;
    }
    
    // Simple text search - in a real implementation, you'd want to get bounding boxes from OCR
    // This is a simplified version that clicks in the center of the screen where text was found
    if (ctx.screenText.find(text) != std::string::npos) {
        // Click in center of screen as a fallback
        // TODO: Use actual OCR bounding box data for precise clicking
        int centerX = ctx.screenWidth / 2;
        int centerY = ctx.screenHeight / 2;
        
        logDebug("Input", "Clicking on text '" + text + "' at estimated position");
        clickAt(centerX, centerY, button);
    } else {
        logError("Input", "Text '" + text + "' not found on screen");
    }
}

void PerceptionContextManager::clickOnObject(const std::string& objectLabel, const std::string& button) {
    // Get current context and find object
    auto ctx = getCurrentContext(true); // Force refresh
    
    if (!ctx.isValid || ctx.detectedObjects.empty()) {
        logError("Input", "Cannot click on object - no object detection data available");
        return;
    }
    
    // Find matching object
    for (const auto& obj : ctx.detectedObjects) {
        if (obj.label.find(objectLabel) != std::string::npos) {
            // Click in center of detected object
            int centerX = obj.x + obj.width / 2;
            int centerY = obj.y + obj.height / 2;
            
            logDebug("Input", "Clicking on object '" + objectLabel + "' at (" + 
                       std::to_string(centerX) + ", " + std::to_string(centerY) + ")");
            clickAt(centerX, centerY, button);
            return;
        }
    }
    
    logError("Input", "Object '" + objectLabel + "' not found on screen");
}

} // namespace Perception
} // namespace GRIM
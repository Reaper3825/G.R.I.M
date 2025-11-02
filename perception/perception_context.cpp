#include "perception_context.hpp"
#include "perception.hpp"
#include "multi_monitor.hpp" // ✅ Multi-monitor support
#include "logger.hpp"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <ctime>
#include <regex> // ✅ For monitor number parsing

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include <opencv2/opencv.hpp>

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
        
        // Step 5: OCR if enabled
        if (pImpl->featureOCR) {
            performOCR(ctx);
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
        
        LOG_DEBUG("PerceptionContext", "Context analysis complete: " + ctx.toSummary());
        
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
    } else if (monitorIndex >= 0 && monitorIndex < monitorCount) {
        // Capture specific monitor
        if (g_multiMonitor) {
            screenshot = g_multiMonitor->captureMonitor(monitorIndex);
            ctx.monitorIndex = monitorIndex;
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
    
    LOG_DEBUG("PerceptionContext", "Captured monitor " + std::to_string(ctx.monitorIndex) + 
              " (" + std::to_string(ctx.screenWidth) + "x" + std::to_string(ctx.screenHeight) + ")");
    
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
    
    LOG_DEBUG("PerceptionContext", "Active window: \"" + ctx.activeWindowTitle + 
              "\" (" + ctx.activeProcessName + ")");
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
        LOG_DEBUG("PerceptionContext", "Object detection result: " + detectionResult);
        
        // TODO: Parse detectionResult and populate ctx.detectedObjects
        // For now, just log that detection was attempted
    }
}

void PerceptionContextManager::analyzeWithVisionAI(VisualContext& ctx) {
    if (ctx.screenshot.empty()) {
        return;
    }
    
    // TODO: Integrate with multimodal vision AI
    // This will be implemented in the next phase
    // For now, placeholder
    LOG_DEBUG("PerceptionContext", "Vision AI analysis not yet implemented");
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
        
        LOG_DEBUG("PerceptionContext", "Visual characteristics - Brightness: " + 
                  std::to_string(ctx.brightnessMean) + ", Contrast: " + 
                  std::to_string(ctx.contrastScore) + ", TextDensity: " +
                  std::to_string(ctx.textDensity));
                  
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
        
        LOG_DEBUG("PerceptionContext", "Change score: " + std::to_string(changeScore));
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
        ctx = getMonitorContext(requestedMonitor, false);
    } else {
        ctx = getCurrentContext(false);
    }
    
    if (!ctx.isValid) {
        return "I cannot see your screen right now. " + ctx.errorMessage;
    }
    
    std::string lowerQClean = question;
    std::transform(lowerQClean.begin(), lowerQClean.end(), lowerQClean.begin(), ::tolower);
    
    std::ostringstream response;
    
    // General "what's on my screen" questions
    if (lowerQClean.find("what") != std::string::npos && 
        (lowerQClean.find("screen") != std::string::npos || lowerQClean.find("see") != std::string::npos ||
         lowerQClean.find("monitor") != std::string::npos)) {
        
        response << "I can see ";
        
        // ✅ Indicate which monitor if multi-monitor
        if (ctx.isMultiMonitor && ctx.monitorIndex >= 0) {
            response << "monitor " << (ctx.monitorIndex + 1) << " (of " << ctx.totalMonitors << "):\n\n";
        } else {
            response << "your screen:\n\n";
        }
        
        response << "You're viewing: " << sceneTypeToString(ctx.sceneType) << "\n";
        
        if (!ctx.activeWindowTitle.empty()) {
            response << "Active window: \"" << ctx.activeWindowTitle << "\"\n";
        }
        
        if (ctx.isDarkTheme) {
            response << "Theme: Dark mode\n";
        }
        
        if (ctx.hasText && !ctx.screenText.empty()) {
            response << "\nVisible text:\n";
            // Show first 500 characters of text
            std::string preview = ctx.screenText.substr(0, std::min<size_t>(500, ctx.screenText.length()));
            response << preview;
            if (ctx.screenText.length() > 500) {
                response << "\n... (and more)";
            }
        }
        
        if (!ctx.detectedObjects.empty()) {
            response << "\n\nDetected objects:\n";
            for (const auto& obj : ctx.detectedObjects) {
                response << "  - " << obj.label << " (" << (int)(obj.confidence * 100) << "%)\n";
            }
        }
        
        if (ctx.hasAIAnalysis && !ctx.aiDescription.empty()) {
            response << "\n" << ctx.aiDescription;
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
            return response.str();
        } else {
            return "Object detection didn't identify specific objects on your screen.";
        }
    }
    
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

} // namespace Perception
} // namespace GRIM

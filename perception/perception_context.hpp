#pragma once

// Unified perception context management for GRIM
// Provides intelligent caching, multi-monitor support, and continuous awareness
//
// USAGE EXAMPLES:
//
// 1. Basic screen analysis:
//    auto ctx = g_contextManager->getCurrentContext();
//    std::cout << ctx.toSummary() << "\n";
//
// 2. Multi-monitor specific query:
//    std::string answer = g_contextManager->answerVisionQuestion("what's on monitor 2?");
//
// 3. Continuous capture (background awareness):
//    ContinuousCaptureConfig config;
//    config.frameSkip = 30;           // Every 30 frames
//    config.changeThreshold = 0.05f;  // 5% change detection
//    g_contextManager->startContinuousCapture(config);
//    // ... later ...
//    auto latestCtx = g_contextManager->getLatestContext();
//    g_contextManager->stopContinuousCapture();

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <atomic>
#include <opencv2/core.hpp>

namespace GRIM {
namespace Perception {


struct ContinuousCaptureConfig {
    bool enabled = false;
    int frameSkip = 120;              // Capture every 60th frame (approx 2/sec at 30fps)
    int captureIntervalMs = 1000;    // Or time-based: capture every 1000ms
    bool useFrameSkip = true;        // true = frame-based, false = time-based
    bool captureAllMonitors = false; // Capture all monitors or just active
    int maxCacheAge = 10000;         // Invalidate cache after 10 seconds
    float changeThreshold = 0.25f;   // 5% change triggers new analysis
    bool useVisionAI = false;        // ✅ Vision AI is too slow for continuous capture (disabled by default)
};

// Represents the current visual context of what the user is looking at
struct VisualContext {
    // Screen capture metadata
    int screenWidth = 0;
    int screenHeight = 0;
    std::chrono::system_clock::time_point captureTime;
    
    // ✅ Multi-monitor support
    int monitorIndex = -1;      // -1 = all monitors, 0+ = specific monitor
    int totalMonitors = 1;
    bool isMultiMonitor = false;
    
    // Active window info
    std::string activeWindowTitle;
    std::string activeProcessName;
    int activeWindowX = 0;
    int activeWindowY = 0;
    int activeWindowWidth = 0;
    int activeWindowHeight = 0;
    
    // OCR results
    std::string screenText;
    float ocrConfidence = 0.0f;
    bool hasText = false;
    
    // Object detection results
    struct DetectedObject {
        std::string label;
        float confidence;
        int x, y, width, height;
    };
    std::vector<DetectedObject> detectedObjects;
    
    // Vision AI interpretation
    std::string aiDescription;
    float aiConfidence = 0.0f;
    bool hasAIAnalysis = false;
    
    // ✅ Vision AI detailed results
    std::string visionAIDescription;
    float visionAIConfidence = 0.0f;
    std::string visionModelUsed;
    
    // Scene analysis
    enum class SceneType {
        Unknown,
        Desktop,
        WebBrowser,
        IDE_Code,
        Terminal,
        Document,
        Image_Video,
        Game,
        Chat_Messaging
    };
    SceneType sceneType = SceneType::Unknown;
    
    // Visual characteristics
    float brightnessMean = 0.0f;
    float contrastScore = 0.0f;
    float textDensity = 0.0f;
    bool isDarkTheme = false;
    
    // Change detection
    float changeScore = 1.0f; // 0.0 = no change, 1.0 = completely different
    
    // Raw screenshot (optional - can be large)
    cv::Mat screenshot; // Empty by default to save memory
    
    // Validity
    bool isValid = false;
    std::string errorMessage;
    
    // Convert to human-readable summary
    std::string toSummary() const;
    
    // Convert to detailed JSON-like string
    std::string toDetailedString() const;
};

// Manages perception context with intelligent caching and updates
class PerceptionContextManager {
public:
    PerceptionContextManager();
    ~PerceptionContextManager();
    
    // Initialize the context manager
    bool init();
    
    // Shutdown and cleanup
    void shutdown();
    
    // Get current visual context (may return cached if recent enough)
    VisualContext getCurrentContext(bool forceRefresh = false);
    
    // ✅ Get context for specific monitor
    VisualContext getMonitorContext(int monitorIndex, bool forceRefresh = false);
    
    // Answer vision-related questions using the current context1
    std::string answerVisionQuestion(const std::string& question);
    
    // Capture and analyze screen with full detail
    VisualContext captureAndAnalyze(bool includeAI = true, bool saveScreenshot = false);
    
    // ✅ Capture and analyze specific monitor
    VisualContext captureAndAnalyzeMonitor(int monitorIndex, bool includeAI = true, bool saveScreenshot = false);
    
    // Get context change since last capture (0.0 = no change, 1.0 = completely different)
    float getChangeScore();
    
    // Enable/disable different perception features
    void setFeatureEnabled(const std::string& feature, bool enabled);
    
    // ✅ NEW: Continuous capture control
    void startContinuousCapture(const ContinuousCaptureConfig& config);
    void stopContinuousCapture();
    bool isContinuousCaptureRunning() const;
    void updateCaptureConfig(const ContinuousCaptureConfig& config);
    
    // ✅ Get latest captured context (from continuous capture thread)
    VisualContext getLatestContext() const;
    
    // ✅ NEW: Memory integration - store visual context for recall
    void storeContextInMemory(const VisualContext& ctx, const std::string& description = "");
    
    // ✅ NEW: Window/Application awareness across monitors
    struct AppLocation {
        std::string appName;
        std::string windowTitle;
        int monitorIndex;
        std::string monitorDescription; // "Monitor 1", "left monitor", etc.
        bool isActive;
    };
    std::vector<AppLocation> findApplication(const std::string& appName);
    std::string describeApplicationLocations();
    
    // Get status of perception systems
    struct PerceptionStatus {
        bool screenCaptureAvailable;
        bool ocrAvailable;
        bool objectDetectionAvailable;
        bool visionAIAvailable;
        bool windowTrackingAvailable;
        std::string ocrEngine;
        std::string objectDetectionModel;
        std::string visionAIModel;
    };
    PerceptionStatus getStatus();
    
    // =========================================================
    // INPUT CONTROL - Simulate keyboard/mouse for action execution
    // =========================================================
    
    // Mouse control
    void moveMouseTo(int x, int y);
    void clickMouse(const std::string& button = "left"); // "left", "right", "middle"
    void doubleClickMouse(const std::string& button = "left");
    void scrollMouse(int delta);
    
    // Keyboard control
    void typeText(const std::string& text, int delayMs = 10);
    void pressKey(const std::string& key);
    void releaseKey(const std::string& key);
    void tapKey(const std::string& key);
    void pressKeyCombo(const std::vector<std::string>& keys); // e.g., {"ctrl", "c"}
    
    // High-level actions (perception + input)
    void clickAt(int x, int y, const std::string& button = "left");
    void clickOnText(const std::string& text, const std::string& button = "left");
    void clickOnObject(const std::string& objectLabel, const std::string& button = "left");

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Internal methods
    VisualContext captureScreen(int monitorIndex = -1); // -1 = active monitor
    void analyzeWindowContext(VisualContext& ctx);
    void performOCR(VisualContext& ctx);
    void performOCREnhanced(VisualContext& ctx); // ✅ NEW: Better OCR with preprocessing
    void detectObjects(VisualContext& ctx);
    void analyzeWithVisionAI(VisualContext& ctx);
    void classifyScene(VisualContext& ctx);
    void computeVisualCharacteristics(VisualContext& ctx);
    float computeChangeScore(const cv::Mat& current, const cv::Mat& previous);
    
    // ✅ NEW: Continuous capture thread
    void continuousCaptureThread();
    std::atomic<bool> m_captureThreadRunning{false};
    std::unique_ptr<std::thread> m_captureThread;
};

// Global context manager instance
extern std::unique_ptr<PerceptionContextManager> g_contextManager;

// Initialize global context manager
void initContextManager();

// Get current visual context
VisualContext getCurrentVisualContext(bool forceRefresh = false);

// Answer vision questions using current context
std::string answerVisionQuestionWithContext(const std::string& question);

} // namespace Perception
} // namespace GRIM

#pragma once
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <opencv2/core.hpp>

namespace GRIM {
namespace Perception {

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
    float changeScore = 0.0f; // 0.0 = no change, 1.0 = completely different
    
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
    
    // Answer vision-related questions using the current context
    std::string answerVisionQuestion(const std::string& question);
    
    // Capture and analyze screen with full detail
    VisualContext captureAndAnalyze(bool includeAI = true, bool saveScreenshot = false);
    
    // ✅ Capture and analyze specific monitor
    VisualContext captureAndAnalyzeMonitor(int monitorIndex, bool includeAI = true, bool saveScreenshot = false);
    
    // Get context change since last capture (0.0 = no change, 1.0 = completely different)
    float getChangeScore();
    
    // Enable/disable different perception features
    void setFeatureEnabled(const std::string& feature, bool enabled);
    
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
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Internal methods
    VisualContext captureScreen(int monitorIndex = -1); // -1 = active monitor
    void analyzeWindowContext(VisualContext& ctx);
    void performOCR(VisualContext& ctx);
    void detectObjects(VisualContext& ctx);
    void analyzeWithVisionAI(VisualContext& ctx);
    void classifyScene(VisualContext& ctx);
    void computeVisualCharacteristics(VisualContext& ctx);
    float computeChangeScore(const cv::Mat& current, const cv::Mat& previous);
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

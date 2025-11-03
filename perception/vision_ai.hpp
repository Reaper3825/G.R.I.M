#pragma once
#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace GRIM {
namespace Perception {

// Vision AI backend types
enum class VisionAIBackend {
    None,
    ONNX_Vision,       // ✅ Fast ONNX quantized vision model (recommended)
    Ollama_LLaVA,      // Local LLaVA via Ollama (slow)
    Ollama_BakLLaVA,   // BakLLaVA variant (slow)
    Ollama_LLaVA_Phi,  // Microsoft Phi-3 Vision (slow)
    OpenAI_GPT4Vision, // OpenAI GPT-4 Vision API
    Azure_GPT4Vision,  // Azure OpenAI GPT-4 Vision
    GitHub_Models      // GitHub Models API
};

// Vision analysis request
struct VisionAnalysisRequest {
    cv::Mat image;
    std::string prompt;
    VisionAIBackend backend = VisionAIBackend::Ollama_LLaVA;
    std::string modelName; // Optional override
    float temperature = 0.7f;
    int maxTokens = 500;
    bool includeDetailedDescription = true;
};

// Vision analysis result
struct VisionAnalysisResult {
    bool success = false;
    std::string description;
    float confidence = 0.0f;
    std::string modelUsed;
    VisionAIBackend backend = VisionAIBackend::None;
    std::string errorMessage;
    
    // Structured extraction (optional)
    std::vector<std::string> detectedElements;
    std::vector<std::string> detectedText;
    std::string dominantActivity;
    std::string contextType; // e.g., "coding", "browsing", "gaming"
};

// Vision AI Manager
class VisionAIManager {
public:
    VisionAIManager();
    ~VisionAIManager();
    
    // Initialize with preferred backend
    bool init(VisionAIBackend preferredBackend = VisionAIBackend::Ollama_LLaVA);
    
    // Shutdown
    void shutdown();
    
    // Check if vision AI is available
    bool isAvailable() const;
    
    // Get available backends
    std::vector<VisionAIBackend> getAvailableBackends();
    
    // Analyze image with vision AI
    VisionAnalysisResult analyzeImage(const VisionAnalysisRequest& request);
    
    // Simplified interface: analyze with default prompt
    VisionAnalysisResult describeScreen(const cv::Mat& screenshot);
    
    // Answer specific question about image
    VisionAnalysisResult answerAboutImage(const cv::Mat& image, const std::string& question);
    
    // Set default backend
    void setDefaultBackend(VisionAIBackend backend);
    
    // Get current backend
    VisionAIBackend getCurrentBackend() const;
    
    // ✅ Warm up the model on startup (async, non-blocking)
    void warmupModel();
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    
    // Backend-specific implementations
    VisionAnalysisResult analyzeWithONNX(const VisionAnalysisRequest& request);
    VisionAnalysisResult analyzeWithOllama(const VisionAnalysisRequest& request);
    VisionAnalysisResult analyzeWithOpenAI(const VisionAnalysisRequest& request);
    VisionAnalysisResult analyzeWithGitHub(const VisionAnalysisRequest& request);
    VisionAnalysisResult analyzeWithHybridVision(const VisionAnalysisRequest& request);
    VisionAnalysisResult analyzeWithFastVision(const VisionAnalysisRequest& request);
    
    // Helper: encode image to base64
    std::string encodeImageToBase64(const cv::Mat& image);
    
    // Helper: check if backend is available
    bool isBackendAvailable(VisionAIBackend backend);
};

// Global vision AI manager
extern std::unique_ptr<VisionAIManager> g_visionAI;

// Initialize global vision AI
void initVisionAI(VisionAIBackend backend = VisionAIBackend::Ollama_LLaVA);

// Get vision AI description of current screen
VisionAnalysisResult describeCurrentScreen();

} // namespace Perception
} // namespace GRIM

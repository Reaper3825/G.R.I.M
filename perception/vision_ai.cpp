#include "vision_ai.hpp"
#include "logger.hpp"
#include "ai.hpp" // For existing AI infrastructure
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <wininet.h>
#pragma comment(lib, "wininet.lib")
#endif

namespace GRIM {
namespace Perception {

std::unique_ptr<VisionAIManager> g_visionAI = nullptr;

// Convert backend enum to string
static std::string backendToString(VisionAIBackend backend) {
    switch (backend) {
        case VisionAIBackend::Ollama_LLaVA: return "Ollama (LLaVA)";
        case VisionAIBackend::Ollama_BakLLaVA: return "Ollama (BakLLaVA)";
        case VisionAIBackend::Ollama_LLaVA_Phi: return "Ollama (LLaVA-Phi3)";
        case VisionAIBackend::OpenAI_GPT4Vision: return "OpenAI GPT-4 Vision";
        case VisionAIBackend::Azure_GPT4Vision: return "Azure GPT-4 Vision";
        case VisionAIBackend::GitHub_Models: return "GitHub Models";
        default: return "None";
    }
}

struct VisionAIManager::Impl {
    bool initialized = false;
    VisionAIBackend defaultBackend = VisionAIBackend::Ollama_LLaVA;
    std::vector<VisionAIBackend> availableBackends;
    
    // Configuration
    std::string ollamaBaseURL = "http://localhost:11434";
    std::string openaiAPIKey;
    std::string azureEndpoint;
    std::string githubToken;
};

VisionAIManager::VisionAIManager()
    : pImpl(std::make_unique<Impl>()) {
}

VisionAIManager::~VisionAIManager() {
    shutdown();
}

bool VisionAIManager::init(VisionAIBackend preferredBackend) {
    if (pImpl->initialized) {
        return true;
    }
    
    LOG_DEBUG("VisionAI", "Initializing Vision AI manager");
    
    pImpl->defaultBackend = preferredBackend;
    
    // Detect available backends
    pImpl->availableBackends.clear();
    
    // Check Ollama backends
    if (isBackendAvailable(VisionAIBackend::Ollama_LLaVA)) {
        pImpl->availableBackends.push_back(VisionAIBackend::Ollama_LLaVA);
    }
    if (isBackendAvailable(VisionAIBackend::Ollama_BakLLaVA)) {
        pImpl->availableBackends.push_back(VisionAIBackend::Ollama_BakLLaVA);
    }
    if (isBackendAvailable(VisionAIBackend::Ollama_LLaVA_Phi)) {
        pImpl->availableBackends.push_back(VisionAIBackend::Ollama_LLaVA_Phi);
    }
    
    // Check API backends
    // TODO: Check for API keys in environment or config
    
    if (pImpl->availableBackends.empty()) {
        LOG_DEBUG("VisionAI", "No vision AI backends available");
        pImpl->initialized = true; // Still mark as initialized, just no backends
        return false;
    }
    
    LOG_DEBUG("VisionAI", "Found " + std::to_string(pImpl->availableBackends.size()) + 
              " available vision AI backends");
    
    pImpl->initialized = true;
    return true;
}

void VisionAIManager::shutdown() {
    if (!pImpl->initialized) {
        return;
    }
    
    LOG_DEBUG("VisionAI", "Shutting down Vision AI manager");
    pImpl->initialized = false;
}

bool VisionAIManager::isAvailable() const {
    return pImpl->initialized && !pImpl->availableBackends.empty();
}

std::vector<VisionAIBackend> VisionAIManager::getAvailableBackends() {
    return pImpl->availableBackends;
}

bool VisionAIManager::isBackendAvailable(VisionAIBackend backend) {
    // For Ollama backends, check if the model is available
    if (backend == VisionAIBackend::Ollama_LLaVA ||
        backend == VisionAIBackend::Ollama_BakLLaVA ||
        backend == VisionAIBackend::Ollama_LLaVA_Phi) {
        
        // Try to query Ollama for available models
        // For now, assume Ollama is available if we can reach it
        // TODO: Implement actual HTTP check
        return true; // Optimistic - will fail gracefully if not available
    }
    
    // For API backends, check for credentials
    if (backend == VisionAIBackend::OpenAI_GPT4Vision) {
        return !pImpl->openaiAPIKey.empty();
    }
    
    if (backend == VisionAIBackend::Azure_GPT4Vision) {
        return !pImpl->azureEndpoint.empty();
    }
    
    if (backend == VisionAIBackend::GitHub_Models) {
        return !pImpl->githubToken.empty();
    }
    
    return false;
}

std::string VisionAIManager::encodeImageToBase64(const cv::Mat& image) {
    if (image.empty()) {
        return "";
    }
    
    try {
        // Encode image to PNG format
        std::vector<unsigned char> buffer;
        std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 9};
        
        cv::imencode(".png", image, buffer, params);
        
        // Convert to base64
        static const char* base64_chars = 
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789+/";
        
        std::string ret;
        int i = 0;
        int j = 0;
        unsigned char char_array_3[3];
        unsigned char char_array_4[4];
        
        size_t bufferSize = buffer.size();
        const unsigned char* bytes_to_encode = buffer.data();
        
        while (bufferSize--) {
            char_array_3[i++] = *(bytes_to_encode++);
            if (i == 3) {
                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;
                
                for (i = 0; i < 4; i++)
                    ret += base64_chars[char_array_4[i]];
                i = 0;
            }
        }
        
        if (i) {
            for (j = i; j < 3; j++)
                char_array_3[j] = '\0';
            
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            
            for (j = 0; j < i + 1; j++)
                ret += base64_chars[char_array_4[j]];
            
            while (i++ < 3)
                ret += '=';
        }
        
        return ret;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("VisionAI", "Failed to encode image to base64: " + std::string(e.what()));
        return "";
    }
}

VisionAnalysisResult VisionAIManager::analyzeImage(const VisionAnalysisRequest& request) {
    VisionAnalysisResult result;
    
    if (!pImpl->initialized) {
        result.success = false;
        result.errorMessage = "Vision AI not initialized";
        return result;
    }
    
    if (request.image.empty()) {
        result.success = false;
        result.errorMessage = "Empty image provided";
        return result;
    }
    
    // Determine which backend to use
    VisionAIBackend backend = request.backend;
    if (backend == VisionAIBackend::None) {
        backend = pImpl->defaultBackend;
    }
    
    // Check if backend is available
    if (std::find(pImpl->availableBackends.begin(), pImpl->availableBackends.end(), backend) 
        == pImpl->availableBackends.end()) {
        
        // Try to fall back to first available backend
        if (!pImpl->availableBackends.empty()) {
            backend = pImpl->availableBackends[0];
            LOG_DEBUG("VisionAI", "Requested backend not available, using " + 
                      backendToString(backend));
        } else {
            result.success = false;
            result.errorMessage = "No vision AI backends available";
            return result;
        }
    }
    
    // Route to appropriate backend
    result.backend = backend;
    
    switch (backend) {
        case VisionAIBackend::Ollama_LLaVA:
        case VisionAIBackend::Ollama_BakLLaVA:
        case VisionAIBackend::Ollama_LLaVA_Phi:
            return analyzeWithOllama(request);
        
        case VisionAIBackend::OpenAI_GPT4Vision:
        case VisionAIBackend::Azure_GPT4Vision:
            return analyzeWithOpenAI(request);
        
        case VisionAIBackend::GitHub_Models:
            return analyzeWithGitHub(request);
        
        default:
            result.success = false;
            result.errorMessage = "Unsupported backend";
            return result;
    }
}

VisionAnalysisResult VisionAIManager::analyzeWithOllama(const VisionAnalysisRequest& request) {
    VisionAnalysisResult result;
    result.backend = request.backend;
    
    try {
        // Determine model name
        std::string modelName = request.modelName;
        if (modelName.empty()) {
            switch (request.backend) {
                case VisionAIBackend::Ollama_LLaVA:
                    modelName = "llava";
                    break;
                case VisionAIBackend::Ollama_BakLLaVA:
                    modelName = "bakllava";
                    break;
                case VisionAIBackend::Ollama_LLaVA_Phi:
                    modelName = "llava-phi3";
                    break;
                default:
                    modelName = "llava";
            }
        }
        result.modelUsed = modelName;
        
        // Encode image to base64
        std::string base64Image = encodeImageToBase64(request.image);
        if (base64Image.empty()) {
            result.success = false;
            result.errorMessage = "Failed to encode image";
            return result;
        }
        
        // Build JSON request for Ollama
        std::ostringstream jsonRequest;
        jsonRequest << "{\n";
        jsonRequest << "  \"model\": \"" << modelName << "\",\n";
        jsonRequest << "  \"prompt\": \"" << request.prompt << "\",\n";
        jsonRequest << "  \"images\": [\"" << base64Image << "\"],\n";
        jsonRequest << "  \"stream\": false,\n";
        jsonRequest << "  \"options\": {\n";
        jsonRequest << "    \"temperature\": " << request.temperature << ",\n";
        jsonRequest << "    \"num_predict\": " << request.maxTokens << "\n";
        jsonRequest << "  }\n";
        jsonRequest << "}";
        
        std::string requestBody = jsonRequest.str();
        
        // Make HTTP POST request to Ollama
        // For now, we'll use a simplified approach - in production, use proper HTTP library
        LOG_DEBUG("VisionAI", "Sending vision request to Ollama (model: " + modelName + ")");
        
        // TODO: Implement actual HTTP request
        // For now, return a placeholder
        result.success = false;
        result.errorMessage = "Ollama vision integration not yet fully implemented - HTTP client needed";
        
        // TEMPORARY: If you have the existing AI system that can call Ollama,
        // we could potentially use that with a modified approach
        
        return result;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
        LOG_ERROR("VisionAI", result.errorMessage);
        return result;
    }
}

VisionAnalysisResult VisionAIManager::analyzeWithOpenAI(const VisionAnalysisRequest& request) {
    VisionAnalysisResult result;
    result.success = false;
    result.errorMessage = "OpenAI/Azure vision integration not yet implemented";
    return result;
}

VisionAnalysisResult VisionAIManager::analyzeWithGitHub(const VisionAnalysisRequest& request) {
    VisionAnalysisResult result;
    result.success = false;
    result.errorMessage = "GitHub Models vision integration not yet implemented";
    return result;
}

VisionAnalysisResult VisionAIManager::describeScreen(const cv::Mat& screenshot) {
    VisionAnalysisRequest request;
    request.image = screenshot;
    request.prompt = "Describe what you see on this screen in detail. "
                     "Identify the main application or activity, visible UI elements, "
                     "any text you can read, and what the user appears to be doing.";
    request.includeDetailedDescription = true;
    
    return analyzeImage(request);
}

VisionAnalysisResult VisionAIManager::answerAboutImage(const cv::Mat& image, 
                                                         const std::string& question) {
    VisionAnalysisRequest request;
    request.image = image;
    request.prompt = question;
    
    return analyzeImage(request);
}

void VisionAIManager::setDefaultBackend(VisionAIBackend backend) {
    pImpl->defaultBackend = backend;
    LOG_DEBUG("VisionAI", "Default backend set to: " + backendToString(backend));
}

VisionAIBackend VisionAIManager::getCurrentBackend() const {
    return pImpl->defaultBackend;
}

// Global initialization
void initVisionAI(VisionAIBackend backend) {
    if (!g_visionAI) {
        g_visionAI = std::make_unique<VisionAIManager>();
        g_visionAI->init(backend);
    }
}

VisionAnalysisResult describeCurrentScreen() {
    if (!g_visionAI) {
        initVisionAI();
    }
    
    // Get current screen capture
    // This would integrate with perception_context.hpp
    VisionAnalysisResult result;
    result.success = false;
    result.errorMessage = "Not yet integrated with screen capture";
    return result;
}

} // namespace Perception
} // namespace GRIM

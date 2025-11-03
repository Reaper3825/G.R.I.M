#include "vision_ai.hpp"
#include "logger.hpp"
#include "ai.hpp" // For existing AI infrastructure and aiConfig
#include <opencv2/opencv.hpp>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

// ONNX Runtime for fast vision inference
#include <onnxruntime_cxx_api.h>

// Hybrid ONNX + Ollama Vision System
#include "vision/hybrid_vision.hpp"

// Fast Vision Interpreter (ONNX + heuristics, no Ollama)
#include "vision/fast_vision_interpreter.hpp"

// Access to global aiConfig from ai.hpp
extern nlohmann::json aiConfig;

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
        case VisionAIBackend::ONNX_Vision: return "ONNX Runtime (Local)";
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
    
    // Configuration - loaded from aiConfig
    std::string ollamaBaseURL;
    std::string visionModel;
    std::string openaiAPIKey;
    std::string azureEndpoint;
    std::string githubToken;
    
    // ONNX Runtime components
    std::unique_ptr<Ort::Env> onnxEnv;
    std::unique_ptr<Ort::Session> onnxSession;
    std::unique_ptr<Ort::SessionOptions> onnxSessionOptions;
    std::string onnxModelPath;
    bool onnxInitialized = false;
    
    // Hybrid Vision System (ONNX + Ollama)
    std::unique_ptr<grim::vision::HybridVisionSystem> hybridVision;
    bool useHybridVision = false;  // Slow, disabled by default
    
    // Fast Vision Interpreter (ONNX only, no Ollama)
    bool useFastVision = true;  // Fast, enabled by default
    
    void loadConfig() {
        // Load from aiConfig.json
        ollamaBaseURL = aiConfig.value("ollama_url", "http://127.0.0.1:11434");
        visionModel = aiConfig.value("vision_model", "llama3.2-vision:11b");
        
        // ✅ ONNX vision encoder model (llama3.2-vision converted from GGUF)
        onnxModelPath = aiConfig.value("onnx_vision_model", 
            "D:/G.R.I.M/data/models/vision/llama3.2-vision-onnx/vision_encoder.onnx");
        
        if (aiConfig.contains("api_keys")) {
            openaiAPIKey = aiConfig["api_keys"].value("openai", "");
            azureEndpoint = aiConfig["api_keys"].value("azure", "");
        }
    }
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
    
    // Load configuration from aiConfig
    pImpl->loadConfig();
    
    LOG_DEBUG("VisionAI", "Using Ollama URL: " + pImpl->ollamaBaseURL);
    LOG_DEBUG("VisionAI", "Vision model: " + pImpl->visionModel);
    
    pImpl->defaultBackend = preferredBackend;
    
    // Detect available backends
    pImpl->availableBackends.clear();
    
    // Check ONNX backend first (recommended - fast!)
    if (isBackendAvailable(VisionAIBackend::ONNX_Vision)) {
        pImpl->availableBackends.push_back(VisionAIBackend::ONNX_Vision);
        LOG_DEBUG("VisionAI", "✅ ONNX Runtime vision backend available");
    }
    
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
    
    // Initialize Fast Vision System (ONNX + heuristics, no slow Ollama calls)
    if (pImpl->useFastVision) {
        try {
            bool success = grim::vision::FastVisionManager::getInstance().initialize(pImpl->onnxModelPath);
            if (success) {
                LOG_DEBUG("VisionAI", "✅ Fast Vision System initialized (<500ms responses)");
            } else {
                LOG_ERROR("VisionAI", "Failed to initialize Fast Vision System");
                pImpl->useFastVision = false;
            }
        } catch (const std::exception& e) {
            LOG_ERROR("VisionAI", "Fast Vision initialization error: " + std::string(e.what()));
            pImpl->useFastVision = false;
        }
    }
    
    // Initialize Hybrid Vision System (ONNX + Ollama) - optional, slow
    if (pImpl->useHybridVision) {
        try {
            grim::vision::VisionConfig visionConfig;
            visionConfig.onnx_model_path = pImpl->onnxModelPath;
            visionConfig.ollama_model = pImpl->visionModel;
            visionConfig.ollama_url = pImpl->ollamaBaseURL;
            visionConfig.use_onnx_preprocessing = true;
            visionConfig.use_gpu = true;  // Use RTX 3080Ti acceleration
            
            pImpl->hybridVision = std::make_unique<grim::vision::HybridVisionSystem>(visionConfig);
            
            LOG_DEBUG("VisionAI", "✅ Hybrid Vision System initialized (slow, for detailed analysis)");
        } catch (const std::exception& e) {
            LOG_ERROR("VisionAI", "Failed to initialize Hybrid Vision System: " + std::string(e.what()));
            pImpl->useHybridVision = false;
        }
    }
    
    pImpl->initialized = true;
    
    // ✅ Warm up the model in background (async, non-blocking)
    if (preferredBackend == VisionAIBackend::Ollama_LLaVA || 
        preferredBackend == VisionAIBackend::Ollama_BakLLaVA ||
        preferredBackend == VisionAIBackend::Ollama_LLaVA_Phi) {
        warmupModel();
    }
    
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
    // Check ONNX backend - verify model file exists
    if (backend == VisionAIBackend::ONNX_Vision) {
        std::ifstream modelFile(pImpl->onnxModelPath);
        if (modelFile.good()) {
            modelFile.close();
            
            // Try to initialize ONNX Runtime if not already done
            if (!pImpl->onnxInitialized) {
                try {
                    LOG_DEBUG("VisionAI", "Initializing ONNX Runtime...");
                    
                    // Create ONNX environment
                    pImpl->onnxEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "GRIM_Vision");
                    
                    // Create session options
                    pImpl->onnxSessionOptions = std::make_unique<Ort::SessionOptions>();
                    pImpl->onnxSessionOptions->SetIntraOpNumThreads(4);
                    pImpl->onnxSessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                    
                    // Try CUDA first, fallback to CPU
                    try {
                        OrtCUDAProviderOptions cuda_options{};
                        cuda_options.device_id = 0;
                        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
                        cuda_options.gpu_mem_limit = SIZE_MAX;
                        cuda_options.arena_extend_strategy = 0;
                        cuda_options.do_copy_in_default_stream = 1;
                        
                        pImpl->onnxSessionOptions->AppendExecutionProvider_CUDA(cuda_options);
                        LOG_DEBUG("VisionAI", "Using CUDA execution provider (RTX 3080Ti)");
                    } catch (...) {
                        LOG_DEBUG("VisionAI", "CUDA not available, using CPU execution provider");
                    }
                    
                    // Create session
#ifdef _WIN32
                    std::wstring wModelPath(pImpl->onnxModelPath.begin(), pImpl->onnxModelPath.end());
                    pImpl->onnxSession = std::make_unique<Ort::Session>(
                        *pImpl->onnxEnv, wModelPath.c_str(), *pImpl->onnxSessionOptions);
#else
                    pImpl->onnxSession = std::make_unique<Ort::Session>(
                        *pImpl->onnxEnv, pImpl->onnxModelPath.c_str(), *pImpl->onnxSessionOptions);
#endif
                    
                    pImpl->onnxInitialized = true;
                    LOG_DEBUG("VisionAI", "✅ ONNX Runtime initialized successfully");
                    LOG_DEBUG("VisionAI", "Model: " + pImpl->onnxModelPath);
                    
                    return true;
                } catch (const Ort::Exception& e) {
                    LOG_ERROR("VisionAI", "Failed to initialize ONNX Runtime: " + std::string(e.what()));
                    pImpl->onnxInitialized = false;
                    return false;
                }
            }
            return pImpl->onnxInitialized;
        }
        return false;
    }
    
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
    
    // Use fast vision system for near-instant results (<500ms)
    if (pImpl->useFastVision && grim::vision::FastVisionManager::getInstance().isReady()) {
        LOG_DEBUG("VisionAI", "Using Fast Vision System (ONNX + heuristics)");
        return analyzeWithFastVision(request);
    }
    
    // Fallback to hybrid system (slow, 20-60 seconds)
    if (pImpl->useHybridVision && pImpl->hybridVision) {
        LOG_DEBUG("VisionAI", "Using Hybrid Vision System (ONNX + Ollama - SLOW)");
        return analyzeWithHybridVision(request);
    }
    
    // No vision systems available
    result.success = false;
    result.errorMessage = "No vision systems available";
    LOG_ERROR("VisionAI", result.errorMessage);
    return result;
}

VisionAnalysisResult VisionAIManager::analyzeWithOllama(const VisionAnalysisRequest& request) {
    VisionAnalysisResult result;
    result.backend = request.backend;
    
    try {
        // Determine model name - use config or fallback
        std::string modelName = request.modelName;
        if (modelName.empty()) {
            // Use configured vision model from aiConfig
            modelName = pImpl->visionModel;
            
            // If still empty or user wants specific backend variant, override
            if (request.backend == VisionAIBackend::Ollama_BakLLaVA) {
                modelName = "bakllava";
            } else if (request.backend == VisionAIBackend::Ollama_LLaVA_Phi) {
                modelName = "llava-phi3";
            }
        }
        result.modelUsed = modelName;
        
        // ✅ Resize large images to speed up vision processing
        cv::Mat imageToEncode = request.image;
        if (request.image.cols > 1280 || request.image.rows > 720) {
            // Resize to max 1280x720 (720p) - sufficient for screen analysis
            double scaleW = 1280.0 / request.image.cols;
            double scaleH = 720.0 / request.image.rows;
            double scale = (scaleW < scaleH) ? scaleW : scaleH;
            int newWidth = static_cast<int>(request.image.cols * scale);
            int newHeight = static_cast<int>(request.image.rows * scale);
            
            cv::resize(request.image, imageToEncode, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
            LOG_DEBUG("VisionAI", "Resized image from " + std::to_string(request.image.cols) + "x" + 
                      std::to_string(request.image.rows) + " to " + 
                      std::to_string(newWidth) + "x" + std::to_string(newHeight) + " for faster processing");
        }
        
        // Encode image to base64
        std::string base64Image = encodeImageToBase64(imageToEncode);
        if (base64Image.empty()) {
            result.success = false;
            result.errorMessage = "Failed to encode image";
            return result;
        }
        
        LOG_DEBUG("VisionAI", "Encoded image to base64 (" + 
                  std::to_string(base64Image.length()) + " chars)");
        
        // Build JSON request for Ollama vision API
        nlohmann::json jsonRequest = {
            {"model", modelName},
            {"prompt", request.prompt},
            {"images", nlohmann::json::array({base64Image})},
            {"stream", false},
            {"options", {
                {"temperature", request.temperature},
                {"num_predict", request.maxTokens}
            }}
        };
        
        std::string requestBody = jsonRequest.dump();
        
        // Make HTTP POST request to Ollama
        LOG_DEBUG("VisionAI", "Sending vision request to Ollama (model: " + modelName + ")");
        LOG_DEBUG("VisionAI", "Request size: " + std::to_string(requestBody.length()) + " bytes");
        LOG_DEBUG("VisionAI", "Waiting for response (this may take 10-30 seconds for vision models)...");
        
        auto startTime = std::chrono::steady_clock::now();
        
        auto resp = cpr::Post(
            cpr::Url{pImpl->ollamaBaseURL + "/api/generate"},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{requestBody},
            cpr::Timeout{120000} // 2 minute timeout for vision models
        );
        
        auto endTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        LOG_DEBUG("VisionAI", "Ollama response received after " + std::to_string(duration) + "ms");
        
        if (resp.status_code == 200) {
            LOG_DEBUG("VisionAI", "Ollama vision response received (" + 
                     std::to_string(resp.text.size()) + " bytes)");
            
            auto j = nlohmann::json::parse(resp.text, nullptr, false);
            
            if (j.is_discarded()) {
                result.success = false;
                result.errorMessage = "Failed to parse Ollama JSON response";
                LOG_ERROR("VisionAI", result.errorMessage + ": " + resp.text.substr(0, 500));
                return result;
            }
            
            std::string response = j.value("response", "");
            if (response.empty()) {
                result.success = false;
                result.errorMessage = "Ollama returned empty response field";
                LOG_ERROR("VisionAI", result.errorMessage + ". Full JSON: " + resp.text.substr(0, 500));
                return result;
            }
            
            result.success = true;
            result.description = response;
            result.confidence = 0.85f; // Vision models are generally high confidence
            
            // Try to extract structured information if available
            // Ollama might include additional context in the response
            if (j.contains("done") && j["done"].get<bool>()) {
                // Successfully completed
                LOG_DEBUG("VisionAI", "Vision analysis completed successfully");
            }
            
            // Parse for common context types
            std::string lowerDesc = response;
            std::transform(lowerDesc.begin(), lowerDesc.end(), lowerDesc.begin(), ::tolower);
            
            if (lowerDesc.find("code") != std::string::npos || 
                lowerDesc.find("programming") != std::string::npos ||
                lowerDesc.find("ide") != std::string::npos) {
                result.contextType = "coding";
            } else if (lowerDesc.find("browser") != std::string::npos || 
                       lowerDesc.find("web") != std::string::npos) {
                result.contextType = "browsing";
            } else if (lowerDesc.find("game") != std::string::npos || 
                       lowerDesc.find("playing") != std::string::npos) {
                result.contextType = "gaming";
            } else if (lowerDesc.find("document") != std::string::npos || 
                       lowerDesc.find("writing") != std::string::npos) {
                result.contextType = "document";
            } else {
                result.contextType = "general";
            }
            
            return result;
        } else {
            result.success = false;
            result.errorMessage = "Ollama HTTP error: " + std::to_string(resp.status_code);
            if (!resp.text.empty()) {
                result.errorMessage += " - " + resp.text;
            }
            LOG_ERROR("VisionAI", result.errorMessage);
            return result;
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
        LOG_ERROR("VisionAI", result.errorMessage);
        return result;
    }
}

VisionAnalysisResult VisionAIManager::analyzeWithONNX(const VisionAnalysisRequest& request) {
    VisionAnalysisResult result;
    result.backend = VisionAIBackend::ONNX_Vision;
    result.modelUsed = "llama3.2-vision-11b-encoder (ONNX)";
    
    if (!pImpl->onnxInitialized || !pImpl->onnxSession) {
        result.success = false;
        result.errorMessage = "ONNX Runtime not initialized";
        return result;
    }
    
    try {
        auto startTime = std::chrono::steady_clock::now();
        
        LOG_DEBUG("VisionAI", "🚀 Analyzing image with ONNX Runtime (llama3.2-vision encoder)");
        
        // ✅ Preprocess image for llama3.2-vision (560x560)
        // Model expects pixel_values: [1, 3, 560, 560]
        cv::Mat preprocessed;
        cv::resize(request.image, preprocessed, cv::Size(560, 560), 0, 0, cv::INTER_CUBIC);
        
        // Convert BGR to RGB
        cv::cvtColor(preprocessed, preprocessed, cv::COLOR_BGR2RGB);
        
        // Normalize to [0, 1] and apply ImageNet normalization
        preprocessed.convertTo(preprocessed, CV_32FC3, 1.0 / 255.0);
        
        // ImageNet mean and std (standard for vision transformers)
        cv::Scalar mean(0.485f, 0.456f, 0.406f);
        cv::Scalar std(0.229f, 0.224f, 0.225f);
        cv::subtract(preprocessed, mean, preprocessed);
        cv::divide(preprocessed, std, preprocessed);
        
        // Convert to CHW format (ONNX expects channels-first)
        std::vector<cv::Mat> channels(3);
        cv::split(preprocessed, channels);
        
        // ✅ Prepare input tensor [1, 3, 560, 560]
        std::vector<float> inputTensorValues;
        const size_t pixels = 560 * 560;
        inputTensorValues.reserve(3 * pixels);
        
        for (int c = 0; c < 3; ++c) {
            inputTensorValues.insert(inputTensorValues.end(),
                                     (float*)channels[c].data,
                                     (float*)channels[c].data + pixels);
        }
        
        // ✅ Create input tensor for llama3.2-vision encoder
        // Expected: pixel_values [1, 3, 560, 560]
        
        std::array<int64_t, 4> pixelValuesShape = {1, 3, 560, 560};
        
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value pixelValuesTensor = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensorValues.data(),
            inputTensorValues.size(),
            pixelValuesShape.data(),
            pixelValuesShape.size()
        );
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t numInputNodes = pImpl->onnxSession->GetInputCount();
        size_t numOutputNodes = pImpl->onnxSession->GetOutputCount();
        
        // ✅ FIX: Store allocated strings to keep them alive during inference
        std::vector<Ort::AllocatedStringPtr> inputNodeNamesAllocated;
        std::vector<Ort::AllocatedStringPtr> outputNodeNamesAllocated;
        std::vector<const char*> inputNodeNames;
        std::vector<const char*> outputNodeNames;
        
        for (size_t i = 0; i < numInputNodes; i++) {
            auto inputName = pImpl->onnxSession->GetInputNameAllocated(i, allocator);
            inputNodeNames.push_back(inputName.get());
            inputNodeNamesAllocated.push_back(std::move(inputName));
        }
        
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto outputName = pImpl->onnxSession->GetOutputNameAllocated(i, allocator);
            outputNodeNames.push_back(outputName.get());
            outputNodeNamesAllocated.push_back(std::move(outputName));
        }
        
        // ✅ DEBUG: Log input/output names to verify model structure
        LOG_DEBUG("VisionAI", "Model has " + std::to_string(numInputNodes) + " inputs, " + 
                  std::to_string(numOutputNodes) + " outputs");
        for (size_t i = 0; i < numInputNodes; i++) {
            LOG_DEBUG("VisionAI", "  Input[" + std::to_string(i) + "]: " + std::string(inputNodeNames[i]));
        }
        for (size_t i = 0; i < numOutputNodes; i++) {
            LOG_DEBUG("VisionAI", "  Output[" + std::to_string(i) + "]: " + std::string(outputNodeNames[i]));
        }
        
        // ✅ Prepare input tensor array (1 input expected)
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(pixelValuesTensor));
        
        // Run inference
        LOG_DEBUG("VisionAI", "Running ONNX inference on vision encoder...");
        
        auto outputTensors = pImpl->onnxSession->Run(
            Ort::RunOptions{nullptr},
            inputNodeNames.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNodeNames.data(),
            numOutputNodes
        );
        
        auto inferenceTime = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            inferenceTime - startTime).count();
        
        LOG_DEBUG("VisionAI", "✅ ONNX inference completed in " + std::to_string(duration) + "ms");
        
        // ✅ Process output embeddings [1, 1600, 1280] - vision features
        if (outputTensors.size() > 0) {
            auto& outputTensor = outputTensors[0];
            auto tensorInfo = outputTensor.GetTensorTypeAndShapeInfo();
            auto shape = tensorInfo.GetShape();
            
            std::string shapeStr = "[";
            for (size_t i = 0; i < shape.size(); i++) {
                shapeStr += std::to_string(shape[i]);
                if (i < shape.size() - 1) shapeStr += ", ";
            }
            shapeStr += "]";
            
            LOG_DEBUG("VisionAI", "Output shape: " + shapeStr);
            LOG_DEBUG("VisionAI", "Vision embeddings extracted: " + std::to_string(shape[1]) + 
                     " tokens × " + std::to_string(shape[2]) + " dimensions");
            
            // ✅ These embeddings can now be sent to Ollama's language model for text generation
            result.success = true;
            result.description = "Vision embeddings extracted successfully. " +
                               std::to_string(shape[1]) + " visual tokens with " + 
                               std::to_string(shape[2]) + " dimensions each. " +
                               "These can be sent to llama3.2-vision language model for image understanding.";
            result.confidence = 0.95f; // High confidence - embeddings extracted
            result.contextType = "vision_embeddings";
            
            // TODO: Send embeddings to Ollama's language model for actual text generation
            // For now, we have the encoder working - language model integration is next step
        } else {
            result.success = false;
            result.errorMessage = "No output tensors from ONNX model";
        }
        
        LOG_DEBUG("VisionAI", "Vision analysis: " + result.description);
        
        return result;
        
    } catch (const Ort::Exception& e) {
        result.success = false;
        result.errorMessage = "ONNX Runtime error: " + std::string(e.what());
        LOG_ERROR("VisionAI", result.errorMessage);
        return result;
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = "Exception: " + std::string(e.what());
        LOG_ERROR("VisionAI", result.errorMessage);
        return result;
    }
}

VisionAnalysisResult VisionAIManager::analyzeWithHybridVision(const VisionAnalysisRequest& request) {
    VisionAnalysisResult result;
    result.backend = VisionAIBackend::Ollama_LLaVA;  // Hybrid uses Ollama for generation
    
    if (!pImpl->hybridVision) {
        result.success = false;
        result.errorMessage = "Hybrid Vision System not initialized";
        return result;
    }
    
    try {
        // Convert cv::Mat to ImageData
        grim::vision::ImageData imageData;
        imageData.width = request.image.cols;
        imageData.height = request.image.rows;
        imageData.channels = request.image.channels();
        
        // OpenCV uses BGR by default, convert to RGB
        cv::Mat rgb;
        if (request.image.channels() == 3) {
            cv::cvtColor(request.image, rgb, cv::COLOR_BGR2RGB);
        } else {
            rgb = request.image.clone();
        }
        
        // Copy pixel data
        imageData.pixels.assign(rgb.data, rgb.data + (rgb.total() * rgb.channels()));
        
        // Query the hybrid vision system
        auto visionResult = pImpl->hybridVision->query(request.prompt, imageData);
        
        if (visionResult.success) {
            result.success = true;
            result.description = visionResult.response;
            result.confidence = 0.95f;  // High confidence for hybrid system
            
            // Log performance metrics
            LOG_DEBUG("VisionAI", "Hybrid Vision Analysis:");
            if (visionResult.timings.onnx_preprocess_ms > 0) {
                LOG_DEBUG("VisionAI", "  ONNX preprocessing: " + 
                         std::to_string(visionResult.timings.onnx_preprocess_ms) + "ms");
            }
            LOG_DEBUG("VisionAI", "  Image optimization: " + 
                     std::to_string(visionResult.timings.image_optimize_ms) + "ms");
            LOG_DEBUG("VisionAI", "  Ollama processing: " + 
                     std::to_string(visionResult.timings.ollama_total_ms) + "ms");
            LOG_DEBUG("VisionAI", "  Total time: " + 
                     std::to_string(visionResult.timings.total_ms) + "ms");
            
        } else {
            result.success = false;
            result.errorMessage = visionResult.error;
            LOG_ERROR("VisionAI", "Hybrid vision analysis failed: " + visionResult.error);
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = "Hybrid vision exception: " + std::string(e.what());
        LOG_ERROR("VisionAI", result.errorMessage);
    }
    
    return result;
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

VisionAnalysisResult VisionAIManager::analyzeWithFastVision(const VisionAnalysisRequest& request) {
    VisionAnalysisResult result;
    result.backend = VisionAIBackend::ONNX_Vision;
    
    try {
        // Convert cv::Mat to ImageData
        grim::vision::ImageData imageData;
        imageData.width = request.image.cols;
        imageData.height = request.image.rows;
        imageData.channels = request.image.channels();
        
        // OpenCV uses BGR, convert to RGB
        cv::Mat rgb;
        if (request.image.channels() == 3) {
            cv::cvtColor(request.image, rgb, cv::COLOR_BGR2RGB);
        } else {
            rgb = request.image.clone();
        }
        
        imageData.pixels.assign(rgb.data, rgb.data + (rgb.total() * rgb.channels()));
        
        // Get OCR text from context if available (will be passed separately)
        std::string ocr_text = "";  // TODO: Pass OCR text from perception context
        
        // Use fast vision interpreter
        auto fastResult = grim::vision::FastVisionManager::getInstance().interpret(
            imageData,
            ocr_text,
            request.prompt
        );
        
        if (fastResult.success) {
            result.success = true;
            result.description = fastResult.description;
            result.confidence = fastResult.scene.confidence;
            result.modelUsed = "ONNX Fast Vision";
            
            LOG_DEBUG("VisionAI", "Fast Vision Analysis:");
            LOG_DEBUG("VisionAI", "  Processing time: " + std::to_string(fastResult.processing_time_ms) + "ms");
            LOG_DEBUG("VisionAI", "  Scene: " + fastResult.description);
            LOG_DEBUG("VisionAI", "  Activity: " + fastResult.activity);
            
        } else {
            result.success = false;
            result.errorMessage = fastResult.error;
            LOG_ERROR("VisionAI", "Fast vision failed: " + fastResult.error);
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = "Fast vision exception: " + std::string(e.what());
        LOG_ERROR("VisionAI", result.errorMessage);
    }
    
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

// ✅ Warm up the vision model by sending a small test image
void VisionAIManager::warmupModel() {
    LOG_DEBUG("VisionAI", "Starting model warmup (async)...");
    
    // Run warmup in a separate thread to not block initialization
    std::thread([this]() {
        try {
            // Create a small dummy image (100x100 black square)
            cv::Mat dummyImage(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
            
            // Simple warmup request
            VisionAnalysisRequest warmupRequest;
            warmupRequest.image = dummyImage;
            warmupRequest.prompt = "test";
            warmupRequest.backend = pImpl->defaultBackend;
            warmupRequest.maxTokens = 10; // Minimal tokens for warmup
            
            LOG_DEBUG("VisionAI", "Sending warmup request to load model into memory...");
            auto result = analyzeImage(warmupRequest);
            
            if (result.success) {
                LOG_DEBUG("VisionAI", "Model warmup completed successfully - model is now cached");
            } else {
                LOG_DEBUG("VisionAI", "Model warmup completed (model should be loaded even if warmup failed)");
            }
        } catch (const std::exception& e) {
            LOG_DEBUG("VisionAI", "Model warmup exception (non-critical): " + std::string(e.what()));
        }
    }).detach(); // Detach thread - runs in background
}

// Global initialization
void initVisionAI(VisionAIBackend backend) {
    if (!g_visionAI) {
        LOG_DEBUG("VisionAI", "Model warming up...");
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

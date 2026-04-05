/**
 * @file hybrid_vision.cpp
 * @brief Implementation of Hybrid ONNX + Ollama Vision System
 */

#include "hybrid_vision.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

// STB image library (implementation is in popup_ui/stb_image_impl.cpp)
#include <stb/stb_image.h>

// Use OpenCV for image processing (resize, JPEG encoding)
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace grim {
namespace vision {

// Base64 encoding helper
static std::string base64_encode(const std::vector<uint8_t>& data) {
    static const char* base64_chars = 
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    
    std::string result;
    result.reserve(((data.size() + 2) / 3) * 4);
    
    size_t i = 0;
    while (i < data.size()) {
        uint32_t octet_a = i < data.size() ? data[i++] : 0;
        uint32_t octet_b = i < data.size() ? data[i++] : 0;
        uint32_t octet_c = i < data.size() ? data[i++] : 0;
        
        uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;
        
        result.push_back(base64_chars[(triple >> 18) & 0x3F]);
        result.push_back(base64_chars[(triple >> 12) & 0x3F]);
        result.push_back(base64_chars[(triple >> 6) & 0x3F]);
        result.push_back(base64_chars[triple & 0x3F]);
    }
    
    // Add padding
    size_t padding = (3 - (data.size() % 3)) % 3;
    for (size_t p = 0; p < padding; p++) {
        result[result.size() - 1 - p] = '=';
    }
    
    return result;
}

// ============================================================================
// HybridVisionSystem Implementation
// ============================================================================

HybridVisionSystem::HybridVisionSystem(const VisionConfig& config)
    : config_(config)
{
    std::cout << "Initializing Hybrid Vision System..." << std::endl;
    
    // Initialize ONNX Runtime
    if (config_.use_onnx_preprocessing) {
        if (init_onnx()) {
            std::cout << "✅ ONNX preprocessing enabled" << std::endl;
        } else {
            std::cout << "⚠️  ONNX preprocessing unavailable, continuing without it" << std::endl;
        }
    }
}

HybridVisionSystem::~HybridVisionSystem() {
    // Cleanup handled by smart pointers
}

bool HybridVisionSystem::init_onnx() {
#ifdef GRIM_HAS_ONNXRUNTIME
    try {
        onnx_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HybridVision");
        
        // Create session options
        onnx_session_options_ = std::make_unique<Ort::SessionOptions>();
        onnx_session_options_->SetIntraOpNumThreads(config_.onnx_threads);
        onnx_session_options_->SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );
        
        // Try to use CUDA if available and requested
        if (config_.use_gpu) {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                onnx_session_options_->AppendExecutionProvider_CUDA(cuda_options);
                using_cuda_ = true;
                std::cout << "✅ Using GPU acceleration (CUDA)" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "⚠️  CUDA not available: " << e.what() << std::endl;
                std::cout << "   Falling back to CPU" << std::endl;
            }
        }
        
        // Load the model
#ifdef _WIN32
        std::wstring model_path_w(config_.onnx_model_path.begin(), 
                                  config_.onnx_model_path.end());
        onnx_session_ = std::make_unique<Ort::Session>(
            *onnx_env_,
            model_path_w.c_str(),
            *onnx_session_options_
        );
#else
        onnx_session_ = std::make_unique<Ort::Session>(
            *onnx_env_,
            config_.onnx_model_path.c_str(),
            *onnx_session_options_
        );
#endif
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t num_inputs = onnx_session_->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = onnx_session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name.get());
        }
        
        size_t num_outputs = onnx_session_->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = onnx_session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name.get());
        }
        
        std::cout << "✅ ONNX model loaded from: " << config_.onnx_model_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize ONNX: " << e.what() << std::endl;
#ifdef GRIM_HAS_ONNXRUNTIME
        onnx_session_.reset();
#endif
        return false;
    }
#else
    return false;
#endif
}

float HybridVisionSystem::preprocess_with_onnx(
    const ImageData& image,
    std::vector<float>& out_embeddings
) {
#ifdef GRIM_HAS_ONNXRUNTIME
    if (!onnx_session_) {
        return 0.0f;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // Resize to 560x560 (llama3.2-vision expected size)
        ImageData resized = image;
        resize_image(resized, 560);
        
        // Normalize and convert to CHW format
        std::vector<float> input_tensor_values(1 * 3 * 560 * 560);
        
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std[3] = {0.229f, 0.224f, 0.225f};
        
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 560; h++) {
                for (int w = 0; w < 560; w++) {
                    int pixel_idx = (h * 560 + w) * 3 + c;
                    int tensor_idx = c * 560 * 560 + h * 560 + w;
                    
                    float pixel = resized.pixels[pixel_idx] / 255.0f;
                    input_tensor_values[tensor_idx] = (pixel - mean[c]) / std[c];
                }
            }
        }
        
        // Create input tensor
        std::vector<int64_t> input_shape = {1, 3, 560, 560};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size()
        );
        
        // Prepare input/output
        std::vector<const char*> input_names_cstr;
        for (const auto& name : input_names_) {
            input_names_cstr.push_back(name.c_str());
        }
        
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = onnx_session_->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            &input_tensor,
            1,
            output_names_cstr.data(),
            output_names_cstr.size()
        );
        
        // Extract embeddings
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t total_elements = type_info.GetElementCount();
        
        out_embeddings.assign(output_data, output_data + total_elements);
        
        auto end = std::chrono::high_resolution_clock::now();
        float duration = std::chrono::duration<float, std::milli>(end - start).count();
        
        return duration;
        
    } catch (const std::exception& e) {
        std::cerr << "ONNX preprocessing error: " << e.what() << std::endl;
        return 0.0f;
    }
#else
    (void)image; (void)out_embeddings;
    return 0.0f;
#endif
}

void HybridVisionSystem::resize_image(ImageData& image, int max_size) {
    if (image.width <= max_size && image.height <= max_size) {
        return;
    }
    
    // Calculate new dimensions maintaining aspect ratio
    float ratio = std::min(
        static_cast<float>(max_size) / image.width,
        static_cast<float>(max_size) / image.height
    );
    
    int new_width = static_cast<int>(image.width * ratio);
    int new_height = static_cast<int>(image.height * ratio);
    
    // Convert to OpenCV Mat
    cv::Mat src(image.height, image.width, CV_8UC3, image.pixels.data());
    cv::Mat dst;
    
    // Resize using OpenCV (high quality Lanczos interpolation)
    cv::resize(src, dst, cv::Size(new_width, new_height), 0, 0, cv::INTER_LANCZOS4);
    
    // Copy back to ImageData
    image.width = new_width;
    image.height = new_height;
    image.pixels.assign(dst.data, dst.data + (new_width * new_height * image.channels));
}

float HybridVisionSystem::optimize_for_ollama(
    const ImageData& image,
    std::string& out_base64
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Resize if needed
    ImageData optimized = image;
    resize_image(optimized, config_.max_image_size);
    
    // Convert to OpenCV Mat
    cv::Mat img_mat(optimized.height, optimized.width, CV_8UC3, optimized.pixels.data());
    
    // Encode to JPEG using OpenCV
    std::vector<uint8_t> jpeg_buffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, config_.jpeg_quality};
    
    if (!cv::imencode(".jpg", img_mat, jpeg_buffer, params)) {
        std::cerr << "Failed to encode JPEG" << std::endl;
        return 0.0f;
    }
    
    // Encode to base64
    out_base64 = base64_encode(jpeg_buffer);
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count();
}

bool HybridVisionSystem::query_ollama(
    const std::string& prompt,
    const std::string& image_b64,
    VisionResult& result
) {
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        std::string url = config_.ollama_url + "/api/generate";
        
        std::cout << "   Ollama URL: " << url << std::endl;
        std::cout << "   Model: " << config_.ollama_model << std::endl;
        std::cout << "   Image size: " << (image_b64.size() / 1024) << " KB (base64)" << std::endl;
        std::cout << "   Timeout: " << config_.timeout_seconds << "s" << std::endl;
        
        // Build JSON payload
        nlohmann::json payload = {
            {"model", config_.ollama_model},
            {"prompt", prompt},
            {"images", {image_b64}},
            {"stream", false}
        };
        
        std::string payload_str = payload.dump();
        std::cout << "   Payload size: " << (payload_str.size() / 1024) << " KB" << std::endl;
        
        std::cout << "   Sending POST request..." << std::endl;
        
        // Send POST request using CPR with appropriate timeout for vision models
        // Vision models can take 20-60 seconds to process, so we need generous timeouts
        cpr::Response response = cpr::Post(
            cpr::Url{url},
            cpr::Header{{"Content-Type", "application/json"}},
            cpr::Body{payload_str},
            cpr::Timeout{std::chrono::milliseconds(config_.timeout_seconds * 1000)},
            cpr::ConnectTimeout{std::chrono::milliseconds(10000)}  // 10 second connect timeout
            // Note: Removed LowSpeed limit - vision models process slowly but are working
        );
        
        auto end = std::chrono::high_resolution_clock::now();
        result.timings.api_roundtrip_ms = 
            std::chrono::duration<float, std::milli>(end - start).count();
        
        std::cout << "   Response received in " << result.timings.api_roundtrip_ms << "ms" << std::endl;
        
        // Check for errors
        if (response.error.code != cpr::ErrorCode::OK) {
            result.error = "Connection error: " + response.error.message;
            std::cout << "   ❌ " << result.error << std::endl;
            return false;
        }
        
        std::cout << "   HTTP Status: " << response.status_code << std::endl;
        
        if (response.status_code != 200) {
            result.error = "HTTP " + std::to_string(response.status_code);
            if (!response.text.empty()) {
                result.error += ": " + response.text.substr(0, 200);  // First 200 chars
            }
            std::cout << "   ❌ " << result.error << std::endl;
            return false;
        }
        
        // Parse response
        std::cout << "   Parsing JSON response..." << std::endl;
        auto response_json = nlohmann::json::parse(response.text);
        
        result.response = response_json.value("response", "");
        
        std::cout << "   Response length: " << result.response.size() << " chars" << std::endl;
        
        // Extract timing information
        if (response_json.contains("total_duration")) {
            result.timings.ollama_total_ms = 
                response_json["total_duration"].get<int64_t>() / 1e6;
        }
        
        if (response_json.contains("load_duration")) {
            result.timings.ollama_load_ms = 
                response_json["load_duration"].get<int64_t>() / 1e6;
        }
        
        if (response_json.contains("prompt_eval_duration")) {
            result.timings.ollama_prompt_eval_ms = 
                response_json["prompt_eval_duration"].get<int64_t>() / 1e6;
        }
        
        if (response_json.contains("eval_duration")) {
            result.timings.ollama_eval_ms = 
                response_json["eval_duration"].get<int64_t>() / 1e6;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        result.error = e.what();
        return false;
    }
}

VisionResult HybridVisionSystem::query(
    const std::string& prompt,
    const ImageData& image
) {
    VisionResult result;
    result.prompt = prompt;
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Querying Vision Model: " << config_.ollama_model << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    if (!image.is_valid()) {
        result.error = "Invalid image data";
        return result;
    }
    
    // Step 1: ONNX preprocessing (optional)
    if (config_.use_onnx_preprocessing && has_onnx_preprocessing()) {
        std::cout << "\n1️⃣  ONNX Preprocessing..." << std::endl;
        
        std::vector<float> embeddings;
        float preprocess_time = preprocess_with_onnx(image, embeddings);
        
        if (preprocess_time > 0 && !embeddings.empty()) {
            result.timings.onnx_preprocess_ms = preprocess_time;
            result.embeddings = std::move(embeddings);
            
            std::cout << "   ✅ Image validated and preprocessed on GPU" << std::endl;
            std::cout << "   Time: " << preprocess_time << "ms" << std::endl;
        }
    }
    
    // Step 2: Optimize image for Ollama
    std::cout << "\n2️⃣  Optimizing image for Ollama..." << std::endl;
    
    std::string image_b64;
    float optimize_time = optimize_for_ollama(image, image_b64);
    
    result.timings.image_optimize_ms = optimize_time;
    
    std::cout << "   Original size: " << image.width << "x" << image.height << std::endl;
    std::cout << "   Encoded size: " << (image_b64.size() / 1024.0) << " KB" << std::endl;
    std::cout << "   Time: " << optimize_time << "ms" << std::endl;
    
    // Step 3: Query Ollama
    std::cout << "\n3️⃣  Sending to Ollama..." << std::endl;
    
    result.success = query_ollama(prompt, image_b64, result);
    
    if (result.success) {
        std::cout << "   ✅ Response received" << std::endl;
    } else {
        std::cout << "   ❌ Error: " << result.error << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    result.timings.total_ms = 
        std::chrono::duration<float, std::milli>(total_end - total_start).count();
    
    return result;
}

std::optional<ImageData> HybridVisionSystem::load_image(const std::string& path) {
    ImageData image;
    
    int width, height, channels;
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    
    if (!data) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return std::nullopt;
    }
    
    image.width = width;
    image.height = height;
    image.channels = 3;  // Force RGB
    image.pixels.assign(data, data + (width * height * 3));
    
    stbi_image_free(data);
    
    return image;
}

std::optional<ImageData> HybridVisionSystem::load_image_from_memory(
    const std::vector<uint8_t>& data
) {
    ImageData image;
    
    int width, height, channels;
    uint8_t* pixels = stbi_load_from_memory(
        data.data(),
        static_cast<int>(data.size()),
        &width, &height, &channels, 3
    );
    
    if (!pixels) {
        std::cerr << "Failed to load image from memory" << std::endl;
        return std::nullopt;
    }
    
    image.width = width;
    image.height = height;
    image.channels = 3;
    image.pixels.assign(pixels, pixels + (width * height * 3));
    
    stbi_image_free(pixels);
    
    return image;
}

// ============================================================================
// VisionSystemManager Implementation
// ============================================================================

std::unique_ptr<HybridVisionSystem> VisionSystemManager::instance_ = nullptr;
bool VisionSystemManager::initialized_ = false;

HybridVisionSystem& VisionSystemManager::get_instance() {
    if (!instance_) {
        instance_ = std::make_unique<HybridVisionSystem>();
        initialized_ = true;
    }
    return *instance_;
}

bool VisionSystemManager::initialize(const VisionConfig& config) {
    try {
        instance_ = std::make_unique<HybridVisionSystem>(config);
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize vision system: " << e.what() << std::endl;
        return false;
    }
}

void VisionSystemManager::shutdown() {
    instance_.reset();
    initialized_ = false;
}

} // namespace vision
} // namespace grim

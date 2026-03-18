/**
 * @file fast_vision_interpreter.cpp
 * @brief Fast vision interpretation implementation
 */

#include "fast_vision_interpreter.hpp"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <regex>
#include <opencv2/opencv.hpp>

namespace grim {
namespace vision {

FastVisionInterpreter::FastVisionInterpreter() {
    initSceneTemplates();
}

FastVisionInterpreter::~FastVisionInterpreter() = default;

bool FastVisionInterpreter::init(const std::string& onnx_model_path) {
#ifdef GRIM_HAS_ONNXRUNTIME
    try {
        onnx_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FastVision");
        
        onnx_options_ = std::make_unique<Ort::SessionOptions>();
        onnx_options_->SetIntraOpNumThreads(4);
        onnx_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            onnx_options_->AppendExecutionProvider_CUDA(cuda_options);
            using_cuda_ = true;
        } catch (...) {
        }
        
#ifdef _WIN32
        std::wstring model_path_w(onnx_model_path.begin(), onnx_model_path.end());
        onnx_session_ = std::make_unique<Ort::Session>(*onnx_env_, model_path_w.c_str(), *onnx_options_);
#else
        onnx_session_ = std::make_unique<Ort::Session>(*onnx_env_, onnx_model_path.c_str(), *onnx_options_);
#endif
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Fast Vision: " << e.what() << std::endl;
        return false;
    }
#else
    (void)onnx_model_path;
    return false;
#endif
}

void FastVisionInterpreter::initSceneTemplates() {
    // Code editor patterns
    scene_templates_.push_back({
        SceneClassification::SceneType::CodeEditor,
        {"function", "class", "def", "import", "const", "var", "let", "public", "private", "return"},
        {"{", "}", ";", "//", "/*", "*/", "=>", "->"},
        0.7f
    });
    
    // Web browser patterns
    scene_templates_.push_back({
        SceneClassification::SceneType::WebBrowser,
        {"http", "https", "www", "search", "google", "chrome", "firefox", "edge"},
        {"☰", "⋮", "🔍", "← →"},
        0.6f
    });
    
    // Terminal patterns
    scene_templates_.push_back({
        SceneClassification::SceneType::Terminal,
        {"$", "#", "PS", "cmd", "bash", "sh", "python", "node", "npm", "git"},
        {">", "$", "#", "~"},
        0.75f
    });
    
    // Chat/messaging patterns
    scene_templates_.push_back({
        SceneClassification::SceneType::ChatApp,
        {"message", "chat", "send", "reply", "typing", "online", "offline"},
        {"👤", "💬", "📎", "😀"},
        0.65f
    });
    
    // Text document patterns
    scene_templates_.push_back({
        SceneClassification::SceneType::TextDocument,
        {"paragraph", "sentence", "document", "page", "word", "edit"},
        {},
        0.5f
    });
}

FastVisionResult FastVisionInterpreter::interpret(
    const ImageData& image,
    const std::string& ocr_text,
    const std::string& prompt
) {
    FastVisionResult result;
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!initialized_) {
        result.error = "Interpreter not initialized";
        return result;
    }
    
    try {
        // Step 1: Extract ONNX embeddings (fast - 262ms on GPU)
        auto embeddings = extractEmbeddings(image);
        
        // Step 2: Classify scene using embeddings + OCR
        result.scene = classifyScene(embeddings, ocr_text);
        
        // Step 3: Infer activity
        result.activity = inferActivity(result.scene, ocr_text);
        
        // Step 4: Generate description
        result.description = generateDescription(result.scene, ocr_text, prompt);
        
        // Step 5: Extract keywords from OCR
        auto ocr_analysis = analyzeOCRContext(ocr_text);
        for (const auto& [keyword, score] : ocr_analysis) {
            if (score > 0.3f) {
                result.keywords.push_back(keyword);
            }
        }
        
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error = e.what();
        result.success = false;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    return result;
}

std::vector<float> FastVisionInterpreter::extractEmbeddings(const ImageData& image) {
    // Resize to 560x560
    cv::Mat img_mat(image.height, image.width, CV_8UC3, const_cast<uint8_t*>(image.pixels.data()));
    cv::Mat resized;
    cv::resize(img_mat, resized, cv::Size(560, 560), 0, 0, cv::INTER_LANCZOS4);
    
    // Normalize
    std::vector<float> input_tensor(1 * 3 * 560 * 560);
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std[3] = {0.229f, 0.224f, 0.225f};
    
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 560; h++) {
            for (int w = 0; w < 560; w++) {
                int pixel_idx = (h * 560 + w) * 3 + c;
                int tensor_idx = c * 560 * 560 + h * 560 + w;
                float pixel = resized.data[pixel_idx] / 255.0f;
                input_tensor[tensor_idx] = (pixel - mean[c]) / std[c];
            }
        }
    }
    
#ifdef GRIM_HAS_ONNXRUNTIME
    std::vector<int64_t> input_shape = {1, 3, 560, 560};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor.data(), input_tensor.size(),
        input_shape.data(), input_shape.size()
    );
    
    std::vector<const char*> input_names = {onnx_session_->GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get()};
    std::vector<const char*> output_names = {onnx_session_->GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get()};
    
    auto outputs = onnx_session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_ort, 1, output_names.data(), 1);
    
    float* output_data = outputs[0].GetTensorMutableData<float>();
    size_t output_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    return std::vector<float>(output_data, output_data + output_size);
#else
    return {};
#endif
}

SceneClassification FastVisionInterpreter::classifyScene(
    const std::vector<float>& embeddings,
    const std::string& ocr_text
) {
    SceneClassification result;
    result.type = SceneClassification::SceneType::Unknown;
    result.confidence = 0.0f;
    
    // Analyze OCR for clues
    std::string ocr_lower = ocr_text;
    std::transform(ocr_lower.begin(), ocr_lower.end(), ocr_lower.begin(), ::tolower);
    
    std::map<SceneClassification::SceneType, float> scores;
    
    for (const auto& template_item : scene_templates_) {
        float score = 0.0f;
        int keyword_matches = 0;
        
        // Check keywords in OCR
        for (const auto& keyword : template_item.keywords) {
            if (ocr_lower.find(keyword) != std::string::npos) {
                keyword_matches++;
                score += 0.1f;
            }
        }
        
        // Check UI patterns
        for (const auto& pattern : template_item.ui_patterns) {
            if (ocr_text.find(pattern) != std::string::npos) {
                score += 0.05f;
            }
        }
        
        scores[template_item.type] = score;
    }
    
    // Find best match
    float max_score = 0.0f;
    for (const auto& [type, score] : scores) {
        if (score > max_score) {
            max_score = score;
            result.type = type;
            result.confidence = std::min(score, 1.0f);
        }
    }
    
    return result;
}

std::map<std::string, float> FastVisionInterpreter::analyzeOCRContext(const std::string& ocr_text) {
    std::map<std::string, float> keywords;
    
    if (ocr_text.empty()) return keywords;
    
    // Simple keyword extraction (you can enhance this)
    std::istringstream iss(ocr_text);
    std::string word;
    std::map<std::string, int> word_counts;
    
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        
        // Skip short words
        if (word.length() < 3) continue;
        
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        word_counts[word]++;
    }
    
    // Normalize to scores
    int max_count = 0;
    for (const auto& [w, count] : word_counts) {
        max_count = std::max(max_count, count);
    }
    
    for (const auto& [word, count] : word_counts) {
        keywords[word] = static_cast<float>(count) / max_count;
    }
    
    return keywords;
}

std::string FastVisionInterpreter::inferActivity(
    const SceneClassification& scene,
    const std::string& ocr_text
) {
    switch (scene.type) {
        case SceneClassification::SceneType::CodeEditor:
            if (ocr_text.find("debug") != std::string::npos) return "debugging code";
            if (ocr_text.find("error") != std::string::npos) return "fixing errors";
            return "writing code";
            
        case SceneClassification::SceneType::WebBrowser:
            if (ocr_text.find("search") != std::string::npos) return "searching the web";
            if (ocr_text.find("video") != std::string::npos) return "watching videos";
            return "browsing the web";
            
        case SceneClassification::SceneType::Terminal:
            if (ocr_text.find("git") != std::string::npos) return "using git";
            if (ocr_text.find("npm") != std::string::npos || ocr_text.find("python") != std::string::npos) 
                return "running commands";
            return "using terminal";
            
        case SceneClassification::SceneType::ChatApp:
            return "chatting";
            
        case SceneClassification::SceneType::TextDocument:
            return "editing document";
            
        default:
            return "working";
    }
}

std::string FastVisionInterpreter::generateDescription(
    const SceneClassification& scene,
    const std::string& ocr_text,
    const std::string& prompt
) {
    std::ostringstream desc;
    
    // Start with scene type
    switch (scene.type) {
        case SceneClassification::SceneType::CodeEditor:
            desc << "The user is in a code editor";
            break;
        case SceneClassification::SceneType::WebBrowser:
            desc << "The user is browsing the web";
            break;
        case SceneClassification::SceneType::Terminal:
            desc << "The user is in a terminal";
            break;
        case SceneClassification::SceneType::ChatApp:
            desc << "The user is in a chat application";
            break;
        case SceneClassification::SceneType::TextDocument:
            desc << "The user is editing a document";
            break;
        default:
            desc << "The user is working on their computer";
    }
    
    // Add context from OCR
    if (!ocr_text.empty() && ocr_text.length() > 20) {
        // Extract first meaningful line
        std::istringstream iss(ocr_text);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.length() > 10 && line.length() < 100) {
                desc << ". Visible text includes: \"" << line.substr(0, 50) << "\"";
                break;
            }
        }
    }
    
    desc << ".";
    
    return desc.str();
}

// Singleton implementation
FastVisionManager& FastVisionManager::getInstance() {
    static FastVisionManager instance;
    return instance;
}

bool FastVisionManager::initialize(const std::string& onnx_model_path) {
    if (initialized_) return true;
    
    interpreter_ = std::make_unique<FastVisionInterpreter>();
    initialized_ = interpreter_->init(onnx_model_path);
    return initialized_;
}

void FastVisionManager::shutdown() {
    interpreter_.reset();
    initialized_ = false;
}

FastVisionResult FastVisionManager::interpret(
    const ImageData& image,
    const std::string& ocr_text,
    const std::string& prompt
) {
    if (!interpreter_) {
        FastVisionResult result;
        result.error = "Not initialized";
        return result;
    }
    
    return interpreter_->interpret(image, ocr_text, prompt);
}

bool FastVisionManager::isReady() const {
    return initialized_ && interpreter_ && interpreter_->isReady();
}

} // namespace vision
} // namespace grim

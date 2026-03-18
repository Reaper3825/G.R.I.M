/**
 * @file fast_vision_interpreter.hpp
 * @brief Fast vision interpretation using ONNX embeddings + heuristics
 * 
 * Provides near-instantaneous (<500ms) vision understanding by:
 * 1. ONNX vision encoder (262ms on RTX 3080Ti)
 * 2. Embedding-based scene classification
 * 3. Heuristic interpretation rules
 * 4. OCR integration for text context
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#ifdef GRIM_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif
#include "hybrid_vision.hpp"

namespace grim {
namespace vision {

/**
 * @brief Scene classification results
 */
struct SceneClassification {
    enum class SceneType {
        Desktop,
        WebBrowser,
        CodeEditor,
        Terminal,
        VideoPlayer,
        ImageViewer,
        TextDocument,
        ChatApp,
        Game,
        Unknown
    };
    
    SceneType type = SceneType::Unknown;
    float confidence = 0.0f;
    std::string description;
    std::vector<std::string> detectedElements;
};

/**
 * @brief Fast vision interpretation result
 */
struct FastVisionResult {
    bool success = false;
    std::string description;          // Quick human-readable description
    SceneClassification scene;
    std::string activity;             // What the user appears to be doing
    std::vector<std::string> keywords; // Key visual elements
    float processing_time_ms = 0.0f;
    std::string error;
};

/**
 * @brief Fast Vision Interpreter
 * 
 * Uses ONNX embeddings + heuristics for near-instant vision understanding.
 * Designed for real-time interaction (<500ms total processing).
 */
class FastVisionInterpreter {
public:
    FastVisionInterpreter();
    ~FastVisionInterpreter();
    
    /**
     * @brief Initialize the interpreter with ONNX model
     * @param onnx_model_path Path to ONNX vision encoder
     * @return true if successful
     */
    bool init(const std::string& onnx_model_path);
    
    /**
     * @brief Interpret an image quickly using embeddings + heuristics
     * @param image Image data
     * @param ocr_text OCR text from image (optional, but recommended)
     * @param prompt User's question (optional, for context)
     * @return Interpretation result
     */
    FastVisionResult interpret(
        const ImageData& image,
        const std::string& ocr_text = "",
        const std::string& prompt = ""
    );
    
    /**
     * @brief Classify scene type from embeddings
     * @param embeddings ONNX vision embeddings
     * @param ocr_text OCR text for context
     * @return Scene classification
     */
    SceneClassification classifyScene(
        const std::vector<float>& embeddings,
        const std::string& ocr_text
    );
    
    /**
     * @brief Generate description from scene and OCR
     * @param scene Scene classification
     * @param ocr_text OCR text
     * @param prompt User's question
     * @return Human-readable description
     */
    std::string generateDescription(
        const SceneClassification& scene,
        const std::string& ocr_text,
        const std::string& prompt
    );
    
    /**
     * @brief Check if initialized
     */
    bool isReady() const { return initialized_; }
    
private:
    bool initialized_ = false;
    
#ifdef GRIM_HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> onnx_env_;
    std::unique_ptr<Ort::Session> onnx_session_;
    std::unique_ptr<Ort::SessionOptions> onnx_options_;
#endif
    bool using_cuda_ = false;
    
    // Scene classification
    struct SceneTemplate {
        SceneClassification::SceneType type;
        std::vector<std::string> keywords;
        std::vector<std::string> ui_patterns;
        float embedding_threshold;
    };
    std::vector<SceneTemplate> scene_templates_;
    
    /**
     * @brief Extract embeddings from image using ONNX
     */
    std::vector<float> extractEmbeddings(const ImageData& image);
    
    /**
     * @brief Analyze OCR text for context clues
     */
    std::map<std::string, float> analyzeOCRContext(const std::string& ocr_text);
    
    /**
     * @brief Detect UI patterns from OCR
     */
    std::vector<std::string> detectUIPatterns(const std::string& ocr_text);
    
    /**
     * @brief Infer activity from scene and text
     */
    std::string inferActivity(
        const SceneClassification& scene,
        const std::string& ocr_text
    );
    
    /**
     * @brief Initialize scene templates
     */
    void initSceneTemplates();
};

/**
 * @brief Singleton manager for fast vision
 */
class FastVisionManager {
public:
    static FastVisionManager& getInstance();
    
    bool initialize(const std::string& onnx_model_path);
    void shutdown();
    
    FastVisionResult interpret(
        const ImageData& image,
        const std::string& ocr_text = "",
        const std::string& prompt = ""
    );
    
    bool isReady() const;
    
private:
    FastVisionManager() = default;
    std::unique_ptr<FastVisionInterpreter> interpreter_;
    bool initialized_ = false;
};

} // namespace vision
} // namespace grim

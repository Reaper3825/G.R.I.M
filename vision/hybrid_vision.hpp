/**
 * @file hybrid_vision.hpp
 * @brief Hybrid ONNX + Ollama Vision System for GRIM
 * 
 * Combines GPU-accelerated ONNX preprocessing with Ollama's vision-language model.
 * This approach provides:
 * - Fast GPU preprocessing via ONNX Runtime (RTX 3080Ti acceleration)
 * - High-quality vision-language understanding via Ollama
 * - Optimized image handling for minimal latency
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <chrono>
#ifdef GRIM_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace grim {
namespace vision {

/**
 * @brief Timing information for vision queries
 */
struct VisionTimings {
    float onnx_preprocess_ms = 0.0f;    ///< ONNX preprocessing time
    float image_optimize_ms = 0.0f;      ///< Image optimization time
    float ollama_load_ms = 0.0f;         ///< Ollama model load time
    float ollama_prompt_eval_ms = 0.0f;  ///< Ollama prompt evaluation time
    float ollama_eval_ms = 0.0f;         ///< Ollama text generation time
    float ollama_total_ms = 0.0f;        ///< Ollama total time
    float api_roundtrip_ms = 0.0f;       ///< Total API round-trip time
    float total_ms = 0.0f;               ///< Total query time
};

/**
 * @brief Result from a vision query
 */
struct VisionResult {
    std::string prompt;           ///< The prompt that was sent
    std::string response;         ///< The model's response
    std::string error;            ///< Error message if failed
    VisionTimings timings;        ///< Timing breakdown
    bool success = false;         ///< Whether the query succeeded
    
    // Optional: Visual embeddings from ONNX
    std::vector<float> embeddings;
    size_t embedding_tokens = 0;
    size_t embedding_dim = 0;
};

/**
 * @brief Image data structure
 */
struct ImageData {
    std::vector<uint8_t> pixels;  ///< Raw pixel data (RGB)
    int width = 0;                ///< Image width
    int height = 0;               ///< Image height
    int channels = 3;             ///< Number of channels (RGB = 3)
    
    bool is_valid() const {
        return !pixels.empty() && width > 0 && height > 0;
    }
};

/**
 * @brief Configuration for the hybrid vision system
 */
struct VisionConfig {
    std::string onnx_model_path = "D:/G.R.I.M/data/models/vision/llama3.2-vision-onnx/vision_encoder.onnx";
    std::string ollama_model = "llama3.2-vision:11b";
    std::string ollama_url = "http://localhost:11434";
    
    bool use_onnx_preprocessing = true;  ///< Use ONNX for GPU preprocessing
    bool use_gpu = true;                 ///< Use GPU acceleration (CUDA)
    bool fast_mode = true;               ///< Use embeddings-only for speed (no Ollama)
    int max_image_size = 896;            ///< Max image dimension for Ollama
    int jpeg_quality = 85;               ///< JPEG compression quality (1-100)
    int onnx_threads = 4;                ///< ONNX intra-op threads
    int timeout_seconds = 60;            ///< API timeout
};

/**
 * @brief Hybrid Vision System
 * 
 * Integrates ONNX Runtime for fast GPU preprocessing with Ollama for
 * vision-language understanding.
 * 
 * Usage:
 * @code
 * VisionConfig config;
 * HybridVisionSystem vision(config);
 * 
 * ImageData img = load_image("photo.jpg");
 * auto result = vision.query("What's in this image?", img);
 * 
 * if (result.success) {
 *     std::cout << result.response << std::endl;
 * }
 * @endcode
 */
class HybridVisionSystem {
public:
    /**
     * @brief Constructor
     * @param config Vision system configuration
     */
    explicit HybridVisionSystem(const VisionConfig& config = VisionConfig{});
    
    /**
     * @brief Destructor
     */
    ~HybridVisionSystem();
    
    // Prevent copying
    HybridVisionSystem(const HybridVisionSystem&) = delete;
    HybridVisionSystem& operator=(const HybridVisionSystem&) = delete;
    
    /**
     * @brief Query the vision model with an image and prompt
     * 
     * @param prompt Text prompt for the vision model
     * @param image Image to analyze
     * @return VisionResult containing response and timing information
     */
    VisionResult query(const std::string& prompt, const ImageData& image);
    
    /**
     * @brief Check if ONNX preprocessing is available
     * @return true if ONNX model is loaded and ready
     */
    bool has_onnx_preprocessing() const {
#ifdef GRIM_HAS_ONNXRUNTIME
        return onnx_session_ != nullptr;
#else
        return false;
#endif
    }
    
    /**
     * @brief Check if GPU acceleration is available
     * @return true if CUDA provider is available
     */
    bool has_gpu_acceleration() const { return using_cuda_; }
    
    /**
     * @brief Get the current configuration
     * @return Reference to configuration
     */
    const VisionConfig& get_config() const { return config_; }
    
    /**
     * @brief Load an image from file
     * @param path Path to image file
     * @return ImageData or empty optional if failed
     */
    static std::optional<ImageData> load_image(const std::string& path);
    
    /**
     * @brief Load an image from memory buffer
     * @param data Image file data (JPEG, PNG, etc.)
     * @return ImageData or empty optional if failed
     */
    static std::optional<ImageData> load_image_from_memory(
        const std::vector<uint8_t>& data
    );
    
private:
    /**
     * @brief Initialize ONNX Runtime session
     * @return true if successful
     */
    bool init_onnx();
    
    /**
     * @brief Preprocess image using ONNX (GPU accelerated)
     * @param image Input image
     * @param out_embeddings Output visual embeddings
     * @return Preprocessing time in milliseconds
     */
    float preprocess_with_onnx(
        const ImageData& image,
        std::vector<float>& out_embeddings
    );
    
    /**
     * @brief Optimize image for Ollama API
     * @param image Input image
     * @param out_base64 Output base64-encoded JPEG
     * @return Optimization time in milliseconds
     */
    float optimize_for_ollama(
        const ImageData& image,
        std::string& out_base64
    );
    
    /**
     * @brief Resize image maintaining aspect ratio
     * @param image Input/output image
     * @param max_size Maximum dimension
     */
    void resize_image(ImageData& image, int max_size);
    
    /**
     * @brief Send query to Ollama API
     * @param prompt Text prompt
     * @param image_b64 Base64-encoded image
     * @param result Output result
     * @return true if successful
     */
    bool query_ollama(
        const std::string& prompt,
        const std::string& image_b64,
        VisionResult& result
    );
    
    VisionConfig config_;
    
#ifdef GRIM_HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> onnx_env_;
    std::unique_ptr<Ort::Session> onnx_session_;
    std::unique_ptr<Ort::SessionOptions> onnx_session_options_;
#endif
    bool using_cuda_ = false;
    
    // Input/output names for ONNX
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

/**
 * @brief Helper class for managing vision system lifecycle
 * 
 * Provides RAII management of the vision system with automatic
 * initialization and cleanup.
 */
class VisionSystemManager {
public:
    /**
     * @brief Get singleton instance
     * @return Reference to vision system
     */
    static HybridVisionSystem& get_instance();
    
    /**
     * @brief Initialize with custom config
     * @param config Configuration
     * @return true if successful
     */
    static bool initialize(const VisionConfig& config);
    
    /**
     * @brief Shutdown vision system
     */
    static void shutdown();
    
private:
    static std::unique_ptr<HybridVisionSystem> instance_;
    static bool initialized_;
};

} // namespace vision
} // namespace grim

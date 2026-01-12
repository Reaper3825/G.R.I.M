//======================================================//
//  grim_language_model_cuda.hpp
//  CUDA-safe declarations for LanguageModel
//  NO IMPLEMENTATIONS - declarations only
//  NO .cpp includes - CUDA compilation safe
//          ONLY SOURCE OF TRUTH
//======================================================//

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include <random>
#include <cmath>
#include <cstdint>
#include <cstddef>

// HyperParameters - Single source of truth for model configuration
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Layers/Quantization/Quantization_GPU.hpp"
#include "../Layers/ScratchBlock/ScratchBlock_GPU.hpp"
#include "../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"
#include "../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../Shared/Loss/LossContext/LossContext.hpp"
#endif

namespace GRIM {

//======================================================//
//  CUDA Error Handling Macros
//======================================================//

#ifdef USE_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__) + " - " + \
                                   cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + ":" + \
                                   std::to_string(__LINE__)); \
        } \
    } while(0)
#endif

//======================================================//
//  Core Data Structures - Minimal Declarations
//======================================================//

// Vector type - minimal definition
class Vector {
public:
    std::vector<float> data;
    
    Vector() = default;
    Vector(size_t size, float val = 0.0f);
    
    size_t size() const;
    float& operator[](size_t idx);
    const float& operator[](size_t idx) const;
    Vector& operator+=(const Vector& other);
    Vector& operator*=(float scalar);
    Vector operator+(const Vector& other) const;
    Vector operator*(float scalar) const;
};

// Matrix type - minimal definition
class Matrix {
public:
    std::vector<Vector> rows;
    int num_rows;
    int num_cols;
    
    Matrix() = default;
    Matrix(int rows, int cols, float init_val = 0.0f, bool random = false);
    
    Vector& operator[](size_t idx);
    const Vector& operator[](size_t idx) const;
};

// Context state for embeddings
struct ContextState {
    std::string domain;
    float sentiment;
    int depth;
    std::vector<std::string> active_tags;
    
    ContextState();
};

//======================================================//
//  Configuration Structures
//  NOTE: All numeric defaults are 0 - caller MUST populate
//  from HyperParameters::loadModelArchitecture() or explicitly
//  DO NOT add hardcoded defaults here - HyperParameters_GPU.hpp
//  is the ONLY source of truth for model architecture values
//======================================================//

// Model execution mode - determines memory allocation strategy
enum class ModelExecutionMode {
    TRAINING,    // Full training state with gradient buffers (~15GB+)
    INFERENCE    // Lightweight inference state with only forward caches (~385MB)
};

struct EncoderConfig {
    // Architecture - MUST be populated from HyperParameters
    int d_model = 0;           // Use HyperParameters::DEFAULT_D_MODEL
    int num_heads = 0;         // Use HyperParameters::DEFAULT_NUM_HEADS
    int num_kv_heads = 0;      // Use HyperParameters::DEFAULT_NUM_KV_HEADS (GQA - Grouped Query Attention)
    int head_dim = 0;          // = d_model / num_heads (set from LanguageModelConfig.head_dim)
    int d_ff = 0;              // Use HyperParameters::DEFAULT_D_FF
    int num_layers = 0;        // Use HyperParameters::DEFAULT_NUM_LAYERS
    int max_seq_len = 0;       // Use HyperParameters::DEFAULT_MAX_SEQ_LEN
    float dropout_rate = 0.0f; // Use HyperParameters::DEFAULT_DROPOUT_RATE
    float attention_dropout = 0.0f; // Use HyperParameters::DEFAULT_ATTENTION_DROPOUT
    
    // Cache limits
    int max_cached_batch = 4;
    int max_cached_seq_len = 8192;
    
    // Fixed config values (not architecture-dependent)
    float rms_epsilon = 1e-3f;  // RMSNorm epsilon - increased for numerical stability during training
    HyperParameters::PositionalEncodingType positional_encoding = HyperParameters::DEFAULT_POSITIONAL_ENCODING;
    bool causal_mask = true;                        
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool use_simd = true;
    int num_threads = 4;
    
    // Flash Attention settings
    bool use_flash_attention = true;  // Use Flash Attention 2 for memory efficiency
    int min_seq_len_for_flash = 128;  // Minimum seq len to activate Flash Attention
    
    // Positional encoding (ALiBi+RoPE hybrid) - pointer to shared state in TrainingState
    // WARNING: If nullptr, attention sees no positional info - positions become equivalent!
    const PBM::PBMSpec* pos_encoding = nullptr;
    
    // CUDA execution
    cudaStream_t stream = nullptr;       // CUDA stream for async execution
    cublasHandle_t cublas_handle = nullptr;  // Centralized cuBLAS handle (Rule 22)
};

struct LMHeadConfig {
    int d_model = 0;        // MUST be populated from HyperParameters
    int vocab_size = 0;     // MUST come from .grmt training data or tokenizer
    bool tie_weights = false;
    const Matrix* embedding_weights = nullptr;
    bool use_bias = true;
    bool use_simd = true;
    float epsilon = 1e-10f;
};

enum class SamplingStrategy {
    GREEDY,
    TOP_K,
    TOP_P,
    BEAM_SEARCH
};

struct GenerationConfig {
    SamplingStrategy strategy = SamplingStrategy::TOP_P;
    int max_new_tokens = 100;
    int min_new_tokens = 0;
    float temperature = 1.0f;
    int top_k = 50;
    float top_p = 0.9f;
    float repetition_penalty = 1.0f;
    float length_penalty = 1.0f;
    int num_beams = 1;
    int num_return_sequences = 1;
    bool early_stopping = false;
    int eos_token_id = 0;
    int pad_token_id = 0;
    int no_repeat_ngram_size = 0;
    bool do_sample = true;
    float diversity_penalty = 0.0f;
    std::vector<int> bad_words_ids;
    unsigned int seed = 0;
};

using GenerationStreamCallback = std::function<void(int token_id, float score)>;

struct ActivationQuantizationConfig {
    bool enabled = false;
    bool apply_to_embeddings = false;
    bool apply_to_encoder_outputs = false;
    bool apply_to_layer_caches = false;
    bool apply_to_qkv_cache = false;
    bool apply_to_logits = false;
    float scale = 1.0f;
    float clip_min = -127.0f;
    float clip_max = 127.0f;
    int zero_point = 0;
    bool symmetric = false;
};

struct LanguageModelConfig {
    // Architecture - MUST be populated from HyperParameters
    // DO NOT add hardcoded defaults here - use HyperParameters::loadModelArchitecture()
    int vocab_size = 0;        // MUST come from .grmt training data or tokenizer
    int d_model = 0;           // Use HyperParameters::DEFAULT_D_MODEL
    int num_heads = 0;         // Use HyperParameters::DEFAULT_NUM_HEADS
    int num_kv_heads = 0;      // Use HyperParameters::DEFAULT_NUM_KV_HEADS (GQA - Grouped Query Attention)
    int d_ff = 0;              // Use HyperParameters::DEFAULT_D_FF
    int num_layers = 0;        // Use HyperParameters::DEFAULT_NUM_LAYERS
    int max_seq_len = 0;       // Use HyperParameters::DEFAULT_MAX_SEQ_LEN
    float dropout_rate = 0.0f; // Use HyperParameters::DEFAULT_DROPOUT_RATE
    float attention_dropout = 0.0f; // Use HyperParameters::DEFAULT_ATTENTION_DROPOUT
    
    // Derived values - computed from above, DO NOT set directly
    // Call computeDerivedValues() after setting d_model/num_heads
    int head_dim = 0;          // = d_model / num_heads (computed)
    
    // Compute derived values from primary values - MUST be called after setting d_model/num_heads
    void computeDerivedValues() {
        if (num_heads > 0 && d_model > 0) {
            head_dim = d_model / num_heads;
        }
    }
    
    // Cache limits
    int max_cached_batch = 4;
    int max_cached_seq_len = 8192;
    int max_tokens_per_batch = 0;  // Optional token budget for training logits/loss
    
    // Positional encoding configuration
    HyperParameters::PositionalEncodingType positional_encoding = HyperParameters::DEFAULT_POSITIONAL_ENCODING;
    
    // Fixed config values (not architecture-dependent)
    bool causal_mask = true;
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool use_simd = true;
    int num_threads = 4;
    bool tie_embeddings = true;
    bool use_bias = true;
    bool use_gpu = true;
    bool use_flash_attention = true;  // Use Flash Attention 2 for memory efficiency
    int min_seq_len_for_flash = 512;   // Minimum sequence length to activate Flash Attention
    std::string vocab_path;            // Optional: source vocab file for auto-detection
    bool infer_vocab_from_file = false; // When true, read vocab size from vocab_path at init
    
    // Execution mode - determines memory allocation strategy
    ModelExecutionMode execution_mode = ModelExecutionMode::INFERENCE;
    
    // ScratchBlock reasoning layer config - populated from ai_config.json
    bool use_scratch_block = true;            // Enable ScratchBlock reasoning layer
    int scratch_block_atom_embedding_dim = 64; // Atom embedding dimension
    int scratch_block_max_atoms = 256;         // Max atoms per sequence
    float scratch_block_atom_scale = 0.1f;     // Scale factor for atom injection

    // Numeric head (side-channel regression for numeric atoms)
    bool numeric_head_enabled = false;
    float numeric_head_loss_weight = 0.1f;
    float numeric_head_huber_delta = 1.0f;
    bool numeric_head_log_scale = true;
    
    GenerationConfig generation;
    ActivationQuantizationConfig activation_quantization;
};

// GPU Configuration - FULL definition (source of truth)
struct GPUConfig {
    bool use_tensor_cores = true;        // Enable FP16 Tensor Core ops
    bool use_cuda_graphs = true;         // Use CUDA graphs for repeated inference
    bool use_pinned_memory = true;       // Pinned host buffers
    bool use_dynamic_batching = true;    // Merge micro-batches
    int max_batch_size = 512;            // Max dynamic batch size
    int micro_batch_size = 64;           // Micro-batch for long sequences
    cudaStream_t stream = 0;             // CUDA stream for async ops
    int device_id = 0;                   // CUDA device to use
};

//======================================================//
//  Forward Declarations
//======================================================//

class FeedForwardNetwork;
class RMSNorm;

// Positional Encoding Type - re-exported from HyperParameters_GPU.hpp
using HyperParameters::PositionalEncodingType;

class ALiBiPositionalBias {
private:

    PBM::PBMState pbm_state_{};   // Unified PBM state (both ALiBi + RoPE)
    int num_heads;                // Number of attention heads
    int head_dim_;                // Dimension per head (= d_model / num_heads)
    bool initialized;             // Initialization status
    PositionalEncodingType type;  // Encoding type

public:
    // DEPRECATED: Use head_dim from LanguageModelConfig instead
    // Kept for backwards compatibility during transition
    int d_head;                   // Dimension per head (for RoPE) - public for initialization
    
    ALiBiPositionalBias();
    ALiBiPositionalBias(const ALiBiPositionalBias&) = delete;
    ALiBiPositionalBias& operator=(const ALiBiPositionalBias&) = delete;
    ALiBiPositionalBias(ALiBiPositionalBias&& other) noexcept;
    ALiBiPositionalBias& operator=(ALiBiPositionalBias&& other) noexcept;
    ~ALiBiPositionalBias();

    // num_kv_heads: REQUIRED. Only GQA (grouped-KV) is supported — pass model's num_kv_heads (<= num_heads).

    void computeSlopes(
        int num_heads,
        int num_kv_heads,
        int d_head,
        PositionalEncodingType type
    );


    float* getSlopes() const;
    float* getRoPEFreqs() const;
    bool isInitialized() const;
    PositionalEncodingType getType() const { return type; }
    void cleanup();
};


class PositionalEncoding {
public:
    std::vector<Vector> encodings;
    int max_seq_len;
    int d_model;
    
    PositionalEncoding() = default;
    PositionalEncoding(int max_len, int dim);
    Vector getEncoding(int position) const;
};


// GrimEmbeddingStack - minimal interface
class GrimEmbeddingStack {
public:
    GrimEmbeddingStack(int vocab_size, int d_model, int max_seq_len);
    
    // num_kv_heads REQUIRED - only GQA is supported (no MHA fallback)
    void enableALiBi(int num_heads, int num_kv_heads);
    void enableHybridPositionalEncoding(int num_heads, int d_head, int num_kv_heads);
    const ALiBiPositionalBias* getALiBiBias() const;
    const Matrix& getTokenEmbeddings() const;
    std::vector<Vector> getBatchEmbeddings(const std::vector<int>& token_ids,
                                           const std::vector<int>& positions,
                                           const ContextState& ctx,
                                           const std::vector<std::string>& tags);
    
    // Public members needed by GPU code
    Matrix token_embed;        // Token embedding matrix [vocab_size x d_model]
    PositionalEncoding pos_encoding;  // Positional encoding
    Vector rms_gamma;          // RMSNorm gamma (scale)
    
private:
    int vocab_size_;
    int d_model_;
    int max_seq_len_;
    std::unique_ptr<ALiBiPositionalBias> alibi_;
};

// FeedForwardNetwork - minimal interface
class FeedForwardNetwork {
public:
    explicit FeedForwardNetwork(const EncoderConfig& config);
    
    std::vector<Vector> forwardBatch(const std::vector<Vector>& input);
    
    // Accessors for GPU weight uploads
    const Matrix& getW1() const;
    const Vector& getB1() const;
    const Matrix& getW2() const;
    const Vector& getB2() const;
    
private:
    struct Impl;
    Impl* pImpl = nullptr;
};

// RMSNorm - minimal interface
// IMPORTANT: This implements RMSNorm (Root Mean Square Layer Normalization)!
// RMSNorm uses only gamma (scale) parameter - no bias term.
class RMSNorm {
public:
    RMSNorm(int d_model, float eps);
    
    std::vector<Vector> forwardBatch(const std::vector<Vector>& input);  // Copy semantics
    std::vector<Vector> forwardBatchMove(std::vector<Vector>&& input);    // Move semantics (zero-copy optimization)
    
    // Accessors for GPU weight uploads
    const Vector& getGamma() const;  // RMSNorm scale parameter

    
private:
    struct Impl;
    Impl* pImpl = nullptr;
};

//======================================================//
//  Generated Sequence
//======================================================//

struct GeneratedSequence {
    std::vector<int> token_ids;
    std::vector<float> token_scores;
    std::vector<float> token_numeric_values;
    std::vector<uint8_t> token_numeric_mask;
    float score = 0.0f;
    bool finished = false;

    float getNormalizedScore(float length_penalty) const;
};

//======================================================//
//  Optimizer State for Training
//======================================================//

struct OptimizerState {
    // AdamW optimizer step counter for bias correction
    // NOTE: Actual moment buffers (m, v) are stored in ParameterGroup
    // AdamW hyperparameters are defined in LanguageModel_Training.cu
    int step = 0;
};

//======================================================//
//  Forward Declarations - Classes Defined Later
//======================================================//

class EncoderLayer;
class GrimEncoder;
class LanguageModelHead;
class TextGenerator;
struct EmbeddingRuntime;
struct TrainingState;  // Forward declare for methods that return references

// Forward declare LossContext namespace and nested struct
namespace LossContext {
    struct LossOptions;
}

//======================================================//
//  Parameter Group for Training
//======================================================//

enum class ParamGroupType : uint8_t {
    EMBEDDING = 0,
    LM_HEAD = 1,
    NUMERIC_HEAD = 2,
    ATTENTION = 3,
    FFN = 4,
    RMSNORM = 5,
    SCRATCHBLOCK = 6,  // Atom type embeddings + projection
    COUNT = 7
};

struct ParameterGroup {
    std::string name;
    float* weights;      // Pointer to actual weights on GPU
    float* grads;        // Pointer to gradients on GPU
    size_t size;         // Number of elements
    float* m_state;      // Adam first moment
    float* v_state;      // Adam second moment
    ParamGroupType type; // Category for fast classification
};

// Forward declare TokenBufferView for method signatures
struct TokenBufferView;

#ifdef USE_CUDA
struct TokenBufferView {
    int* device_token_ids = nullptr;
    float* device_token_numeric_values = nullptr;
    uint8_t* device_token_numeric_mask = nullptr;
    int max_tokens = 0;
    cudaStream_t stream = nullptr;
};
#endif

//======================================================//
//  LanguageModel Class Declaration
//======================================================//

class LanguageModel {
public:
    struct ModelStats {
        size_t total_params = 0;
        size_t embedding_params = 0;
        size_t encoder_params = 0;
        size_t lm_head_params = 0;
        size_t numeric_head_params = 0;
        size_t scratchblock_params = 0;  // Atom type embeddings + projection
        float model_size_mb = 0.0f;
    };

    struct GradComponentMetrics {
        float total_norm = 0.0f;
        float embedding_norm = 0.0f;
        float lm_head_norm = 0.0f;
        float numeric_head_norm = 0.0f;
        float attention_norm = 0.0f;
        float ffn_norm = 0.0f;
        float rmsnorm_norm = 0.0f;      // RMSNorm gamma gradients
        float scratchblock_norm = 0.0f; // ScratchBlock atom embeddings + projection gradients
        float grad_scale = 1.0f;
        float clip_threshold = 0.0f;
        bool clipped = false;
        int valid_token_count = 0;
    };

    struct UpdateProbeResult {
        std::string group_name;
        float parameter_rms = 0.0f;
        float grad_rms = 0.0f;
        float update_rms = 0.0f;
        float relative_update = 0.0f;
        float max_abs_update = 0.0f;
        float learning_rate = 0.0f;
        uint64_t optimizer_step = 0;
        uint32_t sample_size = 0;
    };
    
    // Constructor / Destructor
    explicit LanguageModel(const LanguageModelConfig& config);
    ~LanguageModel();
    
    // Main API
    Vector forward(const std::vector<int>& token_ids,
                   const std::vector<float>& token_numeric_values,
                   const std::vector<uint8_t>& token_numeric_mask);
    Vector getNextTokenLogits(const std::vector<int>& context_tokens,
                              const std::vector<float>& context_numeric_values,
                              const std::vector<uint8_t>& context_numeric_mask);
    std::vector<GeneratedSequence> generate(const std::vector<int>& prompt_tokens,
                                            const std::vector<float>& prompt_numeric_values,
                                            const std::vector<uint8_t>& prompt_numeric_mask,
                                            const GenerationConfig* gen_config = nullptr);
    GeneratedSequence generateStream(const std::vector<int>& prompt_tokens,
                                     const std::vector<float>& prompt_numeric_values,
                                     const std::vector<uint8_t>& prompt_numeric_mask,
                                     GenerationStreamCallback callback,
                                     const GenerationConfig* gen_config = nullptr);
    
    // Training
    float computeLoss(const std::vector<int>& input_ids,
                      const std::vector<int>& target_ids,
                      const std::vector<float>& token_numeric_values,
                      const std::vector<uint8_t>& token_numeric_mask);
    float computeLossBatch(const std::vector<std::vector<int>>& batch_input_ids,
                           const std::vector<std::vector<int>>& batch_target_ids,
                           const std::vector<std::vector<float>>& batch_numeric_values,
                           const std::vector<std::vector<uint8_t>>& batch_numeric_mask,
                           const std::vector<std::vector<uint16_t>>& batch_text_features = {},
                           const std::vector<std::vector<uint8_t>>& batch_text_mask = {});  // Batched training
    Vector forwardWithCache(const std::vector<int>& token_ids,
                            const std::vector<float>& token_numeric_values,
                            const std::vector<uint8_t>& token_numeric_mask,
                            bool tokens_on_device = false,
                            const std::vector<uint16_t>& token_text_features = {},
                            const std::vector<uint8_t>& token_text_mask = {});  // Forward pass with activation caching for training
    
    // =========================================================================
    // INCREMENTAL GENERATION API (KV-Cache Autoregressive)
    // =========================================================================
    // Use these for efficient token-by-token generation:
    //   1. Call forwardInit() once with the prompt tokens - caches K,V for all
    //   2. Call forwardStep() for each new token - computes Q for new token only,
    //      attends to all cached K,V, appends new K,V to cache
    //   3. Call resetKVCache() before starting a new generation session
    // =========================================================================
    
    // Initialize KV cache with prompt tokens (prefill phase)
    // Returns logits for the last prompt token (ready for first sampling)
    Vector forwardInit(const std::vector<int>& prompt_tokens,
                       const std::vector<float>& prompt_numeric_values,
                       const std::vector<uint8_t>& prompt_numeric_mask);
    
    // Process a single new token using cached K,V (decode phase)  
    // Returns logits for this token position (ready for next sampling)
    // Automatically appends new K,V to the cache
    // 
    // forwardStep(): Full recompute (O(n²)) - uses FlashAttention, always correct
    // forwardStepIncremental(): True incremental (O(n)) - uses manual attention kernels
    //                           Faster but may have minor numerical differences
    Vector forwardStep(int new_token, float numeric_value, uint8_t numeric_mask);
    Vector forwardStepIncremental(int new_token, float numeric_value, uint8_t numeric_mask);  // O(n) with GQA support
    
    // Clear KV cache (call before starting new generation)
    void resetKVCache();
    
    // Get current KV cache length (number of tokens with cached K,V)
    int getKVCacheLength() const;
    
    void initCuBLASHandle();   // Initialize cuBLAS handle only (MUST be called before initGPU)
    void initPBM();            // Initialize PBM (ALiBi+RoPE hybrid) - MUST be called before initGPU
    void initTrainingState();  // Initialize training state (allocate GPU buffers + gradients)
    void initInferenceState();  // Initialize inference state (allocate GPU buffers WITHOUT gradients)
    bool parameterGroupsStale() const;
    void backward(float loss, bool accumulate = false, float grad_scale = 1.0f, uint64_t step = 0);
    void updateWeights(float learning_rate,
                       OptimizerState* optimizer_state,
                       float weight_decay = HyperParameters::ADAMW_WEIGHT_DECAY);
    void resetOptimizerMoments();
    void scaleOptimizerMoments(float scale);
    void zeroGrad();
    float computeGradNorm(bool sync_for_host_read = false);  // Default false: async (fast). Set true only when reading result on host.
    void scaleGradients(float scale);
    void clampGradients(float min_val, float max_val);  // Clamp individual gradient values
    void dumpGradients(const std::string& path);  // Dump gradients to binary file for inspection
    void logEmbeddingDiagnostics(const std::string& tag);
    void setSequenceLossWeights(const std::vector<float>& weights);
    void clearSequenceLossWeights();
    
#ifdef USE_CUDA
    void setLossOptions(const LossContext::LossOptions& opts) { loss_options_ = opts; }

    // Training state access (for debugging/diagnostics)
    const TrainingState& getTrainingState() const { return training_state_; }
    TrainingState& getTrainingState() { return training_state_; }
#endif

    // Diagnostics (host-side consumers should poll and clear to stay off the hot path)
    const GradComponentMetrics& gradientMetrics() const { return grad_metrics_; }
    bool hasGradientMetrics() const { return grad_metrics_ready_; }
    void clearGradientMetricsFlag() { grad_metrics_ready_ = false; }
    void recordGradientClip(float clip_threshold, bool clipped);

    const UpdateProbeResult& updateProbe() const { return update_probe_result_; }
    bool hasUpdateProbe() const { return update_probe_ready_; }
    void clearUpdateProbeFlag() { update_probe_ready_ = false; }
    void configureUpdateProbe(const std::string& group_name, size_t sample_elems = 2048);
    void disableUpdateProbe();
    const std::vector<float>& updateProbeWeightsBefore() const { return update_probe_weights_before_; }
    const std::vector<float>& updateProbeWeightsAfter() const { return update_probe_weights_after_; }
    const std::vector<float>& updateProbeGradSample() const { return update_probe_grad_sample_; }
    
    // Utilities
    ModelStats getModelStats() const;
    bool save(const std::string& path);
    bool load(const std::string& path);
    
    // Config access
    const LanguageModelConfig& getConfig() const { return config_; }
    
    // GPU methods
    void initGPU();
    Vector forwardGPU(const std::vector<int>& token_ids,
                      const std::vector<float>& token_numeric_values,
                      const std::vector<uint8_t>& token_numeric_mask);
    Vector getNextTokenLogitsGPU(const std::vector<int>& context_tokens,
                                 const std::vector<float>& context_numeric_values,
                                 const std::vector<uint8_t>& context_numeric_mask);
    TokenBufferView getTokenBufferView();
    void markDevicePromptReady(int token_count);
    
    // Helper accessors for GPU implementation
    GrimEmbeddingStack* getEmbedderPtr() { return embedder_.get(); }
    ALiBiPositionalBias* getAlibiPtr() { return alibi_.get(); }
    
#ifdef USE_CUDA
    // GPU runtime accessors - return references to owned objects (fail loud if not initialized)
    EmbeddingRuntime& getGpuEmbedder();
    const EmbeddingRuntime& getGpuEmbedder() const;
    GPUGrimEncoder& getGpuEncoder();
    const GPUGrimEncoder& getGpuEncoder() const;
#endif

#ifdef USE_CUDA
    // Activation quantization helper for forward phases
    void applyActivationQuantization(float* device_buffer, std::size_t elements);
#endif
    
#ifdef USE_CUDA
    // ScratchBlock reasoning layer access
    ScratchBlockLayer* getScratchBlockLayer() { return scratch_block_layer_.get(); }
    const ScratchBlockLayer* getScratchBlockLayer() const { return scratch_block_layer_.get(); }
    void setScratchBlockEnabled(bool enabled);
    bool isScratchBlockEnabled() const;
    
    // Scratch pool configuration (pinned memory transfers)
    void configureScratchPool(bool enabled);
    bool isScratchPoolInitialized() const;
#endif

#ifndef USE_CUDA
    // CPU component accessors (non-CUDA builds)
    GrimEmbeddingStack& getEmbedder() { return *embedder_; }
    GrimEncoder& getEncoder() { return *encoder_; }
    LanguageModelHead& getLMHead() { return *lm_head_; }
    TextGenerator& getGenerator() { return *generator_; }
#endif
    
    GeneratedSequence generateSequenceGPU(const std::vector<int>& prompt_tokens,
                                          const std::vector<float>& prompt_numeric_values,
                                          const std::vector<uint8_t>& prompt_numeric_mask,
                                          const GenerationConfig& cfg,
                                          GenerationStreamCallback* stream_callback,
                                          std::mt19937& rng);
    
private:
    void buildParameterGroups();  // Build parameter groups for optimizer
    std::vector<Vector> addResidual(const std::vector<Vector>& x, const std::vector<Vector>& residual) const;
    
    LanguageModelConfig config_;
    std::unique_ptr<GrimEmbeddingStack> embedder_;
#ifndef USE_CUDA
    std::unique_ptr<GrimEncoder> encoder_;
    std::unique_ptr<LanguageModelHead> lm_head_;
    std::unique_ptr<TextGenerator> generator_;
#endif
    std::unique_ptr<ALiBiPositionalBias> alibi_;
    
#ifdef USE_CUDA
    // GPU runtime ownership (StreamController model - proper typed ownership, no void*)
    std::unique_ptr<EmbeddingRuntime> gpu_embedder_;
    std::unique_ptr<GPUGrimEncoder> gpu_encoder_;
#endif
    
    bool staged_prompt_ready_ = false;
    int staged_prompt_len_ = 0;
    
    GradComponentMetrics grad_metrics_;
    UpdateProbeResult update_probe_result_;
    bool grad_metrics_ready_ = false;
    bool update_probe_ready_ = false;
    float last_grad_scale_ = 1.0f;
    float last_grad_clip_limit_ = 0.0f;
    std::string update_probe_group_name_;
    size_t update_probe_group_index_ = static_cast<size_t>(-1);
    size_t update_probe_sample_elems_ = 0;
    std::vector<float> update_probe_weights_before_;
    std::vector<float> update_probe_weights_after_;
    std::vector<float> update_probe_grad_sample_;

#ifdef USE_CUDA
    TrainingState training_state_;
    std::vector<ParameterGroup> parameter_groups_;  // Parameter groups for optimizer
    uint64_t last_param_group_arch_hash_ = 0;       // Architecture hash when groups were last built
    uint32_t backward_call_count_ = 0;              // Tracks backward() calls for deterministic diagnostics
    LossContext::LossOptions loss_options_{};
    
    // NOTE: Gradient norm computation moved to TrainingState::gradnorm_ctrl (GradNormController)
    // Old buffers (d_grad_norm_sums_, h_grad_norm_sums_) removed per Rule 20
#endif
    
#ifdef USE_CUDA
    std::unique_ptr<Quantization::QuantizationLayer> activation_quantizer_;
    
    // ScratchBlock reasoning layer (togglable)
    std::unique_ptr<ScratchBlockLayer> scratch_block_layer_;
#endif
};

#ifndef USE_CUDA
class EncoderLayer {
public:
    explicit EncoderLayer(const EncoderConfig& config);
    
    std::vector<Vector> forward(const std::vector<Vector>& input,
                                const ALiBiPositionalBias* alibi = nullptr,
                                std::vector<Vector>* kv_cache_k = nullptr,
                                std::vector<Vector>* kv_cache_v = nullptr);
    
    // Accessors for GPU implementation
    FeedForwardNetwork* getFFN() { return ffn_.get(); }
    RMSNorm* getRMSNorm1() { return rms1_.get(); }
    RMSNorm* getRMSNorm2() { return rms2_.get(); }
    
    const FeedForwardNetwork* getFFN() const { return ffn_.get(); }
    const RMSNorm* getRMSNorm1() const { return rms1_.get(); }
    const RMSNorm* getRMSNorm2() const { return rms2_.get(); }
    
private:
    EncoderConfig config_;
    std::unique_ptr<FeedForwardNetwork> ffn_;
    std::unique_ptr<RMSNorm> rms1_;
    std::unique_ptr<RMSNorm> rms2_;
    std::mt19937 rng_;
};

//======================================================//
//  GrimEncoder Class Declaration
//======================================================//

class GrimEncoder {
public:
    struct EncoderStats {
        size_t total_forward_passes = 0;
        size_t total_tokens_processed = 0;
        double avg_tokens_per_pass = 0.0;
    };
    
    explicit GrimEncoder(const EncoderConfig& config);
    
    std::vector<Vector> forward(const std::vector<Vector>& embeddings,
                                const ALiBiPositionalBias* alibi = nullptr);
    
    std::vector<Vector> forwardWithCache(const std::vector<Vector>& embeddings,
                                         const ALiBiPositionalBias* alibi,
                                         std::vector<std::vector<std::vector<Vector>>>& kv_cache);
    
    Vector getPooledOutput(const std::vector<Vector>& encoder_output,
                          const std::string& pooling = "mean");
    
    const EncoderConfig& getConfig() const { return config_; }
    const EncoderStats& getStats() const { return stats_; }
    
    // Accessors for GPU implementation
    int getNumLayers() const { return static_cast<int>(layers_.size()); }
    EncoderLayer* getLayer(int idx) { return layers_[idx].get(); }
    
private:
    EncoderConfig config_;
    std::vector<std::unique_ptr<EncoderLayer>> layers_;
    EncoderStats stats_;
};

//======================================================//
//  LanguageModelHead Class Declaration
//======================================================//

class LanguageModelHead {
public:
    explicit LanguageModelHead(const LMHeadConfig& config);
    
    Vector forward(const Vector& hidden_state);
    std::vector<Vector> forwardBatch(const std::vector<Vector>& hidden_states);
    Vector getProbabilities(const Vector& logits, float temperature = 1.0f);
    std::vector<std::pair<int, float>> getTopK(const Vector& logits, int k);
    
    const LMHeadConfig& getConfig() const { return config_; }
    const Matrix& getWeights() const;
    const Vector& getBias() const { return b_out_; }
    
    void updateWeights(const std::vector<Vector>& grad_W, const Vector& grad_b, float lr);
    
private:
    void matmulSIMD(const float* input, const Matrix& weight, float* output,
                   int input_size, int output_size, const float* bias, bool transpose) const;
    void softmaxSIMD(float* values, int size, float temperature) const;
    
    LMHeadConfig config_;
    bool tie_weights_;
    Matrix W_out_;
    Vector b_out_;
};
#endif // USE_CUDA

//======================================================//
//  TextGenerator Class Declaration
//======================================================//

#ifndef USE_CUDA
class TextGenerator {
public:
    using StreamCallback = std::function<void(int token_id, float score)>;
    
    TextGenerator(GrimEncoder& encoder, LanguageModelHead& lm_head, const GenerationConfig& config);
    
    std::vector<GeneratedSequence> generate(const std::vector<Vector>& prompt_embeddings,
                                           const ALiBiPositionalBias* alibi = nullptr);
    
    GeneratedSequence generateStream(const std::vector<Vector>& prompt_embeddings,
                                    StreamCallback callback,
                                    const ALiBiPositionalBias* alibi = nullptr);
    
    void setConfig(const GenerationConfig& config) { config_ = config; }
    const GenerationConfig& getConfig() const { return config_; }
    
private:
    GeneratedSequence sampleGenerate(const std::vector<Vector>& prompt_embeddings,
                                    const ALiBiPositionalBias* alibi,
                                    StreamCallback* callback = nullptr);
    
    std::vector<GeneratedSequence> beamSearch(const std::vector<Vector>& prompt_embeddings,
                                             const ALiBiPositionalBias* alibi);
    
    int sampleGreedy(const Vector& logits);
    int sampleTopK(const Vector& logits, int k, float temperature);
    int sampleTopP(const Vector& logits, float p, float temperature);
    
    void applyRepetitionPenalty(Vector& logits, const std::vector<int>& generated_ids);
    void applyBadWords(Vector& logits);
    bool shouldStop(const GeneratedSequence& seq, int current_length);
    
    GrimEncoder& encoder_;
    LanguageModelHead& lm_head_;
    GenerationConfig config_;
    std::mt19937 rng_;
};
#else
class TextGenerator;  // Placeholder to keep pointer type for GPU-only build
using StreamCallback = GenerationStreamCallback;
#endif

//======================================================//
//  GPU Classes (Forward Declarations Only)
//======================================================//

#ifdef USE_CUDA
// GPUEmbeddingStack removed - use EmbeddingRuntime from Layers/Embedding/ instead

// Forward declarations for GPU classes
struct EncoderLayerCache {
    // RMSNorm caches
    float* ln1_input = nullptr;        // Input to RMS1 (layer input)
    float* ln1_output = nullptr;       // RMS1 output
    float* attn_input = nullptr;       // Input to attention (same as ln1_output)
    float* ln2_input = nullptr;        // Input to RMS2 (residual1)
    float* ln2_output = nullptr;       // RMS2 output
    
    // Attention caches (BHSD format)
    float* q = nullptr;                // [batch, num_heads, seq, head_dim]
    float* k = nullptr;                // [batch, num_kv_heads, seq, head_dim]
    float* v = nullptr;                // [batch, num_kv_heads, seq, head_dim]
    float* attn_bhsd = nullptr;        // [batch, num_heads, seq, head_dim] - attention output before W_o
    float* softmax_lse = nullptr;      // [batch, num_heads, seq] - FP32 dense LSE (FA v2)
    float* attn_output = nullptr;      // [total_tokens, d_model] - after W_o projection
    
    // Residual
    float* residual1 = nullptr;        // After first residual add
    
    // FFN caches
    float* ffn_input = nullptr;        // FFN input (same as ln2_output)
    float* ffn_pre_gelu = nullptr;     // FFN pre-GELU activations
    float* ffn_output = nullptr;       // FFN output
    
    // Layer output
    float* layer_output = nullptr;     // Final layer output
};

struct FlashAttentionBF16Scratch {
    __nv_bfloat16* q = nullptr;
    __nv_bfloat16* k = nullptr;
    __nv_bfloat16* v = nullptr;
    __nv_bfloat16* out = nullptr;
    size_t q_elems = 0;
    size_t kv_elems = 0;
};
class EncodingLayer;
using GPUEncoderLayer = EncodingLayer;

// GPUGrimEncoder - Container for encoder layers, manages layer lifecycle
// Forward pass logic is in ForwardPhase2_Encoder.cu::runFullEncoder()
// This class only owns layers and provides access to them
class GPUGrimEncoder {
public:
    explicit GPUGrimEncoder(const EncoderConfig& config);
    
    // Access to layers for training/forward pass
    GPUEncoderLayer* getLayer(int index);
    const GPUEncoderLayer* getLayer(int index) const;
    int getNumLayers() const;
    
    // Configure Flash Attention for all layers
    void setFlashAttention(bool enable, int min_seq_len = 128);
    
private:
    struct Impl;
    Impl* pImpl = nullptr;
};

#endif // USE_CUDA

} // namespace GRIM

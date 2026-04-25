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

// TensorContract - Autograd system (includes ParamGroupType, ParameterGroup, Tensor)
#include "../Shared/TensorContract/TensorContract_GPU.hpp"

// BatchPayload - Single source of truth for per-batch metadata
#include "../Shared/Batching/BatchPayload.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Layers/LMHead/lm_head_GPU.hpp"
#include "../Layers/ReasoningHead/reasoning_head_GPU.hpp"
#include "../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../Shared/Execution/DecodeTimeNumPolicy.hpp"
#include "../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"
#include "../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../Shared/Loss/LossContext/LossContext.hpp"
#endif

namespace GRIM {

//======================================================//
//  Forward Declarations
//======================================================//

// GPUGrimEncoder is defined later in this file but used by LanguageModel class
class GPUGrimEncoder;

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
//  Architecture fields (d_model, num_heads, etc.) are inherited
//  from HyperParameters::ModelArchitecture — the ONLY source of truth.
//  DO NOT redeclare architecture fields here.
//======================================================//

// ModelExecutionMode moved to HyperParameters::ModelExecutionMode
// (Shared/HyperParameters/HyperParameters_GPU.hpp). HyperParameters is the
// single source of truth for all model configuration; this header is the
// model class declaration only.

struct EncoderConfig : public HyperParameters::ModelArchitecture {
    // Architecture fields (d_model, num_heads, num_kv_heads, head_dim, d_ff,
    // num_layers, max_seq_len, dropout_rate, attention_dropout, positional_encoding,
    // tie_embeddings) inherited from HyperParameters::ModelArchitecture

    // Cache limits
    int max_cached_batch = 0;
    int max_cached_seq_len = 0;
    
    // Fixed config values (not architecture-dependent)
    // Issue #104 FIX: Changed from 1e-3 to 1e-5. The old value was too large for Layer 0 embeddings:
    // - Layer 0 input: per_row_rms ≈ 0.006, so mean(x²) ≈ 0.00004
    // - With eps=1e-3: epsilon dominated denominator (25x larger than mean(x²))
    // - Result: output_rms = 0.19 instead of expected 1.0
    // Standard practice (LLaMA, Mistral, GPT): eps = 1e-5 to 1e-6
    float rms_epsilon = 1e-5f;  // RMSNorm epsilon - standard value matching LLaMA/Mistral
    bool causal_mask = true;                        
    bool use_pre_norm = true;
    bool fuse_qkv = true;
    bool use_simd = true;
    int num_threads = 4;
    
    // Flash Attention settings
    bool use_flash_attention = true;  // Use Flash Attention 2 for memory efficiency
    int min_seq_len_for_flash = 0;    // REQUIRED - set from hyperparameters (no defaults)
    
    // Positional encoding (ALiBi+RoPE hybrid) - pointer to shared state in TrainingState
    // WARNING: If nullptr, attention sees no positional info - positions become equivalent!
    const PBM::PBMSpec* pos_encoding = nullptr;
    
    // LayerScale (Issue #109/#129 fix - config propagation)
    // These fields were missing, causing reliance on EncoderLayerConfig defaults
    bool use_layer_scale = false;        // Enable per-sublayer learnable scaling
    float layer_scale_init = 1.0f;       // Issue #129: init=1.0 (NOT CaiT's 0.1 — caused 10x gradient attenuation)
    
    // Per-layer residual centering (Issue #126 fix - can be disabled to improve gradient signal)
    // When true, applies center_columns after each residual add in every encoder layer (24 total).
    // This prevents mode collapse but attenuates gradient signal through 24 centering projections.
    // When false, only the LM head centering (center_hidden_states) helps prevent mode collapse.
    bool center_encoder_residuals = false;
    
    // Bias control - when false, encoder layers skip bias addition (b_qkv, b_o not used)
    bool use_bias = true;

    // QK-Norm: RMSNorm applied to Q and K projections before attention scoring
    bool qk_norm_enabled = false;
    
    // Layer self-allocation (Pattern B): seed and scaling config for weight initialization
    uint64_t weight_seed = 0;         // Base seed for Xavier init (per-layer offset: seed + layer*10)
    float residual_scale = 1.0f;      // Issue #142: 1/sqrt(2*num_layers) for W_o and W2
    float layer_scale_init_value = 1.0f;  // Issue #109: CaiT LayerScale initial scalar value
    
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
    float epsilon = 1e-5f;
};

// SamplingStrategy / GenerationConfig / GenerationStreamCallback /
// LanguageModelConfig / ModelExecutionMode
// are defined in Shared/HyperParameters/HyperParameters_GPU.hpp
// (single source of truth). Refer to them as
// `GRIM::HyperParameters::LanguageModelConfig`, etc.

// GPU Configuration - FULL definition (source of truth)
struct GPUConfig {
    bool use_tensor_cores = true;        // Enable FP16 Tensor Core ops
    bool use_cuda_graphs = true;         // Use CUDA graphs for repeated inference
    bool use_pinned_memory = true;       // Pinned host buffers
    bool use_dynamic_batching = true;    // Merge micro-batches
    int max_batch_size = 1024;            // Max dynamic batch size
    int micro_batch_size = 64;           // Micro-batch for long sequences
    cudaStream_t stream = 0;             // CUDA stream for async ops
    int device_id = 0;                   // CUDA device to use
};

//======================================================//
//  Forward Declarations
//======================================================//

// Positional Encoding Type - re-exported from HyperParameters_GPU.hpp
using HyperParameters::PositionalEncodingType;

class ALiBiPositionalBias {
private:

    PBM::PBMState pbm_state_{};   // Unified PBM state (both ALiBi + RoPE)
    int num_heads;                // Number of attention heads
    bool initialized;             // Initialization status
    PositionalEncodingType type;  // Encoding type

public:
    
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
        int max_seq_len,
        PositionalEncodingType type
    );


    float* getSlopes() const;
    float* getRoPEFreqs() const;
    bool isInitialized() const;
    PositionalEncodingType getType() const { return type; }
    const PBM::PBMState& getPBMState() const { return pbm_state_; }
    void cleanup();
};

// GrimEmbeddingStack - minimal interface
// NOTE: Position embeddings are initialized directly on GPU in TrainingOps.cu
class GrimEmbeddingStack {
public:
    GrimEmbeddingStack(int vocab_size, int d_model, int max_seq_len);
    
    // num_kv_heads REQUIRED - only GQA is supported
    void enableALiBi(int num_heads, int num_kv_heads, int max_seq_len);
    void enableHybridPositionalEncoding(int num_heads, int d_head, int num_kv_heads, int max_seq_len);
    const ALiBiPositionalBias* getALiBiBias() const;
    const Matrix& getTokenEmbeddings() const;
    
    // Public members needed by GPU code
    Matrix token_embed;        // Token embedding matrix [vocab_size x d_model]
    // NOTE: Position embeddings are GPU-only (initialized in TrainingOps.cu)
    
private:
    int vocab_size_;
    int d_model_;
    int max_seq_len_;
    std::unique_ptr<ALiBiPositionalBias> alibi_;
};

//======================================================//
//  Generated Sequence
//======================================================//

struct GeneratedSequence {
    std::vector<int> token_ids;
    std::vector<float> token_scores;
    std::vector<float> token_numeric_values;
    std::vector<uint8_t> token_atom_mask;
    /// Per-token execution slot id (-1 = non-state-bearing); mirrors BatchPayload::token_to_slot_map
    std::vector<int32_t> token_to_slot_map;
    std::shared_ptr<const GRIM::Tokenizer::AtomTable> context_atom_table;  // Atom registry from context (null for generated tokens)
    std::vector<uint32_t> atom_entry_ids;  // Per-token atom entry IDs (kAtomEntryNone = no atom)
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
class TextGenerator;
struct TrainingState;  // Forward declare for methods that return references

// Forward declare LossContext namespace and nested struct
namespace LossContext {
    struct LossOptions;
}

//======================================================//
//  Parameter Group for Training
//  (Defined in TensorContract_GPU.hpp - included above)
//======================================================//
// ParamGroupType and ParameterGroup are now part of the
// unified autograd system in TensorContract_GPU.hpp

// Forward declare TokenBufferView for method signatures
struct TokenBufferView;

#ifdef USE_CUDA
struct TokenBufferView {
    int* device_token_ids = nullptr;
    float* device_token_numeric_values = nullptr;
    uint8_t* device_token_atom_mask = nullptr;
    int32_t* device_token_to_slot_map = nullptr;
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
        size_t scratchblock_params = 0;  // Atom type embeddings + projection
        float model_size_mb = 0.0f;
    };

    // Constructor / Destructor
    explicit LanguageModel(const HyperParameters::LanguageModelConfig& config);
    ~LanguageModel();
    
    // Main API
    Vector forward(const std::vector<int>& token_ids,
                   const std::vector<float>& token_numeric_values,
                   const std::vector<uint8_t>& token_atom_mask,
                   const std::vector<int32_t>& token_to_slot_map = {});
    Vector getNextTokenLogits(const std::vector<int>& context_tokens,
                              const std::vector<float>& context_numeric_values,
                              const std::vector<uint8_t>& context_atom_mask,
                              const std::vector<int32_t>& token_to_slot_map = {});
    std::vector<GeneratedSequence> generate(const std::vector<int>& prompt_tokens,
                                            const std::vector<float>& prompt_numeric_values,
                                            const std::vector<uint8_t>& prompt_atom_mask,
                                            const HyperParameters::GenerationConfig* gen_config = nullptr,
                                            std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table = nullptr,
                                            const std::vector<uint32_t>& prompt_atom_entry_ids = {},
                                            const std::vector<int32_t>& prompt_token_to_slot_map = {});
    GeneratedSequence generateStream(const std::vector<int>& prompt_tokens,
                                     const std::vector<float>& prompt_numeric_values,
                                     const std::vector<uint8_t>& prompt_atom_mask,
                                     HyperParameters::GenerationStreamCallback callback,
                                     const HyperParameters::GenerationConfig* gen_config = nullptr,
                                     std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table = nullptr,
                                     const std::vector<uint32_t>& prompt_atom_entry_ids = {},
                                     const std::vector<int32_t>& prompt_token_to_slot_map = {});
    
    // Training
    float computeLossBatch(const GRIM::Batching::BatchPayload& payload,
                           bool is_training = true);
    
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
                       const std::vector<uint8_t>& prompt_atom_mask,
                       const std::vector<int32_t>& prompt_token_to_slot_map = {});
    
    // Process a single new token using cached K,V (decode phase)  
    // Returns logits for this token position (ready for next sampling)
    // Appends new token to cached sequence and recomputes full forward pass
    Vector forwardStep(int new_token, float numeric_value, uint8_t atom_mask,
                       int32_t new_token_slot_id = -1);

    // Ensure KV cache + decode scratch buffers are allocated.
    // Safe to call repeatedly — skips if already allocated.
    // Required before any incremental generation (prefill + decode).
    void ensureKVCacheAllocated();

    // Clear KV cache (call before starting new generation)
    void resetKVCache();
    
    // Get current KV cache length (number of tokens with cached K,V)
    int getKVCacheLength() const;
    
    void initCuBLASHandle();   // Initialize cuBLAS handle only (MUST be called before initGPU)
    void initPBM();            // Initialize PBM (ALiBi+RoPE hybrid) - MUST be called before initGPU
    void initTrainingState();  // Initialize training state (allocate GPU buffers + gradients)
    void initInferenceState(); // Initialize inference state (allocate GPU buffers WITHOUT gradients)
    // backward() and zeroGrad() DELETED (Rule 26).
    // Backward: Use autogradTrainingStep() which does forward+loss+backward.
    // Zeroing: executeAutogradBackward() zeros all gradients when accumulate=false.
    // updateWeights(), resetOptimizerMoments(), scaleOptimizerMoments() MOVED to
    // AdamW_Kernal_GPU.{hpp,cu} as free functions: launchAdamWStep(), resetAdamWMoments(),
    // scaleAdamWMoments(). AdamW stepping is training infrastructure, not model logic.
    // computeGradNorm(), scaleGradientsByType(), recordGradientClip() DELETED (Rule 26).
    // Phase2 calls GradNorm::measureGradientNorms() + launchScaleGradients() directly.
    
#ifdef USE_CUDA
    void setLossOptions(const LossContext::LossOptions& opts) { loss_options_ = opts; }
    const LossContext::LossOptions& getLossOptions() const { return loss_options_; }

    // Training state access (for debugging/diagnostics)
    const TrainingState& getTrainingState() const { return training_state_; }
    TrainingState& getTrainingState() { return training_state_; }
    
    // Parameter groups accessor (for direct gradient norm / clipping in Phase2)
    const std::vector<ParameterGroup>& parameterGroups() const { return parameter_groups_; }
    std::vector<ParameterGroup>& parameterGroups() { return parameter_groups_; }
    // Build parameter groups for optimizer (must be called during init so Phase2 grad-norm has max_groups > 0)
    void buildParameterGroups();
#endif

    // Utilities
    ModelStats getModelStats() const;
    bool save(const std::string& path);
    bool load(const std::string& path);
    
    // Config access
    const HyperParameters::LanguageModelConfig& getConfig() const { return config_; }
    
    // GPU methods
    void initGPU();
    Vector forwardGPU(const std::vector<int>& token_ids,
                      const std::vector<float>& token_numeric_values,
                      const std::vector<uint8_t>& token_atom_mask,
                      const std::vector<int32_t>& token_to_slot_map = {});
    Vector getNextTokenLogitsGPU(const std::vector<int>& context_tokens,
                                 const std::vector<float>& context_numeric_values,
                                 const std::vector<uint8_t>& context_atom_mask,
                                 const std::vector<int32_t>& token_to_slot_map = {});
    TokenBufferView getTokenBufferView();
    void markDevicePromptReady(int token_count);
    
    // Helper accessors for GPU implementation
    GrimEmbeddingStack* getEmbedderPtr() { return embedder_.get(); }
    
#ifdef USE_CUDA
    // GPU runtime accessors - return references to owned objects (fail loud if not initialized)
    GPUGrimEncoder& getGpuEncoder();
    const GPUGrimEncoder& getGpuEncoder() const;

    
    // ScratchBlock reasoning layer access
    ScratchBlockLayer* getScratchBlockLayer() { return scratch_block_layer_.get(); }
    const ScratchBlockLayer* getScratchBlockLayer() const { return scratch_block_layer_.get(); }
    void setScratchBlockEnabled(bool enabled);
    bool isScratchBlockEnabled() const;

    // Embedding layer access (Pattern B: persistent, self-allocating, owns token + pos weights)
    EmbeddingLayer* getEmbeddingLayer() { return embedding_layer_.get(); }
    const EmbeddingLayer* getEmbeddingLayer() const { return embedding_layer_.get(); }

    // LM Head layer access (Pattern B: persistent, self-allocating)
    LMHeadLayer* getLmHeadLayer() { return lm_head_layer_.get(); }
    const LMHeadLayer* getLmHeadLayer() const { return lm_head_layer_.get(); }

    // Reasoning Head layer access (nullptr when disabled)
    ReasoningHeadLayer* getReasoningHeadLayer() { return reasoning_head_layer_.get(); }
    const ReasoningHeadLayer* getReasoningHeadLayer() const { return reasoning_head_layer_.get(); }

    // Execution Block layer access (nullptr when disabled)
    ExecutionBlockLayer* getExecutionBlockLayer() { return execution_block_layer_.get(); }
    const ExecutionBlockLayer* getExecutionBlockLayer() const { return execution_block_layer_.get(); }

    // Decode-time slot selector layer (nullptr when disabled)
    DecodeTimeSlotSelectorLayer* getDecodeTimeSlotSelectorLayer() { return decode_time_slot_selector_layer_.get(); }
    const DecodeTimeSlotSelectorLayer* getDecodeTimeSlotSelectorLayer() const { return decode_time_slot_selector_layer_.get(); }

    // Decode-time <NUM> policy (nullptr when selector disabled)
    DecodeTimeNumPolicy* getDecodeTimeNumPolicy() { return decode_time_num_policy_.get(); }
    const DecodeTimeNumPolicy* getDecodeTimeNumPolicy() const { return decode_time_num_policy_.get(); }

    // Multi-token prediction (MTP) auxiliary heads - one weight + bias per head (indices 0..mtp_k-1)
    struct MTPHead {
        Tensor weight;  // [vocab_size, d_model] same layout as LM head
        Tensor bias;    // [vocab_size]
    };
    int getMtpK() const { return config_.mtp_enabled ? config_.mtp_k : 0; }
    MTPHead* getMtpHead(int k);
    const MTPHead* getMtpHead(int k) const;
    
    // Scratch pool configuration (pinned memory transfers)
    void configureScratchPool(bool enabled);
    bool isScratchPoolInitialized() const;
#endif
    
    GeneratedSequence generateSequenceGPU(const std::vector<int>& prompt_tokens,
                                          const std::vector<float>& prompt_numeric_values,
                                          const std::vector<uint8_t>& prompt_atom_mask,
                                          const HyperParameters::GenerationConfig& cfg,
                                          HyperParameters::GenerationStreamCallback* stream_callback = nullptr,
                                          std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table = nullptr,
                                          const std::vector<uint32_t>& prompt_atom_entry_ids = {},
                                          const std::vector<int32_t>& prompt_token_to_slot_map = {});
    
private:
    // Core inference forward: assumes data already in cached_* tensors.
    // Runs autograd forward, extracts last-token logits, returns them.
    // All public inference methods (forwardGPU, getNextTokenLogitsGPU,
    // forwardInit, forwardStep) copy their data to cached tensors then call this.
    // When populate_kv_cache=true, extracts K,V from autograd intermediates
    // into BF16 KV cache buffers before clearing intermediates.
    Vector executeInferenceForward_(int seq_len, bool populate_kv_cache = false);

    // KV-cached decode: processes a single token at position token_pos
    // through all encoder layers using cached K,V from prior tokens.
    // Returns logits vector for the new token.
    Vector executeDecodeForward_(int token_pos);

    HyperParameters::LanguageModelConfig config_;
    std::unique_ptr<GrimEmbeddingStack> embedder_;
    
#ifdef USE_CUDA
    // GPU runtime ownership (StreamController model - proper typed ownership, no void*)
    std::unique_ptr<GPUGrimEncoder> gpu_encoder_;
#endif
    
    bool staged_prompt_ready_ = false;
    int staged_prompt_len_ = 0;

#ifdef USE_CUDA
    TrainingState training_state_;
    std::vector<ParameterGroup> parameter_groups_;  // Parameter groups for optimizer
    uint32_t backward_call_count_ = 0;              // Tracks backward() calls for deterministic diagnostics
    LossContext::LossOptions loss_options_{};
    
    // ScratchBlock reasoning layer (togglable)
    std::unique_ptr<ScratchBlockLayer> scratch_block_layer_;

    // Embedding layer (Pattern B: persistent, self-allocating, owns token + position weights)
    std::unique_ptr<EmbeddingLayer> embedding_layer_;

    // LM Head layer (Pattern B: persistent, self-allocating, owns final_rms_gamma)
    std::unique_ptr<LMHeadLayer> lm_head_layer_;

    // Reasoning Head layer (structured reasoning: op + arg selection over atoms)
    std::unique_ptr<ReasoningHeadLayer> reasoning_head_layer_;

    // Execution Block layer (differentiable register machine — internal numeric reasoning)
    std::unique_ptr<ExecutionBlockLayer> execution_block_layer_;

    // Decode-time slot selector (trainable pointer-selector for <NUM> binding)
    std::unique_ptr<DecodeTimeSlotSelectorLayer> decode_time_slot_selector_layer_;

    // Decode-time <NUM> policy (candidate construction + bind-or-mask decisions)
    std::unique_ptr<DecodeTimeNumPolicy> decode_time_num_policy_;

    // Multi-token prediction (MTP) auxiliary heads - K independent linear heads (not tied to embedding)
    std::vector<MTPHead> mtp_heads_;
#endif
};

using StreamCallback = HyperParameters::GenerationStreamCallback;

//======================================================//
//  GPU Classes (Forward Declarations Only)
//======================================================//

#ifdef USE_CUDA
// Forward declarations for GPU classes
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
    void setFlashAttention(bool enable, int min_seq_len);
    
private:
    struct Impl;
    Impl* pImpl = nullptr;
};

#endif // USE_CUDA

} // namespace GRIM

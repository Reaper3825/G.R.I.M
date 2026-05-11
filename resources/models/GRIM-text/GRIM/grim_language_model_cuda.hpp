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

// BatchDeviceBindings - Explicit device pointers per step (replaces the old
// `mutable d_*` fields that used to live on BatchPayload).
#include "../Shared/Batching/BatchDeviceBindings.hpp"

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
#include "../Shared/InferenceState/GenerationState_GPU.hpp"
#include "../Shared/Loss/LossContext/LossContext.hpp"
#endif

namespace GRIM {

//======================================================//
//  Forward Declarations
//======================================================//

// GPUGrimEncoder is defined later in this file but used by LanguageModel class
class GPUGrimEncoder;
namespace HyperParameters { struct EncoderLayerConstructionHP; }

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

//======================================================//
//  Configuration ownership
//
//  All model hyperparameters live in HyperParameters_GPU.hpp:
//    - ModelArchitecture        (d_model, num_heads, ...)
//    - LanguageModelConfig      (full model + feature toggles, inherits ModelArchitecture)
//    - GenerationConfig         (sampling / decoding)
//    - SamplingStrategy
//    - ModelExecutionMode       (TRAINING vs INFERENCE)
//
//  This header MUST NOT redeclare any of those fields. The encoder
//  consumes grouped construction views derived from LanguageModelConfig;
//  the only thing genuinely owned here is the construction-binding struct below —
//  borrowed startup resources that are created at initGPU() time and have
//  no place in a config object or per-forward request.
//======================================================//

#ifdef USE_CUDA
/// Startup device bindings for GPUGrimEncoder construction.
///
/// These are NOT hyperparameters. They are startup/model-assembly inputs.
/// Forward-time stream/cuBLAS handles are carried by the forward payload/request
/// (`AutogradContext` / `Forward::ModelForwardRequest`), never stored on layers.
struct EncoderConstructionBindings {
    /// Shared positional-bias state (ALiBi/RoPE), owned by TrainingState.
    /// REQUIRED: encoder construction throws if null.
    const PBM::PBMSpec* pos_encoding = nullptr;

    /// CUDA stream for startup self-allocation only.
    cudaStream_t init_stream = nullptr;
};
#endif

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

// OptimizerStep (AdamW/RAdamW step counter) lives in
// Shared/Optimizers/OptimizerStep.hpp — it is step bookkeeping, not model state.
// Consumers should include that header directly.

//======================================================//
//  Forward Declarations - Classes Defined Later
//======================================================//

class EncoderLayer;
class TextGenerator;
struct TrainingState;  // Forward declare for methods that return references
struct OptimizerState;

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
    LanguageModel(const HyperParameters::LanguageModelConfig& config,
                  const Config::TrainingHyperparameters& training_hyperparameters);
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
    
    // Training / Eval
    //
    // Two-step contract (Phase2 sync slice):
    //   1) auto bindings = model.uploadBatchToDevice(payload);
    //   2) float loss     = model.computeLossBatch(payload, bindings, is_training);
    //
    // uploadBatchToDevice() performs the H2D copies into TrainingState's
    // reusable cache buffers and returns a BatchDeviceBindings that names the
    // resulting device pointers. computeLossBatch() never writes through
    // payload (BatchPayload is host-only and immutable); device addresses are
    // read only via `bindings`.
    GRIM::Batching::BatchDeviceBindings uploadBatchToDevice(
        const GRIM::Batching::BatchPayload& payload);

    float computeLossBatch(const GRIM::Batching::BatchPayload& payload,
                           const GRIM::Batching::BatchDeviceBindings& bindings,
                           bool is_training = true);
    
    // =========================================================================
    // AUTOREGRESSIVE GENERATION API
    // =========================================================================
    // Use these for token-by-token generation:
    //   1. Call forwardInit() once with the prompt tokens.
    //   2. Call forwardStep() for each sampled token.
    //      - Sequence-local configs use KV-cached single-token decode.
    //      - Sequence-coupled geometry (encoder residual centering, LM-head
    //        hidden centering, PC1 projection) recomputes the full current
    //        sequence so sequence-wise means/projections are mathematically valid.
    //   3. Call resetKVCache() before starting a new generation session.
    // =========================================================================
    
    // Initialize KV cache with prompt tokens (prefill phase)
    // Returns logits for the last prompt token (ready for first sampling)
    Vector forwardInit(const std::vector<int>& prompt_tokens,
                       const std::vector<float>& prompt_numeric_values,
                       const std::vector<uint8_t>& prompt_atom_mask,
                       const std::vector<int32_t>& prompt_token_to_slot_map = {});
    
    // Process one sampled token and return logits for the next sampling step.
    // Uses KV decode only when the active config has no sequence-coupled geometry;
    // otherwise appends the token and reruns the full current sequence.
    Vector forwardStep(int new_token, float numeric_value, uint8_t atom_mask,
                       int32_t new_token_slot_id = -1);

    // Ensure KV cache + decode scratch buffers are allocated.
    // Safe to call repeatedly — skips if already allocated.
    // Required before any incremental generation (prefill + decode).
    void ensureKVCacheAllocated();

    // Clear KV cache (call before starting new generation)
    void resetKVCache();
    
    // Get current generation length. For sequence-local configs this is also
    // the number of tokens with cached K,V; for full-context configs it is only
    // the committed sequence length.
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
    const PBM::PBMSpec& getPBMSpec() const;
    bool isPBMInitialized() const;
    
    // Parameter groups accessor (for direct gradient norm / clipping in Phase2)
    const std::vector<ParameterGroup>& parameterGroups() const { return parameter_groups_; }
    std::vector<ParameterGroup>& parameterGroups() { return parameter_groups_; }
    // Build parameter groups for optimizer (must be called during init so Phase2 grad-norm has max_groups > 0)
    void buildParameterGroups();
    void bindOptimizerState(OptimizerState& optimizer_state, cudaStream_t stream);
#endif

    // Utilities
    ModelStats getModelStats() const;
    bool save(const std::string& path);
    bool load(const std::string& path);
    
    // Config access
    const HyperParameters::LanguageModelConfig& getConfig() const { return config_; }
    // Non-owning Phase1 snapshot; null for inference/test callers that only
    // construct from LanguageModelConfig.
    const Config::TrainingHyperparameters* getTrainingHyperparameters() const { return training_hyperparameters_; }
    bool hasTrainingHyperparameters() const { return training_hyperparameters_ != nullptr; }
    const Config::TrainingHyperparameters& requireTrainingHyperparameters(const char* caller) const;
    
    // GPU methods
    void initGPU(uint64_t weight_init_seed);
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
    LanguageModel(const HyperParameters::LanguageModelConfig& config,
                  const Config::TrainingHyperparameters* training_hyperparameters);

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
    // Phase1 owns the StartupConfig/TrainingHyperparameters for the lifetime of
    // TrainingContext; LanguageModel only keeps a read-only handle to avoid
    // re-slicing training-only knobs into LanguageModelConfig.
    const Config::TrainingHyperparameters* training_hyperparameters_ = nullptr;
    std::unique_ptr<GrimEmbeddingStack> embedder_;
    
#ifdef USE_CUDA
    // GPU runtime ownership (StreamController model - proper typed ownership, no void*)
    std::unique_ptr<GPUGrimEncoder> gpu_encoder_;
#endif
    
    bool staged_prompt_ready_ = false;
    int staged_prompt_len_ = 0;

#ifdef USE_CUDA
    TrainingState training_state_;
    GenerationState generation_state_;
    std::vector<ParameterGroup> parameter_groups_;  // Parameter groups for optimizer
    uint32_t backward_call_count_ = 0;              // Tracks backward() calls for deterministic diagnostics
    LossContext::LossOptions loss_options_{};
    PBM::PBMSpec pbm_spec_{};                       // Non-owning view into GrimEmbeddingStack PBM state
    bool pbm_spec_initialized_ = false;
    
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

// GPUGrimEncoder - Container for encoder layers, manages layer lifecycle.
// Forward pass logic is in ForwardPhase2_Encoder.cu::runFullEncoder().
// This class only owns layers and provides access to them.
//
// Constructor takes the grouped encoder hyperparameter view and startup model-
// assembly bindings. Forward pass runtime handles live on the forward request.
class GPUGrimEncoder {
public:
    GPUGrimEncoder(const HyperParameters::EncoderLayerConstructionHP& config,
                   const EncoderConstructionBindings& bindings,
                   uint64_t weight_seed);

    GPUEncoderLayer* getLayer(int index);
    const GPUEncoderLayer* getLayer(int index) const;
    int getNumLayers() const;

private:
    struct Impl;
    Impl* pImpl = nullptr;
};

#endif // USE_CUDA

} // namespace GRIM

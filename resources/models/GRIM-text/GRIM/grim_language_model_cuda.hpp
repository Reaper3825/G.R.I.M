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
#include <cstdint>
#include <cstddef>

// HyperParameters - Single source of truth for model configuration
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"

// Hyperparameter groupings - construction/read views derived from LanguageModelConfig
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"

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
#include "../Shared/PBM/PBMStateOwner.hpp"
#include "../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../Shared/InferenceState/GenerationState_GPU.hpp"
#endif

namespace GRIM {
class LanguageModel;
}

namespace GRIMText {
namespace Training {
namespace Startup {
struct ModelAssemblyAccess;
} // namespace Startup
} // namespace Training
} // namespace GRIMText

namespace GRIM {

//======================================================//
//  Forward Declarations
//======================================================//

// GPUGrimEncoder is defined later in this file but used by LanguageModel class
class GPUGrimEncoder;

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

//======================================================//
//  Configuration ownership
//
//  All model hyperparameters live in HyperParameters_GPU.hpp:
//    - LanguageModelConfig      (architecture fields + full model feature toggles)
//    - SamplingStrategy
//    - ModelExecutionMode       (TRAINING vs INFERENCE)
//  Generation callsites consume GenerationHP grouped views derived from
//  LanguageModelConfig, never a second config owner.
//
//  This header MUST NOT redeclare any of those fields. The encoder
//  consumes grouped construction views derived from LanguageModelConfig;
//  the only thing genuinely owned here is the construction-binding struct below —
//  borrowed startup resources that are created during startup GPU model assembly and have
//  no place in a config object or per-forward request.
//======================================================//

#ifdef USE_CUDA
/// Startup device bindings for GPUGrimEncoder construction.
///
/// These are NOT hyperparameters. They are startup/model-assembly inputs.
/// Forward-time stream/cuBLAS handles are carried by the forward payload/request
/// (`AutogradContext` / `Forward::ModelForwardRequest`), never stored on layers.
struct EncoderConstructionBindings {
    /// Shared positional-bias state (ALiBi/RoPE), owned by the model PBM RAII owner.
    /// REQUIRED: encoder construction throws if null.
    const PBM::PBMSpec* pos_encoding = nullptr;

    /// CUDA stream for startup self-allocation only.
    cudaStream_t init_stream = nullptr;
};
#endif

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
};

// OptimizerStep (AdamW/RAdamW step counter) lives in
// Shared/Optimizers/OptimizerStep.hpp — it is step bookkeeping, not model state.
// Consumers should include that header directly.

//======================================================//
//  Parameter Group for Training
//  (Defined in TensorContract_GPU.hpp - included above)
//======================================================//
// ParamGroupType and ParameterGroup are now part of the
// unified autograd system in TensorContract_GPU.hpp

//======================================================//
//  LanguageModel Class Declaration
//======================================================//

class LanguageModel {
public:
    // Constructor / Destructor
    explicit LanguageModel(const HyperParameters::LanguageModelConfig& config);
    ~LanguageModel();

    // backward() and zeroGrad() DELETED (Rule 26).
    // Backward: Phase2 runs explicit shared forward, then
    // GRIM::Autograd::computeAutogradLoss() + executeAutogradBackward().
    // Zeroing: executeAutogradBackward() zeros registered ParameterGroup gradients
    // through TensorContract when accumulate=false.
    // updateWeights(), resetOptimizerMoments(), scaleOptimizerMoments() MOVED to
    // AdamW_Kernal_GPU.{hpp,cu} as free functions: launchAdamWStep(), resetAdamWMoments(),
    // scaleAdamWMoments(). AdamW stepping is training infrastructure, not model logic.
    // computeGradNorm(), scaleGradientsByType(), recordGradientClip() DELETED (Rule 26).
    // Phase2 calls GradNorm::measureGradientNorms() + launchScaleGradients() directly.
    
#ifdef USE_CUDA
    // Training state access (for debugging/diagnostics)
    const TrainingState& getTrainingState() const { return training_state_; }
    TrainingState& getTrainingState() { return training_state_; }
    const GenerationState& getGenerationState() const { return generation_state_; }
    GenerationState& getGenerationState() { return generation_state_; }
    const PBM::PBMSpec& getPBMSpec() const;
    const PBM::PBMState& getPBMState() const;
    bool isPBMInitialized() const;
    
    // Parameter groups accessor (for direct gradient norm / clipping in Phase2)
    const std::vector<ParameterGroup>& parameterGroups() const { return parameter_groups_; }
    std::vector<ParameterGroup>& parameterGroups() { return parameter_groups_; }
#endif

    // Utilities
    bool save(const std::string& path);
    bool load(const std::string& path);
    
    // Config access
    const HyperParameters::LanguageModelConfig& getConfig() const { return config_; }
    
#ifdef USE_CUDA
    // GPU runtime accessors - return references to owned objects (fail loud if not initialized)
    GPUGrimEncoder& getGpuEncoder();
    const GPUGrimEncoder& getGpuEncoder() const;

    
    // ScratchBlock reasoning layer access. Presence is config-gated by
    // HyperParameters::scratchBlockConstructionHP(config_); do not runtime-toggle it.
    ScratchBlockLayer* getScratchBlockLayer() { return scratch_block_layer_.get(); }
    const ScratchBlockLayer* getScratchBlockLayer() const { return scratch_block_layer_.get(); }

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
    
private:
    friend struct GRIMText::Training::Startup::ModelAssemblyAccess;

    HyperParameters::LanguageModelConfig config_;
    
#ifdef USE_CUDA
    // GPU runtime ownership (StreamController model - proper typed ownership, no void*)
    std::unique_ptr<GPUGrimEncoder> gpu_encoder_;
#endif
    
#ifdef USE_CUDA
    TrainingState training_state_;
    GenerationState generation_state_;
    PBM::PBMStateOwner pbm_owner_;                   // Durable model-level ALiBi/RoPE buffers
    std::vector<ParameterGroup> parameter_groups_;  // Parameter groups for optimizer
    PBM::PBMSpec pbm_spec_{};                       // Non-owning buffer/event view into pbm_owner_
    bool pbm_spec_initialized_ = false;
    
    // ScratchBlock reasoning layer (config-gated durable topology)
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
    GPUGrimEncoder(const HyperParameters::EncoderLayerConstructionHP& hp,
                   const EncoderConstructionBindings& bindings,
                   uint64_t weight_seed);

    GPUEncoderLayer* getLayer(int index);
    const GPUEncoderLayer* getLayer(int index) const;

private:
    struct Impl;
    Impl* pImpl = nullptr;
};

#endif // USE_CUDA

} // namespace GRIM

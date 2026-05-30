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
#include <stdexcept>

// HyperParameters - single entry point for model configuration snapshot helpers
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"

// Hyperparameter groupings - construction/read views derived from AiConfigSnapshot
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
#include "../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../Shared/Execution/DecodeTimeNumPolicy.hpp"
#include "../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"
#include "../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../Shared/InferenceState/GenerationState_GPU.hpp"
#include "../training/Phases/Startup/Model/ModelGpuState.hpp"
#endif

namespace GRIM {
class LanguageModel;
namespace PBM {
class PBMStateOwner;
}
}

namespace GRIMText {
namespace Training {
namespace Startup {
namespace ModelRegistration::ParameterRegistry {
struct StartupParameterRegistry;
}
void initializeCuBLASHandle(::GRIM::TrainingState& training_state);
void initializePBM(const ::GRIM::Config::AiConfigSnapshot& model_cfg, ::GRIM::TrainingState& training_state, ::GRIM::PBM::PBMStateOwner& pbm_owner);
void assembleGpuModel(const ::GRIM::Config::AiConfigSnapshot& model_cfg, ::GRIM::TrainingState& training_state, const ::GRIM::PBM::PBMStateOwner& pbm_owner, GpuModelState& gpu_model_state, ModelRegistration::ParameterRegistry::StartupParameterRegistry& parameter_registry, std::uint64_t weight_init_seed);
void initializeTrainingRuntime(::GRIM::TrainingState& training_state, const ::GRIM::PBM::PBMStateOwner& pbm_owner);
void initializeInferenceRuntime(const ::GRIM::Config::AiConfigSnapshot& model_cfg, ::GRIM::TrainingState& training_state, ::GRIM::GenerationState& generation_state, const ::GRIM::PBM::PBMStateOwner& pbm_owner, const GpuModelState& gpu_model_state, const ModelRegistration::ParameterRegistry::StartupParameterRegistry& parameter_registry);
} // namespace Startup
} // namespace Training
} // namespace GRIMText

namespace GRIM {

//======================================================//
//  Forward Declarations
//======================================================//

// Concrete encoder layer type lives in Layers/Encoding/Encoding_GPU.hpp.
// Forward declare it here for pointer-only container access; do not add a
// second alias owner in this header.
class EncodingLayer;

// GPUGrimEncoder is defined later in this file but used by LanguageModel class
class GPUGrimEncoder;

//======================================================//
//  Configuration ownership
//
//  All model hyperparameters live in HyperParameters_GPU.hpp:
//    - AiConfigSnapshot         (finalized ai_config.json document)
//    - SamplingStrategy
//    - ModelExecutionMode       (TRAINING vs INFERENCE)
//  Generation callsites consume GenerationHP grouped views derived from
//  AiConfigSnapshot, never a second config owner.
//
//  This header MUST NOT redeclare any of those fields. The encoder
//  consumes grouped construction views derived from AiConfigSnapshot.
//  Borrowed startup resources are passed explicitly by startup GPU model assembly and have
//  no place in a config object or per-forward request.
//======================================================//

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
    explicit LanguageModel(const Config::AiConfigSnapshot& config)
        : config_(config)
    {
        // Startup owns all CUDA-side assembly sequencing. The model constructor
        // only captures the finalized config and fail-loud validates the CUDA
        // execution contract.
#ifdef USE_CUDA
        if (!HyperParameters::snapshotTrainingConfigField<bool>(config_, "use_gpu")) {
            throw std::runtime_error("LanguageModel requires config.use_gpu=true when built with CUDA");
        }
#endif
    }
    ~LanguageModel() = default;

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

    // Startup-owned GPU topology binding. LanguageModel borrows this durable
    // state; it does not own the assembled encoder/layer/MTP objects.
    void bindGpuModelState(GRIMText::Training::Startup::GpuModelState& gpu_model_state) noexcept {
        gpu_model_state_ = &gpu_model_state;
    }

    // Parameter groups accessor (for direct gradient norm / clipping in Phase2)
    const std::vector<ParameterGroup>& parameterGroups() const { return parameter_groups_; }
    std::vector<ParameterGroup>& parameterGroups() { return parameter_groups_; }
#endif

    // Config access
    const Config::AiConfigSnapshot& getConfig() const { return config_; }
    
#ifdef USE_CUDA
    // ScratchBlock reasoning layer access. Presence is config-gated by
    // HyperParameters::scratchBlockConstructionHP(config_); do not runtime-toggle it.
    ScratchBlockLayer* getScratchBlockLayer() { return gpu_model_state_ ? gpu_model_state_->scratch_block_layer.get() : nullptr; }
    const ScratchBlockLayer* getScratchBlockLayer() const { return gpu_model_state_ ? gpu_model_state_->scratch_block_layer.get() : nullptr; }

    // Embedding layer access (Pattern B: persistent, self-allocating, owns token + pos weights)
    EmbeddingLayer* getEmbeddingLayer() { return gpu_model_state_ ? gpu_model_state_->embedding_layer.get() : nullptr; }
    const EmbeddingLayer* getEmbeddingLayer() const { return gpu_model_state_ ? gpu_model_state_->embedding_layer.get() : nullptr; }

    // LM Head layer access (Pattern B: persistent, self-allocating)
    LMHeadLayer* getLmHeadLayer() { return gpu_model_state_ ? gpu_model_state_->lm_head_layer.get() : nullptr; }
    const LMHeadLayer* getLmHeadLayer() const { return gpu_model_state_ ? gpu_model_state_->lm_head_layer.get() : nullptr; }

    // Execution Block layer access (nullptr when disabled)
    ExecutionBlockLayer* getExecutionBlockLayer() { return gpu_model_state_ ? gpu_model_state_->execution_block_layer.get() : nullptr; }
    const ExecutionBlockLayer* getExecutionBlockLayer() const { return gpu_model_state_ ? gpu_model_state_->execution_block_layer.get() : nullptr; }

#endif
    
private:
    Config::AiConfigSnapshot config_;
    
#ifdef USE_CUDA
    // Borrowed startup-owned durable GPU model topology.
    GRIMText::Training::Startup::GpuModelState* gpu_model_state_ = nullptr;
#endif
    
#ifdef USE_CUDA
    TrainingState training_state_;
    GenerationState generation_state_;
    std::vector<ParameterGroup> parameter_groups_;  // Parameter groups for optimizer
#endif
};

using StreamCallback = HyperParameters::GenerationStreamCallback;

//======================================================//
//  GPU Classes (Forward Declarations Only)
//======================================================//

#ifdef USE_CUDA
// GPUGrimEncoder - Container for encoder layers, manages layer lifecycle.
// Forward pass logic is in ForwardPhase2_Encoder.cu::runFullEncoder().
// This class only owns layers and provides access to them.
//
// Constructor takes the grouped encoder hyperparameter view plus explicit
// startup/model-assembly resources. Forward pass runtime handles live on the
// forward request.
class GPUGrimEncoder {
public:
    GPUGrimEncoder(const HyperParameters::EncoderLayerConstructionHP& hp,
                   cudaStream_t init_stream,
                   uint64_t weight_seed);

    EncodingLayer* getLayer(int index);
    const EncodingLayer* getLayer(int index) const;

private:
    struct Impl;
    Impl* pImpl = nullptr;
};

#endif // USE_CUDA

} // namespace GRIM

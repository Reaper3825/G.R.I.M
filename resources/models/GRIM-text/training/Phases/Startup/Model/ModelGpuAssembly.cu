#ifndef USE_CUDA
#define USE_CUDA
#endif

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "ModelGpuAssembly.hpp"
#include "ParameterGroupRegistration.hpp"
#include "../../../../GRIM/grim_language_model_cuda.hpp"
#include "../../../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../../../Shared/Execution/DecodeTimeNumPolicy.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/PBM/PBMStateOwner.hpp"
#include "../../../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIMText {
namespace Training {
namespace Startup {

#ifdef USE_CUDA

GpuModelState::GpuModelState() = default;
GpuModelState::~GpuModelState() = default;
GpuModelState::GpuModelState(GpuModelState&&) noexcept = default;
GpuModelState& GpuModelState::operator=(GpuModelState&&) noexcept = default;

#endif // USE_CUDA

} // namespace Startup
} // namespace Training
} // namespace GRIMText

#ifdef USE_CUDA

//======================================================//
//  Internal startup helpers
//
//  These helpers remove duplicated fail-loud prerequisite checks while keeping
// ownership explicit: Phase1 startup sequences the calls; TrainingContext
// owns PBM separately from GpuModelState; GpuModelState owns durable model
// topology; TrainingState owns runtime workspaces.
//======================================================//

namespace {

constexpr const char* kAssembleGpuModelCaller = "Startup::assembleGpuModel";

void requireRuntimeNotInitialized(const GRIM::TrainingState& training_state,
                                  const char* caller,
                                  const char* runtime_label) {
    if (training_state.initialized) {
        throw std::runtime_error(
            std::string("[") + caller + "] FATAL: training_state_.initialized is already true. "
            "Caller invoked " + runtime_label + " initialization twice (or after the other runtime init). "
            "This is a call-order bug.");
    }
}

cudaStream_t requirePrimaryStream(GRIM::TrainingState& training_state,
                                  const char* caller,
                                  const char* missing_stream_message) {
    if (!training_state.stream_ctrl.isInitialized()) {
        throw std::runtime_error(missing_stream_message);
    }

    cudaStream_t primary_stream = training_state.stream_ctrl.getPrimaryStream();
    GRIM::StreamController::fatalIfDefaultStream(primary_stream, caller);
    return primary_stream;
}

void requireCublasHandle(const GRIM::TrainingState& training_state,
                         const char* caller,
                         const char* missing_handle_message) {
    if (training_state.cublas_handle.get() != nullptr) {
        return;
    }

    const std::string error = std::string("[") + caller + "] " + missing_handle_message;
    std::cerr << error << std::endl;
    throw std::runtime_error(error);
}

void requirePBMReady(bool pbm_initialized,
                     const char* caller,
                     const char* missing_pbm_message) {
    if (!pbm_initialized) {
        throw std::runtime_error(std::string("[") + caller + "] " + missing_pbm_message);
    }
}

void requireReadGateWorkspace(const GRIM::TrainingState& training_state,
                              const char* caller) {
    if (!training_state.read_gate_accum_tensor.data) {
        throw std::runtime_error(
            std::string("[") + caller + "] TrainingState workspace allocation did not create read_gate_accum_tensor");
    }
}

void allocateRuntimeWorkspaces(GRIM::TrainingState& training_state,
                               cudaStream_t primary_stream,
                               const char* caller,
                               const char* allocation_banner) {
    std::cout << allocation_banner << std::endl;
    training_state.allocateReadGateWorkspace(primary_stream);
    requireReadGateWorkspace(training_state, caller);
}

void validatePBMConfigOrThrow(const GRIM::Config::AiConfigSnapshot& model_cfg,
                              const GRIM::HyperParameters::PBMConstructionHP& pbm_hp,
                              const char* caller) {
    const auto positional_encoding = GRIM::HyperParameters::snapshotTrainingConfigField<GRIM::HyperParameters::PositionalEncodingType>(model_cfg, "positional_encoding");
    if (positional_encoding != GRIM::HyperParameters::PositionalEncodingType::ALIBI_ROPE &&
        positional_encoding != GRIM::HyperParameters::PositionalEncodingType::ALIBI) {
        throw std::runtime_error(
            std::string("[") + caller + "] unified PBM requires ALIBI or ALIBI_ROPE, got " +
            GRIM::HyperParameters::positionalEncodingTypeToString(positional_encoding));
    }

    const int expected_d_model = pbm_hp.num_heads * pbm_hp.head_dim;
    const int d_model = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_cfg, "d_model");
    if (d_model != expected_d_model) {
        throw std::runtime_error(
            std::string("[") + caller + "] grouped PBM geometry does not match d_model (d_model=" +
            std::to_string(d_model) + " expected=" + std::to_string(expected_d_model) + ")");
    }
}

void verifyEncoderLayersReady(GRIM::GPUGrimEncoder& encoder,
                              int num_layers,
                              const char* caller) {
    for (int layer = 0; layer < num_layers; ++layer) {
        auto* gpu_layer = encoder.getLayer(layer);
        if (!gpu_layer || !gpu_layer->weightsReady()) {
            throw std::runtime_error(std::string("[") + caller + "] FATAL: Encoder layer " +
                                     std::to_string(layer) + " not ready after self-allocation!");
        }
    }
}

void initializeExecutionSubsystems(
    std::unique_ptr<GRIM::ExecutionBlockLayer>& execution_block_layer,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    std::unique_ptr<GRIM::DecodeTimeSlotSelector>& decode_time_slot_selector,
    const GRIM::Config::AiConfigSnapshot& model_cfg,
    uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(model_cfg);
    if (!execution_hp.enabled) {
        return;
    }

    const uint64_t execution_seed = weight_init_seed + 20;
    GRIMText::Training::Startup::ModelRegistration::initializeExecutionBlockParameterTensors(
        parameter_registry,
        execution_hp,
        execution_seed,
        init_stream);
    execution_block_layer = std::make_unique<GRIM::ExecutionBlockLayer>(
        execution_hp,
        init_stream);
    std::cout << "✓ ExecutionBlock layer created\n";

    const auto selector_hp = GRIM::HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
    if (!selector_hp.enabled) {
        return;
    }

    const uint64_t selector_seed = weight_init_seed + 30;
    decode_time_slot_selector = std::make_unique<GRIM::DecodeTimeSlotSelector>(
        GRIM::createDecodeTimeSlotSelector(selector_hp, selector_seed, init_stream));
    GRIM::validateDecodeTimeNumPolicyConfig(selector_hp);
    std::cout << "✓ DecodeTimeSlotSelector created\n";
}

void initializeMtpHeads(std::vector<GRIM::MtpHeadParameterTensors>& mtp_head_parameter_tensors,
                        const GRIM::HyperParameters::MTPConstructionHP& mtp_hp,
                        uint64_t weight_init_seed,
                        cudaStream_t init_stream) {
    if (!mtp_hp.enabled) {
        return;
    }

    mtp_head_parameter_tensors.resize(static_cast<size_t>(mtp_hp.k));
    for (int k = 0; k < mtp_hp.k; ++k) {
        auto& head = mtp_head_parameter_tensors[static_cast<size_t>(k)];
        const std::string weight_name = "mtp_head_" + std::to_string(k) + ".weight";
        const std::string bias_name = "mtp_head_" + std::to_string(k) + ".bias";

        head.weight = GRIM::Tensor::zeros({mtp_hp.vocab_size, mtp_hp.d_model}, init_stream, weight_name.c_str());
        head.weight.requires_grad_();
        head.weight.ensure_grad();

        const uint64_t mtp_seed = weight_init_seed + 3 + static_cast<uint64_t>(k);
        GRIM::Tensor::xavier_uniform_(head.weight, mtp_seed, init_stream);

        head.bias = GRIM::Tensor::zeros({mtp_hp.vocab_size}, init_stream, bias_name.c_str());
        head.bias.requires_grad_();
        head.bias.ensure_grad();
    }

    std::cout << "✓ MTP auxiliary heads created\n";
}

template <typename LayerT>
void requireWeightsReady(const LayerT* layer,
                         const char* caller,
                         const char* layer_name,
                         const char* failure_detail) {
    if (!layer || !layer->weightsReady()) {
        throw std::runtime_error(
            std::string("[") + caller + "] " + layer_name + " " + failure_detail);
    }
}

const GRIM::LMHeadParameterTensors& requireLmHeadParametersReady(
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const char* caller) {
    const auto& lm_head_parameters = parameter_registry.requireLmHeadParameters(caller);
    if (!GRIM::lmHeadWeightsReady(lm_head_parameters)) {
        throw std::runtime_error(
            std::string("[") + caller + "] LM-head weights are not initialized on the registry owner");
    }
    return lm_head_parameters;
}

const GRIM::EmbeddingParameterTensors& requireEmbeddingParametersReady(
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const char* caller) {
    const auto& embedding_parameters = parameter_registry.requireEmbeddingParameters(caller);
    if (!embedding_parameters.token_weights.data) {
        throw std::runtime_error(
            std::string("[") + caller + "] embedding token weights are not initialized on the registry owner");
    }
    return embedding_parameters;
}

const GRIM::ScratchBlockParameterTensors& requireScratchBlockParametersReady(
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const char* caller) {
    const auto& scratch_block_parameters = parameter_registry.requireScratchBlockParameters(caller);
    if (!scratch_block_parameters.atom_type_embeddings.data ||
        !scratch_block_parameters.atom_projection.data ||
        !scratch_block_parameters.structured_gate_weight.data) {
        throw std::runtime_error(
            std::string("[") + caller + "] ScratchBlock parameter tensors are not initialized on the registry owner");
    }
    return scratch_block_parameters;
}

void logInferenceMemorySummary(const GRIM::Config::AiConfigSnapshot& model_cfg,
                               size_t max_batch_size,
                               size_t max_seq_len_cache,
                               size_t max_tokens) {
    std::cout << "[Startup::initializeInferenceRuntime] ✓ Inference state initialized successfully" << std::endl;
    std::cout << "  Memory allocated for: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache
              << ", tokens=" << max_tokens << std::endl;

    const size_t token_cache_bytes = max_tokens * sizeof(float) * 3;
    const size_t d_model = static_cast<size_t>(GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_cfg, "d_model"));
    const size_t vocab_size = static_cast<size_t>(GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_cfg, "vocab_size"));
    const size_t d_ff = static_cast<size_t>(GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_cfg, "d_ff"));
    const size_t encoder_out_bytes = max_tokens * d_model * sizeof(float);
    const size_t logits_bytes = max_tokens * vocab_size * sizeof(float);
    const size_t workspace_bytes = d_ff * max_seq_len_cache * 4 * sizeof(float);
    const size_t total_bytes = token_cache_bytes + encoder_out_bytes + logits_bytes + workspace_bytes;

    std::cout << "  📊 Total GPU activation memory: ~" << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "      (excludes model weights loaded from checkpoint)" << std::endl;
}

} // namespace

#endif // USE_CUDA

namespace GRIMText::Training::Startup {

#ifdef USE_CUDA

//======================================================//
//  Phase1 startup runtime bootstrap functions
//======================================================//

void initializeCuBLASHandle(::GRIM::TrainingState& training_state) {
    constexpr const char* caller = "Startup::initializeCuBLASHandle";
    cudaStream_t primary_stream = requirePrimaryStream(
        training_state,
        caller,
        "FATAL: StreamController must be initialized before creating cuBLAS handle");

    if (training_state.cublas_handle.get() != nullptr) {
        std::cout << "✓ cuBLAS handle already initialized" << std::endl;
        return;
    }

    cublasStatus_t cublas_err = cublasCreate(training_state.cublas_handle.outParam());
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle: " << cublas_err << std::endl;
        throw std::runtime_error("cuBLAS handle creation failed");
    }

    cublasSetMathMode(training_state.cublas_handle.get(), CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSetStream(training_state.cublas_handle.get(), primary_stream);
    std::cout << "✓ cuBLAS handle bound to Primary stream with Tensor Core acceleration" << std::endl;
}

void initializePBM(const ::GRIM::Config::AiConfigSnapshot& model_cfg,
                   ::GRIM::TrainingState& training_state,
                   ::GRIM::PBM::PBMStateOwner& pbm_owner) {
    constexpr const char* caller = "Startup::initializePBM";
    if (pbm_owner.initialized()) {
        std::cout << "✓ PBM (ALiBi+RoPE) already initialized" << std::endl;
        return;
    }

    cudaStream_t stream = requirePrimaryStream(
        training_state,
        caller,
        "Startup::initializePBM: StreamController is not initialized - ModelAllocationState must initialize stream_ctrl before PBM");

    const auto pbm_hp = GRIM::HyperParameters::pbmConstructionHP(model_cfg);
    validatePBMConfigOrThrow(model_cfg, pbm_hp, caller);

    pbm_owner.initialize(pbm_hp, stream);

    const cudaError_t pbm_sync_err = cudaStreamSynchronize(stream);
    if (pbm_sync_err != cudaSuccess) {
        throw std::runtime_error(std::string("[Startup::initializePBM] cudaStreamSynchronize(primary PBM init stream) failed: ") +
                                 cudaGetErrorString(pbm_sync_err));
    }

    const ::GRIM::PBM::PBMState& pbm_state = pbm_owner.state();
    if (!pbm_state.initialized || !pbm_state.rope_inv_freq || !pbm_state.alibi_slopes) {
        throw std::runtime_error("[Startup::initializePBM] PBM state is invalid");
    }

    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) initialized" << std::endl;
}

void initializeTrainingRuntime(::GRIM::TrainingState& training_state,
                               const ::GRIM::PBM::PBMStateOwner& pbm_owner) {
    constexpr const char* caller = "Startup::initializeTrainingRuntime";
    requireRuntimeNotInitialized(training_state, caller, "training runtime");

    cudaStream_t primary_stream = requirePrimaryStream(
        training_state,
        caller,
        "[Startup::initializeTrainingRuntime] StreamController not initialized! Phase1_Startup must call stream_ctrl.initialize() before training runtime allocation — Rule 20: no silent fallbacks");
    std::cout << "✓ StreamController pre-initialized" << std::endl;

    requireCublasHandle(training_state, caller, "cuBLAS handle not initialized. Call Startup::initializeCuBLASHandle() first!");
    std::cout << "✓ Using pre-initialized cuBLAS handle" << std::endl;

    requirePBMReady(pbm_owner.initialized(), caller, "PBM not initialized! Call Startup::initializePBM() before training runtime allocation — Rule 20: no silent fallbacks");
    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) pre-initialized" << std::endl;

    allocateRuntimeWorkspaces(training_state, primary_stream, caller, "📊 Allocating TrainingState runtime workspaces");

    std::cout << "✓ TrainingState runtime workspaces allocated" << std::endl;
    training_state.initialized = true;
    std::cout << "✓ Training state initialized with runtime workspaces" << std::endl;
}

void initializeInferenceRuntime(const ::GRIM::Config::AiConfigSnapshot& model_cfg,
                               ::GRIM::TrainingState& training_state,
                               ::GRIM::GenerationState& generation_state,
                               const ::GRIM::PBM::PBMStateOwner& pbm_owner,
                               const GpuModelState& gpu_model_state,
                               const ::ParameterRegistry::StartupParameterRegistry& parameter_registry) {
    constexpr const char* caller = "Startup::initializeInferenceRuntime";
    requireRuntimeNotInitialized(training_state, caller, "inference runtime");

    std::cout << "[" << caller << "] Initializing INFERENCE-ONLY state..." << std::endl;
    std::cout << "  → Skipping gradient buffers" << std::endl;
    std::cout << "  → Skipping optimizer state" << std::endl;
    std::cout << "  → Minimal activation caches only" << std::endl;

    cudaStream_t primary_stream = requirePrimaryStream(
        training_state,
        caller,
        "[Startup::initializeInferenceRuntime] StreamController not initialized. Caller must initialize stream_ctrl before inference runtime allocation.");
    requireCublasHandle(training_state, caller, "cuBLAS handle not initialized. Caller must initialize cuBLAS before inference runtime allocation.");
    requirePBMReady(pbm_owner.initialized(), caller, "PBM not initialized. Caller must initialize PBM before inference runtime allocation.");

    if (!gpu_model_state.gpu_encoder) {
        throw std::runtime_error(std::string("[") + caller + "] GPU encoder not initialized. Caller must complete Startup::assembleGpuModel(...) before inference runtime allocation.");
    }
    (void)requireEmbeddingParametersReady(parameter_registry, caller);
    (void)requireLmHeadParametersReady(parameter_registry, caller);

    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(model_cfg);
    if (execution_hp.enabled && !gpu_model_state.execution_block_layer) {
        throw std::runtime_error(std::string("[") + caller + "] ExecutionBlockLayer not assembled by Startup::assembleGpuModel() while execution_block is enabled.");
    }

    const auto selector_hp = GRIM::HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
    if (selector_hp.enabled && !parameter_registry.getDecodeTimeSlotSelector()) {
        throw std::runtime_error(std::string("[") + caller + "] DecodeTimeSlotSelector not assembled by Startup::assembleGpuModel() while selector is enabled.");
    }

    const auto mtp_hp = GRIM::HyperParameters::mtpConstructionHP(model_cfg);
    if (mtp_hp.enabled && static_cast<int>(parameter_registry.mtpHeadParameterTensors().size()) != mtp_hp.k) {
        throw std::runtime_error(std::string("[") + caller + "] MTP heads were not assembled by Startup::assembleGpuModel() while mtp is enabled.");
    }

    cublasSetStream(training_state.cublas_handle.get(), primary_stream);
    std::cout << "  ✓ Using pre-initialized StreamController and cuBLAS handle" << std::endl;

    const size_t max_batch_size = static_cast<size_t>(GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_cfg, "batch_size"));
    const size_t max_seq_len_cache = static_cast<size_t>(GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_cfg, "max_cached_seq_len"));
    const size_t max_tokens = static_cast<size_t>(GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_cfg, "max_tokens_per_batch"));

    std::cout << "  ℹ Allocating activation caches: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache
              << ", total_tokens=" << max_tokens << std::endl;

    allocateRuntimeWorkspaces(training_state, primary_stream, caller, "  ↳ Allocating inference runtime workspaces");

    generation_state.resetSession();
    std::cout << "  ✓ Reset Phase2 inference session state" << std::endl;

    const auto scratch_hp = GRIM::HyperParameters::scratchBlockConstructionHP(model_cfg);
    if (scratch_hp.enabled) {
        if (!gpu_model_state.scratch_block_layer) {
            throw std::runtime_error(std::string("[") + caller + "] ScratchBlockConstructionHP.enabled=true but ScratchBlockLayer was not assembled by Startup::assembleGpuModel()");
        }
        std::cout << "  ✓ ScratchBlock reasoning layer already assembled by Startup::assembleGpuModel() (d_model="
                  << scratch_hp.d_model << ", atom_dim=" << scratch_hp.atom_embedding_dim
                  << ", max_atoms=" << scratch_hp.max_atoms << ")" << std::endl;
    } else if (gpu_model_state.scratch_block_layer) {
        throw std::runtime_error(std::string("[") + caller + "] ScratchBlockLayer exists while ScratchBlockConstructionHP.enabled=false");
    }

    training_state.initialized = true;
    logInferenceMemorySummary(model_cfg, max_batch_size, max_seq_len_cache, max_tokens);

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("[") + caller + "] cudaDeviceSynchronize failed: " +
            std::string(cudaGetErrorString(sync_err)));
    }

    std::cout << "[" << caller << "] ✓ GPU synchronization complete" << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM

namespace GRIM {

#ifdef USE_CUDA

//======================================================//
//  Startup Model Runtime + GPU Assembly
//
//  Phase 1 startup owns the full chronology in this order:
//    1) StreamController
//    2) cuBLAS handle
//    3) PBM
//    4) GPU layer assembly (Startup::assembleGpuModel)
//    5) Parameter registration
//    6) Training/inference runtime workspaces
//
//  This file intentionally holds both the startup free functions and the
//  durable model-layer assembly implementation so startup ordering stays in one
//  place. `ModelAllocationState.cu` remains the only caller that sequences the
//  free-function bootstrap steps around `Startup::assembleGpuModel(...)`.
//======================================================//

} // namespace GRIMText::Training::Startup

namespace GRIMText::Training::Startup {

void assembleGpuModel(const ::GRIM::Config::AiConfigSnapshot& model_cfg,
                      ::GRIM::TrainingState& training_state,
                      const ::GRIM::PBM::PBMStateOwner& pbm_owner,
                      GpuModelState& gpu_model_state,
                      ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
                      uint64_t weight_init_seed) {
    const auto init_hp = GRIM::HyperParameters::gpuModelInitializationHP(model_cfg);

    std::cout << "[assembleGpuModel] Verifying grouped GPU initialization config..." << std::endl;
    if (!init_hp.use_gpu) {
        throw std::runtime_error("[assembleGpuModel] use_gpu=false but GRIM-text REQUIRES GPU - fix ai_config.json");
    }

    try {
        std::cout << "[assembleGpuModel] Assembling GPU model layer objects..." << std::endl;

        const cudaStream_t init_stream = requirePrimaryStream(
            training_state,
            kAssembleGpuModelCaller,
            "FATAL: StreamController not initialized. ModelAllocationState must initialize stream_ctrl before Startup::assembleGpuModel().");
        requireCublasHandle(training_state, kAssembleGpuModelCaller, "cuBLAS handle not initialized. ModelAllocationState must initialize cuBLAS before Startup::assembleGpuModel().");
        requirePBMReady(pbm_owner.initialized(), kAssembleGpuModelCaller, "PBM not initialized before encoder construction. ModelAllocationState must initialize PBM before Startup::assembleGpuModel().");

        //======================================================//
        //  1) Build GPU encoder
        //
        //  Hyperparameters come from EncoderLayerConstructionHP, the grouped
        //  read view owned by HyperparameterGroupings.hpp. Construction resource
        //  inputs — positional encoding and startup init stream — are passed
        //  explicitly. The init seed is an explicit Phase1 RNG input. Forward
        //  stream/cuBLAS live on the forward payload/request,
        //  not on layer configs.
        //======================================================//

        const auto encoder_hp = GRIM::HyperParameters::encoderLayerConstructionHP(model_cfg);

        GRIM::StreamController::fatalIfDefaultStream(init_stream, kAssembleGpuModelCaller);

        std::cout << "[assembleGpuModel] Encoder startup resources prepared" << std::endl;
        std::cout << "✓ Encoder using explicit init stream; PBM stays on the Phase1 forward boundary\n";

        ModelRegistration::initializeFeedForwardParameterTensors(
            parameter_registry.feedForwardParameterTensors(),
            encoder_hp,
            weight_init_seed,
            init_stream);

        gpu_model_state.gpu_encoder = std::make_unique<GRIM::GPUGrimEncoder>(
            encoder_hp,
            init_stream,
            weight_init_seed);
        verifyEncoderLayersReady(*gpu_model_state.gpu_encoder, init_hp.num_layers, kAssembleGpuModelCaller);
        std::cout << "✓ Encoder layers bound registry-owned FFN weights\n";

        //======================================================//
        //  2) Initialize persistent embedding parameters on the registry
        //
        //  Allocates token weights [vocab_size, d_model]. Position
        //  information is injected inside attention via ALiBi/RoPE, so no
        //  separate position-embedding table is allocated (Rule 26).
        //  Must be created BEFORE LM-head init (LM head aliases embedding for tied config).
        //======================================================//
        {
            const auto emb_hp = GRIM::HyperParameters::embeddingLayerConstructionHP(model_cfg);
            ModelRegistration::initializeEmbeddingParameterTensors(
                parameter_registry,
                emb_hp,
                weight_init_seed,
                init_stream,
                true);
            (void)requireEmbeddingParametersReady(parameter_registry, "Startup::assembleGpuModel");
            std::cout << "✓ Embedding parameters created\n";
        }

        //======================================================//
        //  3) Initialize persistent LM-head parameter bundle on the registry
        //======================================================//
        {
            const auto lm_hp = GRIM::HyperParameters::lmHeadLayerConstructionHP(model_cfg);
            auto& embedding_parameters = parameter_registry.requireEmbeddingParameters("Startup::assembleGpuModel");

            GRIM::Tensor* tied_emb = nullptr;
            if (lm_hp.tie_embeddings) {
                tied_emb = &embedding_parameters.token_weights;
                if (!tied_emb->data) {
                    throw std::runtime_error("[assembleGpuModel] FATAL: tied embedding token weights have NULL data");
                }
            }

            ModelRegistration::initializeLmHeadParameterTensors(
                parameter_registry,
                lm_hp,
                weight_init_seed + 1,
                init_stream,
                tied_emb);
            (void)requireLmHeadParametersReady(parameter_registry, "Startup::assembleGpuModel");
            std::cout << "✓ LM head parameters created\n";
        }

        //======================================================//
        //  4) Build optional ScratchBlock reasoning runtime and registry-owned parameters
        //
        //  HyperparameterGroupings owns the static construction contract;
        //  startup model assembly supplies only the init stream/runtime shell,
        //  while ParameterGroupRegistration owns the durable trainable tensors.
        //======================================================//
        const auto scratch_hp = GRIM::HyperParameters::scratchBlockConstructionHP(model_cfg);
        if (scratch_hp.enabled) {
            auto& scratch_block_layer = gpu_model_state.scratch_block_layer;
            scratch_block_layer = std::make_unique<GRIM::ScratchBlockLayer>(
                scratch_hp, init_stream);

            ModelRegistration::initializeScratchBlockParameterTensors(
                parameter_registry,
                scratch_hp,
                weight_init_seed,
                init_stream);
            (void)requireScratchBlockParametersReady(parameter_registry, "Startup::assembleGpuModel");

            const auto& scratch_block_parameters = requireScratchBlockParametersReady(parameter_registry, "Startup::assembleGpuModel");
            if (!scratch_block_parameters.atom_type_embeddings.data ||
                !scratch_block_parameters.atom_projection.data) {
                throw std::runtime_error("[assembleGpuModel] FATAL: registry ScratchBlock tensors not ready after startup allocation");
            }
            std::cout << "✓ ScratchBlock layer created\n";
        }

        //======================================================//
        //  5) Build optional model heads/subsystems
        //======================================================//
        initializeExecutionSubsystems(
            gpu_model_state.execution_block_layer,
            parameter_registry,
            parameter_registry.decode_time_slot_selector,
            model_cfg,
            weight_init_seed,
            init_stream);

        initializeMtpHeads(
            parameter_registry.mtpHeadParameterTensors(),
            GRIM::HyperParameters::mtpConstructionHP(model_cfg),
            weight_init_seed,
            init_stream);

        std::cout << "✓ GPU model layer assembly complete\n";
        std::cout << "  - Attention: GPU-accelerated\n";
        std::cout << "  - FFN: GPU-accelerated SwiGLU with registry-owned parameters\n";
        std::cout << "  - Layer Norm: GPU-accelerated\n";

        if (init_hp.use_flash_attention) {
            std::cout << "✓ Flash Attention configured from grouped encoder HP\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ EXCEPTION in Startup::assembleGpuModel(): " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup

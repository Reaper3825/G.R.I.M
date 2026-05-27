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
#include "../../../../GRIM/grim_language_model_cuda.hpp"
#include "../../../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIMText {
namespace Training {
namespace Startup {

#ifdef USE_CUDA

struct ModelAssemblyAccess {
    static GRIM::TrainingState& trainingState(::GRIM::LanguageModel& model) {
        return model.training_state_;
    }

    static const GRIM::TrainingState& trainingState(const ::GRIM::LanguageModel& model) {
        return model.training_state_;
    }

    static GRIM::GenerationState& generationState(::GRIM::LanguageModel& model) {
        return model.generation_state_;
    }

    static void initializePBM(::GRIM::LanguageModel& model,
                              const ::GRIM::HyperParameters::PBMConstructionHP& pbm_hp,
                              cudaStream_t stream) {
        model.pbm_owner_.initialize(pbm_hp, stream);

        const ::GRIM::PBM::PBMState& pbm_state = model.pbm_owner_.state();
        if (!pbm_state.initialized || !pbm_state.rope_inv_freq || !pbm_state.alibi_slopes || !pbm_state.upload_event) {
            throw std::runtime_error("[Startup::initializePBM] PBM state is invalid");
        }
    }

    static bool pbmInitialized(const ::GRIM::LanguageModel& model) {
        return model.pbm_owner_.initialized();
    }

    static const ::GRIM::PBM::PBMState& pbmState(const ::GRIM::LanguageModel& model) {
        return model.pbm_owner_.state();
    }

    static ::GRIM::GPUGrimEncoder* gpuEncoderPtr(::GRIM::LanguageModel& model) {
        return model.gpu_encoder_.get();
    }

    static std::unique_ptr<::GRIM::GPUGrimEncoder>& gpuEncoder(::GRIM::LanguageModel& model) {
        return model.gpu_encoder_;
    }

    static std::unique_ptr<::GRIM::ScratchBlockLayer>& scratchBlockLayer(::GRIM::LanguageModel& model) {
        return model.scratch_block_layer_;
    }

    static std::unique_ptr<::GRIM::EmbeddingLayer>& embeddingLayer(::GRIM::LanguageModel& model) {
        return model.embedding_layer_;
    }

    static std::unique_ptr<::GRIM::LMHeadLayer>& lmHeadLayer(::GRIM::LanguageModel& model) {
        return model.lm_head_layer_;
    }

    static std::unique_ptr<::GRIM::ExecutionBlockLayer>& executionBlockLayer(::GRIM::LanguageModel& model) {
        return model.execution_block_layer_;
    }

    static std::unique_ptr<::GRIM::DecodeTimeSlotSelectorLayer>& decodeTimeSlotSelectorLayer(::GRIM::LanguageModel& model) {
        return model.decode_time_slot_selector_layer_;
    }

    static std::unique_ptr<::GRIM::DecodeTimeNumPolicy>& decodeTimeNumPolicy(::GRIM::LanguageModel& model) {
        return model.decode_time_num_policy_;
    }

    static std::vector<::GRIM::LanguageModel::MTPHead>& mtpHeads(::GRIM::LanguageModel& model) {
        return model.mtp_heads_;
    }
};

#endif // USE_CUDA

} // namespace Startup
} // namespace Training
} // namespace GRIMText

#ifdef USE_CUDA

//======================================================//
//  Internal startup helpers
//
//  These helpers remove duplicated fail-loud prerequisite checks while keeping
//  ownership explicit: Phase1 startup sequences the calls; LanguageModel owns
//  the durable fields; TrainingState owns runtime workspaces.
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

void requirePBMReady(const GRIM::LanguageModel& model,
                     const char* caller,
                     const char* missing_pbm_message) {
    if (!GRIMText::Training::Startup::ModelAssemblyAccess::pbmInitialized(model)) {
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
                               const GRIM::HyperParameters::TrainingStateWorkspaceHP& workspace_hp,
                               cudaStream_t primary_stream,
                               const char* caller,
                               const char* allocation_banner) {
    std::cout << allocation_banner << std::endl;
    training_state.allocateStepDeviceWorkspaces(workspace_hp, primary_stream);
    requireReadGateWorkspace(training_state, caller);
}

void validatePBMConfigOrThrow(const GRIM::HyperParameters::LanguageModelConfig& model_cfg,
                              const GRIM::HyperParameters::PBMConstructionHP& pbm_hp,
                              const char* caller) {
    if (model_cfg.positional_encoding != GRIM::HyperParameters::PositionalEncodingType::ALIBI_ROPE &&
        model_cfg.positional_encoding != GRIM::HyperParameters::PositionalEncodingType::ALIBI) {
        throw std::runtime_error(
            std::string("[") + caller + "] unified PBM requires ALIBI or ALIBI_ROPE, got " +
            GRIM::HyperParameters::positionalEncodingTypeToString(model_cfg.positional_encoding));
    }

    const int expected_d_model = pbm_hp.num_heads * pbm_hp.head_dim;
    if (model_cfg.d_model != expected_d_model) {
        throw std::runtime_error(
            std::string("[") + caller + "] grouped PBM geometry does not match d_model (d_model=" +
            std::to_string(model_cfg.d_model) + " expected=" + std::to_string(expected_d_model) + ")");
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

GRIM::EncoderConstructionBindings makeEncoderConstructionBindings(const GRIM::PBM::PBMState& pbm_state,
                                                                  cudaStream_t init_stream) {
    GRIM::EncoderConstructionBindings bindings;
    bindings.pos_encoding = &pbm_state;
    bindings.init_stream = init_stream;
    GRIM::StreamController::fatalIfDefaultStream(bindings.init_stream, kAssembleGpuModelCaller);
    return bindings;
}

void initializeExecutionSubsystems(
    std::unique_ptr<GRIM::ExecutionBlockLayer>& execution_block_layer,
    std::unique_ptr<GRIM::DecodeTimeSlotSelectorLayer>& decode_time_slot_selector_layer,
    std::unique_ptr<GRIM::DecodeTimeNumPolicy>& decode_time_num_policy,
    const GRIM::HyperParameters::LanguageModelConfig& model_cfg,
    uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(model_cfg);
    if (!execution_hp.enabled) {
        return;
    }

    const uint64_t execution_seed = weight_init_seed + 20;
    execution_block_layer = std::make_unique<GRIM::ExecutionBlockLayer>(
        execution_hp, execution_seed, init_stream);
    std::cout << "✓ ExecutionBlock layer created\n";

    const auto selector_hp = GRIM::HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
    if (!selector_hp.enabled) {
        return;
    }

    const uint64_t selector_seed = weight_init_seed + 30;
    decode_time_slot_selector_layer = std::make_unique<GRIM::DecodeTimeSlotSelectorLayer>(
        selector_hp, selector_seed, init_stream);
    decode_time_num_policy = std::make_unique<GRIM::DecodeTimeNumPolicy>(selector_hp);
    std::cout << "✓ DecodeTimeSlotSelector created\n";
}

void initializeMtpHeads(std::vector<GRIM::LanguageModel::MTPHead>& mtp_heads,
                        const GRIM::HyperParameters::MTPConstructionHP& mtp_hp,
                        uint64_t weight_init_seed,
                        cudaStream_t init_stream) {
    if (!mtp_hp.enabled) {
        return;
    }

    mtp_heads.resize(static_cast<size_t>(mtp_hp.k));
    for (int k = 0; k < mtp_hp.k; ++k) {
        auto& head = mtp_heads[static_cast<size_t>(k)];
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

void logInferenceMemorySummary(const GRIM::HyperParameters::LanguageModelConfig& model_cfg,
                               size_t max_batch_size,
                               size_t max_seq_len_cache,
                               size_t max_tokens) {
    std::cout << "[Startup::initializeInferenceRuntime] ✓ Inference state initialized successfully" << std::endl;
    std::cout << "  Memory allocated for: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache
              << ", tokens=" << max_tokens << std::endl;

    const size_t token_cache_bytes = max_tokens * sizeof(float) * 3;
    const size_t encoder_out_bytes = max_tokens * static_cast<size_t>(model_cfg.d_model) * sizeof(float);
    const size_t logits_bytes = max_tokens * static_cast<size_t>(model_cfg.vocab_size) * sizeof(float);
    const size_t workspace_bytes = static_cast<size_t>(model_cfg.d_ff) * max_seq_len_cache * 4 * sizeof(float);
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

void initializeCuBLASHandle(::GRIM::LanguageModel& model) {
    auto& training_state = ModelAssemblyAccess::trainingState(model);
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

void initializePBM(::GRIM::LanguageModel& model) {
    constexpr const char* caller = "Startup::initializePBM";
    if (ModelAssemblyAccess::pbmInitialized(model)) {
        std::cout << "✓ PBM (ALiBi+RoPE) already initialized" << std::endl;
        return;
    }

    auto& training_state = ModelAssemblyAccess::trainingState(model);
    const auto& model_cfg = model.getConfig();
    cudaStream_t stream = requirePrimaryStream(
        training_state,
        caller,
        "Startup::initializePBM: StreamController is not initialized - ModelAllocationState must initialize stream_ctrl before PBM");

    const auto pbm_hp = GRIM::HyperParameters::pbmConstructionHP(model_cfg);
    validatePBMConfigOrThrow(model_cfg, pbm_hp, caller);

    ModelAssemblyAccess::initializePBM(model, pbm_hp, stream);

    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) initialized" << std::endl;
}

void initializeTrainingRuntime(::GRIM::LanguageModel& model) {
    constexpr const char* caller = "Startup::initializeTrainingRuntime";
    auto& training_state = ModelAssemblyAccess::trainingState(model);
    requireRuntimeNotInitialized(training_state, caller, "training runtime");

    cudaStream_t primary_stream = requirePrimaryStream(
        training_state,
        caller,
        "[Startup::initializeTrainingRuntime] StreamController not initialized! Phase1_Startup must call stream_ctrl.initialize() before training runtime allocation — Rule 20: no silent fallbacks");
    std::cout << "✓ StreamController pre-initialized" << std::endl;

    requireCublasHandle(training_state, caller, "cuBLAS handle not initialized. Call Startup::initializeCuBLASHandle() first!");
    std::cout << "✓ Using pre-initialized cuBLAS handle" << std::endl;

    requirePBMReady(model, caller, "PBM not initialized! Call Startup::initializePBM() before training runtime allocation — Rule 20: no silent fallbacks");
    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) pre-initialized" << std::endl;

    const auto workspace_hp = GRIM::HyperParameters::trainingStateWorkspaceHP(model.getConfig());
    allocateRuntimeWorkspaces(training_state, workspace_hp, primary_stream, caller, "📊 Allocating TrainingState step workspaces");

    std::cout << "✓ TrainingState step workspaces allocated" << std::endl;
    training_state.initialized = true;
    std::cout << "✓ Training state initialized with runtime workspaces" << std::endl;
}

void initializeInferenceRuntime(::GRIM::LanguageModel& model) {
    constexpr const char* caller = "Startup::initializeInferenceRuntime";
    auto& training_state = ModelAssemblyAccess::trainingState(model);
    requireRuntimeNotInitialized(training_state, caller, "inference runtime");

    const auto& model_cfg = model.getConfig();
    std::cout << "[" << caller << "] Initializing INFERENCE-ONLY state..." << std::endl;
    std::cout << "  → Skipping gradient buffers" << std::endl;
    std::cout << "  → Skipping optimizer state" << std::endl;
    std::cout << "  → Minimal activation caches only" << std::endl;

    cudaStream_t primary_stream = requirePrimaryStream(
        training_state,
        caller,
        "[Startup::initializeInferenceRuntime] StreamController not initialized. Caller must initialize stream_ctrl before inference runtime allocation.");
    requireCublasHandle(training_state, caller, "cuBLAS handle not initialized. Caller must initialize cuBLAS before inference runtime allocation.");
    requirePBMReady(model, caller, "PBM not initialized. Caller must initialize PBM before inference runtime allocation.");

    if (!ModelAssemblyAccess::gpuEncoderPtr(model)) {
        throw std::runtime_error(std::string("[") + caller + "] GPU encoder not initialized. Caller must complete Startup::assembleGpuModel(...) before inference runtime allocation.");
    }
    requireWeightsReady(model.getEmbeddingLayer(), caller, "EmbeddingLayer", "not assembled by Startup::assembleGpuModel().");
    requireWeightsReady(model.getLmHeadLayer(), caller, "LMHeadLayer", "not assembled by Startup::assembleGpuModel().");

    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(model_cfg);
    if (execution_hp.enabled && !model.getExecutionBlockLayer()) {
        throw std::runtime_error(std::string("[") + caller + "] ExecutionBlockLayer not assembled by Startup::assembleGpuModel() while execution_block is enabled.");
    }

    const auto selector_hp = GRIM::HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
    if (selector_hp.enabled && !model.getDecodeTimeSlotSelectorLayer()) {
        throw std::runtime_error(std::string("[") + caller + "] DecodeTimeSlotSelectorLayer not assembled by Startup::assembleGpuModel() while selector is enabled.");
    }

    const auto mtp_hp = GRIM::HyperParameters::mtpConstructionHP(model_cfg);
    if (mtp_hp.enabled && static_cast<int>(ModelAssemblyAccess::mtpHeads(model).size()) != mtp_hp.k) {
        throw std::runtime_error(std::string("[") + caller + "] MTP heads were not assembled by Startup::assembleGpuModel() while mtp is enabled.");
    }

    cublasSetStream(training_state.cublas_handle.get(), primary_stream);
    std::cout << "  ✓ Using pre-initialized StreamController and cuBLAS handle" << std::endl;

    GRIM::HyperParameters::validateRootConfigDocument(model_cfg, caller);
    const auto workspace_hp = GRIM::HyperParameters::trainingStateWorkspaceHP(model_cfg);
    const size_t max_batch_size = static_cast<size_t>(workspace_hp.batch_size);
    const size_t max_seq_len_cache = static_cast<size_t>(model_cfg.max_cached_seq_len);
    const size_t max_tokens = static_cast<size_t>(workspace_hp.max_tokens_per_batch);

    std::cout << "  ℹ Allocating activation caches: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache
              << ", total_tokens=" << max_tokens << std::endl;

    allocateRuntimeWorkspaces(training_state, workspace_hp, primary_stream, caller, "  ↳ Allocating inference runtime workspaces");

    ModelAssemblyAccess::generationState(model).resetSession();
    std::cout << "  ✓ Reset Phase2 inference session state" << std::endl;

    const auto scratch_hp = GRIM::HyperParameters::scratchBlockConstructionHP(model_cfg);
    if (scratch_hp.enabled) {
        if (!model.getScratchBlockLayer()) {
            throw std::runtime_error(std::string("[") + caller + "] ScratchBlockConstructionHP.enabled=true but ScratchBlockLayer was not assembled by Startup::assembleGpuModel()");
        }
        std::cout << "  ✓ ScratchBlock reasoning layer already assembled by Startup::assembleGpuModel() (d_model="
                  << scratch_hp.d_model << ", atom_dim=" << scratch_hp.atom_embedding_dim
                  << ", max_atoms=" << scratch_hp.max_atoms << ")" << std::endl;
    } else if (model.getScratchBlockLayer()) {
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

void assembleGpuModel(::GRIM::LanguageModel& model, uint64_t weight_init_seed) {
    const auto& model_cfg = model.getConfig();
    const auto init_hp = GRIM::HyperParameters::gpuModelInitializationHP(model_cfg);

    std::cout << "[assembleGpuModel] Verifying grouped GPU initialization config..." << std::endl;
    if (!init_hp.use_gpu) {
        throw std::runtime_error("[assembleGpuModel] use_gpu=false but GRIM-text REQUIRES GPU - fix ai_config.json");
    }

    try {
        std::cout << "[assembleGpuModel] Assembling GPU model layer objects..." << std::endl;

        const cudaStream_t init_stream = requirePrimaryStream(
            ModelAssemblyAccess::trainingState(model),
            kAssembleGpuModelCaller,
            "FATAL: StreamController not initialized. ModelAllocationState must initialize stream_ctrl before Startup::assembleGpuModel().");
        requireCublasHandle(ModelAssemblyAccess::trainingState(model), kAssembleGpuModelCaller, "cuBLAS handle not initialized. ModelAllocationState must initialize cuBLAS before Startup::assembleGpuModel().");
        requirePBMReady(model, kAssembleGpuModelCaller, "PBM not initialized before encoder construction. ModelAllocationState must initialize PBM before Startup::assembleGpuModel().");

        //======================================================//
        //  1) Build GPU encoder
        //
        //  Hyperparameters come from EncoderLayerConstructionHP, the grouped
        //  read view owned by HyperparameterGroupings.hpp. Construction resource
        //  bindings — positional encoding and startup init stream — come from
        //  EncoderConstructionBindings. The init seed is an explicit Phase1 RNG
        //  input. Forward stream/cuBLAS live on the forward payload/request,
        //  not on layer configs.
        //======================================================//

        const auto encoder_hp = GRIM::HyperParameters::encoderLayerConstructionHP(model_cfg);
        const GRIM::EncoderConstructionBindings enc_bindings = makeEncoderConstructionBindings(ModelAssemblyAccess::pbmState(model), init_stream);

        std::cout << "[assembleGpuModel] Encoder construction bindings prepared" << std::endl;
        std::cout << "✓ Encoder using TrainingState construction bindings\n";

        auto* encoder_ptr = new GRIM::GPUGrimEncoder(encoder_hp, enc_bindings, weight_init_seed);
        ModelAssemblyAccess::gpuEncoder(model).reset(encoder_ptr);
        verifyEncoderLayersReady(*encoder_ptr, init_hp.num_layers, kAssembleGpuModelCaller);
        std::cout << "✓ Encoder layers self-allocated weights\n";

        //======================================================//
        //  2) Build persistent Embedding layer (Pattern B)
        //
        //  Self-allocates token weights [vocab_size, d_model]. Position
        //  information is injected inside attention via ALiBi/RoPE, so no
        //  separate position-embedding table is allocated (Rule 26).
        //  Must be created BEFORE LMHeadLayer (LM head aliases embedding for tied config).
        //======================================================//
        {
            const auto emb_hp = GRIM::HyperParameters::embeddingLayerConstructionHP(model_cfg);
            auto& embedding_layer = ModelAssemblyAccess::embeddingLayer(model);
            embedding_layer = std::make_unique<GRIM::EmbeddingLayer>(emb_hp, weight_init_seed, enc_bindings.init_stream, true);

            if (!embedding_layer->weightsReady()) {
                throw std::runtime_error("[assembleGpuModel] FATAL: EmbeddingLayer not ready after construction!");
            }
            std::cout << "✓ Embedding layer created\n";
        }

        //======================================================//
        //  3) Build persistent LM Head layer (Pattern B)
        //
        //  Self-allocates weights or aliases embedding for tied config.
        //  Owns final_rms_gamma. Created AFTER EmbeddingLayer because tied
        //  weights need embedding token storage.
        //======================================================//
        {
            const auto lm_hp = GRIM::HyperParameters::lmHeadLayerConstructionHP(model_cfg);
            auto& embedding_layer = ModelAssemblyAccess::embeddingLayer(model);
            auto& lm_head_layer = ModelAssemblyAccess::lmHeadLayer(model);

            GRIM::Tensor* tied_emb = nullptr;
            if (lm_hp.tie_embeddings) {
                tied_emb = &embedding_layer->tokenWeights();
                if (!tied_emb->data) {
                    throw std::runtime_error("[assembleGpuModel] FATAL: tied embedding token weights have NULL data");
                }
            }

            lm_head_layer = std::make_unique<GRIM::LMHeadLayer>(
                lm_hp, weight_init_seed + 1, enc_bindings.init_stream, tied_emb);

            if (!lm_head_layer->weightsReady()) {
                throw std::runtime_error("[assembleGpuModel] FATAL: LMHeadLayer not ready after construction!");
            }
            std::cout << "✓ LM Head layer created\n";
        }

        //======================================================//
        //  4) Build optional ScratchBlock reasoning layer (Pattern B)
        //
        //  HyperparameterGroupings owns the static construction contract;
        //  startup model assembly supplies only the init stream. TrainingState
        //  must not allocate or configure this durable model layer.
        //======================================================//
        const auto scratch_hp = GRIM::HyperParameters::scratchBlockConstructionHP(model_cfg);
        if (scratch_hp.enabled) {
            auto& scratch_block_layer = ModelAssemblyAccess::scratchBlockLayer(model);
            scratch_block_layer = std::make_unique<GRIM::ScratchBlockLayer>(
                scratch_hp, enc_bindings.init_stream);

            if (!scratch_block_layer->atomTypeEmbeddings().data ||
                !scratch_block_layer->atomProjection().data) {
                throw std::runtime_error("[assembleGpuModel] FATAL: ScratchBlockLayer tensors not ready after construction");
            }
            std::cout << "✓ ScratchBlock layer created\n";
        }

        //======================================================//
        //  5) Build optional model heads/subsystems
        //======================================================//
        initializeExecutionSubsystems(
            ModelAssemblyAccess::executionBlockLayer(model),
            ModelAssemblyAccess::decodeTimeSlotSelectorLayer(model),
            ModelAssemblyAccess::decodeTimeNumPolicy(model),
            model_cfg,
            weight_init_seed,
            enc_bindings.init_stream);

        initializeMtpHeads(
            ModelAssemblyAccess::mtpHeads(model),
            GRIM::HyperParameters::mtpConstructionHP(model_cfg),
            weight_init_seed,
            enc_bindings.init_stream);

        std::cout << "✓ GPU model layer assembly complete\n";
        std::cout << "  - Attention: GPU-accelerated\n";
        std::cout << "  - FFN: GPU-accelerated with fused GELU\n";
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

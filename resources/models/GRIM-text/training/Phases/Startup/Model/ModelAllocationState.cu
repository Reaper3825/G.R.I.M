#include "ModelAllocationState.hpp"

#include "ModelGpuAssembly.hpp"
#include "ParameterGroupRegistration.hpp"

#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/LogRecorder/LogRecorder.hpp"

#include <memory>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace GRIMText::Training {

namespace Internal {

std::unique_ptr<GRIM::LanguageModel> initializeModel(
    const GRIM::Config::AiConfigSnapshot& config_snapshot,
    Startup::GpuModelState& gpu_model_state,
    uint64_t weight_init_seed,
    TrainingLogger& logger)
{
    logger.log("Initializing model with weight_init_seed=" + std::to_string(weight_init_seed) + "...");

    if (GRIM::HyperParameters::snapshotTrainingConfigField<GRIM::HyperParameters::HardcodedPattern>(config_snapshot, "hardcoded_hidden_pattern") != GRIM::HyperParameters::HardcodedPattern::DISABLED) {
        logger.log("⚠️  Hardcoded hidden-state diagnostic mode is active; exact config values are listed by ConfigDump.");
        logger.log("⚠️  Encoder output will be REPLACED with synthetic patterns - this is a DIAGNOSTIC MODE ONLY!");
    }

    {
        logger.log("Initializing CUDA device context...");
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaSetDevice(0) failed: ") + cudaGetErrorString(err));
        }

        err = cudaFree(0);
        if (err != cudaSuccess && err != cudaErrorInvalidValue) {
            throw std::runtime_error(std::string("FATAL: CUDA context creation failed: ") + cudaGetErrorString(err));
        }

        int device;
        err = cudaGetDevice(&device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaGetDevice() failed: ") + cudaGetErrorString(err));
        }

        cudaDeviceProp props;
        err = cudaGetDeviceProperties(&props, device);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("FATAL: cudaGetDeviceProperties() failed: ") + cudaGetErrorString(err));
        }

        logger.log("✓ CUDA device initialized: " + std::string(props.name));
    }

    auto model = std::make_unique<GRIM::LanguageModel>(config_snapshot);
    model->bindGpuModelState(gpu_model_state);

    {
        GRIM::StreamControllerConfig stream_config;
        stream_config.verbose = true;

        if (!model->getTrainingState().stream_ctrl.initialize(stream_config)) {
            throw std::runtime_error("FATAL: Failed to initialize StreamController");
        }
        logger.log("✓ StreamController initialized");
    }

    logger.log("Initializing cuBLAS handle...");
    GRIMText::Training::Startup::initializeCuBLASHandle(*model);
    logger.log("✓ cuBLAS handle initialized with Tensor Core acceleration");

    logger.log("Initializing RoPE (required before encoder construction)...");
    GRIMText::Training::Startup::initializePBM(*model);
    logger.log("✓ RoPE initialized");

    logger.log("Assembling GPU model layers with weight_init_seed=" + std::to_string(weight_init_seed) + "...");
    GRIMText::Training::Startup::assembleGpuModel(*model, gpu_model_state, weight_init_seed);
    logger.log("✓ GPU model layers fully assembled");

    logger.log("Registering trainable parameter groups...");
    GRIMText::Training::Startup::ModelRegistration::buildParameterGroups(*model, gpu_model_state);
    logger.log("✓ Trainable parameter groups registered and verified");

    const auto execution_mode = GRIM::HyperParameters::snapshotExecutionMode(config_snapshot);
    if (execution_mode == GRIM::HyperParameters::ModelExecutionMode::TRAINING) {
        logger.log("Initializing TrainingState runtime workspaces...");
        GRIMText::Training::Startup::initializeTrainingRuntime(*model);
        logger.log("✓ TrainingState fully initialized");
    } else if (execution_mode == GRIM::HyperParameters::ModelExecutionMode::INFERENCE) {
        logger.log("Initializing inference runtime workspaces...");
        GRIMText::Training::Startup::initializeInferenceRuntime(*model, gpu_model_state);
        logger.log("✓ Inference runtime fully initialized");
    } else {
        throw std::runtime_error("initializeModel: unsupported execution_mode");
    }

#ifdef USE_CUDA
    {
        auto* gpu_encoder = gpu_model_state.gpu_encoder.get();
        if (!gpu_encoder) {
            throw std::runtime_error(
                "ModelAllocationState::initializeModel: gpu_model_state.gpu_encoder is NULL after GPU model layer assembly");
        }
        const int num_layers = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config_snapshot, "num_layers");
        for (int layer = 0; layer < num_layers; ++layer) {
            auto* enc = gpu_encoder->getLayer(layer);
            if (!enc) {
                throw std::runtime_error("Encoder layer " + std::to_string(layer) + " is NULL after GPU model layer assembly");
            }
        }
        logger.log("✓ Encoder layers verified");
    }
#endif

    logger.log("✓ Model initialized");
    return model;
}

} // namespace Internal

ModelAllocationState captureAndValidateModelAllocationOrThrow(const TrainingContext& ctx) {
    if (!ctx.model) {
        throw std::runtime_error("FATAL: ModelAllocated validation called before model exists");
    }

    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);
    const int max_cached_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "max_cached_seq_len");
    if (max_cached_seq_len != fixed_shape.max_seq_len) {
        throw std::runtime_error("FATAL: model max_cached_seq_len does not match trainingFixedShapeHP (model=" +
                                 std::to_string(max_cached_seq_len) +
                                 " grouping=" + std::to_string(fixed_shape.max_seq_len) + ")");
    }

    ModelAllocationState allocation;
    allocation.model_max_tokens_per_batch = fixed_shape.max_tokens_per_batch;

    return allocation;
}

void ModelAllocated(TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleError;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    ctx.rng = Internal::initializeRNG(ctx.config, *ctx.logging.logger);
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "vocab_size");
    if (vocab_size != static_cast<int>(ctx.data_info.actual_vocab_size)) {
        throw std::runtime_error(
            "FATAL: runtime model vocab_size drifted from startup artifact fact: actual_vocab_size=" +
            std::to_string(ctx.data_info.actual_vocab_size) +
            " config.vocab_size=" + std::to_string(vocab_size));
    }

    try {
        ctx.model = Internal::initializeModel(
            ctx.config,
            ctx.gpu_model,
            ctx.rng.init_seed,
            *ctx.logging.logger);
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, std::string("FATAL: Model initialization failed: ") + e.what(), 0);
        throw;
    }

    ctx.model_allocation = captureAndValidateModelAllocationOrThrow(ctx);
}

} // namespace GRIMText::Training


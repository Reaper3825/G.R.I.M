#include "ModelAllocationState.hpp"

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
    const GRIM::HyperParameters::LanguageModelConfig& config,
    uint64_t weight_init_seed,
    TrainingLogger& logger)
{
    logger.log("Initializing model with weight_init_seed=" + std::to_string(weight_init_seed) + "...");

    if (config.hardcoded_hidden_pattern != GRIM::HyperParameters::HardcodedPattern::DISABLED) {
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

    auto model = std::make_unique<GRIM::LanguageModel>(config);

    {
        GRIM::StreamControllerConfig stream_config;
        stream_config.verbose = true;

        if (!model->getTrainingState().stream_ctrl.initialize(stream_config)) {
            throw std::runtime_error("FATAL: Failed to initialize StreamController");
        }
        logger.log("✓ StreamController initialized");
    }

    logger.log("Initializing cuBLAS handle...");
    model->initCuBLASHandle();
    logger.log("✓ cuBLAS handle initialized with Tensor Core acceleration");

    logger.log("Initializing RoPE (required before encoder construction)...");
    model->initPBM();
    logger.log("✓ RoPE initialized");

    logger.log("Assembling GPU model layers with weight_init_seed=" + std::to_string(weight_init_seed) + "...");
    model->initGPU(weight_init_seed);
    logger.log("✓ GPU model layers fully assembled");

    logger.log("Registering trainable parameter groups...");
    GRIMText::Training::Startup::ModelRegistration::buildParameterGroups(*model);
    logger.log("✓ Trainable parameter groups registered and verified");

    logger.log("Initializing TrainingState runtime workspaces...");
    model->initTrainingState();
    logger.log("✓ TrainingState fully initialized");

#ifdef USE_CUDA
    {
        auto* gpu_encoder = &model->getGpuEncoder();
        for (int layer = 0; layer < config.num_layers; ++layer) {
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

    const auto& state = ctx.model->getTrainingState();
    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);
    if (ctx.model_config.max_cached_seq_len != fixed_shape.max_seq_len) {
        throw std::runtime_error("FATAL: model max_cached_seq_len does not match trainingFixedShapeHP (model=" +
                                 std::to_string(ctx.model_config.max_cached_seq_len) +
                                 " grouping=" + std::to_string(fixed_shape.max_seq_len) + ")");
    }

    const auto& token_ids_shape = state.cached_token_ids_tensor.shape.require("ModelAllocated cached_token_ids_tensor");
    if (!token_ids_shape.is_2d_layout()) {
        throw std::runtime_error("FATAL: cached_token_ids_tensor must be a 2D token-id buffer");
    }

    ModelAllocationState allocation;
    allocation.model_max_tokens_per_batch = token_ids_shape.as_2d().cols;
    if (token_ids_shape.as_2d().cols != fixed_shape.max_tokens_per_batch) {
        throw std::runtime_error("FATAL: cached_token_ids_tensor token capacity does not match trainingFixedShapeHP (tensor=" +
                                 std::to_string(token_ids_shape.as_2d().cols) +
                                 " grouping=" + std::to_string(fixed_shape.max_tokens_per_batch) + ")");
    }

    const auto& targets_shape = state.cached_targets_tensor.shape.require("ModelAllocated cached_targets_tensor");
    if (!targets_shape.is_2d_layout()) {
        throw std::runtime_error("FATAL: cached_targets_tensor must be a 2D target upload buffer");
    }
    if (targets_shape.as_2d().rows != fixed_shape.max_tokens_per_batch) {
        throw std::runtime_error("FATAL: cached_targets_tensor row capacity does not match trainingFixedShapeHP (tensor=" +
                                 std::to_string(targets_shape.as_2d().rows) +
                                 " grouping=" + std::to_string(fixed_shape.max_tokens_per_batch) + ")");
    }

    return allocation;
}

void ModelAllocated(TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleError;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    ctx.rng = Internal::initializeRNG(ctx.config, *ctx.logging.logger);
    ctx.model_config = ctx.config;
    if (ctx.data_info.actual_vocab_size > static_cast<std::uint32_t>(ctx.model_config.vocab_size)) {
        throw std::runtime_error(
            "FATAL: GRMT actual_vocab_size=" + std::to_string(ctx.data_info.actual_vocab_size) +
            " exceeds configured model vocab_size=" + std::to_string(ctx.model_config.vocab_size));
    }

    try {
        ctx.model = Internal::initializeModel(
            ctx.model_config,
            ctx.rng.init_seed,
            *ctx.logging.logger);
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, std::string("FATAL: Model initialization failed: ") + e.what(), 0);
        throw;
    }

    ctx.model_allocation = captureAndValidateModelAllocationOrThrow(ctx);
}

} // namespace GRIMText::Training


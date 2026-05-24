#include "ModelAllocationState.hpp"

#include "ParameterGroupRegistration.hpp"

#include "../InitFacts.hpp"
#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/LogRecorder/LogRecorder.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace Internal {

std::unique_ptr<GRIM::LanguageModel> initializeModel(
    const StartupConfig& config,
    const GRIM::HyperParameters::LanguageModelConfig& model_config,
    uint64_t weight_init_seed,
    TrainingLogger& logger,
    std::string& loaded_checkpoint_path)
{
    loaded_checkpoint_path.clear();

    logger.log("Initializing model with weight_init_seed=" + std::to_string(weight_init_seed) + "...");

    if (model_config.hardcoded_hidden_pattern != GRIM::HyperParameters::HardcodedPattern::DISABLED) {
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

    auto model = std::make_unique<GRIM::LanguageModel>(model_config);

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
        for (int layer = 0; layer < model_config.num_layers; ++layer) {
            auto* enc = gpu_encoder->getLayer(layer);
            if (!enc) {
                throw std::runtime_error("Encoder layer " + std::to_string(layer) + " is NULL after GPU model layer assembly");
            }
        }
        logger.log("✓ Encoder layers verified");
    }
#endif

    std::vector<std::pair<int, std::string>> checkpoint_candidates;
    if (fs::exists(config.paths.checkpoint_dir) && fs::is_directory(config.paths.checkpoint_dir)) {
        for (const auto& entry : fs::directory_iterator(config.paths.checkpoint_dir)) {
            const auto& p = entry.path();
            if (p.extension() == ".bin" && p.stem().string().rfind("checkpoint_epoch_", 0) == 0) {
                std::string stem = p.stem().string();
                std::string epoch_str = stem.substr(std::string("checkpoint_epoch_").size());
                try {
                    int epoch = std::stoi(epoch_str);
                    checkpoint_candidates.emplace_back(epoch, p.string());
                } catch (const std::exception& e) {
                    logger.log("[WARNING] Skipping malformed checkpoint filename: " + stem + " (" + e.what() + ")");
                }
            }
        }
    }

    std::sort(checkpoint_candidates.begin(), checkpoint_candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    bool loaded_checkpoint = false;
    for (const auto& [epoch, checkpoint_path] : checkpoint_candidates) {
        logger.log("Found checkpoint candidate: " + checkpoint_path + " (epoch " + std::to_string(epoch) + ")");
        if (model->load(checkpoint_path)) {
            logger.log("✓ Loaded weights from checkpoint: " + checkpoint_path);
            loaded_checkpoint = true;
            loaded_checkpoint_path = checkpoint_path;
            break;
        }
        logger.log("⚠ Failed to load checkpoint candidate, trying older checkpoint");
    }

    if (!loaded_checkpoint) {
        if (!checkpoint_candidates.empty()) {
            logger.log("⚠ No loadable checkpoint found, starting fresh");
        } else {
            logger.log("No checkpoint found, starting fresh");
        }
    }

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

    ModelAllocationState allocation;
    allocation.model_max_cached_batch = ctx.model_config.max_cached_batch;
    allocation.model_max_tokens_per_batch = ctx.model_config.max_tokens_per_batch;

    if (allocation.model_max_cached_batch != fixed_shape.batch_size) {
        throw std::runtime_error("FATAL: model max_cached_batch does not match trainingFixedShapeHP (model=" +
                                 std::to_string(allocation.model_max_cached_batch) +
                                 " grouping=" + std::to_string(fixed_shape.batch_size) + ")");
    }
    if (ctx.model_config.max_cached_seq_len != fixed_shape.max_seq_len) {
        throw std::runtime_error("FATAL: model max_cached_seq_len does not match trainingFixedShapeHP (model=" +
                                 std::to_string(ctx.model_config.max_cached_seq_len) +
                                 " grouping=" + std::to_string(fixed_shape.max_seq_len) + ")");
    }
    if (allocation.model_max_tokens_per_batch != fixed_shape.max_tokens_per_batch) {
        throw std::runtime_error("FATAL: model max_tokens_per_batch does not match trainingFixedShapeHP (model=" +
                                 std::to_string(allocation.model_max_tokens_per_batch) +
                                 " grouping=" + std::to_string(fixed_shape.max_tokens_per_batch) + ")");
    }

    const auto& token_ids_shape = state.cached_token_ids_tensor.shape.require("ModelAllocated cached_token_ids_tensor");
    if (!token_ids_shape.is_2d_layout()) {
        throw std::runtime_error("FATAL: cached_token_ids_tensor must be a 2D token-id buffer");
    }
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
    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);
    ctx.model_config = GRIM::HyperParameters::startupLanguageModelConfig(
        ctx.config,
        ctx.data_info.actual_vocab_size,
        fixed_shape);

    try {
        ctx.model = Internal::initializeModel(
            ctx.config,
            ctx.model_config,
            ctx.rng.init_seed,
            *ctx.logging.logger,
            ctx.loaded_checkpoint_path);
    } catch (const std::exception& e) {
        EmitModuleError(ModuleId::Training, std::string("FATAL: Model initialization failed: ") + e.what(), 0);
        throw;
    }

    verifyAndDumpInitFacts(ctx);
    ctx.model_allocation = captureAndValidateModelAllocationOrThrow(ctx);

    if (ctx.config.save_test_mode) {
        ctx.logging.logger->log("========================================");
        ctx.logging.logger->log("  SAVE TEST MODE");
        ctx.logging.logger->log("========================================");
        std::string test_save_path = ctx.config.paths.checkpoint_dir + "/save_test.bin";
        ctx.logging.logger->log("Testing model->save() to: " + test_save_path);
        bool save_ok = ctx.model->save(test_save_path);
        if (save_ok) {
            EmitModuleInfo(ModuleId::Checkpoint, "✓ Save test PASSED", 0);
            if (fs::exists(test_save_path)) {
                auto file_size = fs::file_size(test_save_path);
                EmitModuleInfo(ModuleId::Checkpoint,
                    std::string("  File size: ") + std::to_string(file_size) + " bytes", 0);
            }
        } else {
            EmitModuleError(ModuleId::Checkpoint, "✗ Save test FAILED", 0);
        }
        std::exit(save_ok ? 0 : 1);
    }
}

} // namespace GRIMText::Training


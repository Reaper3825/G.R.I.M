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
    const RunCapacity& run_capacity,
    uint32_t vocab_size,
    uint64_t weight_init_seed,
    TrainingLogger& logger,
    std::string& loaded_checkpoint_path)
{
    loaded_checkpoint_path.clear();

    logger.log("Initializing model with weight_init_seed=" + std::to_string(weight_init_seed) + "...");

    const GRIM::HyperParameters::StartupModelCapacityHP model_capacity{
        static_cast<int>(run_capacity.batch_rows),
        static_cast<int>(run_capacity.seq_cap),
        static_cast<int>(run_capacity.max_tokens_per_batch)
    };
    GRIM::HyperParameters::LanguageModelConfig model_config =
        GRIM::HyperParameters::startupLanguageModelConfig(config, vocab_size, model_capacity);

    if (model_config.hardcoded_hidden_pattern != GRIM::HyperParameters::LanguageModelConfig::HardcodedPattern::DISABLED) {
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

    auto model = std::make_unique<GRIM::LanguageModel>(model_config, config.hyperparameters);
    const auto& hp = model->requireTrainingHyperparameters("initializeModel");

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

    logger.log("Initializing TrainingState (grad buffers, activation caches)...");
    model->initTrainingState();
    logger.log("✓ TrainingState fully initialized");

    GRIMText::Training::Startup::ModelRegistration::buildParameterGroups(*model);

    GRIM::LossContext::LossOptions loss_opts{};
    {
        loss_opts.label_smoothing_enabled    = hp.loss_label_smoothing_enabled;
        loss_opts.label_smoothing_epsilon    = hp.loss_label_smoothing_epsilon;
        loss_opts.focal_enabled              = hp.loss_focal_enabled;
        loss_opts.focal_gamma                = hp.loss_focal_gamma;
        loss_opts.focal_alpha                = hp.loss_focal_alpha;
        loss_opts.preference_enabled         = hp.loss_preference_enabled;
        loss_opts.preference_beta            = hp.loss_preference_beta;
        loss_opts.distillation_enabled       = hp.loss_distillation_enabled;
        loss_opts.distillation_temperature   = hp.loss_distillation_temperature;
        loss_opts.distillation_lambda        = hp.loss_distillation_lambda;
        loss_opts.masking_enabled            = hp.loss_masking_enabled;
        loss_opts.masking_tag                = hp.loss_masking_tag;
        loss_opts.entropy_reg_enabled        = hp.loss_entropy_reg_enabled;
        loss_opts.entropy_reg_lambda         = hp.loss_entropy_reg_lambda;
        loss_opts.class_balanced_enabled     = hp.loss_class_balanced_enabled;
        loss_opts.class_balanced_beta        = hp.loss_class_balanced_beta;
    }
    model->setLossOptions(loss_opts);

#ifdef USE_CUDA
    {
        auto* gpu_encoder = &model->getGpuEncoder();
        const auto& cfg = model->getConfig();
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
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

    const auto& model_cfg = ctx.model->getConfig();
    const auto& state = ctx.model->getTrainingState();
    const auto& cap = ctx.run_capacity;

    ModelAllocationState allocation;
    allocation.model_max_cached_batch = model_cfg.max_cached_batch;
    allocation.model_max_cached_seq_len = model_cfg.max_cached_seq_len;
    allocation.model_max_tokens_per_batch = model_cfg.max_tokens_per_batch;

    if (allocation.model_max_cached_batch != static_cast<int>(cap.batch_rows)) {
        throw std::runtime_error("FATAL: model max_cached_batch does not match RunCapacity (model=" +
                                 std::to_string(allocation.model_max_cached_batch) +
                                 " stem=" + std::to_string(cap.batch_rows) + ")");
    }
    if (allocation.model_max_cached_seq_len != cap.seq_cap) {
        throw std::runtime_error("FATAL: model max_cached_seq_len does not match RunCapacity (model=" +
                                 std::to_string(allocation.model_max_cached_seq_len) +
                                 " stem=" + std::to_string(cap.seq_cap) + ")");
    }
    if (allocation.model_max_tokens_per_batch != static_cast<int>(cap.max_tokens_per_batch)) {
        throw std::runtime_error("FATAL: model max_tokens_per_batch does not match RunCapacity (model=" +
                                 std::to_string(allocation.model_max_tokens_per_batch) +
                                 " stem=" + std::to_string(cap.max_tokens_per_batch) + ")");
    }

    const auto& token_shape = state.cached_token_ids_tensor.shape.require("ModelAllocated cached_token_ids_tensor");
    if (!token_shape.is_2d_layout()) {
        throw std::runtime_error("FATAL: cached_token_ids_tensor must be a 2D token buffer");
    }
    const auto& logits_shape = state.cached_logits_tensor.shape.require("ModelAllocated cached_logits_tensor");
    if (!logits_shape.is_2d_layout()) {
        throw std::runtime_error("FATAL: cached_logits_tensor must be a 2D logits buffer");
    }
    if (token_shape.as_2d().cols != static_cast<int>(cap.max_tokens_per_batch)) {
        throw std::runtime_error("FATAL: cached_token_ids_tensor capacity does not match RunCapacity (tensor=" +
                                 std::to_string(token_shape.as_2d().cols) +
                                 " stem=" + std::to_string(cap.max_tokens_per_batch) + ")");
    }
    if (logits_shape.as_2d().rows != static_cast<int>(cap.max_tokens_per_batch)) {
        throw std::runtime_error("FATAL: cached_logits_tensor row capacity does not match RunCapacity (tensor=" +
                                 std::to_string(logits_shape.as_2d().rows) +
                                 " stem=" + std::to_string(cap.max_tokens_per_batch) + ")");
    }

    return allocation;
}

void ModelAllocated(TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleError;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    ctx.rng = Internal::initializeRNG(ctx.config, *ctx.logging.logger);

    try {
        ctx.model = Internal::initializeModel(
            ctx.config,
            ctx.run_capacity,
            ctx.config.actual_vocab_size,
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


#include "OptimizerReady.hpp"

#include "ClassBalancedWeights.hpp"
#include "Model/ParameterGroupRegistration.hpp"
#include "../Phase1_Startup.hpp"
#include "../../OptimizerCheckpoint.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace Internal {

void initializeOptimizer(TrainingContext& ctx) {
    auto& logger = *ctx.logging.logger;
    auto& opt = ctx.optimizer;
    const auto soft_restart_hp =
        GRIM::HyperParameters::softRestartHP(ctx.config);

    logger.log("Initializing optimizer state...");
    opt.optimizer_step.step = 0;
    GRIMText::Training::Startup::ModelRegistration::bindOptimizerState(
        ctx.parameter_registry,
        opt.optimizer_state,
        ctx.requireTrainingState("OptimizerReady::initializeOptimizer").stream_ctrl.getPrimaryStream());

    GRIM::SoftRestart::SoftRestartConfig sr_cfg;
    sr_cfg.cooldown_steps = soft_restart_hp.cooldown_steps;
    opt.soft_restart_controller = GRIM::SoftRestart::SoftRestartController(sr_cfg);

    opt.resetAccumulationSlot();

    auto* gpu_encoder = ctx.gpu_model.gpu_encoder.get();
    if (!gpu_encoder) {
        throw std::runtime_error(
            "OptimizerReady::initializeOptimizer: ctx.gpu_model.gpu_encoder is NULL - "
            "ensure Startup::assembleGpuModel(ctx.config, ctx.training_state, ctx.gpu_model, ctx.parameter_registry, weight_init_seed) completes before optimizer init");
    }
    const int num_layers = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "num_layers");
    for (int layer = 0; layer < num_layers; ++layer) {
        if (!gpu_encoder->getLayer(layer)) {
            throw std::runtime_error("Encoder layer " + std::to_string(layer) + " not initialized - "
                                     "ensure Startup::assembleGpuModel(ctx.config, ctx.training_state, ctx.gpu_model, ctx.parameter_registry, weight_init_seed) completes all layers before training");
        }
    }

    logger.log("✓ Optimizer state initialized");

    if (ctx.loaded_checkpoint_path.empty()) return;

    const std::string opt_path = optimizerSidecarPath(ctx.loaded_checkpoint_path);
    if (!fs::exists(opt_path)) {
        logger.log("⚠ No optimizer sidecar for loaded checkpoint: " + ctx.loaded_checkpoint_path);
        logger.log("  Continuing with fresh optimizer state (moments reset, LR from step 0)");
        return;
    }

    logger.log("Found optimizer sidecar for loaded checkpoint: " + opt_path);
    try {
        loadOptimizerState(ctx, opt_path);
    } catch (const std::exception& e) {
        logger.log(std::string("⚠ Optimizer sidecar load failed: ") + e.what());
        logger.log("  Continuing with fresh optimizer state for checkpoint: " + ctx.loaded_checkpoint_path);
    }
}

} // namespace Internal

void OptimizerReady(TrainingContext& ctx) {
    Internal::initializeOptimizer(ctx);

    const auto loss_config = GRIM::HyperParameters::lossConfigHP(ctx.config);
    if (loss_config.class_balanced_enabled) {
        computeAndUploadClassBalancedWeights(
            ctx.data.train_seqs,
            ctx.data.vocab_size,
            loss_config.class_balanced_beta,
            ctx.requireTrainingState("OptimizerReady"),
            *ctx.logging.logger);
    }
}

} // namespace GRIMText::Training
#include "ResumeState.hpp"

#include "../ClassBalancedWeights.hpp"
#include "../../Phase1_Startup.hpp"
#include "../../../OptimizerCheckpoint.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace Internal {

void initializeOptimizer(TrainingContext& ctx) {
    auto& logger = *ctx.logging.logger;
    auto& opt = ctx.optimizer;
    auto& model = *ctx.model;
    const auto& hp = ctx.config.hyperparameters;

    logger.log("Initializing optimizer state...");
    opt.optimizer_state.step = 0;

    GRIM::SoftRestart::SoftRestartConfig sr_cfg;
    sr_cfg.cooldown_steps = hp.soft_restart_cooldown_steps;
    opt.soft_restart_controller = GRIM::SoftRestart::SoftRestartController(sr_cfg);

    opt.accumulation_position = 0;

    auto* gpu_encoder = &model.getGpuEncoder();
    for (int layer = 0; layer < model.getConfig().num_layers; ++layer) {
        if (!gpu_encoder->getLayer(layer)) {
            throw std::runtime_error("Encoder layer " + std::to_string(layer) + " not initialized - "
                                     "ensure model.initGPU() completes all layers before training");
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

ResumeState captureResumeState(const TrainingContext& ctx) {
    ResumeState st;
    st.loaded_checkpoint_path = ctx.loaded_checkpoint_path;
    st.resumed = !ctx.loaded_checkpoint_path.empty();

    if (st.resumed) {
        st.optimizer_sidecar_path = optimizerSidecarPath(ctx.loaded_checkpoint_path);
    }

    st.optimizer_step = ctx.optimizer.optimizer_state.step;
    st.global_step = ctx.global_step;
    st.best_val_loss = ctx.best_val_loss;
    st.epochs_completed = ctx.epochs_completed;
    st.accumulation_position = ctx.optimizer.accumulation_position;
    return st;
}

void ResumeStateReady(TrainingContext& ctx) {
    Internal::initializeOptimizer(ctx);
    ctx.resume_state = captureResumeState(ctx);

    if (ctx.config.hyperparameters.loss_class_balanced_enabled) {
        computeAndUploadClassBalancedWeights(
            ctx.data.train_seqs,
            ctx.config.actual_vocab_size,
            ctx.config.hyperparameters.loss_class_balanced_beta,
            ctx.model->getTrainingState(),
            *ctx.logging.logger);
    }
}

} // namespace GRIMText::Training


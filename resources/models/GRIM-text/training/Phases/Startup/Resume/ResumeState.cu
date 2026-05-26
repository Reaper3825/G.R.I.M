#include "ResumeState.hpp"

#include "../ClassBalancedWeights.hpp"
#include "../Model/ParameterGroupRegistration.hpp"
#include "../../Phase1_Startup.hpp"
#include "../../../OptimizerCheckpoint.hpp"

#include <cstdint>
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
    const auto soft_restart_hp =
        GRIM::HyperParameters::softRestartHP(ctx.config);

    logger.log("Initializing optimizer state...");
    opt.optimizer_step.step = 0;
    GRIMText::Training::Startup::ModelRegistration::bindOptimizerState(
        model,
        opt.optimizer_state,
        model.getTrainingState().stream_ctrl.getPrimaryStream());

    GRIM::SoftRestart::SoftRestartConfig sr_cfg;
    sr_cfg.cooldown_steps = soft_restart_hp.cooldown_steps;
    opt.soft_restart_controller = GRIM::SoftRestart::SoftRestartController(sr_cfg);

    opt.resetAccumulationSlot();

    auto* gpu_encoder = &model.getGpuEncoder();
    for (int layer = 0; layer < ctx.config.num_layers; ++layer) {
        if (!gpu_encoder->getLayer(layer)) {
            throw std::runtime_error("Encoder layer " + std::to_string(layer) + " not initialized - "
                                     "ensure Startup::assembleGpuModel(*model, weight_init_seed) completes all layers before training");
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

    st.optimizer_step = ctx.optimizer.optimizer_step.step;
    st.global_step = ctx.global_step;
    st.best_val_loss = ctx.best_val_loss;
    st.epochs_completed = ctx.epochs_completed;
    st.accumulation_slot = ctx.optimizer.accumulationSlot();
    return st;
}

void ResumeStateReady(TrainingContext& ctx) {
    Internal::initializeOptimizer(ctx);
    ctx.resume_state = captureResumeState(ctx);

    const auto loss_config = GRIM::HyperParameters::lossConfigHP(ctx.config);
    if (loss_config.class_balanced_enabled) {
        computeAndUploadClassBalancedWeights(
            ctx.data.train_seqs,
            ctx.data_info.actual_vocab_size,
            loss_config.class_balanced_beta,
            ctx.model->getTrainingState(),
            *ctx.logging.logger);
    }
}

} // namespace GRIMText::Training


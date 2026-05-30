#include "../Phase1_Startup.hpp"

#include "CheckpointLoad.hpp"
#include "InitFacts.hpp"

#include "../../../Common/grim_model_serialization.hpp"
#include "../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../Shared/LogRecorder/LogRecorder.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace {

void handleUnusableCheckpointRequest(
    const GRIM::HyperParameters::CheckpointLoadHP& checkpoint_hp,
    TrainingLogger& logger,
    const std::string& reason)
{
    if (checkpoint_hp.execution_mode == GRIM::HyperParameters::ModelExecutionMode::INFERENCE) {
        throw std::runtime_error(
            "CheckpointLoaded: inference requires a usable explicit checkpoint path: " + reason);
    }

    logger.log("Checkpoint request unusable for TRAINING execution mode: " + reason);
    logger.log("Starting fresh model state for TRAINING execution mode");
}

void loadRequestedCheckpoint(TrainingContext& ctx)
{
    if (!ctx.model) {
        throw std::runtime_error("CheckpointLoaded: model is NULL; call ModelAllocated(ctx) before CheckpointLoaded(ctx)");
    }
    if (!ctx.logging.logger) {
        throw std::runtime_error("CheckpointLoaded: logger is NULL; call LoggingReady(ctx) before CheckpointLoaded(ctx)");
    }

    auto& logger = *ctx.logging.logger;
    ctx.loaded_checkpoint_path.clear();

    const auto checkpoint_hp = GRIM::HyperParameters::checkpointLoadHP(
        ctx.config,
        ctx.requested_checkpoint_path,
        GRIM::HyperParameters::snapshotExecutionMode(ctx.config));

    if (checkpoint_hp.checkpoint_path.empty()) {
        handleUnusableCheckpointRequest(
            checkpoint_hp,
            logger,
            "no explicit checkpoint path was provided");
        return;
    }

    const fs::path checkpoint_path(checkpoint_hp.checkpoint_path);
    if (!fs::exists(checkpoint_path)) {
        handleUnusableCheckpointRequest(
            checkpoint_hp,
            logger,
            "requested checkpoint does not exist: " + checkpoint_hp.checkpoint_path);
        return;
    }
    if (!fs::is_regular_file(checkpoint_path)) {
        handleUnusableCheckpointRequest(
            checkpoint_hp,
            logger,
            "requested checkpoint is not a regular file: " + checkpoint_hp.checkpoint_path);
        return;
    }

    logger.log("Loading requested checkpoint: " + checkpoint_hp.checkpoint_path);
    if (!GRIM::loadLanguageModelCheckpoint(*ctx.model, ctx.gpu_model, ctx.parameter_registry, checkpoint_hp.checkpoint_path)) {
        handleUnusableCheckpointRequest(
            checkpoint_hp,
            logger,
            "loadLanguageModelCheckpoint() failed for requested checkpoint: " + checkpoint_hp.checkpoint_path);
        return;
    }

    logger.log("✓ Loaded weights from checkpoint: " + checkpoint_hp.checkpoint_path);
    ctx.loaded_checkpoint_path = checkpoint_hp.checkpoint_path;
}

void runSaveTestIfRequested(TrainingContext& ctx)
{
    if (!GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "save_test_mode")) {
        return;
    }
    if (!ctx.model) {
        throw std::runtime_error("CheckpointLoaded save-test: model is NULL");
    }
    if (!ctx.logging.logger) {
        throw std::runtime_error("CheckpointLoaded save-test: logger is NULL");
    }

    using GRIM::Logging::EmitModuleError;
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);
    ctx.logging.logger->log("========================================");
    ctx.logging.logger->log("  SAVE TEST MODE");
    ctx.logging.logger->log("========================================");
    std::string test_save_path = paths_hp.checkpoint_dir + "/save_test.bin";
    ctx.logging.logger->log("Testing saveLanguageModelCheckpoint() to: " + test_save_path);
    bool save_ok = GRIM::saveLanguageModelCheckpoint(*ctx.model, ctx.gpu_model, ctx.parameter_registry, test_save_path);
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

} // namespace

void CheckpointLoaded(TrainingContext& ctx) {
    loadRequestedCheckpoint(ctx);
    verifyAndDumpInitFacts(ctx);
    runSaveTestIfRequested(ctx);
}

} // namespace GRIMText::Training

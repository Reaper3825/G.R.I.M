#include "CheckpointLoad.hpp"

#include "InitFacts.hpp"
#include "../Phase1_Startup.hpp"

#include "../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../Shared/LogRecorder/LogRecorder.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace {

std::vector<std::pair<int, std::string>> discoverCheckpointCandidates(
    const GRIM::HyperParameters::PathsHP& paths_hp,
    TrainingLogger& logger)
{
    std::vector<std::pair<int, std::string>> checkpoint_candidates;
    if (fs::exists(paths_hp.checkpoint_dir) && fs::is_directory(paths_hp.checkpoint_dir)) {
        for (const auto& entry : fs::directory_iterator(paths_hp.checkpoint_dir)) {
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
    return checkpoint_candidates;
}

void loadMostRecentCheckpoint(TrainingContext& ctx)
{
    if (!ctx.model) {
        throw std::runtime_error("CheckpointLoaded: model is NULL; call ModelAllocated(ctx) before CheckpointLoaded(ctx)");
    }
    if (!ctx.logging.logger) {
        throw std::runtime_error("CheckpointLoaded: logger is NULL; call LoggingReady(ctx) before CheckpointLoaded(ctx)");
    }

    auto& logger = *ctx.logging.logger;
    ctx.loaded_checkpoint_path.clear();

    const auto paths_hp = GRIM::HyperParameters::pathsHP(ctx.config);
    auto checkpoint_candidates = discoverCheckpointCandidates(paths_hp, logger);

    bool loaded_checkpoint = false;
    for (const auto& [epoch, checkpoint_path] : checkpoint_candidates) {
        logger.log("Found checkpoint candidate: " + checkpoint_path + " (epoch " + std::to_string(epoch) + ")");
        if (ctx.model->load(checkpoint_path)) {
            logger.log("✓ Loaded weights from checkpoint: " + checkpoint_path);
            loaded_checkpoint = true;
            ctx.loaded_checkpoint_path = checkpoint_path;
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
}

void runSaveTestIfRequested(TrainingContext& ctx)
{
    if (!ctx.config.save_test_mode) {
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

} // namespace

void CheckpointLoaded(TrainingContext& ctx) {
    loadMostRecentCheckpoint(ctx);
    verifyAndDumpInitFacts(ctx);
    runSaveTestIfRequested(ctx);
}

} // namespace GRIMText::Training

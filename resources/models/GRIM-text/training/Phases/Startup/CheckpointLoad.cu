#include "../Phase1_Startup.hpp"

#include "CheckpointLoad.hpp"
#include "InitFacts.hpp"

#include "../../../Common/grim_model_serialization.hpp"
#include "../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../Shared/LogRecorder/LogRecorder.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace GRIMText::Training {

namespace {

void logFreshRandomInitialization(
    const TrainingContext& ctx,
    TrainingLogger& logger,
    const std::string& reason)
{
    logger.log(
        "[MODEL_INIT] RESULT=RANDOM_INIT checkpoint_loaded=false weight_init_seed=" +
        std::to_string(ctx.rng.init_seed) + " reason=\"" + reason + "\"");
}

std::string trimmed(std::string value) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

// Read training.config.grim_text_checkpoint_select tolerantly — the key is
// optional, so a missing or non-string value yields an empty selection.
std::string checkpointSelectField(const GRIM::Config::AiConfigSnapshot& snapshot) {
    const auto& cfg = GRIM::HyperParameters::snapshotTrainingConfig(snapshot);
    if (!cfg.contains("grim_text_checkpoint_select")) {
        return std::string();
    }
    const auto& value = cfg.at("grim_text_checkpoint_select");
    if (!value.is_string()) {
        return std::string();
    }
    return trimmed(value.get<std::string>());
}

// Parse a trailing integer from a filename stem (e.g. "checkpoint_epoch_3" -> 3).
// Returns -1 when the stem has no trailing digits.
long long trailingEpochNumber(const std::string& stem) {
    std::size_t end = stem.size();
    std::size_t begin = end;
    while (begin > 0 && std::isdigit(static_cast<unsigned char>(stem[begin - 1]))) {
        --begin;
    }
    if (begin == end) {
        return -1;
    }
    try {
        return std::stoll(stem.substr(begin));
    } catch (const std::exception&) {
        return -1;
    }
}

// Ordered list of usable checkpoints in checkpoint_dir, "latest" first.
// Ranking: most recent modification time, then highest parsed epoch number.
// Partial atomic-write artifacts (e.g. "*.bin.transfer.497") and the ".mtp"
// side-cars are skipped by requiring a ".bin" extension.
std::vector<std::string> rankedCheckpoints(const std::string& checkpoint_dir) {
    std::vector<std::string> result;
    if (checkpoint_dir.empty()) {
        return result;
    }
    std::error_code ec;
    if (!fs::is_directory(checkpoint_dir, ec)) {
        return result;
    }

    struct Candidate {
        std::string path;
        fs::file_time_type write_time;
        long long epoch;
    };
    std::vector<Candidate> candidates;
    for (const auto& entry : fs::directory_iterator(checkpoint_dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        const auto& path = entry.path();
        if (path.extension() != ".bin") {
            continue;
        }
        const auto size = fs::file_size(path, ec);
        if (ec || size == 0) {
            ec.clear();
            continue;
        }
        const auto write_time = fs::last_write_time(path, ec);
        if (ec) {
            ec.clear();
            continue;
        }
        candidates.push_back({path.string(), write_time, trailingEpochNumber(path.stem().string())});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) {
                  if (a.write_time != b.write_time) {
                      return a.write_time > b.write_time;
                  }
                  if (a.epoch != b.epoch) {
                      return a.epoch > b.epoch;
                  }
                  return a.path > b.path;
              });

    result.reserve(candidates.size());
    for (auto& c : candidates) {
        result.push_back(std::move(c.path));
    }
    return result;
}

// Resolve the ordered set of checkpoint candidates to attempt, honoring
// training.config.grim_text_checkpoint_select:
//   - ""/"default"    → the configured grim_text_model path (legacy behavior)
//   - "latest"        → every *.bin in the checkpoint dir, newest first
//   - "<name>.bin"    → that file inside the checkpoint dir
//   - "<path>/<name>" → an explicit (possibly relative) path
std::vector<std::string> resolveCheckpointCandidates(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const std::string& requested_path,
    TrainingLogger& logger)
{
    const std::string select = checkpointSelectField(snapshot);

    if (select.empty() || select == "default") {
        if (requested_path.empty()) {
            return {};
        }
        return {requested_path};
    }

    const auto paths_hp = GRIM::HyperParameters::pathsHP(snapshot);

    if (select == "latest") {
        auto candidates = rankedCheckpoints(paths_hp.checkpoint_dir);
        if (candidates.empty()) {
            logger.log("Checkpoint select=\"latest\": no usable checkpoint found in " +
                       paths_hp.checkpoint_dir);
        } else {
            logger.log("Checkpoint select=\"latest\": newest candidate is " + candidates.front());
        }
        return candidates;
    }

    // Specific selection: a bare filename resolves against the checkpoint dir,
    // anything with a separator is treated as an explicit path.
    const fs::path select_path(select);
    std::string resolved;
    if (select_path.has_parent_path() || select_path.is_absolute()) {
        resolved = GRIM::HyperParameters::resolveMappedPath(select);
    } else {
        resolved = (fs::path(paths_hp.checkpoint_dir) / select_path).string();
    }
    logger.log("Checkpoint select=\"" + select + "\": resolved to " + resolved);
    return {resolved};
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

    const auto execution_mode = GRIM::HyperParameters::snapshotExecutionMode(ctx.config);
    const std::string select = checkpointSelectField(ctx.config);
    const std::string effective_select = select.empty() ? "default" : select;
    const std::string requested_path = ctx.requested_checkpoint_path.empty()
        ? "<none>"
        : ctx.requested_checkpoint_path;
    logger.log(
        "[MODEL_INIT] Choosing parameter source | execution_mode=" +
        std::string(GRIM::HyperParameters::modelExecutionModeToJsonString(execution_mode)) +
        " checkpoint_select=\"" + effective_select +
        "\" requested_path=\"" + requested_path + "\"");

    const std::vector<std::string> candidates =
        resolveCheckpointCandidates(ctx.config, ctx.requested_checkpoint_path, logger);

    if (candidates.empty()) {
        logFreshRandomInitialization(
            ctx,
            logger,
            "no checkpoint path was provided or discovered");
        return;
    }

    std::string last_reason;
    for (const auto& candidate : candidates) {
        const fs::path checkpoint_path(candidate);
        if (!fs::exists(checkpoint_path)) {
            last_reason = "requested checkpoint does not exist: " + candidate;
            logger.log(last_reason);
            continue;
        }
        if (!fs::is_regular_file(checkpoint_path)) {
            last_reason = "requested checkpoint is not a regular file: " + candidate;
            logger.log(last_reason);
            continue;
        }

        logger.log("Loading requested checkpoint: " + candidate);
        if (!GRIM::loadLanguageModelCheckpoint(*ctx.model, ctx.requireTrainingState("CheckpointLoaded"), ctx.gpu_model, ctx.parameter_registry, candidate)) {
            last_reason = "loadLanguageModelCheckpoint() failed for requested checkpoint: " + candidate;
            logger.log(last_reason);
            if (candidates.size() > 1) {
                logger.log("Trying next checkpoint candidate...");
            }
            continue;
        }

        logger.log("✓ Loaded weights from checkpoint: " + candidate);
        ctx.loaded_checkpoint_path = candidate;
        logger.log(
            "[MODEL_INIT] RESULT=CHECKPOINT_LOADED checkpoint_loaded=true path=\"" +
            candidate + "\"");
        return;
    }

    logFreshRandomInitialization(
        ctx,
        logger,
        last_reason.empty() ? "no usable checkpoint candidate" : last_reason);
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

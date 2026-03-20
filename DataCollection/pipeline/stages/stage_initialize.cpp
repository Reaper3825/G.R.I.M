#include "stage_initialize.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "DataCollection/io/dataset_io_json.hpp"
#include "DataCollection/collection_state.hpp"
#include "DataCollection/training_paths.hpp"
#include "control/ai_config_paths.hpp"

#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;

StageResult StageInitialize::execute(PipelineContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    StageResult result;

    std::cout << "[Initialize] Resolving paths and loading config...\n";

    // ── Load paths from ai_config.json ──────────────────
    GRIM::Config::GrimTextPaths grimPaths;
    if (GRIM::Config::loadGrimTextPaths(grimPaths)) {
        if (ctx.config.sourceConfigPath.empty() && !grimPaths.source_config.empty())
            ctx.config.sourceConfigPath = grimPaths.source_config;
        if (ctx.config.checkpointDir.empty() && !grimPaths.checkpoints.empty())
            ctx.config.checkpointDir = grimPaths.checkpoints;
        if (ctx.config.rawDir.empty() && !grimPaths.collected.empty())
            ctx.config.rawDir = grimPaths.collected;
        if (ctx.config.verifiedDir.empty() && !grimPaths.verified.empty())
            ctx.config.verifiedDir = grimPaths.verified;
        if (ctx.config.outputDir.empty() && !grimPaths.training_data.empty()) {
            fs::path td(grimPaths.training_data);
            ctx.config.outputDir = td.parent_path().string();
        }
        std::cout << "[Initialize] Loaded paths from ai_config.json\n";
    } else {
        std::cout << "[Initialize] WARNING: Could not load ai_config.json, using defaults\n";
    }

    // Apply defaults where still empty
    if (ctx.config.checkpointDir.empty())  ctx.config.checkpointDir  = "data/checkpoints";
    if (ctx.config.rawDir.empty())         ctx.config.rawDir         = "data/raw";
    if (ctx.config.verifiedDir.empty())    ctx.config.verifiedDir    = "data/verified";
    if (ctx.config.outputDir.empty())      ctx.config.outputDir      = "data";

    // ── Load Q/A JSONL paths from ai_config ─────────────
    try {
        std::ifstream configFile("ai_config.json");
        if (configFile.is_open()) {
            nlohmann::json config = nlohmann::json::parse(configFile);
            if (config.contains("data_collection") &&
                config["data_collection"].contains("qa_jsonl_paths")) {
                auto& qa = config["data_collection"]["qa_jsonl_paths"];
                if (qa.is_array()) {
                    for (const auto& p : qa) {
                        if (p.is_string()) ctx.config.qaJsonlPaths.push_back(p.get<std::string>());
                    }
                }
            }
        }
    } catch (...) {}

    // ── Generate run ID and layout ──────────────────────
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    ctx.run.runId = "run_" + std::to_string(epoch);

    ctx.run.outputRoot   = fs::path(ctx.config.outputDir);
    ctx.run.runRoot      = ctx.run.outputRoot / "pipeline_runs" / ctx.run.runId;
    ctx.run.spoolRoot    = ctx.run.runRoot / "spool";
    ctx.run.manifestPath = ctx.run.outputRoot / "manifest.json";

    std::error_code ec;
    fs::create_directories(ctx.run.spoolRoot, ec);
    fs::create_directories(ctx.run.outputRoot / "shards", ec);

    std::cout << "[Initialize] Run ID: " << ctx.run.runId << "\n";
    std::cout << "[Initialize] Spool:  " << ctx.run.spoolRoot.string() << "\n";
    std::cout << "[Initialize] Output: " << ctx.run.outputRoot.string() << "\n";

    // ── Initialize collection state manager ─────────────
    std::string stateDir = ctx.config.checkpointDir + "/collection_state";
    ctx.stateManager = std::make_unique<GRIM::DataCollection::CollectionStateManager>(stateDir);
    std::cout << "[Initialize] State: " << ctx.stateManager->getTotalUniqueUrls() << " tracked URLs, "
              << ctx.stateManager->getTotalUniqueContent() << " content hashes\n";

    // ── Force rebuild detection ─────────────────────────
    if (ctx.config.mode == PipelineMode::MergeRebuild) {
        ctx.config.forceRebuild = true;
    }

    if (!ctx.config.forceRebuild && ctx.stateManager->getTotalUniqueContent() > 0) {
        fs::path trainingGrmt = fs::path(ctx.config.outputDir) / "training_data.grmt";
        if (!fs::exists(trainingGrmt)) {
            std::cout << "[Initialize] Training data missing but "
                      << ctx.stateManager->getTotalUniqueContent()
                      << " content hashes found. Enabling force-rebuild.\n";
            ctx.config.forceRebuild = true;
        }
    }

    // ── Create storage providers ────────────────────────
    ctx.datasetIO     = std::make_shared<DatasetIOJson>();
    ctx.datasetWriter = std::make_shared<AppendOnlyDatasetWriterJson>();

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    std::cout << "[Initialize] Complete (" << result.durationSeconds << "s)\n\n";
    return result;
}

} // namespace Pipeline
} // namespace GRIM

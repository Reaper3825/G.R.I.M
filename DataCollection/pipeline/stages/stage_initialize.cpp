#include "stage_initialize.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "DataCollection/io/dataset_io_json.hpp"
#include "DataCollection/collection_state.hpp"
#include "DataCollection/training_paths.hpp"
#include "resources.hpp"

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

    const auto resolveFromGrimRoot = [](const std::string& rawPath) {
        fs::path path(rawPath);
        if (path.is_absolute()) {
            return path;
        }
        return fs::path(getGrimRootDir()) / path;
    };

    // ── Load paths from ai_config.json ──────────────────
    const auto& grimTextPaths = aiConfig.at("paths").at("grim_text");
    if (ctx.config.sourceConfigPath.empty())
        ctx.config.sourceConfigPath = resolveFromGrimRoot(grimTextPaths.at("source_config").get<std::string>()).string();
    if (ctx.config.checkpointDir.empty())
        ctx.config.checkpointDir = resolveFromGrimRoot(grimTextPaths.at("checkpoints").get<std::string>()).string();
    if (ctx.config.rawDir.empty())
        ctx.config.rawDir = resolveFromGrimRoot(grimTextPaths.at("collected").get<std::string>()).string();
    if (ctx.config.verifiedDir.empty())
        ctx.config.verifiedDir = resolveFromGrimRoot(grimTextPaths.at("verified").get<std::string>()).string();
    if (ctx.config.outputDir.empty()) {
        fs::path td = resolveFromGrimRoot(grimTextPaths.at("training_data").get<std::string>());
        ctx.config.outputDir = td.parent_path().string();
    }
    std::cout << "[Initialize] Loaded paths from ai_config.json\n";

    if (ctx.config.checkpointDir.empty())  ctx.config.checkpointDir  = "data/checkpoints";
    if (ctx.config.rawDir.empty())         ctx.config.rawDir         = "data/raw";
    if (ctx.config.verifiedDir.empty())    ctx.config.verifiedDir    = "data/verified";
    if (ctx.config.outputDir.empty())      ctx.config.outputDir      = "data";

    // ── Load Q/A JSONL paths from ai_config ─────────────
    const auto& dataCollection = aiConfig.at("data_collection");
    if (dataCollection.contains("qa_jsonl_paths") && dataCollection["qa_jsonl_paths"].is_array()) {
        for (const auto& p : dataCollection["qa_jsonl_paths"]) {
            if (p.is_string()) {
                ctx.config.qaJsonlPaths.push_back(resolveFromGrimRoot(p.get<std::string>()).string());
            }
        }
    }

    // ── Generate run ID and layout ──────────────────────
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    ctx.run.runId = "run_" + std::to_string(epoch);

    ctx.run.outputRoot      = fs::path(ctx.config.outputDir);
    ctx.run.runRoot         = ctx.run.outputRoot / "pipeline_runs" / ctx.run.runId;
    ctx.run.spoolRoot       = ctx.run.runRoot / "spool";
    ctx.run.massDatasetPath = ctx.run.outputRoot / "mass_dataset.jsonl";

    std::error_code ec;
    fs::create_directories(ctx.run.spoolRoot, ec);
    fs::create_directories(ctx.run.outputRoot, ec);

    std::cout << "[Initialize] Run ID: " << ctx.run.runId << "\n";
    std::cout << "[Initialize] Spool:  " << ctx.run.spoolRoot.string() << "\n";
    std::cout << "[Initialize] Output: " << ctx.run.massDatasetPath.string() << "\n";

    // ── Initialize collection state manager ─────────────
    std::string stateDir = ctx.config.checkpointDir + "/collection_state";
    ctx.stateManager = std::make_unique<GRIM::DataCollection::CollectionStateManager>(stateDir);
    std::cout << "[Initialize] State: " << ctx.stateManager->getTotalUniqueUrls() << " tracked URLs, "
              << ctx.stateManager->getTotalUniqueContent() << " content hashes\n";

    // ── Force rebuild (only when mode explicitly requests it) ──
    if (ctx.config.mode == PipelineMode::MergeRebuild) {
        ctx.config.forceRebuild = true;
    }

    // ── Create storage provider ─────────────────────────
    ctx.datasetIO = std::make_shared<DatasetIOJson>();

    size_t existingCount = ctx.datasetIO->countEntries(ctx.run.massDatasetPath);
    std::cout << "[Initialize] Mass dataset: " << existingCount << " existing entries\n";

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    std::cout << "[Initialize] Complete (" << result.durationSeconds << "s)\n\n";
    return result;
}

} // namespace Pipeline
} // namespace GRIM

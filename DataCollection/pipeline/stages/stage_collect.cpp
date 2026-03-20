#include "stage_collect.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "../../web_collector.hpp"
#include "../../../control/ai_config_paths.hpp"

#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;
using GRIM::Training::WebDataCollector;

StageResult StageCollect::execute(PipelineContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    StageResult result;

    std::cout << "=== COLLECTING DATA ===\n\n";

    if (ctx.config.sourceConfigPath.empty()) {
        result.success = false;
        result.errorMessage = "Source config path not specified";
        return result;
    }

    if (!fs::exists(ctx.config.sourceConfigPath)) {
        result.success = false;
        result.errorMessage = "Config file not found: " + ctx.config.sourceConfigPath;
        return result;
    }

    std::cout << "[Collect] Loading sources from " << ctx.config.sourceConfigPath << "...\n";

    WebDataCollector collector;
    collector.setVerbose(true);

    if (ctx.onProgress) {
        collector.setProgressCallback([&ctx](float p) {
            ctx.onProgress(PipelineState::Collect, p * 0.8f, "collecting");
        });
    }

    if (!collector.loadConfigFromJson(ctx.config.sourceConfigPath)) {
        result.success = false;
        result.errorMessage = "Failed to load config from " + ctx.config.sourceConfigPath;
        return result;
    }

    std::string resolvedRawDir = ctx.config.rawDir;
    if (resolvedRawDir.empty()) resolvedRawDir = "data/raw";

    std::error_code ec;
    fs::create_directories(resolvedRawDir, ec);
    collector.setOutputDir(resolvedRawDir);

    std::cout << "[Collect] Collecting data...\n";
    size_t collected = collector.collectData();
    ctx.stats.entriesCollected = collected;
    std::cout << "  Collected entries: " << collected << "\n";

    // Save raw JSONL
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();
    std::string stamp = std::to_string(timestamp);

    fs::path rawOutputPath = fs::path(resolvedRawDir) / ("collected_" + stamp + ".jsonl");
    if (collector.saveToJsonl(rawOutputPath.string())) {
        std::cout << "  Saved raw JSONL: " << rawOutputPath.string() << "\n";
    }

    // Save checkpoint
    if (!ctx.config.checkpointDir.empty()) {
        fs::create_directories(ctx.config.checkpointDir, ec);
        fs::path checkpointPath = fs::path(ctx.config.checkpointDir) / ("checkpoint_" + stamp + ".ckpt");
        if (collector.saveCheckpoint(checkpointPath.string())) {
            std::cout << "  Saved checkpoint: " << checkpointPath.string() << "\n";
        }
    }

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    std::cout << "=== COLLECTION COMPLETE ===\n\n";
    return result;
}

} // namespace Pipeline
} // namespace GRIM

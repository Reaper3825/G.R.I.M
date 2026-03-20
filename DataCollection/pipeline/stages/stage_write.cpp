#include "stage_write.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "DataCollection/io/dataset_io.hpp"
#include "DataCollection/collection_state.hpp"

#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

StageResult StageWrite::execute(PipelineContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    StageResult result;

    std::cout << "=== WRITING DATASET ===\n\n";

    if (!ctx.datasetIO) {
        result.success = false;
        result.errorMessage = "No dataset IO available";
        return result;
    }

    auto reportProgress = [&](float p) {
        if (ctx.onProgress) ctx.onProgress(PipelineState::Write, p, "writing");
    };

    // Collect all tagged entries from chunk spool
    std::vector<TaggedEntry> newEntries;
    size_t totalChunks = ctx.tagCursor.chunkFiles.size();
    size_t chunksProcessed = 0;

    for (const auto& chunkFile : ctx.tagCursor.chunkFiles) {
        if (ctx.stopRequested.load()) {
            result.success = false;
            result.errorMessage = "Stopped by user";
            return result;
        }

        std::ifstream file(chunkFile);
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            try {
                json j = json::parse(line);
                TaggedEntry entry;
                entry.id               = j.value("id", std::string());
                entry.content          = j.value("content", std::string());
                entry.sourceUrl        = j.value("source_url", std::string());
                entry.sourceType       = j.value("source_type", std::string());
                entry.qualityTier      = j.value("quality_tier", std::string());
                entry.subject          = j.value("subject", std::string());
                entry.reliabilityScore = j.value("reliability_score", 0.0f);
                entry.timestamp        = j.value("timestamp", int64_t(0));
                entry.verified         = j.value("verified", true);
                if (j.contains("tags") && j["tags"].is_array()) {
                    for (const auto& t : j["tags"]) {
                        if (t.is_string()) entry.tags.push_back(t.get<std::string>());
                    }
                }
                newEntries.push_back(std::move(entry));
            } catch (...) { continue; }
        }

        chunksProcessed++;
        if (totalChunks > 0) {
            reportProgress(static_cast<float>(chunksProcessed) / static_cast<float>(totalChunks) * 70.0f);
        }
    }

    if (newEntries.empty()) {
        std::cout << "  No new entries to write\n";
        reportProgress(100.0f);
        auto elapsed = std::chrono::steady_clock::now() - startTime;
        result.durationSeconds = std::chrono::duration<float>(elapsed).count();
        return result;
    }

    reportProgress(75.0f);

    // On rebuild: overwrite the entire file; otherwise append
    bool writeOk;
    if (ctx.config.forceRebuild) {
        writeOk = ctx.datasetIO->saveAllEntries(ctx.run.massDatasetPath, newEntries);
        std::cout << "  Rebuild: wrote " << newEntries.size() << " entries (full rewrite)\n";
    } else {
        writeOk = ctx.datasetIO->appendEntries(ctx.run.massDatasetPath, newEntries);
        std::cout << "  Appended " << newEntries.size() << " new entries\n";
    }

    if (!writeOk) {
        result.success = false;
        result.errorMessage = "Failed to write mass dataset";
        return result;
    }

    reportProgress(90.0f);

    ctx.stats.entriesWritten = newEntries.size();

    // Save state manager
    if (ctx.stateManager) {
        ctx.stateManager->saveState();
        std::cout << "  State saved: " << ctx.stateManager->getTotalUniqueUrls() << " URLs, "
                  << ctx.stateManager->getTotalUniqueContent() << " content hashes\n";
    }

    reportProgress(100.0f);

    size_t totalInFile = ctx.datasetIO->countEntries(ctx.run.massDatasetPath);
    std::cout << "  Dataset: " << ctx.run.massDatasetPath.string()
              << " (" << totalInFile << " total entries)\n\n";

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    std::cout << "[Write] Complete (" << result.durationSeconds << "s)\n\n";
    return result;
}

} // namespace Pipeline
} // namespace GRIM

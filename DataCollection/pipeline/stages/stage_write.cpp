#include "stage_write.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "../../io/dataset_io.hpp"

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

    std::cout << "=== WRITING SHARDS ===\n\n";

    if (!ctx.datasetWriter) {
        result.success = false;
        result.errorMessage = "No dataset writer available";
        return result;
    }

    auto reportProgress = [&](float p) {
        if (ctx.onProgress) ctx.onProgress(PipelineState::Write, p, "writing");
    };

    // Begin a new append-only run
    if (!ctx.datasetWriter->beginRun(ctx.run)) {
        result.success = false;
        result.errorMessage = "Failed to begin write run";
        return result;
    }

    std::vector<fs::path> writtenShards;
    size_t totalWritten = 0;
    size_t totalChunks = ctx.tagCursor.chunkFiles.size();
    size_t chunksProcessed = 0;

    for (const auto& chunkFile : ctx.tagCursor.chunkFiles) {
        if (ctx.stopRequested.load()) {
            ctx.datasetWriter->abortRun(ctx.run);
            result.success = false;
            result.errorMessage = "Stopped by user";
            return result;
        }

        // Load tagged entries from the chunk file
        EntryChunk<TaggedEntry> chunk;
        {
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
                    if (j.contains("tags") && j["tags"].is_array()) {
                        for (const auto& t : j["tags"]) {
                            if (t.is_string()) entry.tags.push_back(t.get<std::string>());
                        }
                    }
                    chunk.items.push_back(std::move(entry));
                } catch (...) { continue; }
            }
        }

        if (!chunk.items.empty()) {
            fs::path shardPath;
            if (!ctx.datasetWriter->appendTaggedChunk(chunk, shardPath)) {
                ctx.datasetWriter->abortRun(ctx.run);
                result.success = false;
                result.errorMessage = "Failed to write shard";
                return result;
            }
            writtenShards.push_back(shardPath);
            totalWritten += chunk.items.size();
        }

        chunksProcessed++;
        if (totalChunks > 0) {
            reportProgress(static_cast<float>(chunksProcessed) / static_cast<float>(totalChunks) * 90.0f);
        }
    }

    // Also write the legacy merged_verified_cache.jsonl for backward compatibility
    // with the training pipeline that may still read it
    {
        fs::path cachePath = ctx.run.outputRoot / "merged_verified_cache.jsonl";
        std::ofstream cacheFile(cachePath, std::ios::trunc);
        if (cacheFile.is_open()) {
            for (const auto& shardPath : writtenShards) {
                std::ifstream shard(shardPath);
                std::string line;
                while (std::getline(shard, line)) {
                    if (line.empty()) continue;
                    try {
                        json j = json::parse(line);
                        json out;
                        out["content"] = j.value("content", std::string());
                        cacheFile << out.dump() << "\n";
                    } catch (...) { continue; }
                }
            }
            std::cout << "  Legacy cache written: " << cachePath.string() << "\n";
        }
    }

    // Commit manifest atomically
    if (!ctx.datasetWriter->commitRunManifest(ctx.run, writtenShards)) {
        result.success = false;
        result.errorMessage = "Failed to commit manifest";
        return result;
    }
    reportProgress(95.0f);

    ctx.stats.entriesWritten = totalWritten;
    ctx.stats.shardsWritten = writtenShards.size();

    // Save state manager
    if (ctx.stateManager) {
        ctx.stateManager->saveState();
        std::cout << "  State saved: " << ctx.stateManager->getTotalUniqueUrls() << " URLs, "
                  << ctx.stateManager->getTotalUniqueContent() << " content hashes\n";
    }

    reportProgress(100.0f);

    std::cout << "  Wrote " << totalWritten << " entries across "
              << writtenShards.size() << " shards\n";
    std::cout << "  Manifest: " << ctx.run.manifestPath.string() << "\n\n";

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    std::cout << "[Write] Complete (" << result.durationSeconds << "s)\n\n";
    return result;
}

} // namespace Pipeline
} // namespace GRIM

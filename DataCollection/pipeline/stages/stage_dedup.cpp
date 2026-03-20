#include "stage_dedup.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "DataCollection/collection_state.hpp"

#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_set>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

StageResult StageDedup::execute(PipelineContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    StageResult result;

    std::cout << "=== DEDUPLICATING ===\n\n";

    if (ctx.config.forceRebuild) {
        std::cout << "[Dedup] Force rebuild enabled - ignoring previous processing state\n";
    }

    ChunkSpool spool(ctx.run.spoolRoot);
    const std::string outputSpool = "dedup";
    size_t outputChunkIdx = 0;

    auto reportProgress = [&](float p) {
        if (ctx.onProgress) ctx.onProgress(PipelineState::Deduplicate, p, "deduplicating");
    };

    std::unordered_set<uint64_t> batchHashes;
    size_t totalDeduped = 0;
    size_t duplicatesRemoved = 0;
    size_t totalChunks = ctx.verifyCursor.chunkFiles.size();
    size_t chunksProcessed = 0;

    for (const auto& chunkFile : ctx.verifyCursor.chunkFiles) {
        if (ctx.stopRequested.load()) {
            result.success = false;
            result.errorMessage = "Stopped by user";
            return result;
        }

        std::vector<json> dedupedEntries;
        {
            std::ifstream file(chunkFile);
            std::string line;
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                try {
                    json j = json::parse(line);
                    std::string content = j.value("content", std::string());
                    if (content.empty()) continue;

                    // Cross-run dedup via state manager
                    if (!ctx.config.forceRebuild && ctx.stateManager &&
                        ctx.stateManager->hasMergedContent(content)) {
                        duplicatesRemoved++;
                        continue;
                    }

                    // In-batch dedup
                    uint64_t h = GRIM::DataCollection::computeContentHash(content);
                    if (!batchHashes.insert(h).second) {
                        duplicatesRemoved++;
                        continue;
                    }

                    if (ctx.stateManager) {
                        ctx.stateManager->markContentMerged(content);
                    }

                    dedupedEntries.push_back(std::move(j));
                } catch (...) { continue; }
            }
        }

        if (!dedupedEntries.empty()) {
            fs::path outFile = spool.createChunkFile(outputSpool, outputChunkIdx++);
            std::ofstream out(outFile, std::ios::trunc);
            for (const auto& j : dedupedEntries) {
                out << j.dump() << "\n";
            }
            totalDeduped += dedupedEntries.size();
        }

        chunksProcessed++;
        if (totalChunks > 0) {
            reportProgress(static_cast<float>(chunksProcessed) / static_cast<float>(totalChunks) * 100.0f);
        }
    }

    ctx.stats.duplicatesRemoved = duplicatesRemoved;

    ctx.dedupCursor.stageName = outputSpool;
    ctx.dedupCursor.chunkFiles = spool.enumerateChunks(outputSpool);
    ctx.dedupCursor.nextChunk = 0;

    std::cout << "  Deduplicated: " << totalDeduped << " unique entries";
    if (duplicatesRemoved > 0) {
        std::cout << " (" << duplicatesRemoved << " duplicates removed)";
    }
    std::cout << "\n\n";

    if (totalDeduped == 0) {
        std::cout << "[Dedup] WARNING: No data remaining after deduplication.\n";
        std::cout << "  Use 'Force Rebuild' to reprocess all data.\n";
    }

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    return result;
}

} // namespace Pipeline
} // namespace GRIM

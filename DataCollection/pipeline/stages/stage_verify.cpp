#include "stage_verify.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "../../verifier.hpp"

#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

StageResult StageVerify::execute(PipelineContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    StageResult result;

    std::cout << "=== VERIFYING DATA ===\n\n";

    ChunkSpool spool(ctx.run.spoolRoot);
    const std::string outputSpool = "verify";
    size_t outputChunkIdx = 0;

    auto reportProgress = [&](float p) {
        if (ctx.onProgress) ctx.onProgress(PipelineState::Verify, p, "verifying");
    };

    // Configure verifier
    Config verifierConfig;
    verifierConfig.input_dir = ctx.config.verifiedDir;
    verifierConfig.output_dir = ctx.config.verifiedDir;
    verifierConfig.verbose_logging = false;
    verifierConfig.save_rejected = false;
    verifierConfig.progressive_filtering = true;

    Verifier verifier(verifierConfig);

    size_t totalVerified = 0;
    size_t totalChunks = ctx.ingestCursor.chunkFiles.size();
    size_t chunksProcessed = 0;

    for (const auto& chunkFile : ctx.ingestCursor.chunkFiles) {
        if (ctx.stopRequested.load()) {
            result.success = false;
            result.errorMessage = "Stopped by user";
            return result;
        }

        // Load entries from this chunk
        std::vector<UnverifiedEntry> toVerify;
        {
            std::ifstream file(chunkFile);
            std::string line;
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                try {
                    json j = json::parse(line);
                    UnverifiedEntry ue;
                    ue.content = j.value("content", std::string());
                    ue.source_url = j.value("source_url", std::string());
                    ue.source_type = j.value("source_type", std::string());
                    toVerify.push_back(std::move(ue));
                } catch (...) { continue; }
            }
        }

        if (toVerify.empty()) {
            chunksProcessed++;
            continue;
        }

        // Run verification on this chunk
        auto verified = verifier.verify_entries(toVerify);

        if (!verified.empty()) {
            // Write verified entries to output spool
            fs::path outFile = spool.createChunkFile(outputSpool, outputChunkIdx++);
            std::ofstream out(outFile, std::ios::trunc);
            for (const auto& ve : verified) {
                json j;
                j["content"]           = ve.content;
                j["source_url"]        = ve.source_url;
                j["source_type"]       = ve.source_type;
                j["reliability_score"] = ve.reliability_score;
                j["verification_time"] = ve.verification_time;
                out << j.dump() << "\n";
            }
            totalVerified += verified.size();
        }

        chunksProcessed++;
        if (totalChunks > 0) {
            float progress = static_cast<float>(chunksProcessed) / static_cast<float>(totalChunks) * 100.0f;
            reportProgress(progress);
        }
    }

    ctx.stats.entriesVerified = totalVerified;

    ctx.verifyCursor.stageName = outputSpool;
    ctx.verifyCursor.chunkFiles = spool.enumerateChunks(outputSpool);
    ctx.verifyCursor.nextChunk = 0;

    auto vstats = verifier.get_stats();
    std::cout << "  Verified: " << totalVerified << " entries passed\n";
    std::cout << "  High quality: " << vstats.high_quality_count
              << ", Medium: " << vstats.medium_quality_count
              << ", Low: " << vstats.low_quality_count
              << ", Rejected: " << vstats.failed_verification << "\n\n";

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    return result;
}

} // namespace Pipeline
} // namespace GRIM

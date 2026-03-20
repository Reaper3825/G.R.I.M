#include "stage_ingest.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "../../collection_state.hpp"
#include "../../huggingface_webhook.hpp"
#include "../../verifier.hpp"

#include <nlohmann/json.hpp>
#include <flatbuffers/flatbuffers.h>
#include "../../checkpoint_data_generated.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef GRIM_HAS_POPPLER
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>
#endif

#ifdef GRIM_HAS_LIBZIP
#include <zip.h>
#endif

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;

// ─── Helpers ────────────────────────────────────────────

static bool loadJsonlEntries(const fs::path& filePath,
                             const std::string& sourceTag,
                             std::vector<VerifiedEntry>& out) {
    std::ifstream file(filePath);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            json j = json::parse(line);
            VerifiedEntry ve;
            if (j.contains("content") && j["content"].is_string()) {
                ve.content = j["content"].get<std::string>();
            } else if (j.is_string()) {
                ve.content = j.get<std::string>();
            } else if (j.contains("text") && j["text"].is_string()) {
                ve.content = j["text"].get<std::string>();
            } else {
                continue;
            }
            ve.source_url = j.value("source_url", sourceTag);
            ve.source_type = j.value("source_type", std::string("huggingface"));
            ve.reliability_score = j.value("reliability_score", 0.8f);
            ve.verification_time = j.value("verification_time", time_t(0));
            out.push_back(std::move(ve));
        } catch (...) {
            continue;
        }
    }
    return true;
}

static void writeChunkToSpool(ChunkSpool& spool,
                               const std::string& stageName,
                               size_t& chunkIdx,
                               const std::vector<VerifiedEntry>& entries,
                               size_t chunkSize) {
    for (size_t i = 0; i < entries.size(); i += chunkSize) {
        size_t end = std::min(i + chunkSize, entries.size());
        fs::path chunkFile = spool.createChunkFile(stageName, chunkIdx++);

        std::ofstream out(chunkFile, std::ios::trunc);
        for (size_t j = i; j < end; ++j) {
            json jEntry;
            jEntry["content"]           = entries[j].content;
            jEntry["source_url"]        = entries[j].source_url;
            jEntry["source_type"]       = entries[j].source_type;
            jEntry["reliability_score"] = entries[j].reliability_score;
            jEntry["verification_time"] = entries[j].verification_time;
            out << jEntry.dump() << "\n";
        }
    }
}

StageResult StageIngest::execute(PipelineContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    StageResult result;

    std::cout << "=== INGESTING DATA ===\n\n";

    ChunkSpool spool(ctx.run.spoolRoot);
    size_t chunkIdx = 0;
    const std::string spoolName = "ingest";

    auto reportProgress = [&](float p) {
        if (ctx.onProgress) ctx.onProgress(PipelineState::Ingest, p, "ingesting");
    };

    reportProgress(5.0f);

    // ── 1. Load verified JSONL entries ──────────────────
    std::vector<VerifiedEntry> verifiedEntries;
    if (fs::exists(ctx.config.verifiedDir)) {
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(ctx.config.verifiedDir, ec)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
                std::ifstream file(entry.path());
                std::string line;
                while (std::getline(file, line)) {
                    if (line.empty()) continue;
                    try {
                        json j = json::parse(line);
                        VerifiedEntry ve;
                        ve.content = j["content"].get<std::string>();
                        ve.source_url = j["source_url"].get<std::string>();
                        ve.source_type = j["source_type"].get<std::string>();
                        ve.reliability_score = j.value("reliability_score", 0.8f);
                        ve.verification_time = j.value("verification_time", time_t(0));
                        verifiedEntries.push_back(std::move(ve));
                    } catch (...) { continue; }
                }
            }
        }
    }
    std::cout << "[Ingest] Verified entries: " << verifiedEntries.size() << "\n";
    reportProgress(15.0f);

    // ── 2. Ingest HuggingFace downloads ─────────────────
    fs::path hfRoot = fs::path(ctx.config.outputDir) / "huggingface";
    if (fs::exists(hfRoot)) {
        std::error_code ec;
        size_t hfTotal = 0;
        for (const auto& datasetDir : fs::directory_iterator(hfRoot, ec)) {
            if (!datasetDir.is_directory()) continue;
            for (const auto& fileEntry : fs::recursive_directory_iterator(datasetDir.path(), ec)) {
                if (!fileEntry.is_regular_file()) continue;
                std::string ext = fileEntry.path().extension().string();
                std::string sourceTag = "hf:" + datasetDir.path().filename().string();

                if (ctx.stateManager && !ctx.config.forceRebuild &&
                    ctx.stateManager->hasCollectedUrl(fileEntry.path().string())) {
                    continue;
                }

                if (ext == ".jsonl" || ext == ".json") {
                    loadJsonlEntries(fileEntry.path(), sourceTag, verifiedEntries);
                    hfTotal++;
                } else if (ext == ".txt") {
                    std::ifstream f(fileEntry.path());
                    if (!f.is_open()) continue;
                    std::string content((std::istreambuf_iterator<char>(f)),
                                        std::istreambuf_iterator<char>());
                    if (!content.empty()) {
                        VerifiedEntry ve;
                        ve.content = std::move(content);
                        ve.source_url = sourceTag;
                        ve.source_type = "huggingface";
                        ve.reliability_score = 0.7f;
                        ve.verification_time = 0;
                        verifiedEntries.push_back(std::move(ve));
                        hfTotal++;
                    }
                }

                if (ctx.stateManager) {
                    ctx.stateManager->markUrlCollected(fileEntry.path().string(), "huggingface");
                }
            }
        }
        if (hfTotal > 0) {
            std::cout << "[Ingest] HuggingFace files ingested: " << hfTotal << "\n";
        }
    }
    reportProgress(35.0f);

    // ── 3. Load checkpoints (FlatBuffer) ────────────────
    {
        std::error_code ec;
        if (fs::exists(ctx.config.checkpointDir, ec)) {
            size_t ckptLoaded = 0, ckptSkipped = 0;
            for (const auto& entry : fs::directory_iterator(ctx.config.checkpointDir, ec)) {
                if (!entry.is_regular_file()) continue;
                std::string filename = entry.path().filename().string();
                if (filename.substr(0, 11) != "checkpoint_" ||
                    entry.path().extension() != ".ckpt") continue;

                if (ctx.stateManager && !ctx.config.forceRebuild &&
                    ctx.stateManager->hasCollectedUrl(entry.path().string())) {
                    ckptSkipped++;
                    continue;
                }

                try {
                    std::ifstream file(entry.path(), std::ios::binary);
                    if (!file.is_open()) continue;
                    file.seekg(0, std::ios::end);
                    size_t fileSize = file.tellg();
                    file.seekg(0);
                    std::vector<uint8_t> buffer(fileSize);
                    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

                    auto checkpoint = GRIMCheckpoint::GetCheckpoint(buffer.data());
                    if (!checkpoint || !checkpoint->entries()) continue;

                    for (const auto* fb_entry : *checkpoint->entries()) {
                        if (!fb_entry || !fb_entry->content() || !fb_entry->source_url()) continue;
                        VerifiedEntry ve;
                        ve.content = fb_entry->content()->str();
                        ve.source_url = fb_entry->source_url()->str();
                        ve.source_type = fb_entry->source_type() ? fb_entry->source_type()->str() : "web";
                        ve.reliability_score = fb_entry->reliability_score();
                        ve.verification_time = fb_entry->fetch_timestamp();
                        if (!ve.content.empty()) verifiedEntries.push_back(std::move(ve));
                    }

                    if (ctx.stateManager) {
                        ctx.stateManager->markUrlCollected(entry.path().string(), "checkpoint");
                    }
                    ckptLoaded++;
                } catch (...) { continue; }
            }
            std::cout << "[Ingest] Checkpoints loaded: " << ckptLoaded;
            if (ckptSkipped > 0) std::cout << " (skipped " << ckptSkipped << " already processed)";
            std::cout << "\n";
        }
    }
    reportProgress(55.0f);

    // ── 4. Load previously merged cache ─────────────────
    {
        fs::path cachePath = fs::path(ctx.config.outputDir) / "merged_verified_cache.jsonl";
        if (fs::exists(cachePath)) {
            size_t prevCount = 0;
            std::ifstream prev(cachePath);
            std::string line;
            while (std::getline(prev, line)) {
                if (line.empty()) continue;
                try {
                    json j = json::parse(line);
                    VerifiedEntry ve;
                    ve.content = j["content"].get<std::string>();
                    ve.source_url = j.value("source_url", std::string("merged_cache"));
                    ve.source_type = j.value("source_type", std::string("cached"));
                    ve.reliability_score = j.value("reliability_score", 0.9f);
                    ve.verification_time = j.value("verification_time", time_t(0));
                    verifiedEntries.push_back(std::move(ve));
                    prevCount++;
                } catch (...) { continue; }
            }
            if (prevCount > 0) {
                std::cout << "[Ingest] Previously merged cache: " << prevCount << " entries\n";
            }
        }
    }
    reportProgress(70.0f);

    // ── 5. Write all ingested entries into spool chunks ──
    ctx.stats.entriesIngested = verifiedEntries.size();
    std::cout << "[Ingest] Total ingested entries: " << verifiedEntries.size() << "\n";

    writeChunkToSpool(spool, spoolName, chunkIdx, verifiedEntries, ctx.chunkSize);

    ctx.ingestCursor.stageName = spoolName;
    ctx.ingestCursor.chunkFiles = spool.enumerateChunks(spoolName);
    ctx.ingestCursor.nextChunk = 0;

    reportProgress(100.0f);

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    std::cout << "[Ingest] Complete (" << result.durationSeconds << "s)\n\n";
    return result;
}

} // namespace Pipeline
} // namespace GRIM

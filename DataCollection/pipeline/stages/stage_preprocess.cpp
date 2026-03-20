#include "stage_preprocess.hpp"
#include "../pipeline_context.hpp"
#include "../chunk_spool.hpp"
#include "DataCollection/data_preprocessor.hpp"
#include "control/ai_config_paths.hpp"

#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;
using json = nlohmann::json;
using GRIM::Training::DataPreprocessor;
using GRIM::Training::PreprocessorConfig;

StageResult StagePreprocess::execute(PipelineContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    StageResult result;

    std::cout << "=== PREPROCESSING ===\n\n";

    ChunkSpool spool(ctx.run.spoolRoot);
    const std::string outputSpool = "preprocess";
    size_t outputChunkIdx = 0;

    auto reportProgress = [&](float p) {
        if (ctx.onProgress) ctx.onProgress(PipelineState::Preprocess, p, "preprocessing");
    };

    // Load max_seq_len from training config
    GRIM::Config::TrainingHyperparameters trainParams;
    int maxSeqLen = 900;
    if (GRIM::Config::loadTrainingHyperparameters(trainParams)) {
        maxSeqLen = trainParams.max_seq_len;
        std::cout << "[Preprocess] Using max_seq_len=" << maxSeqLen << " from config\n";
    }

    const int charsPerToken = 4;
    int maxChars = maxSeqLen * charsPerToken;

    PreprocessorConfig prepConfig;
    prepConfig.remove_html = true;
    prepConfig.min_length = 75;
    prepConfig.min_words = 15;
    prepConfig.min_alpha_ratio = 0.5f;
    prepConfig.max_length = 100000;
    prepConfig.max_repetition_ratio = 0.5f;
    prepConfig.dedup_threshold = 0.9f;
    prepConfig.max_token_estimate_chars = maxChars;

    DataPreprocessor preprocessor(prepConfig);

    size_t totalCleaned = 0;
    size_t prunedCount = 0;
    size_t chunksCreated = 0;
    size_t totalChunks = ctx.dedupCursor.chunkFiles.size();
    size_t chunksProcessed = 0;

    for (const auto& chunkFile : ctx.dedupCursor.chunkFiles) {
        if (ctx.stopRequested.load()) {
            result.success = false;
            result.errorMessage = "Stopped by user";
            return result;
        }

        std::vector<std::string> cleanedTexts;
        {
            std::ifstream file(chunkFile);
            std::string line;
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                try {
                    json j = json::parse(line);
                    std::string content = j.value("content", std::string());
                    if (content.empty()) continue;

                    auto rawChunks = preprocessor.chunkLongText(content);
                    for (const auto& rawChunk : rawChunks) {
                        std::string clean = preprocessor.preprocess(rawChunk);
                        auto cleanChunks = preprocessor.chunkLongText(clean);
                        for (const auto& chunk : cleanChunks) {
                            if (!preprocessor.passesQualityFilter(chunk)) {
                                prunedCount++;
                                continue;
                            }
                            if (preprocessor.isDuplicate(chunk)) continue;
                            cleanedTexts.push_back(chunk);
                            if (rawChunks.size() > 1 || cleanChunks.size() > 1) chunksCreated++;
                        }
                    }
                } catch (...) { continue; }
            }
        }

        if (!cleanedTexts.empty()) {
            fs::path outFile = spool.createChunkFile(outputSpool, outputChunkIdx++);
            std::ofstream out(outFile, std::ios::trunc);
            for (const auto& text : cleanedTexts) {
                json j;
                j["content"] = text;
                out << j.dump() << "\n";
            }
            totalCleaned += cleanedTexts.size();
        }

        chunksProcessed++;
        if (totalChunks > 0) {
            reportProgress(static_cast<float>(chunksProcessed) / static_cast<float>(totalChunks) * 100.0f);
        }
    }

    ctx.stats.entriesCleaned = totalCleaned;
    ctx.stats.chunksCreated = chunksCreated;
    ctx.stats.prunedCount = prunedCount;

    ctx.preprocessCursor.stageName = outputSpool;
    ctx.preprocessCursor.chunkFiles = spool.enumerateChunks(outputSpool);
    ctx.preprocessCursor.nextChunk = 0;

    if (chunksCreated > 0) {
        std::cout << "  Split " << chunksCreated << " long texts into multiple chunks\n";
    }
    if (prunedCount > 0) {
        std::cout << "  Pruned " << prunedCount << " corrupted/low-quality chunks\n";
    }
    std::cout << "  Cleaned: " << totalCleaned << " entries\n\n";

    // ── Ingest Q/A JSONL into the output spool ──────────
    if (!ctx.config.qaJsonlPaths.empty()) {
        std::cout << "[Preprocess] Ingesting " << ctx.config.qaJsonlPaths.size() << " Q/A JSONL file(s)...\n";
        size_t qaIngested = 0;

        for (const auto& path : ctx.config.qaJsonlPaths) {
            if (!fs::exists(path)) continue;
            if (!ctx.config.forceRebuild && ctx.stateManager &&
                ctx.stateManager->hasCollectedUrl(path)) continue;

            std::ifstream file(path);
            if (!file.is_open()) continue;

            std::vector<std::string> qaTexts;
            std::string line;
            while (std::getline(file, line)) {
                if (line.empty()) continue;
                try {
                    json j = json::parse(line);
                    std::string content;
                    if (j.contains("question") && j["question"].is_string() &&
                        j.contains("answer") && j["answer"].is_string()) {
                        content = "Q: " + j["question"].get<std::string>()
                                + "\n\nA: " + j["answer"].get<std::string>();
                    } else if (j.contains("conversation") && j["conversation"].is_array()) {
                        for (const auto& turn : j["conversation"]) {
                            if (!turn.contains("role") || !turn.contains("content")) continue;
                            std::string role = turn["role"].get<std::string>();
                            std::string text = turn["content"].get<std::string>();
                            if (!content.empty()) content += "\n\n";
                            if (role == "human" || role == "user")
                                content += "Human: " + text;
                            else if (role == "assistant" || role == "bot")
                                content += "Assistant: " + text;
                            else
                                content += role + ": " + text;
                        }
                    } else if (j.contains("content") && j["content"].is_string()) {
                        content = j["content"].get<std::string>();
                    }
                    if (!content.empty()) qaTexts.push_back(std::move(content));
                } catch (...) { continue; }
            }

            if (!qaTexts.empty()) {
                fs::path outFile = spool.createChunkFile(outputSpool, outputChunkIdx++);
                std::ofstream out(outFile, std::ios::trunc);
                for (const auto& text : qaTexts) {
                    json j;
                    j["content"] = text;
                    out << j.dump() << "\n";
                }
                qaIngested += qaTexts.size();
                totalCleaned += qaTexts.size();
            }

            if (ctx.stateManager) {
                ctx.stateManager->markUrlCollected(path, "qa_jsonl");
            }
        }

        if (qaIngested > 0) {
            std::cout << "  Q/A ingestion: " << qaIngested << " entries added\n";
            ctx.preprocessCursor.chunkFiles = spool.enumerateChunks(outputSpool);
        }
    }

    auto elapsed = std::chrono::steady_clock::now() - startTime;
    result.durationSeconds = std::chrono::duration<float>(elapsed).count();
    std::cout << "[Preprocess] Complete (" << result.durationSeconds << "s)\n\n";
    return result;
}

} // namespace Pipeline
} // namespace GRIM

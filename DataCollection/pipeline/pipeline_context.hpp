#pragma once

#include "pipeline_types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace GRIM {
namespace DataCollection {
    class CollectionStateManager;
}
}

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;

class IDatasetIO;

struct PipelineConfig {
    PipelineMode mode = PipelineMode::Full;
    std::string sourceConfigPath;
    std::string checkpointDir;
    std::string rawDir;
    std::string verifiedDir;
    std::string outputDir;
    bool forceRebuild = false;
    int maxChunkChars = 3600;
    std::vector<std::string> qaJsonlPaths;
};

struct PipelineStats {
    size_t entriesCollected = 0;
    size_t entriesIngested = 0;
    size_t entriesVerified = 0;
    size_t duplicatesRemoved = 0;
    size_t entriesCleaned = 0;
    size_t entriesTagged = 0;
    size_t entriesWritten = 0;
    size_t chunksCreated = 0;
    size_t prunedCount = 0;
    size_t chunksProcessed = 0;
};

struct TaggedEntry {
    std::string id;
    std::string content;
    std::string sourceUrl;
    std::string sourceType;
    std::string qualityTier;
    std::string subject;
    std::vector<std::string> tags;
    float reliabilityScore = 0.0f;
    int64_t timestamp = 0;
    bool verified = false;
};

template <typename T>
struct EntryChunk {
    std::vector<T> items;
    size_t chunkIndex = 0;
    bool isLastChunk = false;

    void clear() {
        items.clear();
        isLastChunk = false;
    }
};

struct ChunkCursor {
    std::string stageName;
    std::vector<fs::path> chunkFiles;
    size_t nextChunk = 0;
};

struct PipelineRunLayout {
    std::string runId;
    fs::path runRoot;
    fs::path spoolRoot;
    fs::path outputRoot;
    fs::path massDatasetPath;
};

struct PipelineContext {
    PipelineConfig config;
    PipelineStats stats;

    std::unique_ptr<GRIM::DataCollection::CollectionStateManager> stateManager;

    PipelineRunLayout run;
    ChunkCursor ingestCursor;
    ChunkCursor verifyCursor;
    ChunkCursor dedupCursor;
    ChunkCursor preprocessCursor;
    ChunkCursor tagCursor;

    EntryChunk<std::string> cleanedChunk;
    EntryChunk<TaggedEntry> taggedChunk;

    size_t chunkSize = 5000;

    std::function<void(PipelineState, float, const std::string&)> onProgress;
    std::atomic<bool> stopRequested{false};

    std::shared_ptr<IDatasetIO> datasetIO;
};

} // namespace Pipeline
} // namespace GRIM

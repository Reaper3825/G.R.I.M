#pragma once

#include "../pipeline/pipeline_context.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;

class IDatasetIO {
public:
    virtual ~IDatasetIO() = default;

    virtual bool iterateShard(const fs::path& shardPath,
                              std::function<bool(const TaggedEntry&)> visitor) = 0;
    virtual size_t countEntries(const fs::path& manifestPath) = 0;
    virtual bool loadAssignments(const fs::path& path, std::vector<std::string>& ids) = 0;
    virtual bool saveAssignments(const fs::path& path, const std::vector<std::string>& ids) = 0;
};

class AppendOnlyDatasetWriter {
public:
    virtual ~AppendOnlyDatasetWriter() = default;

    virtual bool beginRun(const PipelineRunLayout& run) = 0;
    virtual bool appendTaggedChunk(const EntryChunk<TaggedEntry>& chunk,
                                   fs::path& writtenShardPath) = 0;
    virtual bool commitRunManifest(const PipelineRunLayout& run,
                                   const std::vector<fs::path>& newShards) = 0;
    virtual bool abortRun(const PipelineRunLayout& run) = 0;
};

} // namespace Pipeline
} // namespace GRIM

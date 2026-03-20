#pragma once

#include "dataset_io.hpp"

namespace GRIM {
namespace Pipeline {

class DatasetIOJson : public IDatasetIO {
public:
    bool iterateShard(const fs::path& shardPath,
                      std::function<bool(const TaggedEntry&)> visitor) override;
    size_t countEntries(const fs::path& manifestPath) override;
    bool loadAssignments(const fs::path& path, std::vector<std::string>& ids) override;
    bool saveAssignments(const fs::path& path, const std::vector<std::string>& ids) override;
};

class AppendOnlyDatasetWriterJson : public AppendOnlyDatasetWriter {
public:
    bool beginRun(const PipelineRunLayout& run) override;
    bool appendTaggedChunk(const EntryChunk<TaggedEntry>& chunk,
                           fs::path& writtenShardPath) override;
    bool commitRunManifest(const PipelineRunLayout& run,
                           const std::vector<fs::path>& newShards) override;
    bool abortRun(const PipelineRunLayout& run) override;

private:
    size_t shardCounter_ = 0;
    fs::path shardsDir_;
};

} // namespace Pipeline
} // namespace GRIM

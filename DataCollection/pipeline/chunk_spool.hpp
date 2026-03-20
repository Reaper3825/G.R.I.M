#pragma once

#include "pipeline_context.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace GRIM {
namespace Pipeline {

namespace fs = std::filesystem;

class ChunkSpool {
public:
    explicit ChunkSpool(const fs::path& spoolRoot);

    fs::path createChunkFile(const std::string& stageName, size_t chunkIndex);
    std::vector<fs::path> enumerateChunks(const std::string& stageName) const;
    bool cleanupStage(const std::string& stageName);
    bool cleanupRun();

    const fs::path& root() const { return spoolRoot_; }

private:
    fs::path spoolRoot_;
};

} // namespace Pipeline
} // namespace GRIM

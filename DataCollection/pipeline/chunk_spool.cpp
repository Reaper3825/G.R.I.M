#include "chunk_spool.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace GRIM {
namespace Pipeline {

ChunkSpool::ChunkSpool(const fs::path& spoolRoot)
    : spoolRoot_(spoolRoot)
{
    std::error_code ec;
    fs::create_directories(spoolRoot_, ec);
}

fs::path ChunkSpool::createChunkFile(const std::string& stageName, size_t chunkIndex) {
    fs::path stageDir = spoolRoot_ / stageName;
    std::error_code ec;
    fs::create_directories(stageDir, ec);

    std::ostringstream name;
    name << "chunk_" << std::setw(6) << std::setfill('0') << chunkIndex << ".jsonl";
    return stageDir / name.str();
}

std::vector<fs::path> ChunkSpool::enumerateChunks(const std::string& stageName) const {
    std::vector<fs::path> result;
    fs::path stageDir = spoolRoot_ / stageName;

    std::error_code ec;
    if (!fs::exists(stageDir, ec)) return result;

    for (const auto& entry : fs::directory_iterator(stageDir, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
            result.push_back(entry.path());
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

bool ChunkSpool::cleanupStage(const std::string& stageName) {
    fs::path stageDir = spoolRoot_ / stageName;
    std::error_code ec;
    if (!fs::exists(stageDir, ec)) return true;
    fs::remove_all(stageDir, ec);
    return !ec;
}

bool ChunkSpool::cleanupRun() {
    std::error_code ec;
    if (!fs::exists(spoolRoot_, ec)) return true;
    fs::remove_all(spoolRoot_, ec);
    return !ec;
}

} // namespace Pipeline
} // namespace GRIM

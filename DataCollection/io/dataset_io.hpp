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

    virtual bool loadAllEntries(const fs::path& datasetPath,
                                std::vector<TaggedEntry>& entries) = 0;
    virtual bool saveAllEntries(const fs::path& datasetPath,
                                const std::vector<TaggedEntry>& entries) = 0;
    virtual bool appendEntries(const fs::path& datasetPath,
                               const std::vector<TaggedEntry>& entries) = 0;
    virtual size_t countEntries(const fs::path& datasetPath) = 0;

    virtual bool loadAssignments(const fs::path& path, std::vector<std::string>& ids) = 0;
    virtual bool saveAssignments(const fs::path& path, const std::vector<std::string>& ids) = 0;
};

} // namespace Pipeline
} // namespace GRIM

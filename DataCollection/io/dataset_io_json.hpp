#pragma once

#include "dataset_io.hpp"

namespace GRIM {
namespace Pipeline {

class DatasetIOJson : public IDatasetIO {
public:
    bool loadAllEntries(const fs::path& datasetPath,
                        std::vector<TaggedEntry>& entries) override;
    bool saveAllEntries(const fs::path& datasetPath,
                        const std::vector<TaggedEntry>& entries) override;
    bool appendEntries(const fs::path& datasetPath,
                       const std::vector<TaggedEntry>& entries) override;
    size_t countEntries(const fs::path& datasetPath) override;

    bool loadAssignments(const fs::path& path, std::vector<std::string>& ids) override;
    bool saveAssignments(const fs::path& path, const std::vector<std::string>& ids) override;
};

} // namespace Pipeline
} // namespace GRIM

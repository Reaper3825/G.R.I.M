#pragma once

#include "dataset_io.hpp"

namespace GRIM {
namespace Pipeline {

class DatasetIOJson : public IDatasetIO {
public:
    explicit DatasetIOJson(bool strict = false) : strict_(strict) {}
    bool loadAllEntries(const fs::path& datasetPath,
                        std::vector<TaggedEntry>& entries) override;
    bool saveAllEntries(const fs::path& datasetPath,
                        const std::vector<TaggedEntry>& entries) override;
    bool appendEntries(const fs::path& datasetPath,
                       const std::vector<TaggedEntry>& entries) override;
    size_t countEntries(const fs::path& datasetPath) override;

    bool loadAssignments(const fs::path& path, std::vector<std::string>& ids) override;
    bool saveAssignments(const fs::path& path, const std::vector<std::string>& ids) override;
private:
    bool strict_;
};

} // namespace Pipeline
} // namespace GRIM

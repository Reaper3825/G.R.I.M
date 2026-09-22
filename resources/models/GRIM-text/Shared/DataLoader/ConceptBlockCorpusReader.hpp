#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../Curriculum/CurriculumMetadata.hpp"

namespace GRIM {

void loadConceptBlocks(
    const std::filesystem::path& cache_dir,
    std::vector<nlohmann::json>& out,
    CurriculumMetadata& out_metadata,
    const std::string& curriculum_name);

} // namespace GRIM

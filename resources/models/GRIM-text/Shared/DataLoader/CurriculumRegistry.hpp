#pragma once

#include <filesystem>
#include <string>

#include "../Curriculum/CurriculumMetadata.hpp"

namespace GRIM {

CurriculumMetadata LoadCurriculumMetadataFromRegistry(
    const std::filesystem::path& directory,
    const std::string& curriculum_name);

} // namespace GRIM

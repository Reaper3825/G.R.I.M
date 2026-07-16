#pragma once

#include "../concept_block.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace GRIM::ConceptBlockIO {

inline constexpr uint32_t kSchemaVersion = 1;

// Reads and verifies a complete concept-block FlatBuffer. On failure, returns
// false, leaves `blocks` empty, and optionally describes the error.
bool loadFlatBuffer(const std::filesystem::path& path,
                    std::vector<ConceptBlock>& blocks,
                    std::string* error = nullptr);

// Serializes the complete dataset and atomically replaces `path`.
bool saveFlatBuffer(const std::filesystem::path& path,
                    const std::vector<ConceptBlock>& blocks,
                    std::string* error = nullptr);

} // namespace GRIM::ConceptBlockIO

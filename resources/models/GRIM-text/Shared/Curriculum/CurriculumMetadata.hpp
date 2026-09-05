#pragma once

#include <string>
#include <unordered_set>

namespace GRIM {

struct CurriculumMetadata {
    std::string training_stage = "sft";
    std::string name;
    std::string id;
    // Derived union of block IDs owned by the selected curriculum's courses.
    std::unordered_set<std::string> concept_block_ids;

    bool formatAsConcept() const noexcept {
        return training_stage == "sft";
    }
};

} // namespace GRIM

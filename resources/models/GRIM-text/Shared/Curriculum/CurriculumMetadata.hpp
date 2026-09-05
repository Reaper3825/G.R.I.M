#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace GRIM {

struct CourseMetadata {
    std::string id;
    std::vector<std::string> concept_block_ids;
};

struct CurriculumMetadata {
    bool randomize_course_order = false;
    bool randomize_concept_block_order = false;
    std::vector<CourseMetadata> courses;
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

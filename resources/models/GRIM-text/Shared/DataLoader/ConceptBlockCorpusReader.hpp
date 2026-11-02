#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../Curriculum/CurriculumMetadata.hpp"
#include "../Batching/CurriculumOrdering.hpp"
#include "../Goal/GoalContract.hpp"

namespace GRIM {

void loadConceptBlocks(
    const std::filesystem::path& cache_dir,
    std::vector<nlohmann::json>& out,
    CurriculumMetadata& out_metadata,
    const std::string& curriculum_name);

// Runtime assignment data only. No authored prompt, DDEA, answer, or prior state.
struct ScheduledGoal {
    std::string course_id;
    std::string environment_id;
    std::string concept_block_id;
    ConceptBlockGoal goal;
};

// Reuses the corpus reader and canonical block ordering. Decodes only selected
// IDs and Goals; the returned assignments own their data after file buffers die.
// Shared blocks retain their first authored course, matching existing scheduling.
std::vector<ScheduledGoal> loadScheduledGoals(
    const std::filesystem::path& cache_dir,
    const std::string& curriculum_name,
    Batching::CurriculumOrdering ordering,
    uint64_t seed);

} // namespace GRIM

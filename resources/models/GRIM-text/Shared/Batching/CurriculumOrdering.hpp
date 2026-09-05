#pragma once

#include "../Curriculum/CurriculumMetadata.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace GRIM::Batching {

enum class CurriculumOrdering { CURRICULUM, PRESERVE, RANDOM };

inline CurriculumOrdering parseCurriculumOrdering(const std::string& strategy) {
    if (strategy == "CURRICULUM") return CurriculumOrdering::CURRICULUM;
    if (strategy == "PRESERVE") return CurriculumOrdering::PRESERVE;
    if (strategy == "RANDOM") return CurriculumOrdering::RANDOM;
    throw std::runtime_error("batch_strategy must be CURRICULUM, PRESERVE, or RANDOM; got '" + strategy + "'");
}

// Shared blocks belong to their first authored course for scheduling purposes.
// They are exposed once, matching the loader's membership-union semantics.
inline std::vector<std::string> orderedConceptBlocks(
    const CurriculumMetadata& curriculum, CurriculumOrdering ordering, uint64_t seed) {
    if (curriculum.courses.empty())
        throw std::runtime_error("Course scheduling requires ordered course metadata from the registry");
    std::vector<std::vector<std::string>> courses;
    std::unordered_set<std::string> seen;
    for (const auto& course : curriculum.courses) {
        courses.emplace_back();
        for (const auto& id : course.concept_block_ids) {
            if (id.empty()) throw std::runtime_error("Course scheduling encountered an empty concept block ID");
            if (seen.insert(id).second) courses.back().push_back(id);
        }
    }
    std::mt19937_64 rng(seed);
    if (ordering == CurriculumOrdering::CURRICULUM) {
        if (curriculum.randomize_course_order) std::shuffle(courses.begin(), courses.end(), rng);
        if (curriculum.randomize_concept_block_order)
            for (auto& course : courses) std::shuffle(course.begin(), course.end(), rng);
    }
    std::vector<std::string> blocks;
    for (const auto& course : courses) blocks.insert(blocks.end(), course.begin(), course.end());
    // RANDOM ignores both flags and course boundaries. PRESERVE ignores both
    // flags and retains the registry's authored course/block order.
    if (ordering == CurriculumOrdering::RANDOM) std::shuffle(blocks.begin(), blocks.end(), rng);
    return blocks;
}

// Multiple rows/windows belonging to one block stay adjacent in source order.
// Never infer provenance from row position: splitting/filtering changes indices.
inline std::vector<uint32_t> orderedCourseSequences(
    const std::vector<uint32_t>& lengths,
    const std::vector<std::string>& block_ids,
    const CurriculumMetadata& curriculum, CurriculumOrdering ordering, uint64_t seed) {
    if (lengths.size() != block_ids.size() || lengths.size() > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Course scheduling requires one concept block ID per sequence row");
    const auto blocks = orderedConceptBlocks(curriculum, ordering, seed);
    const std::unordered_set<std::string> selected(blocks.begin(), blocks.end());
    std::unordered_map<std::string, std::vector<uint32_t>> rows;
    for (size_t i = 0; i < lengths.size(); ++i) {
        if (lengths[i] == 0) continue;
        if (block_ids[i].empty())
            throw std::runtime_error("Course scheduling blocked: sequence row " + std::to_string(i) +
                " has no concept_block_id. Persist source IDs in GRMT and propagate them through splitting/windowing; rebuild legacy artifacts.");
        if (!selected.count(block_ids[i]))
            throw std::runtime_error("Course scheduling: row references a block outside the selected curriculum: " + block_ids[i]);
        rows[block_ids[i]].push_back(static_cast<uint32_t>(i));
    }
    std::vector<uint32_t> order;
    for (const auto& block : blocks) {
        const auto found = rows.find(block);
        // A block may have no training rows after the train/validation split.
        if (found != rows.end()) order.insert(order.end(), found->second.begin(), found->second.end());
    }
    return order;
}

} // namespace GRIM::Batching

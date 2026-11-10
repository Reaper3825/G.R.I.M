#include "dataset_catalog.hpp"

namespace GRIM::DataCollection {

nlohmann::json DatasetCatalog::curriculumRegistry() const {
    const auto snapshot = view();
    if (!snapshot.catalog) throw std::runtime_error("ConceptBlock database is unavailable: " + lastError());
    nlohmann::json result = {{"curriculums", nlohmann::json::array()}, {"courses", nlohmann::json::array()}};
    for (const auto& curriculum : snapshot.catalog->getCurriculums())
        result["curriculums"].push_back({{"id", curriculum.id}, {"name", curriculum.name}, {"course_ids", curriculum.course_ids}});
    for (const auto& course : snapshot.catalog->getCourses())
        result["courses"].push_back({{"id", course.id}, {"name", course.name}, {"concept_block_ids", course.concept_block_ids}});
    return result;
}

nlohmann::json DatasetCatalog::conceptBlock(const std::string& id) const {
    const auto snapshot = view();
    if (!snapshot.catalog) throw std::runtime_error("ConceptBlock database is unavailable: " + lastError());
    const auto block = snapshot.catalog->getConceptBlockById(id);
    if (block.id.empty()) throw std::runtime_error("Registered ConceptBlock is missing: " + id);
    return {{"source_id", block.id}, {"name", block.name}, {"prompt", block.prompt},
        {"goal", block.goal ? GRIM::goalToJson(*block.goal) : nlohmann::json(nullptr)}};
}

std::vector<nlohmann::json> DatasetCatalog::goalSelections(const std::string& query) const {
    const auto snapshot = view();
    if (!snapshot.catalog) throw std::runtime_error("ConceptBlock database is unavailable: " + lastError());
    std::vector<nlohmann::json> result;
    for (const auto& block : snapshot.catalog->searchGoalConceptBlocks(query)) {
        if (block.goal) result.push_back({{"source_id", block.id}, {"name", block.name},
            {"prompt", block.prompt}, {"goal", GRIM::goalToJson(*block.goal)}});
    }
    return result;
}

void DatasetCatalog::importGoal(const nlohmann::json& selection) {
    GRIM::ConceptBlock block;
    block.id = selection.at("source_id").get<std::string>();
    block.name = selection.value("name", block.id);
    block.goal = GRIM::goalFromJson(selection.at("goal"));
    block.prompt = "Play the selected TextWorld challenge.";
    if (!write([&](DatasetTarget& target) {
        const auto previous = target.getConceptBlockById(block.id);
        if (!previous.id.empty()) {
            if (!previous.goal || GRIM::goalToJson(*previous.goal) != GRIM::goalToJson(*block.goal))
                throw std::runtime_error("Imported Goal ID already exists with different content");
            return true;
        }
        size_t added = 0;
        return target.addConceptBlocks({block}, added);
    })) throw std::runtime_error("Failed to import Goal: " + lastError());
}

} // namespace GRIM::DataCollection

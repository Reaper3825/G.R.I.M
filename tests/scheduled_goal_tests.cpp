#include "resources/models/GRIM-text/Shared/DataLoader/ConceptBlockCorpusReader.hpp"
#include "resources/models/GRIM-text/Shared/DataLoader/CurriculumRegistry.hpp"
#include "DataCollection/concept_block_generated.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using nlohmann::json;
namespace fs = std::filesystem;
using GRIM::Batching::CurriculumOrdering;
namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F action, const std::string& message) {
    try { action(); }
    catch (const std::exception& error) {
        check(std::string(error.what()).find(message) != std::string::npos, "Unexpected diagnostic");
        return;
    }
    throw std::runtime_error("Expected rejection: " + message);
}
void writeJson(const fs::path& path, const json& value) { std::ofstream(path) << value.dump(); }
json registry() {
    return {{"curriculums", json::array({{{"id", "curriculum"}, {"name", "Goals"},
        {"training_stage", "sft"}, {"course_ids", {"course-a", "course-b"}},
        {"randomize_course_order", true}, {"randomize_concept_block_order", true}}})},
        {"courses", json::array({
            {{"id", "course-a"}, {"environment_id", "env-a"}, {"concept_block_ids", {"b", "a"}}},
            {{"id", "course-b"}, {"environment_id", "env-b"}, {"concept_block_ids", {"a", "c"}}}})}};
}
void writeBlocks(const fs::path& directory, const std::vector<std::string>& ids,
                 const std::string& noGoal = {}, uint32_t schema = 3) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<GRIMConcept::ConceptBlock>> blocks;
    for (const auto& id : ids) {
        const auto fbId = builder.CreateString(id);
        const auto target = builder.CreateString("Target " + id);
        const auto goal = id == noGoal ? flatbuffers::Offset<GRIMConcept::Goal>{}
                                      : GRIMConcept::CreateGoal(builder, target);
        const auto hidden = builder.CreateString("AUTHORED OUTPUT MUST STAY OUT OF GOAL PLAN");
        GRIMConcept::ConceptBlockBuilder block(builder);
        block.add_id(fbId); block.add_goal(goal); block.add_prompt(hidden);
        block.add_determine(hidden); block.add_define(hidden); block.add_execute(hidden);
        block.add_update(hidden); block.add_answer(hidden); block.add_raw(hidden);
        blocks.push_back(block.Finish());
    }
    const auto dataset = GRIMConcept::CreateConceptBlockDataset(builder, schema, builder.CreateVector(blocks));
    GRIMConcept::FinishConceptBlockDatasetBuffer(builder, dataset);
    std::ofstream out(directory / "concept_blocks.fb", std::ios::binary);
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
}
std::vector<std::string> ids(const std::vector<GRIM::ScheduledGoal>& plan) {
    std::vector<std::string> result;
    for (const auto& assignment : plan) result.push_back(assignment.concept_block_id);
    return result;
}
}
int main() {
    const auto root = fs::temp_directory_path() / ("grim-scheduled-goals-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(root);
    // Tests own only this newly-created directory and remove it after success.
    auto authored = registry();
    const auto registryPath = root / "curriculum_registry.json";
    writeJson(registryPath, authored);
    writeBlocks(root, {"c", "a", "b", "unselected"});
    const auto metadata = GRIM::LoadCurriculumMetadataFromRegistry(root, "Goals");
    check(metadata.courses[0].environment_id == "env-a", "Course environment lost");
    std::vector<GRIM::ScheduledGoal> retained;
    for (const auto ordering : {CurriculumOrdering::PRESERVE, CurriculumOrdering::CURRICULUM, CurriculumOrdering::RANDOM}) {
        for (uint64_t seed : {0ULL, 9ULL, 123ULL}) {
            auto plan = GRIM::loadScheduledGoals(root, "Goals", ordering, seed);
            check(ids(plan) == GRIM::Batching::orderedConceptBlocks(metadata, ordering, seed), "Schedule diverged");
            check(plan.size() == 3, "Shared block was duplicated");
            for (const auto& item : plan) {
                check(item.goal.target_state == "Target " + item.concept_block_id, "Goal mismatch");
                const bool first = item.concept_block_id != "c";
                check(item.course_id == (first ? "course-a" : "course-b"), "Wrong course owner");
                check(item.environment_id == (first ? "env-a" : "env-b"), "Wrong environment");
            }
            retained = std::move(plan);
        }
    }
    // Existing full-corpus operation still returns authored output.
    std::vector<json> full;
    GRIM::CurriculumMetadata loaded;
    GRIM::loadConceptBlocks(root, full, loaded, "Goals");
    check(full.size() == 3 && !full.front().at("answer").get<std::string>().empty(), "Full reader changed");
    const auto load = [&] { return GRIM::loadScheduledGoals(root, "Goals", CurriculumOrdering::PRESERVE, 0); };
    writeBlocks(root, {"a", "b"});
    rejects(load, "Scheduled ConceptBlock missing: c");
    writeBlocks(root, {"a", "b", "c"}, "b");
    rejects(load, "Scheduled ConceptBlock has no Goal: b");
    writeBlocks(root, {"a", "b", "b", "c"});
    rejects(load, "Duplicate scheduled ConceptBlock ID: b");
    writeBlocks(root, {"a", "b", "c"});
    authored["courses"][0].erase("environment_id"); writeJson(registryPath, authored);
    rejects(load, "Course has no environment_id: course-a");
    authored = registry(); writeJson(registryPath, authored);
    writeBlocks(root, {"a", "b", "c"}, {}, 99);
    rejects(load, "unsupported concept-block schema");
    std::ofstream(root / "concept_blocks.fb", std::ios::binary) << "corrupt";
    rejects(load, "FlatBuffer verification failed");
    fs::remove(root / "concept_blocks.fb");
    // Legacy compatibility may parse a whole JSON line, but retains only Goal data.
    {
        std::ofstream legacy(root / "concept_blocks.jsonl");
        for (const auto& id : {"c", "b", "a"})
            legacy << json{{"id", id}, {"goal", {{"target_state", "Legacy " + std::string(id)}}},
                           {"answer", "PRIVATE ANSWER"}}.dump() << '\n';
    }
    check(ids(load()) == std::vector<std::string>({"b", "a", "c"}), "Legacy schedule changed");
    check(load()[0].goal.target_state == "Legacy b", "Legacy Goal missing");
    fs::remove(root / "concept_blocks.jsonl");
    rejects(load, "Scheduled ConceptBlock missing");
    check(retained.front().goal.target_state == "Target " + retained.front().concept_block_id,
          "Returned Goal borrowed file storage");
    for (const auto& entry : fs::directory_iterator(root)) fs::remove(entry.path());
    fs::remove(root);
    std::cout << "Scheduled Goal loading, ordering, ownership and rejection tests passed\n";
}

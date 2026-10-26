#include "DataCollection/io/concept_block_io_flatbuffer.hpp"
#include "DataCollection/io/goal_flatbuffer.hpp"
#include "../tests/concept_span_test_helpers.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int main() {
    GRIM::ConceptBlock source;
    source.id = "phase-fields-round-trip";
    source.prompt = "Prompt";
    source.determine = "Select the operation needed to produce the requested result.";
    source.define = "Define local terms for this reasoning step.";
    source.execute = "Execute the selected operation.";
    source.update = "Update the working state.";
    source.answer = "Answer";
    source.goal = GRIM::ConceptBlockGoal{"Retrieve the key", {
        {"Carry the key", GRIM::EvidenceCondition{"carrying", "key_1", ""}},
        {"Place the apple", GRIM::EvidenceCondition{"on", "apple_1", "table_1"}},
        {"Manual review", std::nullopt}}, {
        {"Do not eat the apple", GRIM::EvidenceCondition{"eaten", "apple_1", ""}},
        {"Be careful", std::nullopt}}};

    const auto path = std::filesystem::temp_directory_path()
        / "grim_concept_block_phase_fields_test.fb";
    std::string error;
    assert(GRIM::ConceptBlockIO::saveFlatBuffer(path, {source}, &error));

    std::vector<GRIM::ConceptBlock> loaded;
    assert(GRIM::ConceptBlockIO::loadFlatBuffer(path, loaded, &error));
    assert(loaded.size() == 1);
    assert(loaded[0].determine == source.determine);
    assert(loaded[0].define == source.define);
    assert(loaded[0].execute == source.execute);
    assert(loaded[0].update == source.update);
    assert(loaded[0].answer == source.answer);
    assert(GRIM::goalToJson(*loaded[0].goal) == GRIM::goalToJson(*source.goal));

    const auto rendered = GRIM::ConceptCanonical::render(loaded[0], testSpanDefinitions());
    assert(rendered.text.find("Carry the key") != std::string::npos);
    assert(rendered.text.find("Do not eat the apple") != std::string::npos);
    for (const auto* hidden : {"key_1", "apple_1", "table_1", "carrying", "eaten", "<evidence>"})
        assert(rendered.text.find(hidden) == std::string::npos);
    assert(!rendered.findNamedSpan("evidence"));
    assert(rendered.findNamedSpan("determine") != nullptr);
    assert(rendered.findNamedSpan("define") != nullptr);
    assert(rendered.findNamedSpan("execute") != nullptr);
    assert(rendered.findNamedSpan("update") != nullptr);
    assert(rendered.findNamedSpan("determine")->begin < rendered.findNamedSpan("define")->begin);
    assert(rendered.findNamedSpan("define")->begin < rendered.findNamedSpan("execute")->begin);
    assert(rendered.findNamedSpan("execute")->begin < rendered.findNamedSpan("update")->begin);
    assert(rendered.findNamedSpan("update")->begin < rendered.findNamedSpan("answer")->begin);

    // Original v2 slots remain readable without reinterpreting old prose.
    const auto writeLegacy = [&](const std::string& prose) {
        flatbuffers::FlatBufferBuilder builder;
        const auto criterion = GRIMConcept::CreateSuccessCriterion(builder,
            builder.CreateString("Legacy check"), builder.CreateString(prose));
        const auto goal = GRIMConcept::CreateGoal(builder, builder.CreateString("Legacy goal"),
            builder.CreateVector(std::vector{criterion}),
            builder.CreateVector(std::vector{builder.CreateString("Legacy boundary")}));
        GRIMConcept::ConceptBlockBuilder blockBuilder(builder);
        blockBuilder.add_goal(goal);
        const auto block = blockBuilder.Finish();
        const auto root = GRIMConcept::CreateConceptBlockDataset(builder, 2, builder.CreateVector(std::vector{block}));
        GRIMConcept::FinishConceptBlockDatasetBuffer(builder, root);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    };
    writeLegacy("");
    assert(GRIM::ConceptBlockIO::loadFlatBuffer(path, loaded, &error));
    assert(!loaded[0].goal->success_criteria[0].evidence);
    assert(loaded[0].goal->constraints[0].constraint == "Legacy boundary");
    assert(!loaded[0].goal->constraints[0].evidence);
    writeLegacy("Prose is not an executable predicate");
    assert(!GRIM::ConceptBlockIO::loadFlatBuffer(path, loaded, &error));
    assert(loaded.empty() && error.find("migration") != std::string::npos);

    // A malformed new condition cannot overwrite an existing dataset.
    const auto before = std::filesystem::file_size(path);
    source.goal->success_criteria[0].evidence->subject.clear();
    assert(!GRIM::ConceptBlockIO::saveFlatBuffer(path, {source}, &error));
    assert(std::filesystem::file_size(path) == before);

    std::error_code cleanup_error;
    std::filesystem::remove(path, cleanup_error);
    return 0;
}

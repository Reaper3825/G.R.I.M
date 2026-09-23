#include "DataCollection/io/concept_block_io_flatbuffer.hpp"
#include "../tests/concept_span_test_helpers.hpp"

#include <cassert>
#include <filesystem>
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

    const auto rendered = GRIM::ConceptCanonical::render(loaded[0], testSpanDefinitions());
    assert(rendered.findNamedSpan("determine") != nullptr);
    assert(rendered.findNamedSpan("define") != nullptr);
    assert(rendered.findNamedSpan("execute") != nullptr);
    assert(rendered.findNamedSpan("update") != nullptr);
    assert(rendered.findNamedSpan("determine")->begin < rendered.findNamedSpan("define")->begin);
    assert(rendered.findNamedSpan("define")->begin < rendered.findNamedSpan("execute")->begin);
    assert(rendered.findNamedSpan("execute")->begin < rendered.findNamedSpan("update")->begin);
    assert(rendered.findNamedSpan("update")->begin < rendered.findNamedSpan("answer")->begin);

    std::error_code cleanup_error;
    std::filesystem::remove(path, cleanup_error);
    return 0;
}

#include "DataCollection/io/concept_block_io_flatbuffer.hpp"
#include "DataCollection/concept_block_canonical.hpp"

#include <cassert>
#include <filesystem>
#include <string>
#include <vector>

int main() {
    GRIM::ConceptBlock source;
    source.id = "phase-fields-round-trip";
    source.prompt = "Prompt";
    source.determine = "Determine what must be done.";
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
    assert(loaded[0].execute == source.execute);
    assert(loaded[0].update == source.update);
    assert(loaded[0].answer == source.answer);

    const auto rendered = GRIM::ConceptCanonical::render(loaded[0]);
    assert(rendered.determine.present);
    assert(rendered.execute.present);
    assert(rendered.update.present);
    assert(rendered.determine.begin < rendered.execute.begin);
    assert(rendered.execute.begin < rendered.update.begin);
    assert(rendered.update.begin < rendered.answer.begin);

    std::error_code cleanup_error;
    std::filesystem::remove(path, cleanup_error);
    return 0;
}

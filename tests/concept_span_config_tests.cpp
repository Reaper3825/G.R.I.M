#include "concept_span_test_helpers.hpp"
#include "../resources/models/GRIM-text/Shared/HyperParameters/HyperParameters_GPU.hpp"
#include <iostream>

int main() {
    const auto definitions = testSpanDefinitions();
    GRIM::Config::AiConfigSnapshot snapshot;
    snapshot.document["training"]["config"]["concept_spans"] = definitions;
    // Inference contexts own the raw snapshot, not LanguageModelConfig.
    const auto spans =
        GRIM::HyperParameters::snapshotTrainingConfigField<GRIM::NamedConceptSpanDefinitions>(
            snapshot, "concept_spans");
    GRIM::ConceptBlock supplied_state;
    supplied_state.prompt = "Question";
    assert(GRIM::ConceptCanonical::renderReasoningPrompt(supplied_state, spans) ==
           "<prompt>\nQuestion\n</prompt>\n\n");
    assert(GRIM::conceptSpanLayout(spans) == GRIM::conceptSpanLayout(definitions));
    std::cout << "snapshot span access and structured inference rendering passed\n";
}

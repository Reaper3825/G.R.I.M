#include "concept_span_test_helpers.hpp"
#include "../resources/models/GRIM-text/training/Phases/Startup/SlidingWindow.hpp"
#include "../resources/models/GRIM-text/training/Phases/Startup/ConceptSupervision.hpp"
#include <iostream>
int main() try {
    using Policy = GRIM::ConceptSpanSupervision;
    auto definitions = testSpanDefinitions();
    const nlohmann::json data = {
        {"prompt", "Question"}, {"goal", {
            {"target_state", "Outcome"},
            {"success_criteria", {{{"criterion", "Check"}, {"evidence", "Support"}},
                                  {{"criterion", "Other"}, {"evidence", "More"}}}},
            {"constraints", {"Limit"}}}},
        {"knowns", {"Known"}}, {"unknowns", {"Unknown"}},
        {"explanation", {"Legacy"}}, {"determine", "Work"}, {"define", "Terms"},
        {"execute", "Result"}, {"update", "State"}, {"answer", "Response"}};
    std::filesystem::create_directories(".codex-build/concept-spans");
    TrainingLogger logger(".codex-build/concept-spans", "spans");
    // Each declared root must support exactly the same policy, including prompt.
    for (size_t selected = 0; selected < definitions.size(); ++selected) {
        auto policy = definitions;
        for (auto& d : policy) d.supervision = Policy::Context;
        policy[selected].supervision = Policy::Supervised;
        auto input = testSpanSequence(data, policy);
        const auto source_span = input.named_concept_spans->find(policy[selected].name)->span;
        std::vector<GRIM::TokenizerArtifacts::GrmtSequence> rows{input};
        GRIMText::Training::applySlidingWindows(rows, "uniform", GRIM::HyperParameters::TrainingStage::SFT,
            policy, 2048, 1024, 1, true, true, logger);
        assert(rows.size() == 1);
        const auto& row = rows.front();
        for (int32_t i = 1; i < static_cast<int32_t>(input.token_ids.size()) + 1; ++i) {
            const bool supervised = i >= source_span.begin + 1 && i < source_span.end + 1;
            assert(row.targets[i - 1] == (supervised ? row.token_ids[i] : -1));
        }
        GRIM::validateNamedConceptSpans(*row.named_concept_spans, row.token_ids.size());
    }
    // Child overrides mask the whole tagged child; repeated pairs resolve under the same parent.
    for (auto& d : definitions) d.supervision = Policy::Context;
    auto& criteria = definitions[2];
    criteria.supervision = Policy::Supervised;
    criteria.children[0].children[1].supervision = Policy::Context;
    auto row = testSpanSequence(data, definitions);
    assert(GRIMText::Training::projectConceptSupervision(row, definitions, "override"));
    for (const auto& e : row.named_concept_spans->entries)
        if (e.name == "evidence")
            for (int i = e.span.begin; i < e.span.end; ++i) assert(row.targets[i - 1] == -1);
    // Ignore removes data and remaps the tree and all token channels.
    criteria.children[0].children[1].supervision = Policy::Ignore;
    row = testSpanSequence(data, definitions);
    const auto before = row.token_ids.size();
    assert(GRIMText::Training::projectConceptSupervision(row, definitions, "ignore"));
    assert(row.token_ids.size() < before && row.named_concept_spans->count("evidence") == 0);
    GRIM::validateNamedConceptSpans(*row.named_concept_spans, row.token_ids.size());
    assert(row.token_ids.size() == row.token_local_atom_indices.size());
    // Short windows clip parents and children rather than dropping or orphaning them.
    std::vector<GRIM::TokenizerArtifacts::GrmtSequence> rows{testSpanSequence(data, definitions)};
    GRIMText::Training::applySlidingWindows(rows, "clip", GRIM::HyperParameters::TrainingStage::SFT,
        definitions, 120, 100, 1, true, false, logger);
    assert(rows.size() > 1);
    for (const auto& w : rows) {
        assert(w.token_ids.size() <= 120 && w.targets.back() == -1);
        if (w.named_concept_spans) GRIM::validateNamedConceptSpans(*w.named_concept_spans, w.token_ids.size());
    }
    auto stale = testSpanSequence(data, definitions);
    definitions[0].open_delimiter = "<changed>";
    expectSpanError([&] { GRIMText::Training::projectConceptSupervision(stale, definitions, "stale"); });
    std::cout << "uniform supervision, overrides, ignore, windows and layout guard passed\n";
} catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }

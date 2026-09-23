#include "concept_span_test_helpers.hpp"
#include <iostream>
#include <unordered_map>
int main() {
    auto definitions = testSpanDefinitions();
    const nlohmann::json row = {
        {"prompt", "Question"}, {"goal", {
            {"target_state", "Outcome"},
            {"success_criteria", {{{"criterion", "Check one"}, {"evidence", "Support"}},
                                  {{"criterion", "Check two"}, {"evidence", ""}}}},
            {"constraints", {"Limit one", "Limit two"}}}},
        {"knowns", {"Known one", "Known two"}}, {"unknowns", {"Unknown"}},
        {"explanation", {"Legacy line"}}, {"determine", "Work"},
        {"define", "Terms"}, {"execute", "Result"}, {"update", "State"}, {"answer", "Response"}};
    size_t lookups = 0;
    std::unordered_map<std::string, int> ids;
    GRIM::resolveConceptSpanDelimiters(definitions, [&](const std::string& text) {
        ++lookups;
        return ids.emplace(text, 1000 + static_cast<int>(ids.size())).first->second;
    });
    const auto rendered = GRIM::ConceptCanonical::render(row, definitions);
    const auto again = GRIM::ConceptCanonical::render(row, definitions);
    assert(rendered.text == again.text && lookups == ids.size());
    assert(rendered.text.find("<prompt>\nQuestion\n</prompt>\n\n") == 0);
    assert(rendered.text.find("<reasoning>") == std::string::npos);
    assert(rendered.text.find("Legacy line\n<determine>") != std::string::npos);
    const auto sequence = testSpanSequence(row, definitions);
    const auto& tree = *sequence.named_concept_spans;
    GRIM::validateNamedConceptSpans(tree, sequence.token_ids.size());
    assert(tree.count("knowns") == 2 && tree.count("success_criterion") == 2);
    assert(tree.count("evidence") == 1 && tree.count("constraint") == 2);
    for (const auto& e : tree.entries) {
        if (e.name == "evidence") {
            const auto& pair = tree.entries.at(e.parent_entry_index);
            assert(pair.name == "success_criterion");
            assert(tree.entries.at(pair.parent_entry_index).name == "success_criteria_and_evidence");
        }
    }
    for (const auto& d : rendered.delimiters)
        assert(ids.at(rendered.text.substr(d.begin, d.end - d.begin)) == d.token_id);
    auto prefix = row;
    for (const auto* name : {"determine", "define", "execute", "update", "answer"}) prefix.erase(name);
    assert(GRIM::ConceptCanonical::renderReasoningPrompt(prefix, definitions) ==
           rendered.text.substr(0, rendered.findNamedSpan("determine")->begin));
    // Adding a field needs no enum/renderer/schema edits.
    auto added = definitions.back();
    added.name = "future_field";
    added.source_path = "/new_payload";
    definitions.push_back(added);
    auto extended = row; extended["new_payload"] = "Novel";
    assert(GRIM::ConceptCanonical::render(extended, definitions).findNamedSpan("future_field"));
    expectSpanError([&] { auto invalid = definitions; invalid[0].close_delimiter.clear();
                         GRIM::validateConceptSpanDefinitions(invalid); });
    expectSpanError([&] { auto invalid = definitions; invalid.push_back(invalid[0]);
                         GRIM::validateConceptSpanDefinitions(invalid); });
    expectSpanError([&] { GRIM::resolveConceptSpanDelimiters(definitions, [](const auto&) { return -1; }); });
    std::cout << "recursive rendering, delimiter cache, hierarchy and prefix passed\n";
}

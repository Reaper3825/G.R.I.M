#include "concept_span_test_helpers.hpp"
#include "../resources/models/GRIM-text/training/Diagnostics/TokenizerDiagnostics.hpp"
#include <iostream>
#include <unordered_map>
int main() {
    auto definitions = testSpanDefinitions();
    const nlohmann::json row = {
        {"prompt", "Question"}, {"goal", {
            {"target_state", "Outcome"},
            {"success_criteria", {{{"criterion", "Check one"}, {"evidence", {{"predicate", "carrying"}, {"subject", "key_1"}}}},
                                  {{"criterion", "Check two"}}}},
            {"constraints", {{{"constraint", "Limit one"}}, {{"constraint", "Limit two"}}}}}},
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
    assert(rendered.text.find("<prompt>\nQuestion\n</prompt>\n") == 0);
    assert(rendered.text.find("<reasoning>") == std::string::npos);
    assert(rendered.text.find("Legacy line\n<determine>") != std::string::npos);
    const auto sequence = testSpanSequence(row, definitions);
    const auto& tree = *sequence.named_concept_spans;
    GRIM::validateNamedConceptSpans(tree, sequence.token_ids.size());
    assert(tree.count("knowns") == 2 && tree.count("success_criterion") == 2);
    assert(tree.count("evidence") == 0 && tree.count("constraint") == 2);
    assert(rendered.text.find("key_1") == std::string::npos);
    for (const auto& e : tree.entries) {
        if (e.name == "criterion") {
            const auto& pair = tree.entries.at(e.parent_entry_index);
            assert(pair.name == "success_criterion");
            assert(tree.entries.at(pair.parent_entry_index).name == "success_criteria_and_evidence");
        }
    }
    for (const auto& d : rendered.delimiters)
        assert(ids.at(rendered.text.substr(d.begin, d.end - d.begin)) == d.token_id);
    // The diagnostics input has only rendered text, including nested and repeated markers.
    const auto lookup = [&](const std::string& marker) { return ids.at(marker); };
    const auto marked = GRIM::SpanText::markConfiguredDelimiters(rendered.text, definitions, lookup);
    assert(marked.delimiters.size() == rendered.delimiters.size());
    for (std::size_t i = 0; i < marked.delimiters.size(); ++i) {
        const auto& detected = marked.delimiters[i];
        const auto& authored = rendered.delimiters[i];
        assert(detected.begin == authored.begin && detected.end == authored.end &&
               detected.token_id == authored.token_id);
    }
    assert(GRIM::SpanText::markConfiguredDelimiters("Question <unknown_tag>", definitions, lookup)
               .delimiters.empty());
    GRIM::NamedConceptSpanDefinition context;
    context.open_delimiter = "<environment_context>";
    context.close_delimiter = "</environment_context>";
    auto preview_definitions = definitions;
    preview_definitions.push_back(context);
    const std::string preview = "<prompt>Question</prompt>\n<environment_context>Observation</environment_context>";
    const auto preview_marked = GRIM::SpanText::markConfiguredDelimiters(preview, preview_definitions,
        [&](const auto& marker) {
            const auto id = ids.find(marker);
            return id == ids.end() ? -1 : id->second;
        });
    assert(preview_marked.text == preview && preview_marked.delimiters.size() == 2);
    assert(preview_marked.delimiters[0].token_id == ids.at("<prompt>"));
    assert(preview_marked.delimiters[1].token_id == ids.at("</prompt>"));
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

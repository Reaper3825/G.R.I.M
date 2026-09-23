#include "../tests/concept_span_test_helpers.hpp"
#include <iostream>
int main() {
    const auto definitions = testSpanDefinitions();
    using namespace GRIM::ConceptCanonical;
    const nlohmann::json minimal{{"prompt", "Question?"}, {"knowns", {""}},
                                 {"goal", {{"success_criteria", {{{"criterion", ""}, {"evidence", ""}}}}}}};
    const auto result = render(minimal, definitions);
    assert(result.text == "<prompt>\nQuestion?\n</prompt>\n\n");
    assert(result.named_spans.size() == 1 && result.delimiters.size() == 2);
    assert(renderReasoningPrompt(minimal, definitions) == result.text);
    const nlohmann::json raw{{"raw", "literal raw text"}};
    assert(render(raw, definitions).text == "literal raw text");
    expectSpanError([&] { renderReasoningPrompt(raw, definitions); });
    expectSpanError([&] { render(nlohmann::json{{"knowns", "not an array"}}, definitions); });
    expectSpanError([&] { render(nlohmann::json{{"answer", 123}}, definitions); });
    std::cout << "empty/malformed sources and raw bypass passed\n";
}

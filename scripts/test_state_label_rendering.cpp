#include "../DataCollection/concept_block_canonical.hpp"
#include <cassert>
#include <iostream>
using namespace GRIM::ConceptCanonical;
using nlohmann::json;

int main() {
    json state = {{"id", "cb_metadata_only"}, {"prompt", "Question?"},
        {"knowns", {"capacity = 120", "remaining = 84"}},
        {"unknowns", {"used volume"}},
        {"goal", {{"target_state", "Find used volume"},
                  {"success_criteria", {{{"criterion", "One assignment"}, {"evidence", "Subtraction applies"}},
                                        {{"criterion", "Correct units"}, {"evidence", ""}}}},
                  {"constraints", {"Use liters", "Do not invent values"}}}},
        {"explanation", {"Supplied reasoning context"}}, {"answer", "used = 36"}};
    const auto original = state;
    const auto rendered = render(state);
    auto value = [&](const LogicalByteSpan& span) {
        assert(span.present && span.begin < span.end && span.end <= rendered.text.size());
        return rendered.text.substr(span.begin, span.end - span.begin);
    };
    assert(value(rendered.knowns[0]) == "capacity = 120");
    assert(value(rendered.knowns[1]) == "remaining = 84");
    assert(value(rendered.unknowns[0]) == "used volume");
    assert(value(rendered.target_state) == "Find used volume");
    assert(value(rendered.success_criteria[0].criterion) == "One assignment");
    assert(value(rendered.success_criteria[0].evidence) == "Subtraction applies");
    assert(!rendered.success_criteria[1].evidence.present);
    assert(value(rendered.constraints[0]) == "Use liters");
    assert(value(rendered.constraints[1]) == "Do not invent values");
    assert(value(rendered.answer) == "used = 36");
    for (const std::string tag : {"knowns", "unknowns", "target_state", "criteria", "criterion", "evidence", "constraints"}) {
        const auto open = rendered.text.find("<" + tag + ">");
        const auto close = rendered.text.find("</" + tag + ">");
        assert(open != std::string::npos && close > open && close < rendered.answer.begin);
    }
    assert(rendered.text.find("cb_metadata_only") == std::string::npos);
    assert(renderReasoningPrompt(state) == rendered.text.substr(0, rendered.answer.begin));
    assert(renderReasoningPrompt(state).find("used = 36") == std::string::npos);
    assert(state == original);
    // Missing collections, empty entries, and optional evidence.
    json minimal = {{"prompt", "Question?"}, {"answer", "Answer"}};
    assert(render(minimal).text == "Question?\nAnswer");
    assert(renderReasoningPrompt(minimal) == "Question?\n");
    minimal["knowns"] = {""};
    assert(!render(minimal).knowns[0].present);
    assert(render(minimal).text.find("<knowns>") == std::string::npos);
    // No annotation leakage into ordinary raw corpus rendering.
    json raw = {{"raw", "literal raw text"}};
    assert(render(raw).text == "literal raw text");
    bool rejected = false;
    try { renderReasoningPrompt(raw); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    GRIM::ConceptBlock block;
    block.prompt = "Question?";
    block.knowns = {"value"};
    block.answer = "Answer";
    assert(renderReasoningPrompt(block) == render(block).text.substr(0, render(block).answer.begin));
    std::cout << "State labels, value spans, metadata exclusion, and inference/training prefix tests passed\n";
}

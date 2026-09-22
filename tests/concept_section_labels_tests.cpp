#include "../DataCollection/concept_block_canonical.hpp"
#include "../resources/models/GRIM-text/Shared/ConceptBlock/NamedConceptSpans.hpp"

#include <cassert>
#include <iostream>
#include <set>
#include <string>

int main() {
    const nlohmann::json row{
        {"prompt", "What is 10 plus 2?"},
        {"goal", {
            {"target_state", "The total is known."},
            {"success_criteria", nlohmann::json::array({
                {{"criterion", "The response reports the sum."},
                 {"evidence", "The two addends are supplied."}}
            })},
            {"constraints", nlohmann::json::array({
                "Use both supplied quantities.",
                "Report one total."
            })}
        }},
        {"knowns", nlohmann::json::array({"a = 10", "b = 2"})},
        {"unknowns", nlohmann::json::array({"sum", "final response"})},
        {"determine", "Compute the total by adding the two supplied quantities."},
        {"define", "Let a and b be the supplied addends; sum is their total."},
        {"execute", "<TOOL>(${a} + ${b})</TOOL> -> ${sum}"},
        {"update", ""},
        {"answer", "There are ${sum} in all."},
    };
    const auto rendered = GRIM::ConceptCanonical::render(row);
    const auto section = [&](const GRIM::ConceptCanonical::LogicalByteSpan& span) {
        assert(span.present && span.end > span.begin);
        return rendered.text.substr(span.begin, span.end - span.begin);
    };
    assert(section(rendered.determine) ==
           "<determine>\nCompute the total by adding the two supplied quantities.\n</determine>\n\n");
    assert(section(rendered.define) ==
           "<define>\nLet a and b be the supplied addends; sum is their total.\n</define>\n\n");
    assert(section(rendered.execute) ==
           "<execute>\n<TOOL>(${a} + ${b})</TOOL> -> ${sum}\n</execute>\n\n");
    assert(section(rendered.answer) ==
           "<answer>\nThere are ${sum} in all.\n</answer>");
    assert(rendered.determine.end == rendered.define.begin);
    assert(rendered.define.end == rendered.execute.begin);
    assert(rendered.execute.end == rendered.answer.begin);

    const std::vector<std::string> expected_names{
        "prompt",
        "target_state",
        "success_criteria",
        "constraints",
        "knowns",
        "unknowns",
        "determine",
        "define",
        "execute",
        "answer",
    };
    assert(rendered.named_spans.size() == expected_names.size());
    std::set<std::string> unique_names;
    for (std::size_t index = 0; index < expected_names.size(); ++index) {
        const auto& named = rendered.named_spans[index];
        assert(named.name == expected_names[index]);
        assert(named.span.present && named.span.end > named.span.begin);
        assert(unique_names.insert(named.name).second);
        assert(rendered.findNamedSpan(named.name) == &named.span);
        if (index > 0) {
            assert(rendered.named_spans[index - 1].span.end == named.span.begin);
        }
    }
    assert(rendered.named_spans.front().span.begin == 0);
    assert(rendered.named_spans.back().span.end == rendered.text.size());

    const auto* constraints = rendered.findNamedSpan("constraints");
    assert(constraints != nullptr && rendered.constraints.size() == 2);
    assert(constraints->begin < rendered.constraints.front().begin);
    assert(constraints->end > rendered.constraints.back().end);
    const auto constraints_text = section(*constraints);
    assert(constraints_text.find("<constraints>\n") == 0);
    assert(constraints_text.find("Use both supplied quantities.") != std::string::npos);
    assert(constraints_text.find("Report one total.") != std::string::npos);

    const auto* knowns = rendered.findNamedSpan("knowns");
    assert(knowns != nullptr && rendered.knowns.size() == 2);
    assert(knowns->begin < rendered.knowns.front().begin);
    assert(knowns->end > rendered.knowns.back().end);
    const auto knowns_text = section(*knowns);
    assert(knowns_text.find("a = 10") != std::string::npos);
    assert(knowns_text.find("b = 2") != std::string::npos);

    const auto* unknowns = rendered.findNamedSpan("unknowns");
    assert(unknowns != nullptr && rendered.unknowns.size() == 2);
    assert(unknowns->begin < rendered.unknowns.front().begin);
    assert(unknowns->end > rendered.unknowns.back().end);
    const auto unknowns_text = section(*unknowns);
    assert(unknowns_text.find("sum") != std::string::npos);
    assert(unknowns_text.find("final response") != std::string::npos);

    GRIM::NamedConceptSpans token_spans;
    token_spans.entries.push_back(
        {"knowns", {1, 2}, GRIM::GoalTokenSpan{3, 5}});
    assert(token_spans.size() == 1);
    assert(token_spans.count("knowns") == 1);
    assert(token_spans.find("knowns") == &token_spans.entries.front());

    auto supplied_state = row;
    supplied_state.erase("determine");
    supplied_state.erase("define");
    supplied_state.erase("execute");
    supplied_state.erase("answer");
    const auto inference_prefix =
        GRIM::ConceptCanonical::renderReasoningPrompt(supplied_state);
    assert(inference_prefix == rendered.text.substr(0, rendered.determine.begin));
    assert(rendered.text.find("<answer>", rendered.answer.end) == std::string::npos);
    std::cout << "concept section labels and inference prefix verified\n";
}

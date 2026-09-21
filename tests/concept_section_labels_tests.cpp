#include "../DataCollection/concept_block_canonical.hpp"

#include <cassert>
#include <iostream>
#include <string>

int main() {
    const nlohmann::json row{
        {"prompt", "What is 10 plus 2?"},
        {"knowns", nlohmann::json::array({"a = 10", "b = 2"})},
        {"unknowns", nlohmann::json::array({"sum"})},
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

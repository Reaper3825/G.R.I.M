//======================================================//
//  Shared ConceptBlock training-text renderer.
//
//  GRIM-text corpus compilation uses the model-visible renderer below.
//  State tags are shared by training and structured inference. DataHub
//  displays the same config-authored tree as corpus compilation.
//======================================================//

#pragma once

#include "concept_block.hpp"
#include "../resources/models/GRIM-text/Shared/ConceptBlock/SpanTextRenderer.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace GRIM::ConceptCanonical {

using SpanText::LogicalByteSpan;
using SpanText::NamedLogicalByteSpan;
using SpanText::DelimiterByteSpan;
using SpanText::RenderResult;
using SpanText::render;

inline RenderResult renderReasoningPromptWithSpans(
    const nlohmann::json& state, const NamedConceptSpanDefinitions& definitions) {
    return SpanText::renderInputWithSpans(state, definitions);
}
inline std::string renderReasoningPrompt(
    const nlohmann::json& state, const NamedConceptSpanDefinitions& definitions) {
    return SpanText::renderInput(state, definitions);
}

// Plain-text authoring adapter used for raw/PT exports and the data editor.
// This has no structured span metadata or supervision semantics.
inline std::string renderPlainText(const nlohmann::json& j) {
    if (j.contains("raw") && !j.at("raw").get<std::string>().empty())
        return j.at("raw").get<std::string>();
    std::string text;
    const auto append = [&](const char* field) {
        if (j.contains(field) && !j.at(field).get<std::string>().empty())
            text += j.at(field).get<std::string>() + "\n";
    };
    append("prompt");
    const auto lines = j.contains("explanation") ? j.at("explanation") :
                       j.value("intermediates", nlohmann::json::array());
    for (const auto& line : lines) text += line.get<std::string>() + "\n";
    for (const auto* phase : {"determine", "define", "execute", "update"}) append(phase);
    text += j.value("answer", std::string{});
    return text;
}

inline nlohmann::json toCanonicalJson(const ConceptBlock& cb) {
    nlohmann::json j{
        {"prompt", cb.prompt},
        {"knowns", cb.knowns},
        {"unknowns", cb.unknowns},
        {"explanation", cb.explanation.empty() ? cb.intermediates : cb.explanation},
        {"determine", cb.determine},
        {"define", cb.define},
        {"execute", cb.execute},
        {"update", cb.update},
        {"answer", cb.answer},
        {"raw", cb.raw}
    };
    if (cb.goal.has_value()) {
        nlohmann::json goal{{"target_state", cb.goal->target_state}};
        goal["success_criteria"] = nlohmann::json::array();
        for (const auto& entry : cb.goal->success_criteria) {
            goal["success_criteria"].push_back(nlohmann::json{
                {"criterion", entry.criterion},
                {"evidence", entry.evidence}
            });
        }
        goal["constraints"] = cb.goal->constraints;
        j["goal"] = std::move(goal);
    }
    if (!cb.execution.empty()) {
        j["execution"] = nlohmann::json::array();
        for (const auto& step : cb.execution) {
            j["execution"].push_back(nlohmann::json{
                {"op", step.op},
                {"args", step.args},
                {"arg_slots", step.arg_slots},
                {"result", step.result}
            });
        }
    }
    return j;
}

inline RenderResult render(const ConceptBlock& cb, const NamedConceptSpanDefinitions& definitions) {
    return render(toCanonicalJson(cb), definitions);
}

inline std::string renderReasoningPrompt(const ConceptBlock& supplied_state, const NamedConceptSpanDefinitions& definitions) {
    return renderReasoningPrompt(toCanonicalJson(supplied_state), definitions);
}

inline std::string renderPlainText(const ConceptBlock& cb) {
    return renderPlainText(toCanonicalJson(cb));
}

}  // namespace GRIM::ConceptCanonical

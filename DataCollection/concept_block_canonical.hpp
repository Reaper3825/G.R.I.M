//======================================================//
//  Shared ConceptBlock training-text renderer.
//
//  DataHub preview and GRIM-text corpus compilation both call this renderer,
//  preventing the authoring preview from drifting from the trained text.
//======================================================//

#pragma once

#include "concept_block.hpp"

#include <nlohmann/json.hpp>

#include <sstream>
#include <string>

namespace GRIM::ConceptCanonical {

struct RenderResult {
    std::string text;
    // Logical <prompt>...</prompt> boundary. The delimiters are metadata only
    // and are never emitted into model-visible text.
    size_t prompt_byte_end = 0;
};

inline RenderResult render(const nlohmann::json& j) {
    RenderResult result;
    std::ostringstream out;

    if (j.contains("prompt") && j["prompt"].is_string()
        && !j["prompt"].get<std::string>().empty()) {
        out << j["prompt"].get<std::string>();
        result.prompt_byte_end = static_cast<size_t>(out.tellp());
        // Keep human-readable content separated while leaving the newline
        // outside the logical prompt span.
        out << "\n";
    }

    const nlohmann::json* explanation = nullptr;
    if (j.contains("explanation") && j["explanation"].is_array()) {
        explanation = &j["explanation"];
    } else if (j.contains("intermediates") && j["intermediates"].is_array()) {
        explanation = &j["intermediates"];
    }
    if (explanation) {
        for (const auto& step : *explanation) {
            if (step.is_string()) out << "EXP: " << step.get<std::string>() << "\n";
        }
    }

    // Answers are training content independently of whether an arithmetic
    // result exists. This is required for NOOP-supervised Q/A blocks.
    if (j.contains("answer") && j["answer"].is_string()
        && !j["answer"].get<std::string>().empty()) {
        out << "A: " << j["answer"].get<std::string>() << "\n";
    }

    result.text = out.str();
    return result;
}

inline RenderResult renderPlainTextWithPromptBoundary(const nlohmann::json& j) {
    RenderResult result;
    std::ostringstream out;
    if (j.contains("prompt") && j["prompt"].is_string()
        && !j["prompt"].get<std::string>().empty()) {
        out << j["prompt"].get<std::string>();
        result.prompt_byte_end = static_cast<size_t>(out.tellp());
        out << "\n";
    }
    const nlohmann::json* explanation = nullptr;
    if (j.contains("explanation") && j["explanation"].is_array()) {
        explanation = &j["explanation"];
    } else if (j.contains("intermediates") && j["intermediates"].is_array()) {
        explanation = &j["intermediates"];
    }
    if (explanation) {
        for (const auto& step : *explanation) {
            if (step.is_string()) out << step.get<std::string>() << "\n";
        }
    }
    if (j.contains("answer") && j["answer"].is_string()
        && !j["answer"].get<std::string>().empty()) {
        out << j["answer"].get<std::string>() << "\n";
    }
    result.text = out.str();
    return result;
}

inline std::string renderPlainText(const nlohmann::json& j) {
    return renderPlainTextWithPromptBoundary(j).text;
}

inline nlohmann::json toCanonicalJson(const ConceptBlock& cb) {
    nlohmann::json j{
        {"prompt", cb.prompt},
        {"explanation", cb.explanation.empty() ? cb.intermediates : cb.explanation},
        {"answer", cb.answer}
    };
    if (!cb.state_0.type.empty() || !cb.state_0.atoms.empty()) {
        j["state_0"] = nlohmann::json{
            {"type", cb.state_0.type},
            {"atoms", cb.state_0.atoms}
        };
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
    if (cb.state_1.has_result) {
        j["state_1"] = nlohmann::json{{"result", cb.state_1.result}};
    }
    return j;
}

inline RenderResult render(const ConceptBlock& cb) {
    return render(toCanonicalJson(cb));
}

inline std::string renderPlainText(const ConceptBlock& cb) {
    return renderPlainText(toCanonicalJson(cb));
}

}  // namespace GRIM::ConceptCanonical

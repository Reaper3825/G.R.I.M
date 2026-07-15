//======================================================//
//  Shared ConceptBlock training-text renderer.
//
//  DataHub preview and GRIM-text corpus compilation both call this renderer,
//  preventing the authoring preview from drifting from the trained text.
//======================================================//

#pragma once

#include "concept_block.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace GRIM::ConceptCanonical {

struct LiteralSpan {
    int binding_index = -1;
    size_t byte_start = 0;
    size_t byte_end = 0;
};

struct RenderResult {
    std::string text;
    std::vector<LiteralSpan> literal_spans;
    size_t execution_prompt_byte_end = 0;
};

inline std::string formatNumber(double value) {
    if (value == std::floor(value) && std::fabs(value) < 1e12) {
        return std::to_string(static_cast<long long>(value));
    }
    std::ostringstream out;
    out << value;
    return out.str();
}

inline RenderResult render(const nlohmann::json& j) {
    RenderResult result;
    std::ostringstream out;

    if (j.contains("question") && j["question"].is_string()
        && !j["question"].get<std::string>().empty()) {
        out << "Q: " << j["question"].get<std::string>() << "\n";
        result.execution_prompt_byte_end = static_cast<size_t>(out.tellp());
    }

    if (j.contains("state_0") && j["state_0"].is_object()) {
        const auto& state = j["state_0"];
        out << "STATE0";
        if (state.contains("type") && state["type"].is_string()
            && !state["type"].get<std::string>().empty()) {
            out << " type=" << state["type"].get<std::string>();
        }
        if (state.contains("atoms") && state["atoms"].is_array()) {
            int binding_index = 0;
            for (const auto& atom : state["atoms"]) {
                if (!atom.is_number()) continue;
                out << " ";
                const size_t byte_start = static_cast<size_t>(out.tellp());
                out << formatNumber(atom.get<double>());
                const size_t byte_end = static_cast<size_t>(out.tellp());
                result.literal_spans.push_back({binding_index++, byte_start, byte_end});
            }
        }
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

    if (j.contains("execution") && j["execution"].is_array()) {
        for (const auto& step : j["execution"]) {
            if (!step.is_object()) continue;
            out << "EXEC " << step.value("op", std::string());
            if (step.contains("args") && step["args"].is_array()) {
                for (const auto& arg : step["args"]) {
                    if (arg.is_number()) out << " " << formatNumber(arg.get<double>());
                }
            }
            if (step.contains("result") && step["result"].is_number()) {
                out << " => " << formatNumber(step["result"].get<double>());
            }
            out << "\n";
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

inline std::string renderPlainText(const nlohmann::json& j) {
    std::ostringstream out;
    if (j.contains("question") && j["question"].is_string()
        && !j["question"].get<std::string>().empty()) {
        out << j["question"].get<std::string>() << "\n";
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
    return out.str();
}

inline nlohmann::json toCanonicalJson(const ConceptBlock& cb) {
    nlohmann::json j{
        {"question", cb.question},
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

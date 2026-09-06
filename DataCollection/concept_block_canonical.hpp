//======================================================//
//  Shared ConceptBlock training-text renderer.
//
//  GRIM-text corpus compilation uses the model-visible renderer below.
//  State tags are shared by training and structured inference. DataHub
//  additionally displays inspection-only prompt/answer wrappers.
//======================================================//

#pragma once

#include "concept_block.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace GRIM::ConceptCanonical {

struct LogicalByteSpan {
    size_t begin = 0;
    size_t end = 0;
    bool present = false;
};

struct SuccessCriterionByteSpans {
    LogicalByteSpan criterion;
    LogicalByteSpan evidence;
};

struct RenderResult {
    std::string text;
    // State labels are model-visible. Field ranges are half-open byte spans
    // over values, excluding their opening/closing labels.
    LogicalByteSpan target_state;
    LogicalByteSpan criteria;
    std::vector<SuccessCriterionByteSpans> success_criteria;
    // Mirrors the criteria collection: one outer <constraints> span plus an
    // independent <constraint> span per entry. Constraints have no evidence
    // pairing, so entries are bare spans rather than paired records.
    LogicalByteSpan constraints_span;
    std::vector<LogicalByteSpan> constraints;
    // Top-level ConceptBlock collections. Each entry owns an independent span;
    // neither collection has an outer span.
    std::vector<LogicalByteSpan> knowns;
    std::vector<LogicalByteSpan> unknowns;
    // The only model-visible response field supervised during SFT.
    LogicalByteSpan answer;
    // Logical <prompt>...</prompt> boundary. The delimiters are metadata only
    // and are never emitted into model-visible text.
    size_t prompt_byte_begin = 0;
    size_t prompt_byte_end = 0;
};

inline void appendLogicalSpan(std::ostringstream& out,
                              const std::string& text,
                              LogicalByteSpan& span) {
    if (text.empty()) {
        return;
    }
    span.begin = static_cast<size_t>(out.tellp());
    out << text;
    span.end = static_cast<size_t>(out.tellp());
    span.present = true;
}

inline void appendStateField(std::ostringstream& out, const char* label,
                             const std::string& value, LogicalByteSpan& span) {
    if (value.empty()) return;
    out << "<" << label << ">\n";
    appendLogicalSpan(out, value, span);
    out << "\n</" << label << ">";
}

inline RenderResult render(const nlohmann::json& j) {
    RenderResult result;
    if (j.contains("raw") && j["raw"].is_string()
        && !j["raw"].get<std::string>().empty()) {
        result.text = j["raw"].get<std::string>();
        return result;
    }
    std::ostringstream out;
    const bool has_goal_decomposition =
        j.contains("goal") && j["goal"].is_object() &&
        ((j["goal"].contains("target_state") &&
          j["goal"]["target_state"].is_string() &&
          !j["goal"]["target_state"].get<std::string>().empty()) ||
         (j["goal"].contains("success_criteria") &&
          j["goal"]["success_criteria"].is_array() &&
          !j["goal"]["success_criteria"].empty()) ||
         (j["goal"].contains("constraints") &&
          j["goal"]["constraints"].is_array() &&
          !j["goal"]["constraints"].empty()));
    const bool has_concept_entries =
        (j.contains("knowns") && j["knowns"].is_array() &&
         !j["knowns"].empty()) ||
        (j.contains("unknowns") && j["unknowns"].is_array() &&
         !j["unknowns"].empty());

    if (j.contains("prompt") && j["prompt"].is_string()
        && !j["prompt"].get<std::string>().empty()) {
        result.prompt_byte_begin = static_cast<size_t>(out.tellp());
        out << j["prompt"].get<std::string>();
        result.prompt_byte_end = static_cast<size_t>(out.tellp());
        // Keep the pinned prompt visually separate from goal decomposition
        // while leaving the separator outside the logical prompt span.
        out << ((has_goal_decomposition || has_concept_entries) ? "\n\n" : "\n");
    }

    auto append_entry_collection = [&out](
        const nlohmann::json& source,
        std::vector<LogicalByteSpan>& spans, const char* label) {
        spans.reserve(source.size());
        for (const auto& source_entry : source) {
            LogicalByteSpan entry;
            if (source_entry.is_string()) {
                appendStateField(
                    out, label, source_entry.get<std::string>(), entry);
                if (entry.present) {
                    out << "\n\n";
                }
            }
            spans.push_back(entry);
        }
    };
    if (j.contains("knowns") && j["knowns"].is_array()) {
        append_entry_collection(j["knowns"], result.knowns, "knowns");
    }
    if (j.contains("unknowns") && j["unknowns"].is_array()) {
        append_entry_collection(j["unknowns"], result.unknowns, "unknowns");
    }

    if (j.contains("goal") && j["goal"].is_object()) {
        const auto& goal = j["goal"];
        if (goal.contains("target_state") && goal["target_state"].is_string()) {
            appendStateField(
                out, "target_state", goal["target_state"].get<std::string>(), result.target_state);
            if (result.target_state.present) {
                out << "\n\n";
            }
        }

        if (goal.contains("success_criteria") &&
            goal["success_criteria"].is_array() &&
            !goal["success_criteria"].empty()) {
            out << "<criteria>\n";
            result.criteria.begin = static_cast<size_t>(out.tellp());
            size_t criteria_content_end = result.criteria.begin;
            result.success_criteria.reserve(goal["success_criteria"].size());
            for (size_t index = 0; index < goal["success_criteria"].size(); ++index) {
                const auto& source_entry = goal["success_criteria"][index];
                SuccessCriterionByteSpans entry;
                if (source_entry.is_object()) {
                    if (source_entry.contains("criterion") &&
                        source_entry["criterion"].is_string()) {
                        appendStateField(
                            out, "criterion",
                            source_entry["criterion"].get<std::string>(),
                            entry.criterion);
                        if (entry.criterion.present) {
                            criteria_content_end = static_cast<size_t>(out.tellp());
                        }
                    }
                    if (entry.criterion.present) {
                        out << "\n";
                    }
                    if (source_entry.contains("evidence") &&
                        source_entry["evidence"].is_string()) {
                        appendStateField(
                            out, "evidence",
                            source_entry["evidence"].get<std::string>(),
                            entry.evidence);
                        if (entry.evidence.present) {
                            criteria_content_end = static_cast<size_t>(out.tellp());
                        }
                    }
                }
                result.success_criteria.push_back(entry);
                if (index + 1 < goal["success_criteria"].size()) {
                    out << "\n\n";
                }
            }
            result.criteria.end = criteria_content_end;
            result.criteria.present =
                result.criteria.end > result.criteria.begin;
            out << "\n</criteria>\n\n";
        }

        if (goal.contains("constraints") &&
            goal["constraints"].is_array() &&
            !goal["constraints"].empty()) {
            out << "<constraints>\n";
            result.constraints_span.begin = static_cast<size_t>(out.tellp());
            size_t constraints_content_end = result.constraints_span.begin;
            result.constraints.reserve(goal["constraints"].size());
            for (size_t index = 0; index < goal["constraints"].size(); ++index) {
                const auto& source_constraint = goal["constraints"][index];
                LogicalByteSpan constraint;
                if (source_constraint.is_string()) {
                    appendStateField(
                        out, "constraint",
                        source_constraint.get<std::string>(),
                        constraint);
                    if (constraint.present) {
                        constraints_content_end = static_cast<size_t>(out.tellp());
                    }
                }
                result.constraints.push_back(constraint);
                if (index + 1 < goal["constraints"].size()) {
                    out << "\n\n";
                }
            }
            result.constraints_span.end = constraints_content_end;
            result.constraints_span.present =
                result.constraints_span.end > result.constraints_span.begin;
            out << "\n</constraints>\n\n";
        }
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

    // Answers are training content independently of whether an arithmetic
    // result exists. This is required for NOOP-supervised Q/A blocks.
    if (j.contains("answer") && j["answer"].is_string()
        && !j["answer"].get<std::string>().empty()) {
        appendLogicalSpan(
            out, j["answer"].get<std::string>(), result.answer);
    }

    result.text = out.str();
    return result;
}

// Supplied state is context, never a target. Removing the answer through the
// same renderer guarantees an exact match to the SFT prefix byte layout.
inline std::string renderReasoningPrompt(const nlohmann::json& supplied_state) {
    if (supplied_state.contains("raw") && supplied_state["raw"].is_string() &&
        !supplied_state["raw"].get_ref<const std::string&>().empty())
        throw std::invalid_argument("Structured reasoning requires state fields, not an opaque raw sequence");
    auto context = supplied_state;
    context.erase("answer");
    return render(context).text;
}

inline RenderResult renderPlainTextWithPromptBoundary(const nlohmann::json& j) {
    RenderResult result;
    if (j.contains("raw") && j["raw"].is_string()
        && !j["raw"].get<std::string>().empty()) {
        result.text = j["raw"].get<std::string>();
        return result;
    }
    std::ostringstream out;
    if (j.contains("prompt") && j["prompt"].is_string()
        && !j["prompt"].get<std::string>().empty()) {
        result.prompt_byte_begin = static_cast<size_t>(out.tellp());
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
        appendLogicalSpan(
            out, j["answer"].get<std::string>(), result.answer);
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
        {"knowns", cb.knowns},
        {"unknowns", cb.unknowns},
        {"explanation", cb.explanation.empty() ? cb.intermediates : cb.explanation},
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

inline RenderResult render(const ConceptBlock& cb) {
    return render(toCanonicalJson(cb));
}

inline std::string renderReasoningPrompt(const ConceptBlock& supplied_state) {
    return renderReasoningPrompt(toCanonicalJson(supplied_state));
}

// Human-facing inspection form. State tags also appear in model input;
// prompt/answer wrappers remain inspection-only.
inline std::string renderLogicalTrainingPreview(const ConceptBlock& cb) {
    if (cb.format_type == "raw" || !cb.raw.empty()) {
        return cb.raw;
    }
    std::ostringstream out;

    if (!cb.prompt.empty()) {
        out << "<prompt>\n" << cb.prompt << "\n</prompt>\n\n";
    }

    for (const auto& known : cb.knowns) {
        out << "<knowns>\n"
            << known
            << "\n</knowns>\n\n";
    }
    for (const auto& unknown : cb.unknowns) {
        out << "<unknowns>\n"
            << unknown
            << "\n</unknowns>\n\n";
    }

    if (cb.goal.has_value()) {
        if (!cb.goal->target_state.empty()) {
            out << "<target_state>\n"
                << cb.goal->target_state
                << "\n</target_state>\n\n";
        }
        if (!cb.goal->success_criteria.empty()) {
            out << "<criteria>\n";
            for (size_t index = 0;
                 index < cb.goal->success_criteria.size();
                 ++index) {
                const auto& entry = cb.goal->success_criteria[index];
                out << "    <criterion>\n"
                    << "    " << entry.criterion << "\n"
                    << "    </criterion>\n";
                if (!entry.evidence.empty()) {
                    out << "    <evidence>\n"
                        << "    " << entry.evidence << "\n"
                        << "    </evidence>\n";
                }
                if (index + 1 < cb.goal->success_criteria.size()) {
                    out << "\n";
                }
            }
            out << "</criteria>\n\n";
        }
        if (!cb.goal->constraints.empty()) {
            out << "<constraints>\n";
            for (size_t index = 0;
                 index < cb.goal->constraints.size();
                 ++index) {
                out << "    <constraint>\n"
                    << "    " << cb.goal->constraints[index] << "\n"
                    << "    </constraint>\n";
                if (index + 1 < cb.goal->constraints.size()) {
                    out << "\n";
                }
            }
            out << "</constraints>\n\n";
        }
    }

    const auto& explanation = cb.explanation.empty()
        ? cb.intermediates
        : cb.explanation;
    if (!explanation.empty() || !cb.answer.empty()) {
        out << "<answer>\n";
        for (const auto& step : explanation) {
            out << step << "\n";
        }
        if (!cb.answer.empty()) {
            out << cb.answer << "\n";
        }
        out << "</answer>\n";
    }
    return out.str();
}

inline std::string renderPlainText(const ConceptBlock& cb) {
    return renderPlainText(toCanonicalJson(cb));
}

}  // namespace GRIM::ConceptCanonical

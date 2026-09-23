//======================================================//
//  Shared ConceptBlock training-text renderer.
//
//  GRIM-text corpus compilation uses the model-visible renderer below.
//  State tags are shared by training and structured inference. DataHub
//  displays the same config-authored tree as corpus compilation.
//======================================================//

#pragma once

#include "concept_block.hpp"
#include "../resources/models/GRIM-text/Shared/ConceptBlock/NamedConceptSpans.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace GRIM::ConceptCanonical {

struct LogicalByteSpan {
    size_t begin = 0;
    size_t end = 0;
    bool present = false;
};

struct NamedLogicalByteSpan {
    std::string name;
    LogicalByteSpan span;
    std::uint32_t parent_entry_index = kNoNamedConceptSpanEntry;
    std::vector<std::uint32_t> child_entry_indices;
};

struct DelimiterByteSpan {
    size_t begin;
    size_t end;
    std::int32_t token_id;
};

struct RenderResult {
    std::string text;
    std::vector<DelimiterByteSpan> delimiters;
    // Depth-first ordered tree; repeated entries retain their configured name.
    std::vector<NamedLogicalByteSpan> named_spans;
    const LogicalByteSpan* findNamedSpan(std::string_view name) const noexcept {
        for (const auto& entry : named_spans) {
            if (std::string_view(entry.name) == name) {
                return &entry.span;
            }
        }
        return nullptr;
    }
};

// Every configured node uses this traversal. Source storage is addressed by
// relative JSON pointers; arrays repeat only when the definition requests it.
inline RenderResult render(const nlohmann::json& source,
                           const NamedConceptSpanDefinitions& definitions) {
    validateConceptSpanDefinitions(definitions);
    RenderResult result;
    if (source.contains("raw") && !source.at("raw").get<std::string>().empty()) {
        result.text = source.at("raw").get<std::string>();
        return result;
    }
    std::function<void(const NamedConceptSpanDefinition&, const nlohmann::json&,
                       std::uint32_t, bool)> emit;
    emit = [&](const auto& d, const auto& parent_value, std::uint32_t parent, bool selected) {
        const auto pointer = nlohmann::json::json_pointer(d.source_path);
        if (!selected && !parent_value.contains(pointer)) return;
        const auto& value = selected ? parent_value : parent_value.at(pointer);
        if (!selected && d.repeat) {
            if (!value.is_array()) throw std::runtime_error("Repeated concept span needs an array: " + d.name);
            for (const auto& item : value) emit(d, item, parent, true);
            return;
        }
        if (value.is_null() || value.empty()) return;
        const size_t begin = result.text.size();
        const size_t delimiter_begin = result.delimiters.size();
        const auto index = static_cast<std::uint32_t>(result.named_spans.size());
        result.named_spans.push_back({d.name, {}, parent, {}});
        const auto delimiter = [&](const std::string& text, std::int32_t id) {
            if (text.empty()) return;
            const auto start = result.text.size();
            result.text += text;
            result.delimiters.push_back({start, result.text.size(), id});
        };
        delimiter(d.open_delimiter, d.open_delimiter_id);
        if (!d.open_delimiter.empty()) result.text += "\n";
        const size_t content_begin = result.text.size();
        if (!d.children.empty()) {
            for (const auto& child : d.children) emit(child, value, index, false);
        } else {
            const auto append = [&](const auto& scalar) {
                if (!scalar.is_string()) throw std::runtime_error("Concept span leaf needs text: " + d.name);
                const auto text = scalar.template get<std::string>();
                if (!text.empty()) result.text += text + "\n";
            };
            if (value.is_array()) for (const auto& item : value) append(item);
            else append(value);
        }
        if (result.text.size() == content_begin) {
            result.text.resize(begin);
            result.delimiters.resize(delimiter_begin);
            result.named_spans.resize(index);
            return;
        }
        delimiter(d.close_delimiter, d.close_delimiter_id);
        if (!d.close_delimiter.empty()) result.text += "\n\n";
        result.named_spans[index].span = {begin, result.text.size(), true};
        if (parent != kNoNamedConceptSpanEntry)
            result.named_spans[parent].child_entry_indices.push_back(index);
    };
    for (const auto& d : definitions) emit(d, source, kNoNamedConceptSpanEntry, false);
    return result;
}

inline std::string renderReasoningPrompt(
    const nlohmann::json& supplied_state, const NamedConceptSpanDefinitions& definitions) {
    if (supplied_state.contains("raw") && !supplied_state.at("raw").get<std::string>().empty())
        throw std::invalid_argument("Structured reasoning input cannot contain raw text");
    const auto rendered = render(supplied_state, definitions);
    const auto policies = conceptSpanPolicies(rendered.named_spans, definitions);
    std::vector<bool> keep(rendered.text.size(), true);
    for (size_t i = 0; i < rendered.named_spans.size(); ++i) {
        const auto& span = rendered.named_spans[i].span;
        std::fill(keep.begin() + span.begin, keep.begin() + span.end,
                  policies[i] != ConceptSpanSupervision::Ignore);
    }
    std::string result;
    for (size_t i = 0; i < keep.size(); ++i) if (keep[i]) result += rendered.text[i];
    return result;
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

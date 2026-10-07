#pragma once

// Configured text and byte-span rendering shared by data authoring and inference.
// This layer has no ConceptBlock or corpus dependency.
#include "NamedConceptSpans.hpp"
#include <nlohmann/json.hpp>
#include <string_view>
#include <utility>

namespace GRIM::SpanText {

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

inline RenderResult renderInputWithSpans(
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
    RenderResult result;
    std::vector<size_t> boundary_map(keep.size() + 1, 0);
    for (size_t i = 0; i < keep.size(); ++i) {
        if (keep[i]) result.text += rendered.text[i];
        boundary_map[i + 1] = result.text.size();
    }
    std::vector<std::uint32_t> entries(rendered.named_spans.size(), kNoNamedConceptSpanEntry);
    for (size_t i = 0; i < rendered.named_spans.size(); ++i) {
        auto entry = rendered.named_spans[i];
        entry.span.begin = boundary_map[entry.span.begin];
        entry.span.end = boundary_map[entry.span.end];
        if (entry.span.begin == entry.span.end) continue;
        if (entry.parent_entry_index != kNoNamedConceptSpanEntry)
            entry.parent_entry_index = entries.at(entry.parent_entry_index);
        entry.child_entry_indices.clear();
        entries[i] = static_cast<std::uint32_t>(result.named_spans.size());
        if (entry.parent_entry_index != kNoNamedConceptSpanEntry)
            result.named_spans.at(entry.parent_entry_index).child_entry_indices.push_back(entries[i]);
        result.named_spans.push_back(std::move(entry));
    }
    for (const auto& delimiter : rendered.delimiters) {
        const auto begin = boundary_map[delimiter.begin], end = boundary_map[delimiter.end];
        if (begin == end) continue;
        if (end - begin != delimiter.end - delimiter.begin)
            throw std::runtime_error("Concept policy partially removed a delimiter");
        result.delimiters.push_back({begin, end, delimiter.token_id});
    }
    return result;
}

inline std::string renderInput(
    const nlohmann::json& supplied_state, const NamedConceptSpanDefinitions& definitions) {
    return renderInputWithSpans(supplied_state, definitions).text;
}


} // namespace GRIM::SpanText

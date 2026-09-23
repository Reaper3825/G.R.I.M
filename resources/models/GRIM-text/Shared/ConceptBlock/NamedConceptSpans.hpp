#pragma once

#include "../Goal/GoalTokenSpan.hpp"
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <memory>
#include <algorithm>

namespace GRIM {

inline constexpr std::uint32_t kNoNamedConceptSpanEntry =
    std::numeric_limits<std::uint32_t>::max();
inline constexpr std::int32_t kUnresolvedConceptSpanDelimiterId = -1;

// Training policy authored on a span definition. Nested definitions inherit
// their nearest explicit ancestor policy unless they override it.
enum class ConceptSpanSupervision : std::uint8_t {
    Inherit = 0,
    Context,
    Supervised,
    Ignore,
};

// Config-authored recursive schema for one model-visible ConceptBlock field.
// Delimiter text is authored in JSON. The corresponding token IDs are resolved
// exactly once after tokenizer initialization and reused for every row. A node
// with children but no delimiters is a transparent structural grouping node.
struct NamedConceptSpanDefinition {
    std::string name;
    ConceptSpanSupervision supervision = ConceptSpanSupervision::Inherit;
    // JSON pointer relative to the parent source value. Empty means that value.
    std::string source_path;
    bool repeat = false;

    std::string open_delimiter;
    std::string close_delimiter;
    std::int32_t open_delimiter_id = kUnresolvedConceptSpanDelimiterId;
    std::int32_t close_delimiter_id = kUnresolvedConceptSpanDelimiterId;

    std::vector<NamedConceptSpanDefinition> children;
};

using NamedConceptSpanDefinitions =
    std::vector<NamedConceptSpanDefinition>;

inline void from_json(const nlohmann::json& j, NamedConceptSpanDefinition& d) {
    d = NamedConceptSpanDefinition{};
    d.name = j.at("name").get<std::string>();
    d.source_path = j.at("source_path").get<std::string>();
    d.repeat = j.at("repeat").get<bool>();
    d.open_delimiter = j.at("open_delimiter").get<std::string>();
    d.close_delimiter = j.at("close_delimiter").get<std::string>();
    const auto policy = j.at("supervision").get<std::string>();
    if (policy == "inherit") d.supervision = ConceptSpanSupervision::Inherit;
    else if (policy == "context") d.supervision = ConceptSpanSupervision::Context;
    else if (policy == "supervised") d.supervision = ConceptSpanSupervision::Supervised;
    else if (policy == "ignore") d.supervision = ConceptSpanSupervision::Ignore;
    else throw std::runtime_error("Unknown concept span supervision: " + policy);
    d.children = j.at("children").get<NamedConceptSpanDefinitions>();
}

inline void to_json(nlohmann::json& j, const NamedConceptSpanDefinition& d) {
    const char* policy = nullptr;
    switch (d.supervision) {
        case ConceptSpanSupervision::Inherit: policy = "inherit"; break;
        case ConceptSpanSupervision::Context: policy = "context"; break;
        case ConceptSpanSupervision::Supervised: policy = "supervised"; break;
        case ConceptSpanSupervision::Ignore: policy = "ignore"; break;
        default: throw std::runtime_error("Invalid concept span supervision");
    }
    j = {{"name", d.name}, {"source_path", d.source_path}, {"repeat", d.repeat},
         {"supervision", policy}, {"open_delimiter", d.open_delimiter},
         {"close_delimiter", d.close_delimiter}, {"children", d.children}};
}

inline void validateConceptSpanDefinitions(const NamedConceptSpanDefinitions& definitions,
                                           bool root = true) {
    if (root && definitions.empty()) throw std::runtime_error("concept_spans must not be empty");
    std::unordered_set<std::string> names;
    for (const auto& d : definitions) {
        if (d.name.empty() || d.name.size() > 1024 || !names.insert(d.name).second)
            throw std::runtime_error("Concept span names must be nonempty and unique among siblings: " + d.name);
        (void)nlohmann::json::json_pointer(d.source_path);
        if (d.open_delimiter.empty() != d.close_delimiter.empty())
            throw std::runtime_error("Concept span requires both delimiters or neither: " + d.name);
        if (root && d.supervision == ConceptSpanSupervision::Inherit)
            throw std::runtime_error("Root concept span needs explicit supervision: " + d.name);
        validateConceptSpanDefinitions(d.children, false);
    }
}

// The caller supplies the tokenizer's exact piece lookup. No manual-vocabulary
// parsing and no encoding/fallback are involved. Call once per loaded vocabulary.
template<class Lookup>
inline void resolveConceptSpanDelimiters(NamedConceptSpanDefinitions& definitions, Lookup&& lookup) {
    validateConceptSpanDefinitions(definitions);
    std::function<void(NamedConceptSpanDefinitions&)> visit = [&](auto& nodes) {
        for (auto& d : nodes) {
            auto resolve = [&](const std::string& text) -> std::int32_t {
                if (text.empty()) return kUnresolvedConceptSpanDelimiterId;
                const auto id = lookup(text);
                if (id < 0) throw std::runtime_error("Missing exact concept delimiter token: " + text);
                return id;
            };
            d.open_delimiter_id = resolve(d.open_delimiter);
            d.close_delimiter_id = resolve(d.close_delimiter);
            visit(d.children);
        }
    };
    visit(definitions);
}

// Structural identity excludes policy and resolved IDs: policies may change
// without recompiling data; layout/source changes require corpus regeneration.
inline std::string conceptSpanLayout(const NamedConceptSpanDefinitions& definitions) {
    nlohmann::json j = definitions;
    std::function<void(nlohmann::json&)> strip = [&](auto& nodes) {
        for (auto& node : nodes) {
            node.erase("supervision");
            strip(node["children"]);
        }
    };
    strip(j);
    return j.dump();
}

template<class Entries>
inline std::vector<ConceptSpanSupervision> conceptSpanPolicies(
    const Entries& entries, const NamedConceptSpanDefinitions& definitions) {
    std::vector<const NamedConceptSpanDefinition*> resolved(entries.size());
    std::vector<ConceptSpanSupervision> policies(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        const auto p = e.parent_entry_index;
        if (p != kNoNamedConceptSpanEntry && p >= i)
            throw std::runtime_error("Concept span parent must precede child");
        const auto& candidates = p == kNoNamedConceptSpanEntry ? definitions : resolved[p]->children;
        const auto found = std::find_if(candidates.begin(), candidates.end(),
            [&](const auto& d) { return d.name == e.name; });
        if (found == candidates.end()) throw std::runtime_error("Unknown configured concept span: " + e.name);
        resolved[i] = &*found;
        policies[i] = found->supervision == ConceptSpanSupervision::Inherit
            ? policies.at(p) : found->supervision;
    }
    return policies;
}

// One configurable model-visible span node after canonical rendering and
// tokenization. Root nodes may own child nodes for collection entries and
// nested scalar fields.
struct NamedConceptSpan {
    std::string name;
    GoalTokenSpan span;
    // Entry indices within the owning NamedConceptSpans::entries vector.
    // Roots use kNoNamedConceptSpanEntry; children name their direct parent.
    std::uint32_t parent_entry_index = kNoNamedConceptSpanEntry;
    std::vector<std::uint32_t> child_entry_indices;
};

// Ordered row-level span metadata. Entry order is canonical model-visible
// order. Parent/child relationships are expressed only through entry indices.
struct NamedConceptSpans {
    std::vector<NamedConceptSpan> entries;

    bool empty() const noexcept { return entries.empty(); }
    std::size_t size() const noexcept { return entries.size(); }

    std::size_t count(std::string_view name) const noexcept {
        std::size_t result = 0;
        for (const auto& entry : entries) {
            if (std::string_view(entry.name) == name) {
                ++result;
            }
        }
        return result;
    }

    const NamedConceptSpan* find(std::string_view name) const noexcept {
        for (const auto& entry : entries) {
            if (std::string_view(entry.name) == name) {
                return &entry;
            }
        }
        return nullptr;
    }

    NamedConceptSpan* find(std::string_view name) noexcept {
        for (auto& entry : entries) {
            if (std::string_view(entry.name) == name) {
                return &entry;
            }
        }
        return nullptr;
    }
};

inline void validateNamedConceptSpans(const NamedConceptSpans& spans, std::size_t token_count) {
    std::vector<std::vector<std::uint32_t>> children(spans.size());
    std::vector<std::uint32_t> stack;
    std::int32_t previous_root_end = 0;
    for (std::uint32_t i = 0; i < spans.size(); ++i) {
        const auto& e = spans.entries[i];
        if (e.name.empty() || e.name.size() > 1024 || !e.span.valid() || static_cast<std::size_t>(e.span.end) > token_count)
            throw std::runtime_error("Invalid named concept span: " + e.name);
        while (!stack.empty() && stack.back() != e.parent_entry_index) stack.pop_back();
        if (e.parent_entry_index == kNoNamedConceptSpanEntry) {
            if (e.span.begin < previous_root_end) throw std::runtime_error("Overlapping concept span roots");
            previous_root_end = e.span.end;
        } else {
            if (stack.empty() || e.parent_entry_index >= i)
                throw std::runtime_error("Concept span tree is not in parent-first depth-first order");
            const auto& p = spans.entries[e.parent_entry_index];
            auto& siblings = children[e.parent_entry_index];
            if (e.span.begin < p.span.begin || e.span.end > p.span.end ||
                (!siblings.empty() && e.span.begin < spans.entries[siblings.back()].span.end))
                throw std::runtime_error("Concept child span is outside its parent or overlaps a sibling");
            siblings.push_back(i);
        }
        stack.push_back(i);
    }
    for (std::size_t i = 0; i < spans.size(); ++i)
        if (children[i] != spans.entries[i].child_entry_indices)
            throw std::runtime_error("Concept span parent/child links disagree");
}

// boundary_map[i] is the output position preceding old token i. A removed
// token has identical adjacent positions. This supports clipping, windowing,
// BOS insertion and selective loading without any field-specific handling.
inline std::shared_ptr<const NamedConceptSpans> remapNamedConceptSpans(
    const std::shared_ptr<const NamedConceptSpans>& source,
    const std::vector<std::int32_t>& boundary_map) {
    if (!source) return nullptr;
    auto result = std::make_shared<NamedConceptSpans>();
    std::vector<std::uint32_t> indices(source->size(), kNoNamedConceptSpanEntry);
    for (std::size_t i = 0; i < source->size(); ++i) {
        auto e = source->entries[i];
        if (!e.span.valid() || static_cast<std::size_t>(e.span.end) >= boundary_map.size())
            throw std::runtime_error("Cannot remap invalid concept span");
        e.span = {boundary_map[e.span.begin], boundary_map[e.span.end]};
        if (!e.span.valid()) continue;
        if (e.parent_entry_index != kNoNamedConceptSpanEntry) {
            if (e.parent_entry_index >= i || indices[e.parent_entry_index] == kNoNamedConceptSpanEntry)
                throw std::runtime_error("Cannot remap orphan concept span");
            e.parent_entry_index = indices[e.parent_entry_index];
        }
        e.child_entry_indices.clear();
        indices[i] = static_cast<std::uint32_t>(result->size());
        if (e.parent_entry_index != kNoNamedConceptSpanEntry)
            result->entries[e.parent_entry_index].child_entry_indices.push_back(indices[i]);
        result->entries.push_back(std::move(e));
    }
    return result->empty() ? nullptr : result;
}

} // namespace GRIM

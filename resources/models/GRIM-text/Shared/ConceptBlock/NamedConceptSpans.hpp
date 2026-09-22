#pragma once

#include "../Goal/GoalTokenSpan.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

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
};

// Config-authored recursive schema for one model-visible ConceptBlock field.
// Delimiter text is authored in JSON. The corresponding token IDs are resolved
// exactly once after tokenizer initialization and reused for every row. A node
// with children but no delimiters is a transparent structural grouping node.
struct NamedConceptSpanDefinition {
    std::string name;
    ConceptSpanSupervision supervision = ConceptSpanSupervision::Inherit;

    std::string open_delimiter;
    std::string close_delimiter;
    std::int32_t open_delimiter_id = kUnresolvedConceptSpanDelimiterId;
    std::int32_t close_delimiter_id = kUnresolvedConceptSpanDelimiterId;

    std::vector<NamedConceptSpanDefinition> children;
};

using NamedConceptSpanDefinitions =
    std::vector<NamedConceptSpanDefinition>;

// One configurable model-visible span node after canonical rendering and
// tokenization. Root nodes may own child nodes for collection entries and
// nested scalar fields.
struct NamedConceptSpan {
    std::string name;
    std::vector<std::int32_t> token_ids;
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

} // namespace GRIM

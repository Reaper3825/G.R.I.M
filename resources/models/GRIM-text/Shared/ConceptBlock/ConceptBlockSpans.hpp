#pragma once

#include "../Goal/GoalTokenSpan.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace GRIM {

// One top-level ConceptBlock known/unknown entry after canonical rendering and
// tokenization. The span is half-open and indexes the owning sequence.
struct ConceptBlockSpanEntry {
    std::vector<std::int32_t> token_ids;
    GoalTokenSpan span;
};

// Sequence-level ConceptBlock metadata. This intentionally lives outside Goal:
// knowns and unknowns describe the concept input, not its goal identifier.
struct ConceptBlockSpans {
    std::vector<ConceptBlockSpanEntry> knowns;
    std::vector<ConceptBlockSpanEntry> unknowns;
    std::optional<ConceptBlockSpanEntry> reasoning;
    std::optional<ConceptBlockSpanEntry> determine;
    std::optional<ConceptBlockSpanEntry> define;
    std::optional<ConceptBlockSpanEntry> execute;
    std::optional<ConceptBlockSpanEntry> update;

    bool empty() const noexcept {
        return knowns.empty() && unknowns.empty() && !reasoning &&
               !determine && !define && !execute && !update;
    }
};

} // namespace GRIM

#pragma once

#include "../Goal/GoalTokenSpan.hpp"

#include <cstdint>
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

    bool empty() const noexcept {
        return knowns.empty() && unknowns.empty();
    }
};

} // namespace GRIM

#pragma once

#include "GoalTokenSpan.hpp"

#include <cstdint>
#include <vector>

namespace GRIM {

// One verifier contract. Keeping optional evidence in the same record preserves
// its field-level association through corpus/window/batch copies. An empty
// evidence_token_ids vector and invalid evidence_span mean evidence is pending.
struct SuccessCriterion {
    std::vector<std::int32_t> token_ids;
    GoalTokenSpan criterion_span;
    std::vector<std::int32_t> evidence_token_ids;
    GoalTokenSpan evidence_span;
};

struct SuccessCriteria {
    GoalTokenSpan span;
    std::vector<SuccessCriterion> entries;
};

} // namespace GRIM

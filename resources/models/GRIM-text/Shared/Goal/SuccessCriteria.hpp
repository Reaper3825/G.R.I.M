#pragma once

#include "GoalTokenSpan.hpp"

#include <cstdint>
#include <vector>

namespace GRIM {

// One verifier contract. Keeping criterion and evidence in the same record
// preserves their field-level association through corpus/window/batch copies.
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

#pragma once

#include "GoalTokenSpan.hpp"

#include <cstdint>
#include <vector>

namespace GRIM {

struct Constraint {
    std::vector<std::int32_t> token_ids;
    GoalTokenSpan constraint_span;
};

// Mirrors SuccessCriteria: one outer span enclosing the ordered entries
// exactly, plus one span per entry. Constraints carry no evidence pairing.
struct Constraints {
    GoalTokenSpan span;
    std::vector<Constraint> entries;
};

} // namespace GRIM

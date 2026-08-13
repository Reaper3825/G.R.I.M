#pragma once

#include "GoalTokenSpan.hpp"

#include <cstdint>
#include <vector>

namespace GRIM {

struct Constraint {
    std::vector<std::int32_t> token_ids;
    GoalTokenSpan constraint_span;
};

struct Constraints {
    std::vector<Constraint> entries;
};

} // namespace GRIM

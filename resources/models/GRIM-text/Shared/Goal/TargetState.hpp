#pragma once

#include "GoalTokenSpan.hpp"

#include <cstdint>
#include <vector>

namespace GRIM {

struct TargetState {
    std::vector<std::int32_t> token_ids;
    GoalTokenSpan span;
};

} // namespace GRIM

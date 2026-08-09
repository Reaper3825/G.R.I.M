#pragma once

#include "GoalTokenSpan.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cstdint>
#include <vector>

namespace GRIM {

struct TargetState {
    std::vector<std::int32_t> token_ids;
    GoalTokenSpan span;
    Tensor norm_mean_pool;
};

} // namespace GRIM

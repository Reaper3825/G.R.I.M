#pragma once

#include "TargetState.hpp"
#include "SuccessCriteria.hpp"
#include <optional>

namespace GRIM {

struct Goal {
    std::optional<TargetState> target_state;
    std::optional<SuccessCriteria> success_criteria;
};

} // namespace GRIM

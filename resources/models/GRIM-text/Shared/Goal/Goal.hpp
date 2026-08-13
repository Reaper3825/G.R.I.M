#pragma once

#include "Constraints.hpp"
#include "SuccessCriteria.hpp"
#include "TargetState.hpp"

#include <optional>

namespace GRIM {

struct Goal {
    std::optional<TargetState> target_state;
    std::optional<SuccessCriteria> success_criteria;
    std::optional<Constraints> constraints;
};

} // namespace GRIM

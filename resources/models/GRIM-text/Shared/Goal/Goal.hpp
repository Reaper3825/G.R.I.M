#pragma once

#include "TargetState.hpp"

#include <optional>

namespace GRIM {

struct Goal {
    std::optional<TargetState> target_state;
};

} // namespace GRIM

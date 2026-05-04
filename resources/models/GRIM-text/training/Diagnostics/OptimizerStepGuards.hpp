//======================================================//
//  OptimizerStepGuards.hpp
//  Optimizer-owned finite checks around the step boundary.
//
//   1. checkPostOptimizerWeightsFinite (Rule 20)
//        After the optimizer step, sample one weight from
//        every parameter group and throw on NaN/Inf.
//======================================================//

#pragma once

#include <cstddef>

namespace GRIM {
    struct ParameterGroup;
}

namespace GRIM::Diagnostics {

void checkPostOptimizerWeightsFinite(
    const GRIM::ParameterGroup* groups,
    std::size_t group_count,
    int optimizer_step,
    float learning_rate,
    int batch_idx);

} // namespace GRIM::Diagnostics

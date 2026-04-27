//======================================================//
//  OptimizerState.hpp
//  Step counter for AdamW / RAdamW bias correction.
//
//  This is the single source of truth for the runtime
//  optimizer step count. The actual moment buffers (m, v)
//  live on each ParameterGroup (see TensorContract_GPU.hpp);
//  AdamW hyperparameters live in HyperParameters_GPU.hpp.
//======================================================//

#pragma once

namespace GRIM {

/// Runtime optimizer step counter.
///
/// Used by launchAdamWStep() / launchRAdamWStep() for bias correction.
/// Incremented once per applied optimizer step (after gradient accumulation).
struct OptimizerState {
    int step = 0;
};

} // namespace GRIM

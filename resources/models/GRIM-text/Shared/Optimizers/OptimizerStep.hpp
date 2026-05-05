//======================================================//
//  OptimizerStep.hpp
//  Step counter for AdamW / RAdamW bias correction.
//
//  This is the single source of truth for the runtime
//  optimizer step count. The actual GPU optimizer state
//  (Adam/RAdam moment tensors) lives in OptimizerState_GPU.hpp.
//======================================================//

#pragma once

namespace GRIM {

/// Runtime optimizer step counter.
///
/// Used by launchAdamWStep() / launchRAdamWStep() for bias correction.
/// Incremented once per applied optimizer step (after gradient accumulation).
struct OptimizerStep {
    int step = 0;
};

} // namespace GRIM
//======================================================//
//  MemoryMeasurer.hpp
//  Phase 2 GPU-memory checkpoint measurement.
//======================================================//

#pragma once

#include "../../Shared/Diagnostics/MemoryAllocationTracker.hpp"

namespace GRIM::Forward {
struct ModelForwardOutputs;
}

namespace GRIMText::Training {

struct TrainingContext;

namespace Memory {

// Shared compile-time gate for both allocation hooks and Phase 2 checkpoints.
// Set MemoryAccounting::Enabled to false to disable both together.
inline constexpr bool EnablePeakMemoryMeasurer = GRIM::MemoryAccounting::Enabled;

// Samples device-wide CUDA memory at a named ownership boundary, updates the
// run-wide high-water mark, and immediately logs both the current sample and
// the matching owner of the running maximum. Observability failures never
// terminate or otherwise change a training step.
void measurePeakMemory(
    TrainingContext& ctx,
    const char* owner,
    int batch_idx,
    int accumulation_slot) noexcept;

// Measures the device-wide forward boundary and emits the per-tensor retained
// activation inventory owned by this forward pass.
void measureForwardMemory(
    TrainingContext& ctx,
    const GRIM::Forward::ModelForwardOutputs& forward_outputs,
    int batch_idx,
    int accumulation_slot) noexcept;

} // namespace Memory
} // namespace GRIMText::Training

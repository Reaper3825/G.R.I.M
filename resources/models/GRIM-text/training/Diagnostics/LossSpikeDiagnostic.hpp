//======================================================//
//  LossSpikeDiagnostic.hpp
//  Per-sequence breakdown when batch loss spikes far
//  above the captured baseline (initial) loss.
//  Lifted from Phase2_TrainingLoop.cu.
//======================================================//

#pragma once

namespace GRIM::Batching { struct BatchPayload; }
namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

// Spike threshold = baseline_loss * kLossSpikeBaselineMultiplier.
// Chosen so that a model that has already learned anything non-trivial
// (loss well below the random baseline) will flag any regression back
// toward, and well past, the random-init regime as a spike.
constexpr float kLossSpikeBaselineMultiplier = 1.5f;

void runLossSpikeDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    float loss,
    int batch_idx,
    float baseline_loss);

} // namespace GRIM::Diagnostics

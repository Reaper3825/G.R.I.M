//======================================================//
//  LossSpikeDiagnostic.hpp
//  Per-sequence breakdown when batch loss spikes far
//  above the captured baseline (initial) loss.
//  Lifted from Phase2_TrainingLoop.cu.
//======================================================//

#pragma once

namespace GRIM::Batching { struct BatchPayload; }
namespace GRIMText { namespace Training { struct TrainingContext; } }
namespace GRIM::Loss { class LossSignalBus; }

namespace GRIM::Diagnostics {

// Spike detection lives in GRIM::Loss::LossSignalBus (see
// Shared/Loss/LossSignals/LossSignals.hpp). This diagnostic is now a pure
// LOG SUBSCRIBER: it fires when bus.latest().baseline_spike is set and emits
// the per-sequence breakdown. The threshold/baseline are sourced from the
// bus so every loss-spike consumer sees the same numbers.
void runLossSpikeDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    float loss,
    int batch_idx,
    const GRIM::Loss::LossSignalBus& bus);

} // namespace GRIM::Diagnostics

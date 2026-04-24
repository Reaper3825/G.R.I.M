//======================================================//
//  LossSpikeDiagnostic.hpp
//  Per-sequence breakdown when batch loss > 50.
//  Lifted verbatim from Phase2_TrainingLoop.cu.
//======================================================//

#pragma once

namespace GRIM::Batching { struct BatchPayload; }
namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

void runLossSpikeDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    float loss,
    int batch_idx);

} // namespace GRIM::Diagnostics

//======================================================//
//  LossStatsDiagnostic.hpp
//  Per-batch loss equation logging ([BATCH_LOSS]) and
//  loss/token statistics line ([LossStats]). Lifted
//  verbatim from Phase2_TrainingLoop.cu processBatch.
//======================================================//

#pragma once

namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct BatchResult;
} }
namespace GRIM::Batching { struct BatchPayload; }

namespace GRIM::Diagnostics {

void runLossStatsDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    const GRIMText::Training::BatchResult& result,
    int batch_idx);

} // namespace GRIM::Diagnostics

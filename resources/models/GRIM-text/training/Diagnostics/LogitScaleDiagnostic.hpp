#pragma once
//======================================================//
//  LogitScaleDiagnostic.hpp
//  Per-batch logit-scale equation, h↔W alignment,
//  unigram-direction collapse detector, LM-head row-norm
//  spot check, and the embedded computeRhoDiagnostic() call.
//
//  Lifted verbatim from Phase2_TrainingLoop.cu (the
//  "TRAINING SIGNAL: Logit Statistics" scope).
//======================================================//

#include "../../Shared/Batching/BatchPayload.hpp"

namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

void runLogitScaleDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx);

} // namespace GRIM::Diagnostics

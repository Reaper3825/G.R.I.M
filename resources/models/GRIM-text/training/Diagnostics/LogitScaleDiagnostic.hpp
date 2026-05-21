#pragma once
//======================================================//
//  LogitScaleDiagnostic.hpp
//  Per-batch LM-valid logit-scale equation, h↔W alignment,
//  unigram-direction collapse detector, LM-head row-norm
//  spot check, and the embedded computeRhoDiagnostic() call.
//======================================================//

#include "../../Shared/Batching/BatchPayload.hpp"

namespace GRIMText { namespace Training { struct TrainingContext; } }
namespace GRIM { struct Tensor; }

namespace GRIM::Diagnostics {

void runLogitScaleDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Tensor& logits_tensor,
    const GRIM::Tensor& lm_head_input_tensor,
    int batch_idx);

} // namespace GRIM::Diagnostics

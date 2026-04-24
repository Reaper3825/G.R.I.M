//======================================================//
//  MtpDiagnostic.hpp
//  Multi-Token-Prediction telemetry: per-head loss/acc,
//  loss_ratio, alpha_effective, L_total, and the
//  Lk/L0 healthy-range monitor. Lifted verbatim from
//  Phase2_TrainingLoop.cu (runEpoch interval-log scope).
//======================================================//

#pragma once

namespace GRIMText { namespace Training {
struct TrainingContext;
struct BatchResult;
} }

namespace GRIM::Diagnostics {

void runMtpDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIMText::Training::BatchResult& batch_result);

} // namespace GRIM::Diagnostics

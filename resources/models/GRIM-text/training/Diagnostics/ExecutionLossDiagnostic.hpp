//======================================================//
//  ExecutionLossDiagnostic.hpp
//  Execution-objective decomposition diagnostic
//======================================================//

#pragma once

namespace GRIMText::Training {
struct TrainingContext;
}

namespace GRIM::Autograd {
struct LossResult;
}

namespace GRIM::Batching {
struct BatchPayload;
}

namespace GRIM::Forward {
struct ModelForwardOutputs;
}

namespace GRIM::Diagnostics {

/// Reconstruct the execution auxiliary objective from the retained execution
/// logits/probabilities, emit a per-head equation diagnostic, and publish the
/// decomposition to the execution-loss telemetry streams.
///
/// This is diagnostic-only: it does not mutate the live loss tensor or change
/// backward behavior. `residual = loss_result.execution_loss - reconstructed`
/// exposes any mismatch between the live objective and the independently
/// reconstructed terms.
void runExecutionLossDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Forward::ModelForwardOutputs& forward_outputs,
    const GRIM::Autograd::LossResult& loss_result,
    int batch_idx);

} // namespace GRIM::Diagnostics

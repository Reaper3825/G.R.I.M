//======================================================//
//  AutogradExecutionLoss.hpp
//  Autograd-owned execution auxiliary loss assembly
//======================================================//

#pragma once

namespace GRIM {
namespace Batching {
struct BatchPayload;
}
namespace Forward {
struct ModelForwardOutputs;
}
namespace Autograd {

struct AutogradContext;
struct AutogradLossState;

struct ExecutionAuxiliaryLossSummary {
    float structured_ce = 0.0f;      // Unweighted policy-temperature CE averaged over emitted CE tensors
    float entropy_monitor = 0.0f;    // Monitoring-only entropy scalar; not added to loss_tensor
    int active_steps = 0;            // Active non-padded transition targets considered
    int scalar_loss_terms = 0;       // Scalar execution loss terms added before normalization
    int execute_targets = 0;
    int stop_targets = 0;
};

/**
 * Adds execution auxiliary losses into loss_state.loss_tensor.
 *
 * Contract:
 * - Consumes retained Category 1 tensors from Forward::ExecutionBlockStepOutput
 *   entries stored on ModelForwardOutputs::exec_outputs_per_row.
 * - Resolves transition targets from Phase1-authored BatchPayload.transition_targets.
 * - Mutates only Category 1 autograd state in Forward::ModelForwardOutputs + AutogradLossState.
 * - Returns host telemetry for logging / diagnostics.
 * - Throws on missing retained tensors, invalid transition targets, or broken
 *   execution output/payload alignment.
 */
ExecutionAuxiliaryLossSummary addExecutionAuxiliaryLoss(
    AutogradContext& ctx,
    const Batching::BatchPayload& payload,
    Forward::ModelForwardOutputs& forward_outputs,
    AutogradLossState& loss_state);

}  // namespace Autograd
}  // namespace GRIM

//======================================================//
//  AutogradExecutionLoss.hpp
//  Autograd-owned execution auxiliary loss assembly
//======================================================//

#pragma once

namespace GRIM {
namespace Autograd {

struct AutogradContext;
struct AutogradIntermediates;

struct ExecutionAuxiliaryLossSummary {
    float structured_ce = 0.0f;      // Unweighted raw CE monitor averaged over emitted CE tensors
    float entropy_monitor = 0.0f;    // Monitoring-only entropy scalar; not added to loss_tensor
    int active_steps = 0;            // Active teacher/non-padded execution steps considered
    int scalar_loss_terms = 0;       // Scalar execution loss terms added before normalization
};

/**
 * Adds execution auxiliary losses into intermediates.loss_tensor.
 *
 * Contract:
 * - Consumes retained Category 1 tensors from Forward::ExecutionBlockStepOutput
 *   entries stored on ModelForwardOutputs::exec_outputs_per_row.
 * - Resolves teacher targets from Phase1-authored BatchPayload.teacher_steps.
 * - Mutates only Category 1 autograd state in AutogradIntermediates.
 * - Returns host telemetry for logging / diagnostics.
 * - Throws on missing retained tensors, invalid teacher targets, or broken
 *   execution output/payload alignment.
 */
ExecutionAuxiliaryLossSummary addExecutionAuxiliaryLoss(
    AutogradContext& ctx,
    AutogradIntermediates& intermediates);

}  // namespace Autograd
}  // namespace GRIM

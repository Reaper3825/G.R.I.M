//======================================================//
//  AutogradSelectorSupervisionLoss.hpp
//  Decode-time selector supervision loss primitive
//======================================================//

#pragma once

namespace GRIM {
namespace Forward {
struct ModelForwardOutputs;
}
namespace Autograd {

struct AutogradContext;
struct AutogradLossState;

/**
 * Adds final-state decode-time selector supervision CE into loss_state.loss_tensor.
 *
 * Contract:
 * - Consumes Phase1-authored BatchPayload selector targets from ctx.payload.
 * - Mutates only Category 1 autograd state in Forward::ModelForwardOutputs + AutogradLossState.
 * - Owns detached selector input tensors in Forward::ModelForwardOutputs for backward lifetime.
 * - Returns the weighted host scalar contribution added to total loss.
 * - Throws on configured-but-unavailable selector/policy state or unrepresentable targets.
 */
float addSelectorSupervisionLoss(
    AutogradContext& ctx,
    Forward::ModelForwardOutputs& forward_outputs,
    AutogradLossState& loss_state);

}  // namespace Autograd
}  // namespace GRIM

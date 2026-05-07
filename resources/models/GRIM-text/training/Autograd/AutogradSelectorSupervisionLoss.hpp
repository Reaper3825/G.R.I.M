//======================================================//
//  AutogradSelectorSupervisionLoss.hpp
//  Decode-time selector supervision loss primitive
//======================================================//

#pragma once

namespace GRIM {
namespace Autograd {

struct AutogradContext;
struct AutogradIntermediates;

/**
 * Adds final-state decode-time selector supervision CE into intermediates.loss_tensor.
 *
 * Contract:
 * - Consumes Phase1-authored BatchPayload selector targets from ctx.payload.
 * - Mutates only Category 1 autograd state in AutogradIntermediates.
 * - Owns detached selector input tensors in AutogradIntermediates for backward lifetime.
 * - Returns the weighted host scalar contribution added to total loss.
 * - Throws on configured-but-unavailable selector/policy state or unrepresentable targets.
 */
float addSelectorSupervisionLoss(
    AutogradContext& ctx,
    AutogradIntermediates& intermediates);

}  // namespace Autograd
}  // namespace GRIM

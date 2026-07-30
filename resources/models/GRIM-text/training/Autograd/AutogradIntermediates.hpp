//======================================================//
//  AutogradIntermediates.hpp
//
//  Training autograd step-state declarations kept next to the
//  shared-forward sink contract.
//
//  - Training callers own the retained Category 1 shared-forward tensors
//    (`Forward::ModelForwardOutputs`) explicitly for the active step.
//  - Training callers own the training-only scalar loss root explicitly as
//    `AutogradLossState`.
//
//  Separated from AutogradTraining.hpp to avoid circular includes
//  (TrainingState_GPU.hpp←→AutogradTraining.hpp).
//======================================================//

#pragma once

#include "../../Shared/Forward/ModelForwardOutputs.hpp"
namespace GRIM {
namespace Autograd {

struct AutogradLossState {
    Tensor loss_tensor;                // Scalar loss driving backward

    void clear() {
        loss_tensor = Tensor();
    }

    bool hasLoss() const { return loss_tensor.data != nullptr; }
};

}  // namespace Autograd
}  // namespace GRIM

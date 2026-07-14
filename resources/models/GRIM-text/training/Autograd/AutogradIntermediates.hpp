//======================================================//
//  AutogradIntermediates.hpp
//
//  Training autograd step-state declarations kept next to the
//  shared-forward sink contract.
//
//  - Training callers own the retained Category 1 shared-forward tensors
//    (`Forward::ModelForwardOutputs`) explicitly for the active step.
//  - Training callers also own the training-only scalar loss root and
//    execution-loss activity flags explicitly as `AutogradLossState`.
//
//  Separated from AutogradTraining.hpp to avoid circular includes
//  (TrainingState_GPU.hpp←→AutogradTraining.hpp).
//======================================================//

#pragma once

#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

#include <vector>

namespace GRIM {
namespace Autograd {

struct AutogradLossState {
    Tensor loss_tensor;                // Scalar loss driving backward
    bool exec_op_ce_added = false;     // true iff an active op-path execution loss was added to loss_tensor
    bool exec_arg_ce_added = false;    // true iff an active arg-path execution loss was added to loss_tensor
    bool exec_write_ce_added = false;  // true iff active write selection CE was added to loss_tensor
    bool exec_transition_added = false;// true iff active transition/value execution loss was added to loss_tensor
    bool exec_execute_ce_added = false;// true iff EXECUTE/NOOP CE was added
    bool exec_stop_ce_added = false;   // true iff STOP/CONTINUE CE was added

    void clear() {
        loss_tensor = Tensor();
        exec_op_ce_added = false;
        exec_arg_ce_added = false;
        exec_write_ce_added = false;
        exec_transition_added = false;
        exec_execute_ce_added = false;
        exec_stop_ce_added = false;
    }

    bool hasLoss() const { return loss_tensor.data != nullptr; }
};

}  // namespace Autograd
}  // namespace GRIM

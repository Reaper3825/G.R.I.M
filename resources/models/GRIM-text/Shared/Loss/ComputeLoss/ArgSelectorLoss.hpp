#pragma once
//======================================================//
//  ArgSelectorLoss.hpp
//  Forward loss operation for arg/option selector supervision.
//======================================================//

#include "../../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM {
namespace autograd {

// Scalar selection cross-entropy. d_targets is the per-row batch-global
// candidate index (or -1 to ignore). num_valid is the supervised-row count.
Tensor argSelectorLoss(const Tensor& selection_logits,
                       const int* d_targets,
                       int total_tokens,
                       int num_classes,
                       int num_valid,
                       cudaStream_t stream);

}  // namespace autograd
}  // namespace GRIM

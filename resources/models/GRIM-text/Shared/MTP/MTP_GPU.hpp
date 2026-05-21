//======================================================//
//  MTP_GPU.hpp
//  Multi-Token Prediction (MTP) CUDA primitives
//
//  Narrow shared kernels only. Autograd loss assembly lives under
//  training/Autograd so this shared module does not cross model, LM-head,
//  TrainingState, or AutogradContext ownership boundaries. Host launchers
//  consume BatchPayload for payload-owned geometry; only device kernels receive
//  raw scalar ABI values.
//======================================================//

#pragma once

#include <cuda_runtime.h>

namespace GRIM {
namespace Batching {
struct BatchPayload;
}

namespace MTP {

//=============================================================================
// KERNEL LAUNCHERS — accuracy
//=============================================================================

/**
 * Compute MTP head accuracy: correct count and valid count (target != -1).
 * d_correct and d_valid are caller-owned output counters and MUST already be
 * zero-initialized before launch; the kernel accumulates into them.
 * Caller copies d_correct and d_valid to host and computes acc = correct / valid.
 */
void launchMTPAccuracyKernel(
    const float* logits,
    const int* targets,
    const Batching::BatchPayload& payload,
    int* d_correct,
    int* d_valid,
    cudaStream_t stream
);

}  // namespace MTP
}  // namespace GRIM

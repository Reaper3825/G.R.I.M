//======================================================//
//  MTP_GPU.hpp
//  Multi-Token Prediction (MTP) auxiliary heads
//
//  Kernels & loss computation for K auxiliary future-prediction
//  heads (Gloeckle et al. 2024 / DeepSeek-V3 design).
//  Extracted from AutogradLoss + AutogradTraining for modularity.
//======================================================//

#pragma once

#include "../TensorContract/TensorContract_GPU.hpp"
#include "../Loss/ComputeLoss/AutogradLoss.hpp"
#include "MTPDiagnostics.hpp"
#include "../../training/Autograd/AutogradIntermediates.hpp"
#include "../../training/Autograd/AutogradTraining.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../CudaAllocUtils.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace GRIM {
namespace MTP {

//=============================================================================
// KERNEL LAUNCHERS — accuracy
//=============================================================================

/**
 * Compute MTP head accuracy: correct count and valid count (target != -1).
 * Caller copies d_correct and d_valid to host and computes acc = correct / valid.
 */
void launchMTPAccuracyKernel(
    const float* logits,
    const int* targets,
    int total_tokens,
    int vocab_size,
    int* d_correct,
    int* d_valid,
    cudaStream_t stream
);

//=============================================================================
// MTP LOSS COMPUTATION — L_total += α/K * Σ_k L_k
//=============================================================================

/**
 * Compute MTP auxiliary losses and accumulate into intermediates.loss_tensor.
 *
 * This function:
 *  1. Resolves mtp_input (same representation as LM head — A1 fix)
 *  2. For each head k: allocate per-head GPU target buffer, matmul, bias, unified_loss, scale, add
 *  3. Fills diagnostics (head_loss, head_acc, alpha_effective)
 *
 * Reads shifted targets from payload.mtp_shifted_targets[k] (computed by
 * buildBatchPayload) and uploads to per-head GPU buffers stored in
 * intermediates.mtp_shifted_targets_gpu (kept alive for NLLLossGradFn backward).
 * BatchPayload is the single source of truth for batch geometry and masking.
 *
 * @param ctx           AutogradContext with model, config, stream, step, payload
 * @param intermediates AutogradIntermediates owning encoder_output, centered output, loss_tensor, per-head target buffers
 * @param diagnostics   Host-side per-step MTP telemetry payload
 */
void computeMTPAuxiliaryLosses(
    Autograd::AutogradContext& ctx,
    Autograd::AutogradIntermediates& intermediates,
    MTPDiagnostics& diagnostics
);

/**
 * Verify that all MTP head gradients are connected (non-NULL .grad).
 * @return true if all gradients are present (or MTP disabled), false if any missing
 */
bool verifyMTPGradients(const LanguageModel& model);

}  // namespace MTP
}  // namespace GRIM

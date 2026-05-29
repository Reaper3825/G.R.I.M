//======================================================//
//  AutogradMtpAuxiliaryLoss.hpp
//  Autograd-owned Multi-Token Prediction (MTP) loss assembly
//======================================================//

#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace GRIM {

class LanguageModel;
struct Tensor;

namespace Batching {
struct BatchDeviceBindings;
struct BatchPayload;
}

namespace HyperParameters {
struct LossConfigHP;
}

namespace MTP {
struct MTPDiagnostics;
}

namespace Autograd {

/**
 * Add MTP auxiliary losses into the active autograd loss tensor using
 * precomputed shared-forward-owned MTP logits.
 *
 * Ownership boundary:
 * - Caller must have already requested MTP logits from shared forward and kept
 *   the resulting Category 1 tensors alive through loss assembly.
 * - This primitive consumes uploaded BatchDeviceBindings target slices,
 *   appends weighted MTP losses into `loss_tensor`, and returns host
 *   telemetry in `diagnostics`.
 * - It MUST NOT create logits or re-run forward math.
 */
float computeAutogradMtpAuxiliaryLosses(
    LanguageModel& model,
    Tensor& loss_tensor,
    std::vector<Tensor>& mtp_logits_tensors,
    MTP::MTPDiagnostics& diagnostics,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const HyperParameters::LossConfigHP& loss_config,
    const float* d_class_weights,
    cudaStream_t stream,
    float mtp_alpha_effective,
    float text_ce_loss
);

}  // namespace Autograd
}  // namespace GRIM

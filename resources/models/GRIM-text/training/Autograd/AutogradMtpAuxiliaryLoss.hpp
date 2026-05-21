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
 * Add MTP auxiliary losses into the active autograd loss tensor.
 *
 * Ownership boundary:
 * - Caller resolves `mtp_input` from the LM-head/autograd representation policy.
 * - This primitive consumes model-owned MTP heads, Phase1-authored payload
 *   semantics, and uploaded BatchDeviceBindings target slices.
 * - It mutates only Category 1 graph state: `loss_tensor` and
 *   `mtp_logits_tensors`, and returns host telemetry in `diagnostics`.
 * - It never reads AutogradContext, TrainingState, or LMHeadLayer fields.
 */
float computeAutogradMtpAuxiliaryLosses(
    LanguageModel& model,
    Tensor& mtp_input,
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

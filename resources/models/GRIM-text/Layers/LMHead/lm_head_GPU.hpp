//======================================================//
//  LM Head Layer - GPU (registry-owned parameter tensors)
//  Linear projection from hidden states to vocabulary logits
//
//  Borrows: weights [vocab_size, d_model], bias [vocab_size] (optional),
//           final_rms_gamma [d_model] (pre-LM-head normalization).
//
//  Architecture: logits = projected(centered(RMSNorm(encoder_output))) @ W^T + bias
//  Where W is either tied to embedding weights or independently allocated.
//
//  Backward is handled automatically by the autograd tape system:
//    grad_W = centered^T @ grad_logits
//    grad_input = grad_logits @ W  (flows back through centering + RMSNorm ops)
//    grad_bias = sum(grad_logits, dim=0)
//    grad_gamma via RMSNormGradFn
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <cstdint>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

namespace GRIM {

// Local experiment toggle only. Keep this LM-head-local until we decide
// whether token-layout gating should become an authored config field.
//
// IMPORTANT: setting this false disables the LM-head-side hard token-type
// gate without changing embedding lookup, so tied embeddings no longer have
// strict embedding/LM symmetry. That asymmetry is intentional for local
// experiments.
inline constexpr bool kEnableLmHeadTokenTypeGateExperiment = false;

//======================================================//
//  LM-head free functions over registry-owned tensors
//======================================================//

inline Tensor& requireMutableFinalRmsGamma(
    LMHeadParameterTensors& parameter_tensors,
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const char* caller) {
    if (hp.freeze_learned_rms_gammas) {
        throw std::runtime_error(
            std::string("[FROZEN-GAMMA-LEAK] mutable access to final_rms_gamma while "
                        "freeze_learned_rms_gammas=true. caller=") +
            (caller ? caller : "<unknown>"));
    }
    return parameter_tensors.final_rms_gamma;
}

inline bool lmHeadWeightsReady(const LMHeadParameterTensors& parameter_tensors) {
    return parameter_tensors.weights.data != nullptr;
}

/// LM head forward with autograd tracking:
///   0. Optional: RMSNorm(input, final_rms_gamma_frozen_or_trained_) — pre-LM-head normalization
///   1. Optional: center_columns_by_causal_prefix_lengths on normalized input (Issue #125/#132)
///   2. Optional: project_out_pc1 on the current LM input (composes after centering when both are enabled)
///   3. logits = input @ weights^T  (autograd::matmul, transpose_b=true)
///   4. Optional: center_rows on logits (numerical stability)
///   5. Optional: logits += bias  (autograd::broadcast_add)
void forwardLmHead(
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameter_tensors,
    const Tensor& input,
    const Batching::BatchPayload& payload,
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs);

} // namespace GRIM

#endif // USE_CUDA

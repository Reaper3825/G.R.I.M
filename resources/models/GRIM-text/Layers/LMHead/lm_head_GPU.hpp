//======================================================//
//  LM Head Layer - GPU (registry-owned parameter tensors)
//  Linear projection from hidden states to vocabulary logits
//
//  Borrows: weights [vocab_size, d_model], bias [vocab_size] (optional),
//           final_rms_gamma [d_model] (pre-LM-head normalization),
//           mlp_W_gate/mlp_W_up [d_model, mlp_d_ff] + mlp_W_down [mlp_d_ff, d_model]
//           (optional residual SwiGLU adapter, config.lm_head_mlp_enabled).
//
//  Architecture: logits = adapter(RMSNorm(encoder_output)) @ W_eff^T + bias
//  Where W is either tied to embedding weights or independently allocated, and
//  adapter(z) = z + mlp_alpha * (SiLU(z @ W_gate) ⊙ (z @ W_up)) @ W_down.
//
//  Backward is handled automatically by the autograd tape system:
//    grad_W = readout_input^T @ grad_logits
//    grad_input = grad_logits @ W_eff (flows through adapter + RMSNorm)
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
//  LM-head free function over registry-owned tensors
//======================================================//

/// LM head forward with autograd tracking:
///   0.   Optional: RMSNorm(input, final_rms_gamma_frozen_or_trained_) — pre-LM-head normalization
///   0.5. Optional: residual SwiGLU adapter u = z + mlp_alpha * (SiLU(z@W_gate) ⊙ (z@W_up)) @ W_down
///   1. logits = input @ W_eff^T (optional token-type weight gate)
///   2. Optional: logits += bias
///
/// hp.atom_insertion_enabled selects token rows or B*(S-1) atom-gap rows.
void forwardLmHead(
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameter_tensors,
    const Tensor& input,
    const Batching::BatchPayload& payload,
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs);

// Selects real token positions within the existing payload, not a new batch.
struct LMHeadReadoutSelection {
    int batch_row = 0;
    int position_begin = 0;
    int position_end = 0; // exclusive
};

// Same row-local readout as forwardLmHead, without sequence pooling. The full
// hidden rectangle is validated against payload; selection defines its slice.
// Capture passes detached tensors and may reuse one gated weight matrix.
// Ordinary forward calls must continue to use forwardLmHead.
void forwardLmHeadReadoutChunk(
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameter_tensors,
    const Tensor& input,
    const Batching::BatchPayload& payload,
    const LMHeadReadoutSelection& selection,
    const Tensor* prepared_effective_weights,
    cudaStream_t stream, cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs);

} // namespace GRIM

#endif // USE_CUDA

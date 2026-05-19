#pragma once
//======================================================//
//  LMHeadGemmDiagnostics.hpp
//  Focused LM-head GEMM equation tracing for Issue #132.
//
//  Ownership boundary:
//    - LMHead/lm_head_GPU.cu owns the production forward composition.
//    - AutogradAttention.cu owns production matmul backward.
//    - This module owns the optional diagnostic logging around that boundary.
//======================================================//

#include "TensorContract_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM::autograd {

void logLmHeadGemmForwardEquation(const Tensor& lm_input,
                                  const Tensor& effective_weights,
                                  const Tensor& logits,
                                  bool center_hidden_states,
                                  bool project_out_pc1,
                                  bool used_centered_weights,
                                  int total_tokens,
                                  int d_model,
                                  int vocab_size,
                                  cudaStream_t stream);

void logLmHeadGemmBackwardEquation(const Tensor& grad_output,
                                   const float* grad_lm_input,
                                   const float* grad_w_eff,
                                   bool lm_input_requires_grad,
                                   bool w_eff_requires_grad,
                                   const char* lm_input_name,
                                   const char* w_eff_name,
                                   int M,
                                   int K,
                                   int N,
                                   bool transpose_b,
                                   cudaStream_t stream);

}  // namespace GRIM::autograd
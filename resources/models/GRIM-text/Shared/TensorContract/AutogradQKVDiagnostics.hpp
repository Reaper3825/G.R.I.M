#pragma once
//======================================================//
//  AutogradQKVDiagnostics.hpp
//  QKV projection / attention tensor diagnostics for TensorContract autograd.
//
//  Ownership boundary:
//    - Encoding_GPU.cu orchestrates layer forward only.
//    - AutogradAttention.cu owns Q/K/V tape operations.
//    - This module owns QKV-specific diagnostic logging used around that tape path.
//======================================================//

#include "TensorContract_GPU.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace GRIM::autograd {

int qkvDebugLevel();

void checkQKVTensorFinite(const char* tag,
                          const Tensor& tensor,
                          cudaStream_t stream);

void logGradFlowTensorStats(const char* tag,
                            const float* data,
                            std::size_t count,
                            cudaStream_t stream,
                            bool force = false);

void logGradFlowBf16TensorStats(const char* tag,
                                const __nv_bfloat16* data,
                                std::size_t count,
                                cudaStream_t stream,
                                bool force = false);

void logQKVProjectionEquation(const Tensor& ln1_out,
                              const Tensor& W_qkv,
                              const Tensor& b_qkv,
                              const Tensor& qkv_out,
                              const Batching::BatchPayload& payload,
                              const GRIM::HyperParameters::EncoderSelfAttentionHP& hp,
                              cudaStream_t stream,
                              int layer_idx);

}  // namespace GRIM::autograd

//======================================================//
//  Numeric Head GPU - Autograd TensorContract API
//  Side-channel regression head for numeric atoms
//======================================================//

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

/**
 * @brief Autograd NumericHead forward
 * 
 * Computes: predictions = encoder_output @ weights + bias
 * Returns a Tensor with grad_fn that will propagate gradients back through
 * the encoder_output tensor during backward pass.
 * 
 * @param encoder_output [total_tokens, d_model] input tensor (requires_grad)
 * @param weights        [d_model, 1] weight tensor (requires_grad, leaf)
 * @param bias           [1] bias tensor (optional, requires_grad, leaf)
 * @param handle         cuBLAS handle
 * @param stream         CUDA stream
 * @return Tensor [total_tokens, 1] predictions with grad_fn attached
 */
Tensor numeric_head_forward(
    Tensor& encoder_output,
    Tensor& weights,
    Tensor* bias,  // nullptr if no bias
    cublasHandle_t handle,
    cudaStream_t stream
);

} // namespace GRIM

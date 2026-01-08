//======================================================//
//  RMSNorm_Kernel_GPU.hpp
//  CUDA launchers for root-mean-square normalization
//======================================================//

#pragma once

#include <cuda_runtime_api.h>

namespace GRIM {

void launchRMSNorm(const float* input,
				      float* output,
				      const float* gamma,
				      int batch_size,
				      int hidden_dim,
				      float eps,
				      cudaStream_t stream);

// Forward wrapper for compatibility
void launchRMSNormForward(const float* input, const float* gamma,
                          float* output, int tokens, int hidden_dim,
                          float epsilon, cudaStream_t stream);

// Backward: grad_output -> grad_input, grad_gamma accumulation
void launchRMSNormBackward(const float* input,
                           const float* grad_output,
                           const float* gamma,
                           float* grad_input,
                           float* grad_gamma,
                           int batch_size,
                           int hidden_dim,
                           float eps,
                           cudaStream_t stream);

} // namespace GRIM

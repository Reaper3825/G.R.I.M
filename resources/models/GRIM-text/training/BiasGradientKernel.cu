//======================================================//
//  BiasGradientKernel.cu
//  Bias gradient computation kernel (row-wise sum reduction)
//  
//  Computes grad_bias[j] = sum_over_i(grad_output[i,j])
//  
//  This was previously in BackwardKernels.cu which was deleted
//  during the 3-phase backward removal. Re-implementing here
//  for LM head and numeric head backward passes.
//======================================================//

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

// Kernel: Sum gradients across all tokens (batch dimension) to get bias gradients
// grad_output: [total_tokens, hidden_dim] - gradient w.r.t. layer output
// grad_bias: [hidden_dim] - gradient w.r.t. bias (accumulated sum)
__global__ void biasSumGradientKernel(
    const float* grad_output,
    float* grad_bias,
    int total_tokens,
    int hidden_dim
) {
    // Each thread handles one bias element (one output dimension)
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden_dim) return;
    
    // Sum gradients across all tokens for this output dimension
    float sum = 0.0f;
    for (int i = 0; i < total_tokens; ++i) {
        sum += grad_output[i * hidden_dim + j];
    }
    
    grad_bias[j] = sum;
}

} // anonymous namespace

// Public API - NOT extern "C" to match forward declarations in lm_head_GPU.cu, numeric_head_GPU.cu
void launchBiasSumGradient(
    const float* grad_output,
    float* grad_bias,
    int total_tokens,
    int hidden_dim,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (hidden_dim + threads - 1) / threads;
    
    biasSumGradientKernel<<<blocks, threads, 0, stream>>>(
        grad_output, grad_bias, total_tokens, hidden_dim
    );
}

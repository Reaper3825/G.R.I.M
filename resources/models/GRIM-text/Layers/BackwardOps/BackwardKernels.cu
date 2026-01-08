/**
 * @file BackwardKernels.cu
 * @brief Utility CUDA kernels for backward pass
 *
 * Contains essential kernels extracted from legacy codebase:
 * - launchBiasSumGradient: Sum gradients across batch/sequence for bias
 * - launchAttentionOutput: Compute attention output (scores @ V)
 */

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

//======================================================//
//  Bias Sum Gradient Kernel
//=====================================================//

__global__ void biasSumGradientKernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int total_tokens,
    int hidden_dim
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim_idx >= hidden_dim) return;
    
    float sum = 0.0f;
    for (int token = 0; token < total_tokens; token++) {
        sum += grad_output[token * hidden_dim + dim_idx];
    }
    
    atomicAdd(&grad_bias[dim_idx], sum);
}

void launchBiasSumGradient(
    const float* grad_output,
    float* grad_bias,
    int total_tokens,
    int hidden_dim,
    cudaStream_t stream
) {
    // Rule 20: Crash on invalid params
    if (hidden_dim <= 0 || total_tokens <= 0) {
        throw std::runtime_error("launchBiasSumGradient: invalid dims total_tokens=" + std::to_string(total_tokens) +
                                 " hidden_dim=" + std::to_string(hidden_dim));
    }
    if (!grad_output) {
        throw std::runtime_error("launchBiasSumGradient: grad_output is NULL");
    }
    if (!grad_bias) {
        throw std::runtime_error("launchBiasSumGradient: grad_bias is NULL");
    }
    
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    
    if (blocks > 65535) {
        blocks = 65535;  // CUDA limit - actual overflow would be a much larger bug
    }
    
    biasSumGradientKernel<<<blocks, threads, 0, stream>>>(
        grad_output, grad_bias, total_tokens, hidden_dim
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchBiasSumGradient kernel error: " + std::string(cudaGetErrorString(err)));
    }
}

//======================================================//
//  Attention Output Kernel
//======================================================//

__global__ void attentionOutputKernel(
    const float* __restrict__ scores,  // [batch, heads, seq_len, seq_len]
    const float* __restrict__ V,       // [batch, heads, seq_len, d_head]
    float* __restrict__ output,        // [batch, heads, seq_len, d_head]
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head
) {
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int d_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || 
        seq_idx >= seq_len || d_idx >= d_head) {
        return;
    }
    
    const float* score_row = scores + ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * seq_len;
    float sum = 0.0f;
    
    for (int i = 0; i < seq_len; ++i) {
        const float* v_vec = V + ((batch_idx * num_heads + head_idx) * seq_len + i) * d_head;
        sum += score_row[i] * v_vec[d_idx];
    }
    
    output[((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * d_head + d_idx] = sum;
}

extern "C" void launchAttentionOutput(
    const float* scores,
    const float* V,
    float* output,
    int batch_size,
    int num_heads,
    int seq_len,
    int d_head,
    cudaStream_t stream
) {
    // Rule 20: Fail loud on invalid parameters
    if (!scores) throw std::runtime_error("launchAttentionOutput: scores is NULL");
    if (!V) throw std::runtime_error("launchAttentionOutput: V is NULL");
    if (!output) throw std::runtime_error("launchAttentionOutput: output is NULL");
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || d_head <= 0) {
        throw std::runtime_error("launchAttentionOutput: invalid dimensions batch=" + 
            std::to_string(batch_size) + " heads=" + std::to_string(num_heads) + 
            " seq=" + std::to_string(seq_len) + " d_head=" + std::to_string(d_head));
    }

    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(d_head < 1024 ? d_head : 1024);
    
    attentionOutputKernel<<<grid, block, 0, stream>>>(
        scores, V, output, batch_size, num_heads, seq_len, d_head
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchAttentionOutput kernel error: " + std::string(cudaGetErrorString(err)));
    }
}

//======================================================//
//  Residual Add Kernels
//======================================================//

__global__ void residualAddKernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input[idx] + residual[idx];
    }
}

__global__ void residualAddScaledKernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    float scale,
    int total_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        output[idx] = input[idx] + scale * residual[idx];
    }
}

extern "C" void launchResidualAdd(
    const float* input,
    const float* residual,
    float* output,
    int total_size,
    cudaStream_t stream
) {
    // Rule 20: Fail loud on invalid parameters
    if (!input) throw std::runtime_error("launchResidualAdd: input is NULL");
    if (!residual) throw std::runtime_error("launchResidualAdd: residual is NULL");
    if (!output) throw std::runtime_error("launchResidualAdd: output is NULL");
    if (total_size <= 0) {
        throw std::runtime_error("launchResidualAdd: invalid total_size=" + std::to_string(total_size));
    }

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    residualAddKernel<<<blocks, threads, 0, stream>>>(
        input, residual, output, total_size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchResidualAdd kernel error: " + std::string(cudaGetErrorString(err)));
    }
}

extern "C" void launchResidualAddScaled(
    const float* input,
    const float* residual,
    float* output,
    float scale,
    int total_size,
    cudaStream_t stream
) {
    // Rule 20: Fail loud on invalid parameters
    if (!input) throw std::runtime_error("launchResidualAddScaled: input is NULL");
    if (!residual) throw std::runtime_error("launchResidualAddScaled: residual is NULL");
    if (!output) throw std::runtime_error("launchResidualAddScaled: output is NULL");
    if (total_size <= 0) {
        throw std::runtime_error("launchResidualAddScaled: invalid total_size=" + std::to_string(total_size));
    }

    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;
    
    residualAddScaledKernel<<<blocks, threads, 0, stream>>>(
        input, residual, output, scale, total_size
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("launchResidualAddScaled kernel error: " + std::string(cudaGetErrorString(err)));
    }
}

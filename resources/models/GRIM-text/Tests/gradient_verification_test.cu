/**
 * Gradient Verification Test
 * 
 * Standalone test that runs the EXACT same CUDA kernels as training
 * with FIXED known inputs, comparing GPU output against CPU-computed expected values.
 * 
 * FOCUS: Encoder gradient collapse investigation
 *   - Flash Attention backward (softmax Jacobian is prime suspect)
 *   - FFN backward
 *   - Full encoder layer gradient flow
 * 
 * NO modifications to source kernels - we test them as-is.
 * 
 * Build: cmake --build build --config Release --target gradient_verification_test
 * Run: gradient_verification_test.exe
 */

#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>

// Include the ACTUAL kernel headers (same as training uses)
#include "../Shared/TensorContract/TensorContract_GPU.hpp"  // autograd::rms_norm (production path)
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Layers/FeedForward/Feed_Forward_GPU.hpp"
#include "../Shared/TensorConversion/TensorConversion.hpp"

// Bring GRIM types into scope for cleaner test code
using GRIM::Tensor;
namespace autograd = GRIM::autograd;

//==============================================================================
// Configuration - Small model for tractable manual verification
// Note: Flash Attention is built for HEAD_DIM = 64 only in this project
//==============================================================================
namespace TestConfig {
    constexpr int BATCH_SIZE = 2;
    constexpr int SEQ_LEN = 4;
    constexpr int NUM_HEADS = 2;
    constexpr int HEAD_DIM = 64;  // MUST be 64 for this build
    constexpr int D_MODEL = NUM_HEADS * HEAD_DIM;  // 128
    constexpr int D_FF = D_MODEL * 4;  // 512
    constexpr int TOTAL_TOKENS = BATCH_SIZE * SEQ_LEN;  // 8
    constexpr float EPSILON = 1e-5f;
}

//==============================================================================
// Utility
//==============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

void printTensor(const char* label, const float* d_data, int rows, int cols) {
    std::vector<float> h(rows * cols);
    CUDA_CHECK(cudaMemcpy(h.data(), d_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\n%s [%d x %d]:\n", label, rows, cols);
    for (int i = 0; i < std::min(rows, 8); i++) {
        printf("  row %d: [", i);
        for (int j = 0; j < std::min(cols, 8); j++) {
            printf("%9.5f", h[i * cols + j]);
            if (j < cols - 1 && j < 7) printf(", ");
        }
        if (cols > 8) printf(", ...");
        printf("]\n");
    }
    if (rows > 8) printf("  ...\n");
}

void printTensorCPU(const char* label, const float* data, int rows, int cols) {
    printf("\n%s [%d x %d]:\n", label, rows, cols);
    for (int i = 0; i < std::min(rows, 8); i++) {
        printf("  row %d: [", i);
        for (int j = 0; j < std::min(cols, 8); j++) {
            printf("%9.5f", data[i * cols + j]);
            if (j < cols - 1 && j < 7) printf(", ");
        }
        if (cols > 8) printf(", ...");
        printf("]\n");
    }
    if (rows > 8) printf("  ...\n");
}

float computeRmsGPU(const float* d_data, int size) {
    std::vector<float> h(size);
    CUDA_CHECK(cudaMemcpy(h.data(), d_data, size * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += h[i] * h[i];
    return sqrtf(sum / size);
}

float computeRmsCPU(const float* data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += data[i] * data[i];
    return sqrtf(sum / size);
}

void compareSideBySide(const char* label, const float* gpu_data, const float* cpu_expected, 
                       int rows, int cols, float tol = 1e-4f) {
    std::vector<float> gpu(rows * cols);
    CUDA_CHECK(cudaMemcpy(gpu.data(), gpu_data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\n%s - Side by Side [%d x %d]:\n", label, rows, cols);
    printf("  %12s  %12s  %12s  %s\n", "GPU Actual", "CPU Expected", "Diff", "Status");
    printf("  %s\n", std::string(60, '-').c_str());
    
    int mismatches = 0;
    float max_diff = 0.0f;
    int max_diff_idx = 0;
    
    for (int i = 0; i < std::min(rows * cols, 32); i++) {
        float diff = fabsf(gpu[i] - cpu_expected[i]);
        float rel_diff = (fabsf(cpu_expected[i]) > 1e-6f) ? diff / fabsf(cpu_expected[i]) : diff;
        const char* status = (rel_diff < tol) ? "✓" : "✗ MISMATCH";
        
        if (rel_diff >= tol) mismatches++;
        if (diff > max_diff) { max_diff = diff; max_diff_idx = i; }
        
        printf("  [%3d] %12.6f  %12.6f  %12.6f  %s\n", i, gpu[i], cpu_expected[i], diff, status);
    }
    if (rows * cols > 32) printf("  ... (%d more elements)\n", rows * cols - 32);
    
    float gpu_norm = computeRmsCPU(gpu.data(), rows * cols);
    float cpu_norm = computeRmsCPU(cpu_expected, rows * cols);
    
    printf("\n  Summary:\n");
    printf("    GPU norm:      %12.6f\n", gpu_norm);
    printf("    Expected norm: %12.6f\n", cpu_norm);
    printf("    Norm ratio:    %12.6f (should be ~1.0)\n", gpu_norm / cpu_norm);
    printf("    Max diff:      %12.6f at index %d\n", max_diff, max_diff_idx);
    printf("    Mismatches:    %d / %d\n", mismatches, std::min(rows * cols, 32));
}

//==============================================================================
// CPU Reference Implementations (Mathematical Ground Truth)
//==============================================================================

namespace CPURef {

// RMSNorm forward: y = x * gamma / sqrt(mean(x^2) + eps)
void rmsNormForward(const float* x, const float* gamma, float* y,
                    int batch, int dim, float eps) {
    for (int b = 0; b < batch; b++) {
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum_sq += x[b * dim + d] * x[b * dim + d];
        }
        float rms = sqrtf(sum_sq / dim + eps);
        float inv_rms = 1.0f / rms;
        
        for (int d = 0; d < dim; d++) {
            y[b * dim + d] = x[b * dim + d] * gamma[d] * inv_rms;
        }
    }
}

// RMSNorm backward: grad_x given grad_y
// Formula: dx = (dy * gamma - x * sum(dy * gamma * x) / (dim * rms^2)) / rms
void rmsNormBackward(const float* x, const float* grad_y, const float* gamma,
                     float* grad_x, float* grad_gamma,
                     int batch, int dim, float eps) {
    // Zero grad_gamma accumulator
    for (int d = 0; d < dim; d++) grad_gamma[d] = 0.0f;
    
    for (int b = 0; b < batch; b++) {
        // Compute rms for this row
        float sum_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum_sq += x[b * dim + d] * x[b * dim + d];
        }
        float variance = sum_sq / dim;
        float rms = sqrtf(variance + eps);
        float inv_rms = 1.0f / rms;
        float inv_rms_cubed = inv_rms * inv_rms * inv_rms;
        
        // Compute sum(dy * gamma * x) for this row
        float sum_dy_gamma_x = 0.0f;
        for (int d = 0; d < dim; d++) {
            sum_dy_gamma_x += grad_y[b * dim + d] * gamma[d] * x[b * dim + d];
        }
        
        // Compute grad_x for each element
        // dx_i = (dy_i * gamma_i * inv_rms) - (x_i * sum_dy_gamma_x * inv_rms^3 / dim)
        for (int d = 0; d < dim; d++) {
            float dy = grad_y[b * dim + d];
            float g = gamma[d];
            float xi = x[b * dim + d];
            
            grad_x[b * dim + d] = dy * g * inv_rms - xi * sum_dy_gamma_x * inv_rms_cubed / dim;
            
            // Accumulate grad_gamma
            grad_gamma[d] += dy * xi * inv_rms;
        }
    }
}

} // namespace CPURef

//==============================================================================
// Test: Flash Attention Backward (THE PRIME SUSPECT for gradient collapse)
// 
// Softmax Jacobian: dS = P * (dP - sum(P * dP))
// When P is near one-hot (saturated), dS → 0 regardless of dP magnitude!
//==============================================================================

void testFlashAttentionBackward() {
    using namespace TestConfig;
    printf("\n");
    printf("################################################################\n");
    printf("#  TEST: Flash Attention Backward (ACTUAL KERNEL)             #\n");
    printf("#  PRIME SUSPECT: Softmax gradient vanishing                  #\n");
    printf("################################################################\n");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Sizes for attention
    const int batch_heads = BATCH_SIZE * NUM_HEADS;  // 4
    const int qkv_size = batch_heads * SEQ_LEN * HEAD_DIM;  // 4 * 4 * 64 = 1024
    const size_t q_elems = static_cast<size_t>(qkv_size);
    const size_t lse_elems = static_cast<size_t>(BATCH_SIZE) * NUM_HEADS * SEQ_LEN;
    
    // Allocate GPU memory
    float *d_Q, *d_K, *d_V, *d_output;
    float *d_grad_output, *d_grad_Q, *d_grad_K, *d_grad_V;
    __nv_bfloat16 *d_q_bf16, *d_k_bf16, *d_v_bf16, *d_out_bf16;
    __nv_bfloat16 *d_dout_bf16, *d_dq_bf16, *d_dk_bf16, *d_dv_bf16;
    float* d_softmax_lse;
    void* d_dq_accum;
    void* d_dsoftmax_sum;
    
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_Q, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_K, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_V, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_q_bf16, q_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_k_bf16, q_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_v_bf16, q_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out_bf16, q_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dout_bf16, q_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dq_bf16, q_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dk_bf16, q_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_dv_bf16, q_elems * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_softmax_lse, lse_elems * sizeof(float)));

    const size_t dq_accum_bytes = flash_attn_dq_accum_bytes(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM);
    const size_t dsoftmax_sum_bytes = flash_attn_dsoftmax_sum_bytes(BATCH_SIZE, SEQ_LEN, NUM_HEADS);
    CUDA_CHECK(cudaMalloc(&d_dq_accum, dq_accum_bytes));
    CUDA_CHECK(cudaMalloc(&d_dsoftmax_sum, dsoftmax_sum_bytes));
    
    // Initialize Q, K, V with FIXED values
    std::vector<float> h_Q(qkv_size), h_K(qkv_size), h_V(qkv_size);
    std::vector<float> h_grad_output(qkv_size);
    
    // Q: deterministic pattern  
    for (int i = 0; i < qkv_size; i++) {
        h_Q[i] = 0.1f * ((i % 7) - 3);
    }
    // K: slightly different pattern
    for (int i = 0; i < qkv_size; i++) {
        h_K[i] = 0.1f * ((i % 5) - 2);
    }
    // V: another pattern
    for (int i = 0; i < qkv_size; i++) {
        h_V[i] = 0.1f * ((i % 9) - 4);
    }
    // Grad output: uniform small gradients
    for (int i = 0; i < qkv_size; i++) {
        h_grad_output[i] = 0.05f;
    }
    
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grad_Q, 0, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_K, 0, qkv_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_V, 0, qkv_size * sizeof(float)));
    
    printf("\n===== INPUT DATA =====\n");
    printTensorCPU("h_Q (first head)", h_Q.data(), SEQ_LEN, HEAD_DIM);
    printTensorCPU("h_K (first head)", h_K.data(), SEQ_LEN, HEAD_DIM);
    printTensorCPU("h_V (first head)", h_V.data(), SEQ_LEN, HEAD_DIM);
    
    // ===== Forward pass first (need output for backward) =====
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        d_Q, d_q_bf16, BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        d_K, d_k_bf16, BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, stream);
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        d_V, d_v_bf16, BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, stream);
    const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

    flash_attn_fwd_ex(
        d_q_bf16,
        d_k_bf16,
        d_v_bf16,
        d_out_bf16,
        d_softmax_lse,
        nullptr,
        BATCH_SIZE,
        SEQ_LEN,
        NUM_HEADS,
        NUM_HEADS,
        HEAD_DIM,
        softmax_scale,
        true,
        true,
        0.0f,
        0,
        stream);

    TensorConversion::convert_BSHD_bf16_to_BHSD(
        d_out_bf16, d_output, BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("\n===== FORWARD OUTPUT =====\n");
    printTensor("Attention Output", d_output, batch_heads * SEQ_LEN, HEAD_DIM);
    
    // ===== BACKWARD PASS (THE CRITICAL TEST) =====
    TensorConversion::convert_BHSD_to_BSHD_bf16(
        d_grad_output, d_dout_bf16, BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, stream);

    flash_attn_bwd_ex(
        d_q_bf16,
        d_k_bf16,
        d_v_bf16,
        d_out_bf16,
        d_dout_bf16,
        d_softmax_lse,
        nullptr,
        d_dq_bf16,
        d_dk_bf16,
        d_dv_bf16,
        d_dq_accum,
        d_dsoftmax_sum,
        BATCH_SIZE,
        SEQ_LEN,
        NUM_HEADS,
        NUM_HEADS,
        HEAD_DIM,
        softmax_scale,
        true,
        true,
        0.0f,
        0,
        stream);

    TensorConversion::convert_BSHD_bf16_to_BHSD(
        d_dq_bf16, d_grad_Q, BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, stream);
    TensorConversion::convert_BSHD_bf16_to_BHSD(
        d_dk_bf16, d_grad_K, BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, stream);
    TensorConversion::convert_BSHD_bf16_to_BHSD(
        d_dv_bf16, d_grad_V, BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("\n===== BACKWARD PASS =====\n");
    printTensor("grad_Q", d_grad_Q, batch_heads * SEQ_LEN, HEAD_DIM);
    printTensor("grad_K", d_grad_K, batch_heads * SEQ_LEN, HEAD_DIM);
    printTensor("grad_V", d_grad_V, batch_heads * SEQ_LEN, HEAD_DIM);
    
    // ===== CRITICAL: Gradient Magnitude Analysis =====
    printf("\n===== GRADIENT MAGNITUDE ANALYSIS (ATTENTION) =====\n");
    float q_norm = computeRmsCPU(h_Q.data(), qkv_size);
    float k_norm = computeRmsCPU(h_K.data(), qkv_size);
    float v_norm = computeRmsCPU(h_V.data(), qkv_size);
    float grad_out_norm = computeRmsCPU(h_grad_output.data(), qkv_size);
    float grad_q_norm = computeRmsGPU(d_grad_Q, qkv_size);
    float grad_k_norm = computeRmsGPU(d_grad_K, qkv_size);
    float grad_v_norm = computeRmsGPU(d_grad_V, qkv_size);
    
    printf("  Input norms:\n");
    printf("    |Q|: %12.6f\n", q_norm);
    printf("    |K|: %12.6f\n", k_norm);
    printf("    |V|: %12.6f\n", v_norm);
    printf("\n  Gradient norms:\n");
    printf("    |grad_output|: %12.6f\n", grad_out_norm);
    printf("    |grad_Q|:      %12.6f\n", grad_q_norm);
    printf("    |grad_K|:      %12.6f\n", grad_k_norm);
    printf("    |grad_V|:      %12.6f\n", grad_v_norm);
    printf("\n  Attenuation ratios (should be ~1.0 for healthy gradients):\n");
    printf("    |grad_Q|/|grad_out|: %12.6f\n", grad_q_norm / grad_out_norm);
    printf("    |grad_K|/|grad_out|: %12.6f\n", grad_k_norm / grad_out_norm);
    printf("    |grad_V|/|grad_out|: %12.6f\n", grad_v_norm / grad_out_norm);
    
    if (grad_q_norm / grad_out_norm < 0.1f || grad_k_norm / grad_out_norm < 0.1f) {
        printf("\n  *** WARNING: Attention gradients severely attenuated! ***\n");
        printf("  This could indicate softmax saturation (near one-hot attention)\n");
    }
    
    // Cleanup
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_output);
    cudaFree(d_grad_output); cudaFree(d_grad_Q); cudaFree(d_grad_K); cudaFree(d_grad_V);
    cudaFree(d_q_bf16); cudaFree(d_k_bf16); cudaFree(d_v_bf16); cudaFree(d_out_bf16);
    cudaFree(d_dout_bf16); cudaFree(d_dq_bf16); cudaFree(d_dk_bf16); cudaFree(d_dv_bf16);
    cudaFree(d_softmax_lse); cudaFree(d_dq_accum); cudaFree(d_dsoftmax_sum);
    cudaStreamDestroy(stream);
}

void testFFNBackward() {
    printf("\n");
    printf("################################################################\n");
    printf("#  TEST: FFN Backward - SKIPPED (FeedForwardLayer::backwardGPU #\n");
    printf("#        was deleted per Rule 20 - training uses               #\n");
    printf("#        BackwardPhase2_Encoder.cu::computeFFNBackward)        #\n");
    printf("################################################################\n");
}

//==============================================================================
// Test: RMSNorm (tests production autograd::rms_norm path)
//==============================================================================

void testRMSNorm() {
    using namespace TestConfig;
    printf("\n");
    printf("################################################################\n");
    printf("#  TEST: RMSNorm Forward + Backward (ACTUAL KERNEL)           #\n");
    printf("################################################################\n");
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // ===== Allocate GPU memory =====
    float *d_input, *d_gamma, *d_output;
    float *d_grad_output, *d_grad_input, *d_grad_gamma;
    
    CUDA_CHECK(cudaMalloc(&d_input, TOTAL_TOKENS * D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, TOTAL_TOKENS * D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_output, TOTAL_TOKENS * D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_input, TOTAL_TOKENS * D_MODEL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gamma, D_MODEL * sizeof(float)));
    
    // ===== Initialize with FIXED known values =====
    std::vector<float> h_input(TOTAL_TOKENS * D_MODEL);
    std::vector<float> h_gamma(D_MODEL);
    std::vector<float> h_grad_output(TOTAL_TOKENS * D_MODEL);
    
    // Input: deterministic pattern
    for (int i = 0; i < TOTAL_TOKENS * D_MODEL; i++) {
        h_input[i] = 0.1f * ((i % 7) - 3);  // Values in [-0.3, 0.3]
    }
    
    // Gamma: all 1.0 for simplicity (can test with other values)
    for (int i = 0; i < D_MODEL; i++) {
        h_gamma[i] = 1.0f;
    }
    
    // Grad output: deterministic pattern
    for (int i = 0; i < TOTAL_TOKENS * D_MODEL; i++) {
        h_grad_output[i] = 0.05f * ((i % 5) - 2);  // Values in [-0.1, 0.1]
    }
    
    // Copy to GPU
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), TOTAL_TOKENS * D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_output, h_grad_output.data(), TOTAL_TOKENS * D_MODEL * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_grad_gamma, 0, D_MODEL * sizeof(float)));
    
    printf("\n===== INPUT DATA =====\n");
    printTensorCPU("h_input", h_input.data(), TOTAL_TOKENS, D_MODEL);
    printTensorCPU("h_gamma", h_gamma.data(), 1, D_MODEL);
    printTensorCPU("h_grad_output (from next layer)", h_grad_output.data(), TOTAL_TOKENS, D_MODEL);
    
    // ===== CPU Reference: Forward =====
    std::vector<float> cpu_output(TOTAL_TOKENS * D_MODEL);
    CPURef::rmsNormForward(h_input.data(), h_gamma.data(), cpu_output.data(),
                           TOTAL_TOKENS, D_MODEL, EPSILON);
    
    // ===== GPU Actual: Forward (ACTUAL KERNEL via autograd) =====
    // Use autograd::rms_norm - the SAME path production training uses
    Tensor input_tensor = Tensor::from_ptr(
        d_input, TensorContract::TensorShape::make_BSM(TOTAL_TOKENS, D_MODEL), false, false);
    input_tensor.stream = stream;
    
    Tensor gamma_tensor = Tensor::from_ptr(
        d_gamma, TensorContract::TensorShape::make_BSM(1, D_MODEL), false, false);
    gamma_tensor.stream = stream;
    
    // Forward pass using production autograd path
    Tensor output_tensor = autograd::rms_norm(input_tensor, gamma_tensor, EPSILON, stream);
    
    // Copy result to our output buffer for comparison
    CUDA_CHECK(cudaMemcpy(d_output, output_tensor.data, TOTAL_TOKENS * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    printf("\n===== FORWARD PASS =====\n");
    compareSideBySide("RMSNorm Forward Output", d_output, cpu_output.data(), TOTAL_TOKENS, D_MODEL);
    
    // ===== CPU Reference: Backward =====
    std::vector<float> cpu_grad_input(TOTAL_TOKENS * D_MODEL);
    std::vector<float> cpu_grad_gamma(D_MODEL, 0.0f);
    CPURef::rmsNormBackward(h_input.data(), h_grad_output.data(), h_gamma.data(),
                            cpu_grad_input.data(), cpu_grad_gamma.data(),
                            TOTAL_TOKENS, D_MODEL, EPSILON);
    
    // ===== GPU Actual: Backward (via autograd) =====
    // Re-run forward with requires_grad=true to build computation graph
    {
        Tensor input_for_bwd = Tensor::from_ptr(
            d_input, TensorContract::TensorShape::make_BSM(TOTAL_TOKENS, D_MODEL), false, true);  // requires_grad=true
        input_for_bwd.stream = stream;
        input_for_bwd.alloc_grad();  // Allocate gradient buffer
        
        Tensor gamma_for_bwd = Tensor::from_ptr(
            d_gamma, TensorContract::TensorShape::make_BSM(1, D_MODEL), false, true);  // requires_grad=true
        gamma_for_bwd.stream = stream;
        gamma_for_bwd.alloc_grad();
        
        // Forward to build graph
        Tensor out_for_bwd = autograd::rms_norm(input_for_bwd, gamma_for_bwd, EPSILON, stream);
        
        // Create upstream gradient tensor
        Tensor grad_upstream = Tensor::from_ptr(
            d_grad_output, TensorContract::TensorShape::make_BSM(TOTAL_TOKENS, D_MODEL), false, false);
        grad_upstream.stream = stream;
        
        // Run backward
        out_for_bwd.backward(&grad_upstream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Copy gradients to comparison buffers
        CUDA_CHECK(cudaMemcpy(d_grad_input, input_for_bwd.grad_data(), TOTAL_TOKENS * D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_grad_gamma, gamma_for_bwd.grad_data(), D_MODEL * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    
    printf("\n===== BACKWARD PASS =====\n");
    compareSideBySide("RMSNorm grad_input", d_grad_input, cpu_grad_input.data(), TOTAL_TOKENS, D_MODEL);
    compareSideBySide("RMSNorm grad_gamma", d_grad_gamma, cpu_grad_gamma.data(), 1, D_MODEL);
    
    // ===== Gradient Magnitude Analysis =====
    printf("\n===== GRADIENT MAGNITUDE ANALYSIS =====\n");
    float input_norm = computeRmsCPU(h_input.data(), TOTAL_TOKENS * D_MODEL);
    float grad_out_norm = computeRmsCPU(h_grad_output.data(), TOTAL_TOKENS * D_MODEL);
    float grad_in_norm_gpu = computeRmsGPU(d_grad_input, TOTAL_TOKENS * D_MODEL);
    float grad_in_norm_cpu = computeRmsCPU(cpu_grad_input.data(), TOTAL_TOKENS * D_MODEL);
    
    printf("  |input|:              %12.6f\n", input_norm);
    printf("  |grad_output|:        %12.6f\n", grad_out_norm);
    printf("  |grad_input| (GPU):   %12.6f\n", grad_in_norm_gpu);
    printf("  |grad_input| (CPU):   %12.6f\n", grad_in_norm_cpu);
    printf("\n");
    printf("  Ratio GPU/CPU:        %12.6f (should be ~1.0)\n", grad_in_norm_gpu / grad_in_norm_cpu);
    printf("  Ratio |grad_in|/|grad_out|: %8.4f\n", grad_in_norm_gpu / grad_out_norm);
    printf("  (This ratio shows how much RMSNorm attenuates/amplifies gradients)\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_grad_gamma);
    cudaStreamDestroy(stream);
}

//==============================================================================
// Test: Fan-in gradient accumulation regression
//
// Topology uses a single non-leaf trunk feeding N
// independent consumers whose outputs are summed into one root. The trunk's
// upstream (a leaf "embedding") must receive the SUM of all N paths.
//
//   base (leaf, requires_grad)
//     └─ trunk = exp(base)            <- shared node, in-degree N
//         ├─ h_0 = mul_scalar(trunk, c_0)
//         ├─ ...
//         └─ h_{N-1} = mul_scalar(trunk, c_{N-1})
//   total = h_0 + h_1 + ... + h_{N-1} ; total.backward(ones)
//
// Truth:  base.grad[i] = (sum_k c_k) * exp(base[i])
// Legacy DFS recursion (first-wins `applied` guard) collapses the trunk to a
// SINGLE consumer, so base.grad is wrong (one c_k term). The worklist engine
// accumulates all N contributions and matches truth.
//==============================================================================

bool testFanInAccumulation() {
    printf("\n");
    printf("################################################################\n");
    printf("#  TEST: Fan-in accumulation regression                       #\n");
    printf("################################################################\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int N = 16;            // trunk element count
    constexpr int NUM_CONSUMERS = 4;
    const float c[NUM_CONSUMERS] = {1.0f, 2.0f, 3.0f, 4.0f};
    float c_sum = 0.0f;
    for (int k = 0; k < NUM_CONSUMERS; ++k) c_sum += c[k];

    std::vector<float> h_base(N);
    for (int i = 0; i < N; ++i) h_base[i] = 0.1f * ((i % 7) - 3);  // [-0.3, 0.3]

    std::vector<float> h_ones(N, 1.0f);

    float* d_base = nullptr;
    float* d_ones = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ones, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_base, h_base.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ones, h_ones.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // CPU ground truth: (sum_k c_k) * exp(base[i])
    std::vector<float> truth(N);
    for (int i = 0; i < N; ++i) truth[i] = c_sum * expf(h_base[i]);

    auto runBackward = [&](bool use_engine) -> std::vector<float> {
        GRIM::setUseEngineBackward(use_engine);

        Tensor base = Tensor::from_ptr(
            d_base, TensorContract::TensorShape::make_BSM(N, 1), false, true);  // leaf, requires_grad
        base.stream = stream;
        base.alloc_grad();   // fresh zeroed grad accumulator for this run

        // Shared trunk + N independent consumers summed into one root.
        Tensor trunk = autograd::exp(base, stream);
        Tensor total = autograd::mul_scalar(trunk, c[0], stream);
        for (int k = 1; k < NUM_CONSUMERS; ++k) {
            Tensor head = autograd::mul_scalar(trunk, c[k], stream);
            total = autograd::add(total, head, stream);
        }

        Tensor seed = Tensor::from_ptr(
            d_ones, TensorContract::TensorShape::make_BSM(N, 1), false, false);
        seed.stream = stream;

        total.backward(&seed);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<float> grad(N);
        CUDA_CHECK(cudaMemcpy(grad.data(), base.grad_data(), N * sizeof(float), cudaMemcpyDeviceToHost));
        return grad;
    };

    auto maxRelErr = [&](const std::vector<float>& a, const std::vector<float>& b) -> float {
        float worst = 0.0f;
        for (int i = 0; i < N; ++i) {
            float diff = fabsf(a[i] - b[i]);
            float denom = fabsf(b[i]) > 1e-6f ? fabsf(b[i]) : 1.0f;
            worst = std::max(worst, diff / denom);
        }
        return worst;
    };

    std::vector<float> legacy_grad = runBackward(false);
    std::vector<float> engine_grad = runBackward(true);

    const float legacy_err = maxRelErr(legacy_grad, truth);
    const float engine_err = maxRelErr(engine_grad, truth);

    printf("\n  sum(c_k) = %.1f, consumers = %d, trunk elems = %d\n", c_sum, NUM_CONSUMERS, N);
    printf("  %-8s %12s %12s %12s\n", "idx", "truth", "engine", "legacy");
    for (int i = 0; i < std::min(N, 8); ++i) {
        printf("  %-8d %12.6f %12.6f %12.6f\n", i, truth[i], engine_grad[i], legacy_grad[i]);
    }
    printf("\n  engine max rel err: %.6e\n", engine_err);
    printf("  legacy max rel err: %.6e\n", legacy_err);

    const float tol = 1e-4f;
    const bool engine_ok = engine_err < tol;
    // The legacy recursion MUST collapse (be wrong) for this to be a real
    // regression guard. If a future refactor makes legacy correct too, this
    // surfaces it rather than silently passing.
    const bool legacy_collapsed = legacy_err > tol;

    bool pass = engine_ok && legacy_collapsed;
    printf("\n  [%s] engine matches truth: %s\n", engine_ok ? "PASS" : "FAIL", engine_ok ? "yes" : "NO");
    printf("  [%s] legacy collapses (demonstrates bug): %s\n",
           legacy_collapsed ? "PASS" : "WARN", legacy_collapsed ? "yes" : "no");
    printf("  >>> Fan-in test: %s <<<\n", pass ? "PASS" : "FAIL");

    GRIM::setUseEngineBackward(true);  // restore default
    cudaFree(d_base);
    cudaFree(d_ones);
    cudaStreamDestroy(stream);
    return pass;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    printf("\n");
    printf("****************************************************************\n");
    printf("*     GRIM-text Gradient Verification Test                     *\n");
    printf("*     Tests ACTUAL kernels with FIXED inputs                   *\n");
    printf("*     Shows GPU vs CPU Expected side-by-side                   *\n");
    printf("****************************************************************\n");
    
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("\nUsing GPU: %s\n", prop.name);
    
    printf("\nTest Configuration:\n");
    printf("  BATCH_SIZE:   %d\n", TestConfig::BATCH_SIZE);
    printf("  SEQ_LEN:      %d\n", TestConfig::SEQ_LEN);
    printf("  D_MODEL:      %d\n", TestConfig::D_MODEL);
    printf("  NUM_HEADS:    %d\n", TestConfig::NUM_HEADS);
    printf("  HEAD_DIM:     %d\n", TestConfig::HEAD_DIM);
    printf("  D_FF:         %d\n", TestConfig::D_FF);
    printf("  TOTAL_TOKENS: %d\n", TestConfig::TOTAL_TOKENS);
    printf("  EPSILON:      %.2e\n", TestConfig::EPSILON);
    
    // Test the components in order of suspicion for gradient collapse
    testFlashAttentionBackward();  // PRIME SUSPECT: softmax Jacobian
    testFFNBackward();             // Second suspect: GELU derivative chain
    testRMSNorm();                 // Baseline (probably fine)

    // Autograd worklist engine: fan-in accumulation must not collapse.
    const bool fanin_ok = testFanInAccumulation();

    printf("\n");
    printf("****************************************************************\n");
    printf("*                    TEST COMPLETE                             *\n");
    printf("****************************************************************\n");

    return fanin_ok ? 0 : 1;
}

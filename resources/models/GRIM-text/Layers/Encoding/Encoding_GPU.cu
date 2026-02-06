//======================================================//
//  Encoding_GPU.cu
//  PRODUCTION-READY Transformer Encoder Layer
//  
//  This implementation:
//    - Uses Flash Attention directly (NOT GPUMultiHeadAttention)
//    - Uses modular QKV projection from Layers/Attention/
//    - Uses RMSNorm (NOT LayerNorm - no beta parameter!)
//    - Validates EVERYTHING and screams on errors
//    - GQA-native from the start
//======================================================//

#include "Encoding_GPU.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"  // EncoderLayerCache definition
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../FeedForward/Feed_Forward_GPU.hpp"
#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorConversion/TensorConversion.hpp"
#include "../../Shared/EquationLogging/EquationLogging.hpp"  // Centralized equation logging (Rule 21)
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>  // Rule 21 diagnostic: std::min_element, std::max_element

//======================================================//
//  Issue #37 DIAGNOSTIC: Hidden State Alignment Tracker
//  Logs how hidden states align with W[277] at each stage
//  DISABLED by default - set kEnableHiddenAlignDiag = true to enable
//======================================================//
namespace {
    constexpr bool kEnableHiddenAlignDiag = false;  // Set true to enable [HiddenAlign] logs
    constexpr bool kEnableEncoderStepLogs = false;  // Set true to enable [EncoderFwd] step logs
    
    // Use centralized EquationLogger for [*_EQUATION] diagnostic logs (Rule 21)
    inline bool isEquationLoggingEnabled() {
        return GRIM::getEquationLogger().isEnabled();
    }
    
    // Shared W[277] reference - set once per forward pass
    static const float* s_w277_ref = nullptr;
    static int s_w277_d_model = 0;
    static float s_w277_norm = 0.0f;
    static int s_layer_diag_count = 0;
    static constexpr int kMaxDiagLogs = 120;  // First 2 batches * 12 layers
    
    void setW277Reference(const float* lm_weights, int vocab_size, int d_model, cudaStream_t stream) {
        if constexpr (!kEnableHiddenAlignDiag) return;
        constexpr int kToken277 = 277;
        if (!lm_weights || kToken277 >= vocab_size) {
            s_w277_ref = nullptr;
            return;
        }
        // Copy W[277] row to static host buffer
        static std::vector<float> h_w277;
        h_w277.resize(d_model);
        s_w277_d_model = d_model;
        
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_w277.data(), lm_weights + static_cast<size_t>(kToken277) * d_model,
                   d_model * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute norm
        double sum_sq = 0.0;
        for (int i = 0; i < d_model; ++i) sum_sq += h_w277[i] * h_w277[i];
        s_w277_norm = static_cast<float>(std::sqrt(sum_sq));
        s_w277_ref = h_w277.data();
    }
    
    void logHiddenStateAlignment(const char* stage, int layer_idx, 
                                  const float* d_hidden, int total_tokens, int d_model,
                                  cudaStream_t stream) {
        if constexpr (!kEnableHiddenAlignDiag) return;
        if (!s_w277_ref || s_w277_d_model != d_model || s_layer_diag_count >= kMaxDiagLogs) return;
        
        cudaStreamSynchronize(stream);
        
        // Sample first 5 tokens
        const int num_sample = std::min(5, total_tokens);
        std::vector<float> h_hidden(static_cast<size_t>(num_sample) * d_model);
        cudaMemcpy(h_hidden.data(), d_hidden, h_hidden.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute alignment for each sampled token
        float total_dot = 0.0f, total_cos = 0.0f, total_h_norm = 0.0f;
        for (int t = 0; t < num_sample; ++t) {
            const float* h = h_hidden.data() + t * d_model;
            double dot = 0.0, h_norm_sq = 0.0;
            for (int i = 0; i < d_model; ++i) {
                dot += h[i] * s_w277_ref[i];
                h_norm_sq += h[i] * h[i];
            }
            float h_norm = static_cast<float>(std::sqrt(h_norm_sq));
            float cos_sim = (h_norm > 1e-8f && s_w277_norm > 1e-8f) 
                           ? static_cast<float>(dot) / (h_norm * s_w277_norm) : 0.0f;
            total_dot += static_cast<float>(dot);
            total_cos += cos_sim;
            total_h_norm += h_norm;
        }
        
        fprintf(stderr, "[HiddenAlign] layer=%d stage=%s tokens=%d | "
                "mean_dot=%.4f mean_cos=%.4f mean_h_norm=%.4f w277_norm=%.4f\n",
                layer_idx, stage, total_tokens,
                total_dot / num_sample, total_cos / num_sample, 
                total_h_norm / num_sample, s_w277_norm);
    }
    
    void resetLayerDiagCount() { s_layer_diag_count = 0; }
    void incrementLayerDiagCount() { ++s_layer_diag_count; }

    // QKV debug logging (GRIM_DEBUG_QKV).
    int qkvDebugLevel() {
        static int level = []() {
            const char* raw = std::getenv("GRIM_DEBUG_QKV");
            return (raw && *raw) ? std::atoi(raw) : 0;
        }();
        return level;
    }

    struct NonFiniteStats {
        int nan_count;
        int inf_count;
        int first_nan_idx;
        int first_inf_idx;
        float first_nan_val;
        float first_inf_val;
    };

    __global__ void scanNonFiniteKernel(const float* data, int count, NonFiniteStats* stats) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count) {
            return;
        }
        const float v = data[idx];
        if (isnan(v)) {
            atomicAdd(&stats->nan_count, 1);
            const int old = atomicCAS(&stats->first_nan_idx, -1, idx);
            if (old == -1) {
                stats->first_nan_val = v;
            }
        } else if (isinf(v)) {
            atomicAdd(&stats->inf_count, 1);
            const int old = atomicCAS(&stats->first_inf_idx, -1, idx);
            if (old == -1) {
                stats->first_inf_val = v;
            }
        }
    }

    void logNonFiniteStats(const char* tag,
                           const float* data,
                           int count,
                           cudaStream_t stream,
                           bool always_log) {
        if (!data || count <= 0) {
            fprintf(stderr, "[QKV_DEBUG] %s invalid (ptr=%p count=%d)\n",
                    tag ? tag : "<null>", static_cast<const void*>(data), count);
            return;
        }

        NonFiniteStats init{};
        init.first_nan_idx = -1;
        init.first_inf_idx = -1;

        NonFiniteStats* d_stats = nullptr;
        cudaError_t err = cudaMalloc(&d_stats, sizeof(NonFiniteStats));
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s cudaMalloc failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            return;
        }

        err = cudaMemcpyAsync(d_stats, &init, sizeof(init), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s cudaMemcpyAsync H2D failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            cudaFree(d_stats);
            return;
        }

        constexpr int kThreads = 256;
        const int blocks = (count + kThreads - 1) / kThreads;
        scanNonFiniteKernel<<<blocks, kThreads, 0, stream>>>(data, count, d_stats);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s scanNonFiniteKernel launch failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            cudaFree(d_stats);
            return;
        }

        NonFiniteStats out{};
        err = cudaMemcpyAsync(&out, d_stats, sizeof(out), cudaMemcpyDeviceToHost, stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s cudaMemcpyAsync D2H failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            cudaFree(d_stats);
            return;
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "[QKV_DEBUG] %s cudaStreamSynchronize failed: %s\n",
                    tag ? tag : "<null>", cudaGetErrorString(err));
            cudaFree(d_stats);
            return;
        }

        cudaFree(d_stats);

        if (!always_log && out.nan_count == 0 && out.inf_count == 0) {
            return;
        }

        fprintf(stderr, "[QKV_DEBUG] %s n=%d nan=%d inf=%d",
                tag ? tag : "<null>", count, out.nan_count, out.inf_count);
        if (out.nan_count > 0) {
            fprintf(stderr, " first_nan_idx=%d first_nan_val=%g",
                    out.first_nan_idx, out.first_nan_val);
        }
        if (out.inf_count > 0) {
            fprintf(stderr, " first_inf_idx=%d first_inf_val=%g",
                    out.first_inf_idx, out.first_inf_val);
        }
        fprintf(stderr, "\n");
    }

    void logTensorNonFinite(const char* tag,
                            const GRIM::Tensor& tensor,
                            cudaStream_t stream,
                            bool always_log) {
        if (!tensor.data) {
            fprintf(stderr, "[QKV_DEBUG] %s invalid (ptr=%p)\n",
                    tag ? tag : "<null>", static_cast<const void*>(tensor.data));
            return;
        }
        const int count = static_cast<int>(tensor.numel());
        logNonFiniteStats(tag, tensor.data, count, stream, always_log);
    }
    
    // ========================================================================
    // ISSUE #77 DIAGNOSTIC: Log ln1_out statistics during FORWARD pass
    // This helps track what values are being cached for backward GEMM
    // ========================================================================
    
    static bool g_issue77_fwd_diag_enabled = false;  // Set true to enable, adds D2H sync overhead
    static int g_issue77_fwd_layer_count = 0;
    static int g_issue77_fwd_batch_count = 0;
    static float g_diag_input_avg_rms = 0.0f;  // [RMSNORM_EQUATION] temp for cross-block communication
    
    // ========================================================================
    // ISSUE #100 FIX: Proper float atomics for min/max (Issue #15 bug class)
    // 
    // Using atomicMin/Max on reinterpret_cast<int*>(float*) FAILS for negative
    // floats because IEEE 754 bit patterns don't sort correctly as integers.
    // Example: -0.8472 as int = -1085276358, -0.2155 as int = -1089811251
    // atomicMin would pick -1089811251 (the LESS negative float), giving wrong result.
    // 
    // FIX: Use CAS loop with proper float comparison.
    // ========================================================================
    __device__ __forceinline__ void atomicMinFloat_local(float* addr, float value) {
        int* addr_as_int = reinterpret_cast<int*>(addr);
        int old = *addr_as_int;
        int expected;
        do {
            expected = old;
            float old_float = __int_as_float(expected);
            if (value >= old_float) return;  // Our value is not smaller, exit
            old = atomicCAS(addr_as_int, expected, __float_as_int(value));
        } while (expected != old);
    }
    
    __device__ __forceinline__ void atomicMaxFloat_local(float* addr, float value) {
        int* addr_as_int = reinterpret_cast<int*>(addr);
        int old = *addr_as_int;
        int expected;
        do {
            expected = old;
            float old_float = __int_as_float(expected);
            if (value <= old_float) return;  // Our value is not larger, exit
            old = atomicCAS(addr_as_int, expected, __float_as_int(value));
        } while (expected != old);
    }
    
    // Kernel to compute min/max/sum/sum_sq for ln1_out diagnostics  
    __global__ void diagLn1OutKernel(
        const float* __restrict__ data,
        int count,
        float* __restrict__ out_min,
        float* __restrict__ out_max,
        float* __restrict__ out_sum,
        float* __restrict__ out_sum_sq
    ) {
        __shared__ float s_min, s_max, s_sum, s_sum_sq;
        
        if (threadIdx.x == 0) {
            s_min = FLT_MAX;
            s_max = -FLT_MAX;
            s_sum = 0.0f;
            s_sum_sq = 0.0f;
        }
        __syncthreads();
        
        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
            float val = data[i];
            if (!isnan(val) && !isinf(val)) {
                local_min = fminf(local_min, val);
                local_max = fmaxf(local_max, val);
                local_sum += val;
                local_sum_sq += val * val;
            }
        }
        
        // Reduce within block - ISSUE #100 FIX: Use proper float atomics
        atomicMinFloat_local(&s_min, local_min);
        atomicMaxFloat_local(&s_max, local_max);
        atomicAdd(&s_sum, local_sum);
        atomicAdd(&s_sum_sq, local_sum_sq);
        __syncthreads();
        
        if (threadIdx.x == 0) {
            atomicMinFloat_local(out_min, s_min);
            atomicMaxFloat_local(out_max, s_max);
            atomicAdd(out_sum, s_sum);
            atomicAdd(out_sum_sq, s_sum_sq);
        }
    }
    
    void logLn1OutStats(int layer_idx, const float* data, int total_tokens, int d_model, cudaStream_t stream) {
        if (!g_issue77_fwd_diag_enabled) return;
        // Only log first 24 calls (2 batches * 12 layers)
        if (g_issue77_fwd_layer_count >= 24) return;
        
        g_issue77_fwd_layer_count++;
        
        const int count = total_tokens * d_model;
        
        float h_min = FLT_MAX, h_max = -FLT_MAX, h_sum = 0.0f, h_sum_sq = 0.0f;
        float* d_min, *d_max, *d_sum, *d_sum_sq;
        cudaMalloc(&d_min, sizeof(float));
        cudaMalloc(&d_max, sizeof(float));
        cudaMalloc(&d_sum, sizeof(float));
        cudaMalloc(&d_sum_sq, sizeof(float));
        
        float init_min = FLT_MAX, init_max = -FLT_MAX, init_zero = 0.0f;
        cudaMemcpyAsync(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sum, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sum_sq, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
        
        int threads = 256;
        int blocks = std::min((count + threads - 1) / threads, 256);
        diagLn1OutKernel<<<blocks, threads, 0, stream>>>(data, count, d_min, d_max, d_sum, d_sum_sq);
        
        cudaStreamSynchronize(stream);
        cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);
        
        float mean = h_sum / count;
        float variance = h_sum_sq / count - mean * mean;
        float stddev = sqrtf(fmaxf(variance, 0.0f));
        
        fprintf(stderr, "[Issue77-FWD-ln1_out] layer=%d batch=%d tokens=%d d_model=%d: "
                "min=%.6f max=%.6f mean=%.6f std=%.6f\n",
                layer_idx, g_issue77_fwd_batch_count, total_tokens, d_model,
                h_min, h_max, mean, stddev);
        
        // Detect first layer of new batch
        if (layer_idx == 0 && g_issue77_fwd_layer_count > 1) {
            g_issue77_fwd_batch_count++;
        }
        
        cudaFree(d_min);
        cudaFree(d_max);
        cudaFree(d_sum);
        cudaFree(d_sum_sq);
    }
    
    void resetIssue77FwdDiag() {
        g_issue77_fwd_layer_count = 0;
        g_issue77_fwd_batch_count = 0;
    }

    // ISSUE #91 DIAGNOSTIC: Log RMSNorm INPUT statistics (before normalization)
    // This tells us what goes INTO RMSNorm, helping diagnose why Layer 0 output has std=0.747
    void logRmsNormInputStats(int layer_idx, const float* data, int total_tokens, int d_model, cudaStream_t stream) {
        if (!g_issue77_fwd_diag_enabled) return;
        // Only log first 24 calls (2 batches * 12 layers)
        if (g_issue77_fwd_layer_count >= 24) return;
        
        const int count = total_tokens * d_model;
        
        float h_min = FLT_MAX, h_max = -FLT_MAX, h_sum = 0.0f, h_sum_sq = 0.0f;
        float* d_min, *d_max, *d_sum, *d_sum_sq;
        cudaMalloc(&d_min, sizeof(float));
        cudaMalloc(&d_max, sizeof(float));
        cudaMalloc(&d_sum, sizeof(float));
        cudaMalloc(&d_sum_sq, sizeof(float));
        
        float init_min = FLT_MAX, init_max = -FLT_MAX, init_zero = 0.0f;
        cudaMemcpyAsync(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sum, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sum_sq, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
        
        int threads = 256;
        int blocks = std::min((count + threads - 1) / threads, 256);
        diagLn1OutKernel<<<blocks, threads, 0, stream>>>(data, count, d_min, d_max, d_sum, d_sum_sq);
        
        cudaStreamSynchronize(stream);
        cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);
        
        float mean = h_sum / count;
        float variance = h_sum_sq / count - mean * mean;
        float stddev = sqrtf(fmaxf(variance, 0.0f));
        float rms = sqrtf(h_sum_sq / count);  // RMS = sqrt(mean(x^2))
        
        fprintf(stderr, "[Issue91-FWD-rms_input] layer=%d batch=%d tokens=%d d_model=%d: "
                "min=%.10f max=%.10f mean=%.10f std=%.10f rms=%.10f\n",
                layer_idx, g_issue77_fwd_batch_count, total_tokens, d_model,
                h_min, h_max, mean, stddev, rms);
        
        cudaFree(d_min);
        cudaFree(d_max);
        cudaFree(d_sum);
        cudaFree(d_sum_sq);
    }

    // ========================================================================
    // ISSUE #93 DIAGNOSTIC: DebugQKVExpectations
    // Computes EXPECTED Q/K/V output statistics using known GEMM math,
    // then compares with ACTUAL output to identify where the explosion happens.
    // 
    // GEMM math for output = input @ W^T:
    //   - Each output element = sum over K of (input[i] * weight[i])
    //   - For uncorrelated inputs: output_rms ≈ input_rms * weight_rms * sqrt(K)
    //   - Where K = d_model = 768
    //
    // This is guarded by g_debug_qkv_expectations_enabled and runs first 2 batches
    // ========================================================================
    
    static bool g_debug_qkv_expectations_enabled = false;  // Set true to enable, VERY expensive D2H copies
    static int g_debug_qkv_call_count = 0;
    static constexpr int kMaxDebugQKVCalls = 24;  // 2 batches * 12 layers
    
    // Kernel to compute statistics: min, max, sum, sum_sq
    __global__ void computeStatsKernel(
        const float* __restrict__ data,
        int count,
        float* __restrict__ out_min,
        float* __restrict__ out_max,
        float* __restrict__ out_sum,
        float* __restrict__ out_sum_sq
    ) {
        __shared__ float s_min, s_max, s_sum, s_sum_sq;
        
        if (threadIdx.x == 0) {
            s_min = FLT_MAX;
            s_max = -FLT_MAX;
            s_sum = 0.0f;
            s_sum_sq = 0.0f;
        }
        __syncthreads();
        
        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
            float val = data[i];
            if (!isnan(val) && !isinf(val)) {
                local_min = fminf(local_min, val);
                local_max = fmaxf(local_max, val);
                local_sum += val;
                local_sum_sq += val * val;
            }
        }
        
        atomicMin(reinterpret_cast<int*>(&s_min), __float_as_int(local_min));
        atomicMax(reinterpret_cast<int*>(&s_max), __float_as_int(local_max));
        atomicAdd(&s_sum, local_sum);
        atomicAdd(&s_sum_sq, local_sum_sq);
        __syncthreads();
        
        if (threadIdx.x == 0) {
            atomicMin(reinterpret_cast<int*>(out_min), __float_as_int(s_min));
            atomicMax(reinterpret_cast<int*>(out_max), __float_as_int(s_max));
            atomicAdd(out_sum, s_sum);
            atomicAdd(out_sum_sq, s_sum_sq);
        }
    }
    
    struct TensorStats {
        float min_val;
        float max_val;
        float mean;
        float rms;
        float stddev;
    };
    
    TensorStats computeTensorStats(const float* data, int count, cudaStream_t stream) {
        TensorStats stats{};
        if (!data || count <= 0) return stats;
        
        float* d_min, *d_max, *d_sum, *d_sum_sq;
        cudaMalloc(&d_min, sizeof(float));
        cudaMalloc(&d_max, sizeof(float));
        cudaMalloc(&d_sum, sizeof(float));
        cudaMalloc(&d_sum_sq, sizeof(float));
        
        float init_min = FLT_MAX, init_max = -FLT_MAX, init_zero = 0.0f;
        cudaMemcpyAsync(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sum, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sum_sq, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
        
        int threads = 256;
        int blocks = std::min((count + threads - 1) / threads, 256);
        computeStatsKernel<<<blocks, threads, 0, stream>>>(data, count, d_min, d_max, d_sum, d_sum_sq);
        
        float h_min, h_max, h_sum, h_sum_sq;
        cudaStreamSynchronize(stream);
        cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_min);
        cudaFree(d_max);
        cudaFree(d_sum);
        cudaFree(d_sum_sq);
        
        stats.min_val = h_min;
        stats.max_val = h_max;
        stats.mean = h_sum / count;
        stats.rms = sqrtf(h_sum_sq / count);
        float variance = h_sum_sq / count - stats.mean * stats.mean;
        stats.stddev = sqrtf(fmaxf(variance, 0.0f));
        
        return stats;
    }
    
    // ========================================================================
    // ISSUE #106 DIAGNOSTIC: Input-Weight Directional Alignment Analysis
    // Rule 21 Equation-Based Diagnostic for QKV Magnitude Explosion
    // ========================================================================
    // 
    // EQUATION: Y[i,j] = ||x_i|| × ||w_j|| × cos(θ_ij)
    // 
    // The QKV output magnitude depends on THREE factors:
    // 1. Input row norms ||x_i|| - magnitude of input vectors
    // 2. Weight row norms ||w_j|| - magnitude of weight rows
    // 3. cos(θ_ij) - DIRECTIONAL ALIGNMENT between input row i and weight row j
    //
    // For RANDOM vectors: E[cos(θ)] = 0, Var[cos(θ)] = 1/K (where K = d_model)
    // For STRUCTURED vectors: E[cos(θ)] ≠ 0 if there's systematic alignment!
    //
    // Layer 0 embeddings may have STRUCTURE that systematically aligns with 
    // W_qkv rows (both learned from same vocabulary/position semantics).
    // This would cause E[cos(θ)] >> 0 and explain the 12x explosion.
    // ========================================================================
    
    __global__ void inputWeightCosineSampleKernel(
        const float* __restrict__ input,     // [tokens, d_model] - ln1_out
        const float* __restrict__ weight,    // [qkv_dim, d_model] - W_qkv
        float* __restrict__ cosine_stats,    // [6] = {sum, sum_sq, min, max, count, sum_abs}
        int d_model,
        int sample_input_rows,               // How many input rows to sample
        int sample_weight_rows               // How many weight rows to sample
    ) {
        // Compute pairwise cosine similarity between sampled input rows and weight rows
        // Each block handles one (input_row, weight_row) pair
        const int input_idx = blockIdx.x;
        const int weight_idx = blockIdx.y;
        
        if (input_idx >= sample_input_rows || weight_idx >= sample_weight_rows) return;
        
        // Shared memory for parallel dot product reduction
        __shared__ float s_dot[256];
        __shared__ float s_input_norm_sq[256];
        __shared__ float s_weight_norm_sq[256];
        
        float local_dot = 0.0f, local_in_sq = 0.0f, local_w_sq = 0.0f;
        
        for (int k = threadIdx.x; k < d_model; k += blockDim.x) {
            float a = input[input_idx * d_model + k];
            float b = weight[weight_idx * d_model + k];
            local_dot += a * b;
            local_in_sq += a * a;
            local_w_sq += b * b;
        }
        
        s_dot[threadIdx.x] = local_dot;
        s_input_norm_sq[threadIdx.x] = local_in_sq;
        s_weight_norm_sq[threadIdx.x] = local_w_sq;
        __syncthreads();
        
        // Reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_dot[threadIdx.x] += s_dot[threadIdx.x + stride];
                s_input_norm_sq[threadIdx.x] += s_input_norm_sq[threadIdx.x + stride];
                s_weight_norm_sq[threadIdx.x] += s_weight_norm_sq[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            float dot = s_dot[0];
            float input_norm = sqrtf(s_input_norm_sq[0]);
            float weight_norm = sqrtf(s_weight_norm_sq[0]);
            float cosine = dot / (input_norm * weight_norm + 1e-8f);
            
            // Accumulate statistics atomically
            atomicAdd(&cosine_stats[0], cosine);           // sum
            atomicAdd(&cosine_stats[1], cosine * cosine);  // sum_sq  
            // Use CAS for min/max (proper float atomics)
            // For simplicity, just use atomicAdd for now and compute min/max on host
            atomicAdd(&cosine_stats[4], 1.0f);             // count
            atomicAdd(&cosine_stats[5], fabsf(cosine));    // sum_abs (for mean |cos|)
        }
    }
    
    // Manual GEMM to compute expected output for a small sample of rows
    // This bypasses cuBLAS entirely to verify the math
    __global__ void manualGemmSampleKernel(
        const float* __restrict__ input,    // [tokens, d_model] row-major
        const float* __restrict__ weight,   // [qkv_dim, d_model] row-major (we transpose)
        float* __restrict__ output,         // [sample_rows, qkv_dim] - manual result
        int d_model,                         // K dimension (768)
        int qkv_dim,                         // N dimension (1280)
        int sample_rows                      // How many rows to compute (e.g., 4)
    ) {
        // Each block handles one output element
        // Grid: [sample_rows, qkv_dim]
        const int row = blockIdx.x;   // Which input token
        const int col = blockIdx.y;   // Which qkv_dim output
        
        if (row >= sample_rows || col >= qkv_dim) return;
        
        // Compute dot product: input[row, :] @ weight[col, :] (W transposed)
        // = sum over k of input[row * d_model + k] * weight[col * d_model + k]
        
        __shared__ float s_partial[256];
        float local_sum = 0.0f;
        
        for (int k = threadIdx.x; k < d_model; k += blockDim.x) {
            float a = input[row * d_model + k];
            float b = weight[col * d_model + k];  // W is [qkv_dim, d_model]
            local_sum += a * b;
        }
        
        s_partial[threadIdx.x] = local_sum;
        __syncthreads();
        
        // Reduce within block
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            output[row * qkv_dim + col] = s_partial[0];
        }
    }
    
    void DebugQKVExpectations(
        int layer_idx,
        const float* ln1_out,        // [tokens, d_model] - Input to QKV projection
        const float* W_qkv,          // [qkv_dim, d_model] - Weight matrix
        const float* qkv_out,        // [tokens, qkv_dim] - ACTUAL output (from cuBLAS)
        const float* b_qkv,          // [qkv_dim] - Bias (can be nullptr)
        int total_tokens,
        int d_model,                 // 768
        int qkv_dim,                 // 1280 (d_model + 2*kv_dim)
        int kv_dim,                  // 256 (for Q/K/V split)
        cudaStream_t stream
    ) {
        if (!g_debug_qkv_expectations_enabled) return;
        if (g_debug_qkv_call_count >= kMaxDebugQKVCalls) return;
        g_debug_qkv_call_count++;
        
        // 1. Compute input statistics
        TensorStats input_stats = computeTensorStats(ln1_out, total_tokens * d_model, stream);
        
        // 2. Compute weight statistics  
        TensorStats weight_stats = computeTensorStats(W_qkv, qkv_dim * d_model, stream);
        
        // 3. Compute ACTUAL output statistics (from cuBLAS matmul) - THIS IS PRE-BIAS
        TensorStats actual_stats = computeTensorStats(qkv_out, total_tokens * qkv_dim, stream);
        
        // 4. Calculate EXPECTED output statistics using GEMM math
        // 
        // UPDATED FORMULA (Issue #91):
        // The old formula assumed uncorrelated, zero-mean, symmetric inputs:
        //   expected_rms = r_in * r_w * sqrt(K)  ≈ 0.866
        //
        // This is WRONG for transformer activations because:
        // 1. RMSNorm output has POSITIVE SKEW (max >> |min|)
        // 2. Input rows are normalized to fixed norm, creating element correlation
        // 3. The skew causes dot products to be systematically positive
        //
        // ========================================================================
        // WHY LAYER 0/1 ARE DIFFERENT FROM LAYERS 2-11 (Issue #93):
        // ========================================================================
        // Layer 0 (and Layer 1 in batched mode) receives fundamentally different input:
        //
        // LAYER 0 INPUT = token_embedding + position_embedding (then RMSNorm)
        // - Token embeddings: Xavier init, scaled by sqrt(d_model) (Issue #92)
        // - Position embeddings: Xavier init, scaled by sqrt(d_model) (Issue #92)
        // - Both have STRUCTURED patterns (vocabulary semantics, position encoding)
        // - When added together, they create COHERENT row vectors that align similarly
        // - Result: LOW alignment ratio (~0.03) but LARGE dot products with W_qkv
        //
        // LAYERS 2-11 INPUT = previous_layer_output (already encoder-processed)
        // - Residual connections mix information across all positions/features
        // - Multiple attention + FFN passes DECORRELATE the row vectors  
        // - Result: HIGHER alignment ratio (~0.15-0.21) but smaller QKV magnitudes
        //
        // THE PARADOX: Lower alignment SHOULD mean smaller output (our formula says so)
        // But Layer 0/1 has LOWEST alignment yet HIGHEST QKV magnitude!
        //
        // EXPLANATION: The alignment_ratio metric measures |mean(rows)|² / mean(|rows|²)
        // This captures whether rows POINT IN SAME DIRECTION.
        // But embedding rows can have STRUCTURE that creates large W_qkv dot products
        // even when they don't point in the same direction (low alignment_ratio).
        // Specifically: embedding weights and W_qkv weights may share learned structure
        // that causes E[cos(θ_embedding_row, W_qkv_row)] >> E[cos(θ_encoder_row, W_qkv_row)]
        // ========================================================================
        //
        // NEW APPROACH: Compute expected based on input/weight row norms and 
        // empirical alignment. For Y = X @ W^T:
        //   Y[i,j] = ||x_i|| * ||w_j|| * cos(θ_ij)
        //   
        // For random orthogonal vectors: E[cos²(θ)] = 1/K
        // But for skewed/aligned data: observed ratio is 3-12x higher
        //
        // We compute BOTH the theoretical (uncorrelated) and row-norm-based expectations
        // to diagnose where the discrepancy comes from.
        
        // Theoretical (old formula, for reference):
        float theoretical_rms = input_stats.rms * weight_stats.rms * sqrtf(static_cast<float>(d_model));
        
        // Will compute row-norm-based expected after we have row statistics below
        float expected_rms = theoretical_rms;  // Placeholder, updated later
        
        // Expected max assuming Gaussian: ~4 sigma = 4 * rms (for large count)
        float expected_max = expected_rms * 4.0f;
        
        // 5. Manual GEMM verification on small sample (4 rows, all qkv_dim cols)
        const int sample_rows = 4;
        float* d_manual_out = nullptr;
        cudaMalloc(&d_manual_out, sample_rows * qkv_dim * sizeof(float));
        cudaMemsetAsync(d_manual_out, 0, sample_rows * qkv_dim * sizeof(float), stream);
        
        dim3 grid(sample_rows, qkv_dim);
        dim3 block(256);
        manualGemmSampleKernel<<<grid, block, 0, stream>>>(
            ln1_out, W_qkv, d_manual_out, d_model, qkv_dim, sample_rows);
        
        // Get manual output stats
        TensorStats manual_stats = computeTensorStats(d_manual_out, sample_rows * qkv_dim, stream);
        
        // 6. Sample actual first few values to compare with manual
        std::vector<float> h_actual_sample(sample_rows * qkv_dim);
        std::vector<float> h_manual_sample(sample_rows * qkv_dim);
        cudaMemcpyAsync(h_actual_sample.data(), qkv_out, h_actual_sample.size() * sizeof(float), 
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_manual_sample.data(), d_manual_out, h_manual_sample.size() * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        cudaFree(d_manual_out);
        
        // ========================================================================
        // ADDITIONAL DIAGNOSTICS - What was missing before
        // ========================================================================
        
        // 7a. BIAS statistics - could be the source of explosion!
        TensorStats bias_stats{};
        if (b_qkv) {
            bias_stats = computeTensorStats(b_qkv, qkv_dim, stream);
        }
        
        // 7b. Q/K/V SPLIT statistics - check if Q, K, V have different magnitudes
        // qkv_out layout: [tokens, qkv_dim] where qkv_dim = d_model + 2*kv_dim
        // Q: columns [0, d_model)
        // K: columns [d_model, d_model + kv_dim)  
        // V: columns [d_model + kv_dim, qkv_dim)
        
        // We'll sample first 100 tokens for Q/K/V split analysis
        const int split_sample_tokens = std::min(100, total_tokens);
        std::vector<float> h_qkv_sample(static_cast<size_t>(split_sample_tokens) * qkv_dim);
        cudaMemcpy(h_qkv_sample.data(), qkv_out, h_qkv_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Compute separate stats for Q, K, V
        double q_sum_sq = 0, k_sum_sq = 0, v_sum_sq = 0;
        float q_min = FLT_MAX, q_max = -FLT_MAX;
        float k_min = FLT_MAX, k_max = -FLT_MAX;
        float v_min = FLT_MAX, v_max = -FLT_MAX;
        
        for (int t = 0; t < split_sample_tokens; ++t) {
            const float* row = h_qkv_sample.data() + t * qkv_dim;
            // Q portion: [0, d_model)
            for (int i = 0; i < d_model; ++i) {
                float v = row[i];
                q_sum_sq += v * v;
                q_min = fminf(q_min, v);
                q_max = fmaxf(q_max, v);
            }
            // K portion: [d_model, d_model + kv_dim)
            for (int i = d_model; i < d_model + kv_dim; ++i) {
                float v = row[i];
                k_sum_sq += v * v;
                k_min = fminf(k_min, v);
                k_max = fmaxf(k_max, v);
            }
            // V portion: [d_model + kv_dim, qkv_dim)
            for (int i = d_model + kv_dim; i < qkv_dim; ++i) {
                float v = row[i];
                v_sum_sq += v * v;
                v_min = fminf(v_min, v);
                v_max = fmaxf(v_max, v);
            }
        }
        
        float q_rms = sqrtf(static_cast<float>(q_sum_sq / (split_sample_tokens * d_model)));
        float k_rms = sqrtf(static_cast<float>(k_sum_sq / (split_sample_tokens * kv_dim)));
        float v_rms = sqrtf(static_cast<float>(v_sum_sq / (split_sample_tokens * kv_dim)));
        
        // 7c. PER-ROW RMS variance - detect outlier tokens
        std::vector<float> row_rms_values(split_sample_tokens);
        float row_rms_min = FLT_MAX, row_rms_max = -FLT_MAX;
        double row_rms_sum = 0, row_rms_sum_sq = 0;
        
        for (int t = 0; t < split_sample_tokens; ++t) {
            const float* row = h_qkv_sample.data() + t * qkv_dim;
            double row_sum_sq = 0;
            for (int i = 0; i < qkv_dim; ++i) {
                row_sum_sq += row[i] * row[i];
            }
            float rms = sqrtf(static_cast<float>(row_sum_sq / qkv_dim));
            row_rms_values[t] = rms;
            row_rms_min = fminf(row_rms_min, rms);
            row_rms_max = fmaxf(row_rms_max, rms);
            row_rms_sum += rms;
            row_rms_sum_sq += rms * rms;
        }
        float row_rms_mean = static_cast<float>(row_rms_sum / split_sample_tokens);
        float row_rms_std = sqrtf(static_cast<float>(row_rms_sum_sq / split_sample_tokens - row_rms_mean * row_rms_mean));
        
        // 7d. WEIGHT ROW NORMS - check if some W_qkv rows have larger norms
        std::vector<float> h_weight_sample(static_cast<size_t>(qkv_dim) * d_model);
        cudaMemcpy(h_weight_sample.data(), W_qkv, h_weight_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        float w_row_norm_min = FLT_MAX, w_row_norm_max = -FLT_MAX;
        double w_row_norm_sum = 0;
        
        for (int r = 0; r < qkv_dim; ++r) {
            const float* w_row = h_weight_sample.data() + r * d_model;
            double norm_sq = 0;
            for (int c = 0; c < d_model; ++c) {
                norm_sq += w_row[c] * w_row[c];
            }
            float norm = sqrtf(static_cast<float>(norm_sq));
            w_row_norm_min = fminf(w_row_norm_min, norm);
            w_row_norm_max = fmaxf(w_row_norm_max, norm);
            w_row_norm_sum += norm;
        }
        float w_row_norm_mean = static_cast<float>(w_row_norm_sum / qkv_dim);
        
        // 7e. INPUT ROW NORMS - check if input rows have varying norms
        std::vector<float> h_input_sample(static_cast<size_t>(split_sample_tokens) * d_model);
        cudaMemcpy(h_input_sample.data(), ln1_out, h_input_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        float in_row_norm_min = FLT_MAX, in_row_norm_max = -FLT_MAX;
        double in_row_norm_sum = 0;
        
        for (int t = 0; t < split_sample_tokens; ++t) {
            const float* in_row = h_input_sample.data() + t * d_model;
            double norm_sq = 0;
            for (int c = 0; c < d_model; ++c) {
                norm_sq += in_row[c] * in_row[c];
            }
            float norm = sqrtf(static_cast<float>(norm_sq));
            in_row_norm_min = fminf(in_row_norm_min, norm);
            in_row_norm_max = fmaxf(in_row_norm_max, norm);
            in_row_norm_sum += norm;
        }
        float in_row_norm_mean = static_cast<float>(in_row_norm_sum / split_sample_tokens);
        
        // 7f. INPUT ROW CORRELATION - THE KEY DIAGNOSTIC!
        // If input rows are all pointing in similar directions, GEMM output explodes
        // Check pairwise cosine similarity between first N rows
        const int corr_sample = std::min(10, split_sample_tokens);
        double corr_sum = 0;
        int corr_count = 0;
        float corr_min = FLT_MAX, corr_max = -FLT_MAX;
        
        for (int i = 0; i < corr_sample; ++i) {
            for (int j = i + 1; j < corr_sample; ++j) {
                const float* row_i = h_input_sample.data() + i * d_model;
                const float* row_j = h_input_sample.data() + j * d_model;
                double dot = 0, norm_i_sq = 0, norm_j_sq = 0;
                for (int k = 0; k < d_model; ++k) {
                    dot += row_i[k] * row_j[k];
                    norm_i_sq += row_i[k] * row_i[k];
                    norm_j_sq += row_j[k] * row_j[k];
                }
                float cosine = static_cast<float>(dot / (sqrt(norm_i_sq) * sqrt(norm_j_sq) + 1e-8));
                corr_sum += cosine;
                corr_count++;
                corr_min = fminf(corr_min, cosine);
                corr_max = fmaxf(corr_max, cosine);
            }
        }
        float corr_mean = (corr_count > 0) ? static_cast<float>(corr_sum / corr_count) : 0.0f;
        
        // 7g. INPUT COLUMN VARIANCE - if low, rows are similar
        // Column mean and variance across tokens
        std::vector<double> col_sums(d_model, 0.0);
        std::vector<double> col_sum_sqs(d_model, 0.0);
        for (int t = 0; t < split_sample_tokens; ++t) {
            const float* row = h_input_sample.data() + t * d_model;
            for (int c = 0; c < d_model; ++c) {
                col_sums[c] += row[c];
                col_sum_sqs[c] += row[c] * row[c];
            }
        }
        float col_var_min = FLT_MAX, col_var_max = -FLT_MAX;
        double col_var_sum = 0;
        for (int c = 0; c < d_model; ++c) {
            float mean = static_cast<float>(col_sums[c] / split_sample_tokens);
            float var = static_cast<float>(col_sum_sqs[c] / split_sample_tokens - mean * mean);
            col_var_min = fminf(col_var_min, var);
            col_var_max = fmaxf(col_var_max, var);
            col_var_sum += var;
        }
        float col_var_mean = static_cast<float>(col_var_sum / d_model);
        
        // ========================================================================
        // Issue #106: INPUT-WEIGHT COSINE SIMILARITY ANALYSIS (Rule 21)
        // This is the KEY diagnostic for understanding why Layer 0 explodes!
        // ========================================================================
        // We compute: cos(θ_ij) = (x_i · w_j) / (||x_i|| × ||w_j||)
        // For RANDOM vectors: E[cos(θ)] = 0, Var[cos] = 1/K, E[|cos|] = sqrt(2/(πK))
        // If Layer 0 embeddings share STRUCTURE with W_qkv (both learned from same
        // vocabulary/semantic space), we'll see E[cos²] >> 1/K (systematic alignment)
        // ========================================================================
        
        const int cos_sample_input = std::min(32, split_sample_tokens);
        const int cos_sample_weight = std::min(64, qkv_dim);
        
        // Allocate GPU buffer for cosine statistics
        float* d_cos_stats = nullptr;
        cudaMalloc(&d_cos_stats, 6 * sizeof(float));
        cudaMemsetAsync(d_cos_stats, 0, 6 * sizeof(float), stream);
        
        // Launch kernel to compute input-weight cosine similarities
        dim3 cos_grid(cos_sample_input, cos_sample_weight);
        inputWeightCosineSampleKernel<<<cos_grid, 256, 0, stream>>>(
            ln1_out, W_qkv, d_cos_stats, d_model, cos_sample_input, cos_sample_weight);
        
        // Get statistics back to host
        float h_cos_stats[6];
        cudaMemcpy(h_cos_stats, d_cos_stats, 6 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_cos_stats);
        
        float cos_count = h_cos_stats[4];
        float cos_mean = (cos_count > 0) ? h_cos_stats[0] / cos_count : 0.0f;
        float cos_variance = (cos_count > 1) ? 
            (h_cos_stats[1] / cos_count - cos_mean * cos_mean) : 0.0f;
        float cos_std = sqrtf(fmaxf(cos_variance, 0.0f));
        float cos_abs_mean = (cos_count > 0) ? h_cos_stats[5] / cos_count : 0.0f;
        
        // EXPECTED VALUES for random vectors in K=768 dimensions:
        // E[cos] = 0
        // Var[cos] = 1/K = 1/768 = 0.0013
        // E[|cos|] = sqrt(2/(π*K)) = sqrt(2/(π*768)) ≈ 0.0287
        float expected_cos_variance = 1.0f / static_cast<float>(d_model);
        float expected_cos_abs_mean = sqrtf(2.0f / (3.14159f * static_cast<float>(d_model)));
        
        // Compute actual-to-expected ratios
        float cos_var_ratio = cos_variance / (expected_cos_variance + 1e-12f);
        float cos_abs_ratio = cos_abs_mean / (expected_cos_abs_mean + 1e-6f);
        
        // 7h. Check for "dominant direction" - compute mean row and its norm
        std::vector<float> mean_row(d_model, 0.0f);
        for (int t = 0; t < split_sample_tokens; ++t) {
            const float* row = h_input_sample.data() + t * d_model;
            for (int c = 0; c < d_model; ++c) {
                mean_row[c] += row[c];
            }
        }
        double mean_row_norm_sq = 0;
        for (int c = 0; c < d_model; ++c) {
            mean_row[c] /= split_sample_tokens;
            mean_row_norm_sq += mean_row[c] * mean_row[c];
        }
        float mean_row_norm = sqrtf(static_cast<float>(mean_row_norm_sq));
        // Ratio: mean_row_norm / avg_row_norm tells us how aligned rows are
        // If rows random: mean_row_norm → 0 (cancellation)
        // If rows identical: mean_row_norm = avg_row_norm
        float alignment_ratio = mean_row_norm / (in_row_norm_mean + 1e-8f);
        
        // ========================================================================
        // Issue #91: Compute row-norm-based expected RMS
        // ========================================================================
        // For Y = X @ W^T where X rows have norm r_x and W rows have norm r_w:
        // If X and W rows are random orthogonal:
        //   Y[i,j] = ||x_i|| * ||w_j|| * cos(θ) where E[cos²(θ)] = 1/K
        //   E[Y²] = r_x² * r_w² / K
        //   E[RMS] = r_x * r_w / sqrt(K)
        // 
        // But if X has a "dominant direction" (mean_row_norm > 0), the dot products
        // along that direction sum coherently instead of canceling:
        //   Y[i,j] ≈ (mean_component · w_j) + (random_component · w_j)
        // The mean component contributes: alignment_ratio * ||x_i|| * ||w_j|| (worst case cos=1)
        // The random component contributes: (1 - alignment_ratio) * ||x_i|| * ||w_j|| / sqrt(K)
        
        // Row-norm-based expected (for perfectly random/orthogonal X and W rows):
        float rownorm_expected_rms = in_row_norm_mean * w_row_norm_mean / sqrtf(static_cast<float>(d_model));
        
        // Adjusted for alignment - this accounts for the "dominant direction" in inputs:
        // The aligned portion projects coherently, the random portion cancels partially
        float aligned_contrib = alignment_ratio * in_row_norm_mean * w_row_norm_mean;
        float random_contrib = (1.0f - alignment_ratio) * rownorm_expected_rms;
        float adjusted_expected_rms = sqrtf(aligned_contrib * aligned_contrib + random_contrib * random_contrib);
        
        // Update expected_rms to use the adjusted formula
        expected_rms = adjusted_expected_rms;
        expected_max = expected_rms * 4.0f;
        
        // ========================================================================
        // Log comprehensive comparison
        // ========================================================================
        fprintf(stderr, "\n[DebugQKVExpectations] layer=%d call=%d tokens=%d\n", layer_idx, g_debug_qkv_call_count, total_tokens);
        fprintf(stderr, "  INPUT (ln1_out)  [%d × %d]: min=%.4f max=%.4f mean=%.4f rms=%.4f std=%.4f\n",
                total_tokens, d_model, input_stats.min_val, input_stats.max_val, 
                input_stats.mean, input_stats.rms, input_stats.stddev);
        fprintf(stderr, "  WEIGHT (W_qkv)   [%d × %d]: min=%.4f max=%.4f mean=%.4f rms=%.4f std=%.4f\n",
                qkv_dim, d_model, weight_stats.min_val, weight_stats.max_val,
                weight_stats.mean, weight_stats.rms, weight_stats.stddev);
        
        if (b_qkv) {
            fprintf(stderr, "  BIAS (b_qkv)     [%d]: min=%.4f max=%.4f mean=%.4f rms=%.4f std=%.4f\n",
                    qkv_dim, bias_stats.min_val, bias_stats.max_val,
                    bias_stats.mean, bias_stats.rms, bias_stats.stddev);
        } else {
            fprintf(stderr, "  BIAS (b_qkv)     [nullptr]\n");
        }
        
        // Show multiple expected formulas for comparison (Issue #91)
        fprintf(stderr, "  EXPECTED OUTPUT (theoretical): rms=%.4f  (formula: elem_rms_in * elem_rms_w * sqrt(%d), assumes independent elements)\n",
                theoretical_rms, d_model);
        fprintf(stderr, "  EXPECTED OUTPUT (row-norm):    rms=%.4f  (formula: row_norm_in * row_norm_w / sqrt(%d), assumes orthogonal rows)\n",
                rownorm_expected_rms, d_model);
        fprintf(stderr, "  EXPECTED OUTPUT (aligned):     rms=%.4f  (adjusted for alignment_ratio=%.4f)\n",
                adjusted_expected_rms, alignment_ratio);
        fprintf(stderr, "  ACTUAL OUTPUT    [%d × %d]: min=%.4f max=%.4f mean=%.4f rms=%.4f std=%.4f\n",
                total_tokens, qkv_dim, actual_stats.min_val, actual_stats.max_val,
                actual_stats.mean, actual_stats.rms, actual_stats.stddev);
        fprintf(stderr, "  MANUAL GEMM      [%d × %d]: min=%.4f max=%.4f mean=%.4f rms=%.4f std=%.4f\n",
                sample_rows, qkv_dim, manual_stats.min_val, manual_stats.max_val,
                manual_stats.mean, manual_stats.rms, manual_stats.stddev);
        
        // Q/K/V split stats
        fprintf(stderr, "  Q/K/V SPLIT (first %d tokens):\n", split_sample_tokens);
        fprintf(stderr, "    Q [cols 0-%d]:       min=%.4f max=%.4f rms=%.4f\n", d_model-1, q_min, q_max, q_rms);
        fprintf(stderr, "    K [cols %d-%d]:   min=%.4f max=%.4f rms=%.4f\n", d_model, d_model+kv_dim-1, k_min, k_max, k_rms);
        fprintf(stderr, "    V [cols %d-%d]: min=%.4f max=%.4f rms=%.4f\n", d_model+kv_dim, qkv_dim-1, v_min, v_max, v_rms);
        
        // Per-row variance
        fprintf(stderr, "  OUTPUT ROW RMS (first %d tokens): min=%.4f max=%.4f mean=%.4f std=%.4f  (outlier ratio=%.2fx)\n",
                split_sample_tokens, row_rms_min, row_rms_max, row_rms_mean, row_rms_std,
                row_rms_max / (row_rms_min + 1e-8f));
        
        // Weight row norms
        fprintf(stderr, "  WEIGHT ROW NORMS: min=%.4f max=%.4f mean=%.4f (variance=%.2fx)\n",
                w_row_norm_min, w_row_norm_max, w_row_norm_mean, 
                w_row_norm_max / (w_row_norm_min + 1e-8f));
        
        // Input row norms
        fprintf(stderr, "  INPUT ROW NORMS (first %d tokens): min=%.4f max=%.4f mean=%.4f (variance=%.2fx)\n",
                split_sample_tokens, in_row_norm_min, in_row_norm_max, in_row_norm_mean,
                in_row_norm_max / (in_row_norm_min + 1e-8f));
        
        // NEW: Input row correlation - THE KEY DIAGNOSTIC
        fprintf(stderr, "  INPUT ROW CORRELATION (first %d tokens pairwise):\n", corr_sample);
        fprintf(stderr, "    cosine: min=%.4f max=%.4f mean=%.4f\n", corr_min, corr_max, corr_mean);
        fprintf(stderr, "    (random vectors: ~0.0, identical: 1.0)\n");
        
        // NEW: Input column variance
        fprintf(stderr, "  INPUT COLUMN VARIANCE: min=%.4f max=%.4f mean=%.4f range_ratio=%.2fx\n",
                col_var_min, col_var_max, col_var_mean, col_var_max / (col_var_min + 1e-8f));
        fprintf(stderr, "    (low range_ratio = ISOTROPIC columns, high ratio = HETEROGENEOUS columns)\n");
        
        // ========================================================================
        // ISSUE #106 RULE 21 OUTPUT: INPUT-WEIGHT COSINE SIMILARITY ANALYSIS
        // ========================================================================
        if (isEquationLoggingEnabled()) {
            fprintf(stderr, "\n[QKV_EQUATION] layer=%d: Y[i,j] = ||x_i|| × ||w_j|| × cos(θ_ij)\n", layer_idx);
            fprintf(stderr, "  INPUT x (ln1_out): shape=[%d,%d] row_norm_mean=%.4f\n", 
                    total_tokens, d_model, in_row_norm_mean);
            fprintf(stderr, "  WEIGHT w (W_qkv):  shape=[%d,%d] row_norm_mean=%.4f\n",
                    qkv_dim, d_model, w_row_norm_mean);
            fprintf(stderr, "  PARAMETERS: d_model=%d, sampled_pairs=%d×%d=%d\n",
                    d_model, cos_sample_input, cos_sample_weight, static_cast<int>(cos_count));
            fprintf(stderr, "  \n");
            fprintf(stderr, "  [COSINE DISTRIBUTION] cos(θ) between input rows and weight rows:\n");
            fprintf(stderr, "    OBSERVED: mean=%.6f std=%.6f mean_abs=%.6f\n", cos_mean, cos_std, cos_abs_mean);
            fprintf(stderr, "    EXPECTED (random K=%d): mean=0.0 var=%.6f mean_abs=%.6f\n",
                    d_model, expected_cos_variance, expected_cos_abs_mean);
            fprintf(stderr, "    RATIO: variance=%.2fx expected, |cos|=%.2fx expected\n", 
                    cos_var_ratio, cos_abs_ratio);
            fprintf(stderr, "  \n");
            fprintf(stderr, "  [QKV_OUTPUT] Expected from GEMM formula:\n");
            fprintf(stderr, "    If cos uniformly random: Y_rms ≈ ||x|| × ||w|| × √(1/K) = %.4f × %.4f × %.4f = %.4f\n",
                    in_row_norm_mean, w_row_norm_mean, 1.0f/sqrtf(static_cast<float>(d_model)), rownorm_expected_rms);
            fprintf(stderr, "    If cos systematic bias: Y_rms ≈ ||x|| × ||w|| × |cos_mean| + random ≈ (formula complex)\n");
            fprintf(stderr, "    ADJUSTED formula (alignment_ratio=%.4f): %.4f\n", alignment_ratio, adjusted_expected_rms);
            fprintf(stderr, "    ACTUAL Y_rms: %.4f\n", actual_stats.rms);
            fprintf(stderr, "  \n");
            
            // Structured logging via centralized EquationLogger
            EQ_LOG_HOST(
                "QKV_COSINE_EQUATION",
                "Y[i,j] = ||x_i|| * ||w_j|| * cos(theta_ij)",
                "x_row_norm=" + std::to_string(in_row_norm_mean) + " w_row_norm=" + std::to_string(w_row_norm_mean) + " cos_mean=" + std::to_string(cos_mean),
                "cos_var_ratio=" + std::to_string(cos_var_ratio) + " cos_abs_ratio=" + std::to_string(cos_abs_ratio),
                "expected_rms=" + std::to_string(adjusted_expected_rms),
                "actual_rms=" + std::to_string(actual_stats.rms),
                0, layer_idx, 0, GRIM::EquationPhase::ATTENTION_SCORE);
            
            // Root cause analysis
            if (cos_var_ratio > 2.0f || cos_abs_ratio > 2.0f) {
                fprintf(stderr, "  [ANOMALY] INPUT-WEIGHT COSINE SIMILARITY IS %.1fx HIGHER THAN RANDOM!\n", 
                        fmaxf(cos_var_ratio, cos_abs_ratio));
                fprintf(stderr, "            This means input vectors SYSTEMATICALLY ALIGN with weight rows.\n");
                fprintf(stderr, "            For Layer 0 (embeddings): Embedding weights and W_qkv may share structure!\n");
                fprintf(stderr, "            Both learned from same vocabulary → similar semantic directions.\n");
            }
            
            // Column variance analysis (isotropic vs heterogeneous)
            float col_var_range_ratio = col_var_max / (col_var_min + 1e-8f);
            if (col_var_range_ratio < 3.0f) {
                fprintf(stderr, "  [ANOMALY] INPUT COLUMNS ARE ISOTROPIC (range_ratio=%.2fx < 3x)!\n", col_var_range_ratio);
                fprintf(stderr, "            Isotropic columns cause COHERENT summation in GEMM → output explosion.\n");
                fprintf(stderr, "            Layer 0 embeddings are often isotropic (uniform variance across dimensions).\n");
            }
            
            fprintf(stderr, "\n");
        }
        
        // NEW: Alignment ratio - most important metric!
        fprintf(stderr, "  INPUT ALIGNMENT: mean_row_norm=%.4f, avg_row_norm=%.4f, RATIO=%.4f\n",
                mean_row_norm, in_row_norm_mean, alignment_ratio);
        fprintf(stderr, "    (ratio ~0: random directions, ratio ~1: all rows parallel!)\n");
        
        // First 4 values comparison
        fprintf(stderr, "  FIRST 4 VALUES COMPARISON (actual vs manual):\n");
        float max_diff = 0.0f;
        for (int i = 0; i < 4 && i < sample_rows * qkv_dim; ++i) {
            float diff = fabsf(h_actual_sample[i] - h_manual_sample[i]);
            max_diff = fmaxf(max_diff, diff);
            fprintf(stderr, "    [%d]: actual=%.6f manual=%.6f diff=%.6f\n",
                    i, h_actual_sample[i], h_manual_sample[i], diff);
        }
        
        // Verdict - show ratio to both theoretical and adjusted expected
        float rms_ratio_theoretical = actual_stats.rms / (theoretical_rms + 1e-8f);
        float rms_ratio_adjusted = actual_stats.rms / (adjusted_expected_rms + 1e-8f);
        fprintf(stderr, "  VERDICT:\n");
        fprintf(stderr, "    vs theoretical:  actual/expected = %.2fx (mismatch due to positive skew in input)\n",
                rms_ratio_theoretical);
        fprintf(stderr, "    vs adjusted:     actual/expected = %.2fx (accounting for alignment ratio=%.3f)\n",
                rms_ratio_adjusted, alignment_ratio);
        fprintf(stderr, "    manual_vs_actual_max_diff = %.6f\n", max_diff);
        
        // Anomaly detection - NOW uses adjusted expected (should be close to 1.0x if formula is correct)
        bool has_anomaly = false;
        if (rms_ratio_adjusted > 2.0f) {
            fprintf(stderr, "  *** ANOMALY: Actual output is %.1fx larger than ADJUSTED expected! ***\n", rms_ratio_adjusted);
            fprintf(stderr, "       (Theoretical ratio was %.1fx - the difference shows alignment effect)\n", rms_ratio_theoretical);
            has_anomaly = true;
        } else if (rms_ratio_adjusted < 0.5f) {
            fprintf(stderr, "  *** ANOMALY: Actual output is %.1fx SMALLER than adjusted expected! ***\n", rms_ratio_adjusted);
            has_anomaly = true;
        }
        if (max_diff > 1e-3f) {
            fprintf(stderr, "  *** ANOMALY: Manual GEMM doesn't match cuBLAS! Possible bug! ***\n");
            has_anomaly = true;
        }
        if (b_qkv && bias_stats.rms > 1.0f) {
            fprintf(stderr, "  *** ANOMALY: Bias RMS=%.4f is too large! Should be ~0 or small. ***\n", bias_stats.rms);
            has_anomaly = true;
        }
        if (row_rms_max / (row_rms_min + 1e-8f) > 10.0f) {
            fprintf(stderr, "  *** ANOMALY: Outlier tokens detected! Max/min row RMS ratio=%.1fx ***\n",
                    row_rms_max / (row_rms_min + 1e-8f));
            has_anomaly = true;
        }
        if (in_row_norm_max / (in_row_norm_min + 1e-8f) > 10.0f) {
            fprintf(stderr, "  *** ANOMALY: Input has outlier rows! Max/min norm ratio=%.1fx ***\n",
                    in_row_norm_max / (in_row_norm_min + 1e-8f));
            has_anomaly = true;
        }
        // NEW anomaly: high alignment ratio indicates correlated inputs
        if (alignment_ratio > 0.3f) {
            fprintf(stderr, "  *** ANOMALY: Input rows are HIGHLY ALIGNED (ratio=%.2f)! ***\n", alignment_ratio);
            fprintf(stderr, "      This explains the output explosion - GEMM formula assumes uncorrelated inputs!\n");
            has_anomaly = true;
        }
        if (corr_mean > 0.5f) {
            fprintf(stderr, "  *** ANOMALY: High pairwise correlation (mean=%.2f)! ***\n", corr_mean);
            has_anomaly = true;
        }
        if (!has_anomaly) {
            fprintf(stderr, "  ✓ No obvious anomalies detected\n");
        }
        fprintf(stderr, "\n");
    }
    
    void resetDebugQKVExpectations() {
        g_debug_qkv_call_count = 0;
    }
    
    // ========================================================================
    // ISSUE #93 DIAGNOSTIC: logForwardStageStats
    // Logs min/max/mean/rms statistics for intermediate tensors during forward pass
    // This helps identify exactly WHERE values explode through the layer
    // ========================================================================
    static int g_fwd_stage_call_count = 0;
    static constexpr int kMaxFwdStageCalls = 48;  // 2 batches * 12 layers * 2 stages (attn+ffn)
    
    void logForwardStageStats(
        const char* stage_name,
        int layer_idx,
        const float* data,
        int count,
        cudaStream_t stream
    ) {
        if (g_fwd_stage_call_count >= kMaxFwdStageCalls) return;
        g_fwd_stage_call_count++;
        
        // Allocate reduction buffers
        float h_min = FLT_MAX, h_max = -FLT_MAX, h_sum = 0.0f, h_sum_sq = 0.0f;
        float* d_min, *d_max, *d_sum, *d_sum_sq;
        cudaMalloc(&d_min, sizeof(float));
        cudaMalloc(&d_max, sizeof(float));
        cudaMalloc(&d_sum, sizeof(float));
        cudaMalloc(&d_sum_sq, sizeof(float));
        
        float init_min = FLT_MAX, init_max = -FLT_MAX, init_zero = 0.0f;
        cudaMemcpyAsync(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sum, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_sum_sq, &init_zero, sizeof(float), cudaMemcpyHostToDevice, stream);
        
        int threads = 256;
        int blocks = std::min((count + threads - 1) / threads, 256);
        diagLn1OutKernel<<<blocks, threads, 0, stream>>>(data, count, d_min, d_max, d_sum, d_sum_sq);
        
        cudaStreamSynchronize(stream);
        cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_sum_sq, d_sum_sq, sizeof(float), cudaMemcpyDeviceToHost);
        
        float mean = h_sum / count;
        float variance = h_sum_sq / count - mean * mean;
        float stddev = sqrtf(fmaxf(variance, 0.0f));
        float rms = sqrtf(h_sum_sq / count);
        
        // Also compute absolute max for quick diagnosis
        float abs_max = fmaxf(fabsf(h_min), fabsf(h_max));
        
        fprintf(stderr, "[Issue93-FWD-%s] layer=%d: min=%.4f max=%.4f abs_max=%.4f mean=%.4f rms=%.4f\n",
                stage_name, layer_idx, h_min, h_max, abs_max, mean, rms);
        
        cudaFree(d_min);
        cudaFree(d_max);
        cudaFree(d_sum);
        cudaFree(d_sum_sq);
    }
    
    void resetForwardStageStats() {
        g_fwd_stage_call_count = 0;
    }
}
//======================================================// 

// External kernel declaration (global scope - defined in BackwardKernels.cu)
void launchBiasSumGradient(const float* grad_output, float* grad_bias,
                          int total_tokens, int hidden_dim,
                          cudaStream_t stream);

namespace GRIM {

// ISSUE #77 DIAGNOSTIC: Expose reset function from anonymous namespace
void resetIssue77FwdDiag() {
    ::resetIssue77FwdDiag();  // Call the anonymous namespace version
}

static_assert(!GRIM::HyperParameters::QK_NORMALIZATION_ENABLED,
              "FlashAttention v2 forward does not support QK normalization.");
static_assert(GRIM::HyperParameters::SOFTMAX_TEMPERATURE == 1.0f,
              "FlashAttention v2 forward requires softmax_temperature=1.0f.");

//======================================================//
//  CUDA Error Checking
//======================================================//

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char msg[512]; \
        snprintf(msg, sizeof(msg), "CUDA ERROR at %s:%d - %s: %s", \
                 __FILE__, __LINE__, #call, cudaGetErrorString(err)); \
        throw std::runtime_error(msg); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        char msg[256]; \
        snprintf(msg, sizeof(msg), "cuBLAS ERROR at %s:%d - %s: status=%d", \
                 __FILE__, __LINE__, #call, (int)status); \
        throw std::runtime_error(msg); \
    } \
} while(0)

//======================================================//
//  Extern declarations - use shared kernels from BackwardKernels.cu
//  Rule 20: No duplicate code, use centralized implementations
//======================================================//

// From BackwardKernels.cu
extern "C" void launchResidualAdd(
    const float* input,
    const float* residual, 
    float* output,
    int total_size,
    cudaStream_t stream
);

//======================================================//
//  Local kernels (not duplicated elsewhere)
//======================================================//

__global__ void fillOnesKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

__global__ void addBiasKernel(float* data, const float* bias, int tokens, int dim) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    if (token_idx < tokens && dim_idx < dim) {
        data[token_idx * dim + dim_idx] += bias[dim_idx];
    }
}

static void launchAddBias(float* data, const float* bias, int tokens, int dim, 
                          cudaStream_t stream) {
    dim3 grid(tokens);
    dim3 block(dim);
    if (dim > 1024) {
        // For large dims, use 2D grid
        grid = dim3(tokens, (dim + 255) / 256);
        block = dim3(256);
    }
    addBiasKernel<<<grid, block, 0, stream>>>(data, bias, tokens, dim);
    CUDA_CHECK(cudaGetLastError());
}

//======================================================//
//  RMSNorm Forward/Backward (calls into RMSNorm_Kernel_GPU.cu)
//======================================================//

// NOTE: RMSNorm forward/backward are declared in RMSNorm_Kernel_GPU.hpp
// Include that header if needed. The extern declarations below are REMOVED
// as backward pass is now handled via TensorView-based RMSNormBackwardParams.

//======================================================//
//  EncodingLayer Implementation
//======================================================//

EncodingLayer::EncodingLayer(const EncodingConfig& cfg) {
    setConfig(cfg);
}

EncodingLayer::~EncodingLayer() {
    freeWeights();
    // NOTE: config_.cublas_handle is NOT owned by EncodingLayer (Rule 22)
    // TrainingState owns it - do NOT destroy!
}

EncodingLayer::EncodingLayer(EncodingLayer&& other) noexcept
    : config_(other.config_)
    , weights_allocated_(other.weights_allocated_)
    , rms1_gamma_(std::move(other.rms1_gamma_))
    , rms2_gamma_(std::move(other.rms2_gamma_))
    , W_qkv_(std::move(other.W_qkv_))
    , b_qkv_(std::move(other.b_qkv_))
    , W_o_(std::move(other.W_o_))
    , b_o_(std::move(other.b_o_))
    , ffn_(std::move(other.ffn_))
    , workspace_(other.workspace_)
    , workspace_bytes_(other.workspace_bytes_)
{
    // Null out the moved-from object
    other.config_.cublas_handle = nullptr;
    other.weights_allocated_ = false;
    other.workspace_ = nullptr;
    other.workspace_bytes_ = 0;
}

EncodingLayer& EncodingLayer::operator=(EncodingLayer&& other) noexcept {
    if (this != &other) {
        freeWeights();
        // NOTE: config_.cublas_handle is NOT owned - do NOT destroy (Rule 22)
        
        config_ = other.config_;
        weights_allocated_ = other.weights_allocated_;
        config_.cublas_handle = other.config_.cublas_handle;
        rms1_gamma_ = std::move(other.rms1_gamma_);
        rms2_gamma_ = std::move(other.rms2_gamma_);
        W_qkv_ = std::move(other.W_qkv_);
        b_qkv_ = std::move(other.b_qkv_);
        W_o_ = std::move(other.W_o_);
        b_o_ = std::move(other.b_o_);
        ffn_ = std::move(other.ffn_);
        workspace_ = other.workspace_;
        workspace_bytes_ = other.workspace_bytes_;
        
        other.config_.cublas_handle = nullptr;
        other.weights_allocated_ = false;
        other.workspace_ = nullptr;
        other.workspace_bytes_ = 0;
    }
    return *this;
}

void EncodingLayer::setConfig(const EncodingConfig& cfg) {
    cfg.validate("EncodingLayer::setConfig");
    config_ = cfg;
}

void EncodingLayer::freeWeights() {
    // Tensor handles its own memory cleanup via destructor
    // Reset to empty tensors
    rms1_gamma_ = Tensor();
    rms2_gamma_ = Tensor();
    W_qkv_ = Tensor();
    b_qkv_ = Tensor();
    W_o_ = Tensor();
    b_o_ = Tensor();
    ffn_.reset();
    weights_allocated_ = false;
}

void EncodingLayer::allocateWeights() {
    config_.validate("EncodingLayer::allocateWeights");
    
    if (weights_allocated_) {
        throw std::runtime_error("EncodingLayer::allocateWeights: weights already allocated! "
                                 "Call freeWeights() first if you want to reallocate.");
    }
    
    const int d_model = config_.d_model;
    const int kv_dim = config_.kvDim();
    const int d_ff = config_.d_ff;
    
    // Use centralized cuBLAS handle (Rule 22: NO local handle creation)
    if (!config_.cublas_handle) {
        std::cerr << "ERROR in EncodingLayer::allocateWeights: config_.cublas_handle is NULL!" << std::endl;
        std::cerr << "  config_.stream = " << config_.stream << std::endl;
        std::cerr << "  d_model = " << d_model << ", num_heads = " << config_.num_heads << std::endl;
        throw std::runtime_error("EncodingLayer::allocateWeights: config_.cublas_handle is NULL. "
                                 "MUST pass training_state.cublas_handle per Rule 22.");
    }
    if (!config_.stream) {
        std::cerr << "ERROR in EncodingLayer::allocateWeights: config_.stream is NULL!" << std::endl;
        throw std::runtime_error("EncodingLayer::allocateWeights: config_.stream is NULL");
    }
    StreamController::fatalIfDefaultStream(config_.stream, "EncodingLayer::allocateWeights");
    // Rule 20: Don't store copy of handle, use config_.cublas_handle directly
    
    // Create shapes for weight tensors
    // RMSNorm gamma: 1D vector stored as [1, d_model] BSM
    TensorContract::Shape2D gamma_2d{1, d_model};
    TensorContract::TensorShape gamma_shape(TensorContract::Layout::BSM, gamma_2d);
    
    // Attention weights
    const int qkv_out_dim = d_model + 2 * kv_dim;
    TensorContract::Shape2D qkv_weight_2d{qkv_out_dim, d_model};
    TensorContract::Shape2D qkv_bias_2d{1, qkv_out_dim};
    TensorContract::Shape2D o_weight_2d{d_model, d_model};
    TensorContract::Shape2D o_bias_2d{1, d_model};
    
    TensorContract::TensorShape qkv_weight_shape(TensorContract::Layout::BSM, qkv_weight_2d);
    TensorContract::TensorShape qkv_bias_shape(TensorContract::Layout::BSM, qkv_bias_2d);
    TensorContract::TensorShape o_weight_shape(TensorContract::Layout::BSM, o_weight_2d);
    TensorContract::TensorShape o_bias_shape(TensorContract::Layout::BSM, o_bias_2d);
    
    // RMSNorm gamma - initialized to 1.0, then set requires_grad
    rms1_gamma_ = Tensor::zeros(gamma_shape, true, config_.stream);
    rms2_gamma_ = Tensor::zeros(gamma_shape, true, config_.stream);
    
    // Fill gamma with ones via kernel
    int threads = 256;
    int blocks = (d_model + threads - 1) / threads;
    fillOnesKernel<<<blocks, threads, 0, config_.stream>>>(rms1_gamma_.data, d_model);
    fillOnesKernel<<<blocks, threads, 0, config_.stream>>>(rms2_gamma_.data, d_model);
    
    // Xavier initialization for attention weights
    W_qkv_ = Tensor::xavier_uniform(qkv_weight_shape, true, config_.stream);
    b_qkv_ = Tensor::zeros(qkv_bias_shape, true, config_.stream);
    
    // W_o: [d_model, d_model] output projection
    W_o_ = Tensor::xavier_uniform(o_weight_shape, true, config_.stream);
    b_o_ = Tensor::zeros(o_bias_shape, true, config_.stream);
    
    // FFN layer
    FeedForwardConfig ffn_cfg;
    ffn_cfg.d_model = d_model;
    ffn_cfg.d_ff = d_ff;
    ffn_cfg.stream = config_.stream;
    ffn_cfg.cublas_handle = config_.cublas_handle;  // CRITICAL: Must pass handle to FFN (Rule 22)
    ffn_ = std::make_unique<FeedForwardLayer>(ffn_cfg);
    ffn_->ensureWeightStorage();
    
    // AUTOGRAD MIGRATION: Allocate gradient buffers for all trainable tensors
    // This replaces the legacy cudaMalloc in InitTrainingState.cu
    rms1_gamma_.ensure_grad();
    rms2_gamma_.ensure_grad();
    W_qkv_.ensure_grad();
    b_qkv_.ensure_grad();
    W_o_.ensure_grad();
    b_o_.ensure_grad();
    // FFN gradients allocated via ffn_->ensureWeightStorage() -> FFN's own ensure_grad() calls
    // LayerScale: Allocated via TrainingTensors, wired via useExternalWeights()
    
    weights_allocated_ = true;
}

void EncodingLayer::useExternalWeights(
    Tensor& rms1_gamma,
    Tensor& rms2_gamma,
    Tensor& qkv_weight,
    Tensor& qkv_bias,
    Tensor& out_weight,
    Tensor& out_bias,
    Tensor& ffn_w1,
    Tensor& ffn_b1,
    Tensor& ffn_w2,
    Tensor& ffn_b2,
    Tensor* layer_scale1,    // Issue #109: optional LayerScale for attention residual
    Tensor* layer_scale2     // Issue #109: optional LayerScale for FFN residual
) {
    // Rule 20: Fail loud if already allocated own weights (prevents confusion)
    if (weights_allocated_ && !using_external_weights_) {
        throw std::runtime_error("EncodingLayer::useExternalWeights: Cannot switch to external weights "
                                 "after allocating own weights. Use freeWeights() first or never call allocateWeights().");
    }
    
    config_.validate("EncodingLayer::useExternalWeights");
    
    const int d_model = config_.d_model;
    const int kv_dim = config_.kvDim();
    const int d_ff = config_.d_ff;
    const int qkv_out_dim = d_model + 2 * kv_dim;
    
    // Validate shapes
    if (rms1_gamma.numel() != d_model) {
        throw std::invalid_argument("useExternalWeights: rms1_gamma size mismatch. Expected " + 
                                    std::to_string(d_model) + ", got " + std::to_string(rms1_gamma.numel()));
    }
    if (rms2_gamma.numel() != d_model) {
        throw std::invalid_argument("useExternalWeights: rms2_gamma size mismatch");
    }
    if (qkv_weight.numel() != static_cast<std::size_t>(qkv_out_dim) * d_model) {
        throw std::invalid_argument("useExternalWeights: qkv_weight size mismatch. Expected " +
                                    std::to_string(static_cast<std::size_t>(qkv_out_dim) * d_model) + 
                                    ", got " + std::to_string(qkv_weight.numel()));
    }
    if (out_weight.numel() != static_cast<std::size_t>(d_model) * d_model) {
        throw std::invalid_argument("useExternalWeights: out_weight size mismatch");
    }
    if (ffn_w1.numel() != static_cast<std::size_t>(d_ff) * d_model) {
        throw std::invalid_argument("useExternalWeights: ffn_w1 size mismatch");
    }
    if (ffn_w2.numel() != static_cast<std::size_t>(d_model) * d_ff) {
        throw std::invalid_argument("useExternalWeights: ffn_w2 size mismatch");
    }
    
    // Create view Tensors that reference the external buffers
    // NOTE: These Tensors do NOT own the data (owns_data=false)
    // The grad pointers also come from the external Tensors
    // ISSUE #59: Use share_grad() for proper shared_ptr semantics
    
    // RMSNorm gammas
    rms1_gamma_ = Tensor::from_ptr(rms1_gamma.data, rms1_gamma.shape, false, true);
    rms1_gamma_.share_grad(rms1_gamma);
    rms1_gamma_.owns_data = false;
    
    rms2_gamma_ = Tensor::from_ptr(rms2_gamma.data, rms2_gamma.shape, false, true);
    rms2_gamma_.share_grad(rms2_gamma);
    rms2_gamma_.owns_data = false;
    

    
    W_qkv_ = Tensor::from_ptr(qkv_weight.data, qkv_weight.shape, false, true);
    W_qkv_.share_grad(qkv_weight);
    W_qkv_.owns_data = false;
    

    if (qkv_bias.data) {
        b_qkv_ = Tensor::from_ptr(qkv_bias.data, qkv_bias.shape, false, true);
        b_qkv_.share_grad(qkv_bias);
        b_qkv_.owns_data = false;
    }
    
    // Output projection
    W_o_ = Tensor::from_ptr(out_weight.data, out_weight.shape, false, true);
    W_o_.share_grad(out_weight);
    W_o_.owns_data = false;
    
    if (out_bias.data) {
        b_o_ = Tensor::from_ptr(out_bias.data, out_bias.shape, false, true);
        b_o_.share_grad(out_bias);
        b_o_.owns_data = false;
    }
    
    // FFN - need to create the layer and set its external weights
    if (!ffn_) {
        FeedForwardConfig ffn_cfg;
        ffn_cfg.d_model = d_model;
        ffn_cfg.d_ff = d_ff;
        ffn_cfg.stream = config_.stream;
        ffn_cfg.cublas_handle = config_.cublas_handle;
        ffn_ = std::make_unique<FeedForwardLayer>(ffn_cfg);
    }
    ffn_->useExternalWeights(ffn_w1, ffn_b1, ffn_w2, ffn_b2);
    
    // Issue #109: LayerScale tensors (optional, gated by config_.use_layer_scale)
    if (layer_scale1 && layer_scale1->data) {
        layer_scale1_ = Tensor::from_ptr(layer_scale1->data, layer_scale1->shape, false, true);
        layer_scale1_.share_grad(*layer_scale1);
        layer_scale1_.owns_data = false;
    }
    if (layer_scale2 && layer_scale2->data) {
        layer_scale2_ = Tensor::from_ptr(layer_scale2->data, layer_scale2->shape, false, true);
        layer_scale2_.share_grad(*layer_scale2);
        layer_scale2_.owns_data = false;
    }
    
    weights_allocated_ = true;
    using_external_weights_ = true;
    
    fprintf(stderr, "[EncodingLayer] Using external weights: qkv=[%zu], W_o=[%zu], ffn_w1=[%zu], ffn_w2=[%zu]\n",
            W_qkv_.numel(), W_o_.numel(), ffn_w1.numel(), ffn_w2.numel());
}

void EncodingLayer::validateReady(const char* context) const {
    if (!weights_allocated_) {
        throw std::runtime_error(std::string(context) + 
            ": weights not allocated! Call allocateWeights() first.");
    }
    if (!config_.cublas_handle) {
        throw std::runtime_error(std::string(context) + 
            ": cuBLAS handle not initialized in config!");
    }
}

std::size_t EncodingLayer::requiredWorkspaceBytes(int total_tokens, int seq_len) const {
    config_.validate("EncodingLayer::requiredWorkspaceBytes");
    
    if (total_tokens <= 0 || seq_len <= 0) {
        throw std::invalid_argument("requiredWorkspaceBytes: total_tokens and seq_len must be > 0");
    }
    
    const int batch_size = total_tokens / seq_len;
    const int d_model = config_.d_model;
    const int kv_dim = config_.kvDim();
    const int num_heads = config_.num_heads;
    const int num_kv_heads = config_.effectiveKVHeads();
    const int head_dim = config_.headDim();
    const int d_ff = config_.d_ff;
    
    std::size_t bytes = 0;
    
    // RMSNorm intermediates
    bytes += total_tokens * d_model * sizeof(float);  // ln1_out
    bytes += total_tokens * d_model * sizeof(float);  // ln2_out
    
    // QKV projection outputs [tokens, dim]
    bytes += total_tokens * d_model * sizeof(float);  // Q [tokens, d_model]
    bytes += total_tokens * kv_dim * sizeof(float);   // K [tokens, kv_dim]
    bytes += total_tokens * kv_dim * sizeof(float);   // V [tokens, kv_dim]
    
    // QKV reshaped to BHSD for Flash Attention
    bytes += batch_size * num_heads * seq_len * head_dim * sizeof(float);      // Q_bhsd
    bytes += batch_size * num_kv_heads * seq_len * head_dim * sizeof(float);   // K_bhsd
    bytes += batch_size * num_kv_heads * seq_len * head_dim * sizeof(float);   // V_bhsd
    
    // Attention output BHSD
    bytes += batch_size * num_heads * seq_len * head_dim * sizeof(float);      // attn_out_bhsd
    
    // Attention output reshaped [tokens, d_model]
    bytes += total_tokens * d_model * sizeof(float);  // attn_out
    
    // Residual
    bytes += total_tokens * d_model * sizeof(float);  // residual1
    
    // FFN intermediates
    bytes += total_tokens * d_ff * sizeof(float);     // pre_gelu
    bytes += total_tokens * d_ff * sizeof(float);     // post_gelu
    bytes += total_tokens * d_model * sizeof(float);  // ffn_out
    
    return bytes;
}

void EncodingLayer::setWorkspace(float* workspace, std::size_t bytes) {
    workspace_ = workspace;
    workspace_bytes_ = bytes;
}

//======================================================//
//  Forward Pass - Autograd Implementation with ForwardIntermediates (Issue #56 Fix)
//======================================================//

Tensor EncodingLayer::forward(const Tensor& input, int seq_len, cudaStream_t stream,
                               ForwardIntermediates& intermediates) {
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] START total_tokens=%d seq_len=%d\n", 
                input.shape.flat.rows, seq_len);
    }
    validateReady("EncodingLayer::forward");
    
    // CRITICAL: Set autograd cuBLAS handle before any autograd::matmul calls
    // The handle must be set per-call since it's thread_local
    autograd::set_autograd_cublas_handle(config_.cublas_handle);
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] autograd cuBLAS handle set: %p\n", (void*)config_.cublas_handle);
    }
    
    // Validate input
    if (!input.data) {
        throw std::runtime_error("EncodingLayer::forward: input.data is NULL");
    }
    if (!stream) {
        throw std::runtime_error("EncodingLayer::forward: stream is NULL");
    }
    
    const int total_tokens = input.shape.flat.rows;
    const int d_model = config_.d_model;
    
    if (input.shape.flat.cols != d_model) {
        throw std::runtime_error("EncodingLayer::forward: input d_model mismatch. "
                                 "Expected " + std::to_string(d_model) + 
                                 ", got " + std::to_string(input.shape.flat.cols));
    }
    if (total_tokens % seq_len != 0) {
        throw std::runtime_error("EncodingLayer::forward: total_tokens not divisible by seq_len");
    }
    
    const int batch_size = total_tokens / seq_len;
    const int num_heads = config_.num_heads;
    const int num_kv_heads = config_.effectiveKVHeads();
    const int head_dim = config_.headDim();
    const int qkv_debug = qkvDebugLevel();
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] validated: batch=%d heads=%d kv_heads=%d head_dim=%d\n", 
                batch_size, num_heads, num_kv_heads, head_dim);
    }
    
    //--------------------------------------------------
    // 1. RMSNorm1: input -> ln1_out
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------
    // ISSUE #91 DIAGNOSTIC: Log RMSNorm INPUT statistics BEFORE normalization
    logRmsNormInputStats(g_issue77_fwd_layer_count % 12, input.data, total_tokens, d_model, stream);
    
    // ISSUE #91 DIAGNOSTIC: Log gamma weights to verify they're 1.0
    if (g_issue77_fwd_diag_enabled && g_issue77_fwd_layer_count < 24) {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        float gamma_sum = 0.0f, gamma_sum_sq = 0.0f;
        std::vector<float> h_gamma(d_model);
        cudaMemcpy(h_gamma.data(), rms1_gamma_.data, d_model * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < d_model; i++) {
            gamma_sum += h_gamma[i];
            gamma_sum_sq += h_gamma[i] * h_gamma[i];
        }
        float gamma_mean = gamma_sum / d_model;
        float gamma_rms = sqrtf(gamma_sum_sq / d_model);
        fprintf(stderr, "[Issue91-FWD-gamma] layer=%d: gamma_mean=%.10f gamma_rms=%.10f (expected: mean=1.0, rms=1.0)\n",
                layer_idx, gamma_mean, gamma_rms);
        
        // ISSUE #91 DIAGNOSTIC: Log W_qkv weight statistics to check initialization
        const int qkv_elems = W_qkv_.numel();
        const int sample_size = std::min(10000, qkv_elems);
        std::vector<float> h_wqkv(sample_size);
        cudaMemcpy(h_wqkv.data(), W_qkv_.data, sample_size * sizeof(float), cudaMemcpyDeviceToHost);
        float wqkv_min = h_wqkv[0], wqkv_max = h_wqkv[0];
        double wqkv_sum = 0.0, wqkv_sum_sq = 0.0;
        for (int i = 0; i < sample_size; i++) {
            wqkv_min = std::min(wqkv_min, h_wqkv[i]);
            wqkv_max = std::max(wqkv_max, h_wqkv[i]);
            wqkv_sum += h_wqkv[i];
            wqkv_sum_sq += h_wqkv[i] * h_wqkv[i];
        }
        float wqkv_mean = wqkv_sum / sample_size;
        float wqkv_rms = sqrtf(wqkv_sum_sq / sample_size);
        fprintf(stderr, "[Issue91-FWD-W_qkv] layer=%d: min=%.10f max=%.10f mean=%.10f rms=%.10f (expected rms ~0.036 for Xavier)\n",
                layer_idx, wqkv_min, wqkv_max, wqkv_mean, wqkv_rms);
    }
    
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 1: RMSNorm1...\n");
    
    // ISSUE #93 DIAGNOSTIC: Log layer input BEFORE RMSNorm
    // This helps trace where large values originate (embedding vs previous layer output)
    {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        logForwardStageStats("layer_input", layer_idx, input.data, total_tokens * d_model, stream);
    }
    
    // ISSUE #94 DIAGNOSTIC: Detailed RMSNorm input/output comparison
    // Log per-row RMS BEFORE normalization to understand scaling factor
    if (g_issue77_fwd_diag_enabled && g_issue77_fwd_layer_count < 24) {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        
        // Sample first 10 rows to compute their individual RMS values
        const int sample_rows = std::min(10, total_tokens);
        std::vector<float> h_input_rows(sample_rows * d_model);
        cudaMemcpy(h_input_rows.data(), input.data, h_input_rows.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        float min_row_rms = FLT_MAX, max_row_rms = 0.0f;
        double sum_row_rms = 0.0;
        for (int r = 0; r < sample_rows; r++) {
            double row_sum_sq = 0.0;
            for (int c = 0; c < d_model; c++) {
                float val = h_input_rows[r * d_model + c];
                row_sum_sq += val * val;
            }
            float row_rms = sqrtf(row_sum_sq / d_model);
            min_row_rms = std::min(min_row_rms, row_rms);
            max_row_rms = std::max(max_row_rms, row_rms);
            sum_row_rms += row_rms;
        }
        float avg_row_rms = sum_row_rms / sample_rows;
        
        fprintf(stderr, "[Issue94-RMSNorm-INPUT] layer=%d: per_row_rms: min=%.4f max=%.4f avg=%.4f "
                "(Note: output_rms = input_rms * gamma / sqrt(input_rms² + eps))\n",
                layer_idx, min_row_rms, max_row_rms, avg_row_rms);
        
        // Store for equation logging after RMSNorm (using file-scope temp)
        g_diag_input_avg_rms = avg_row_rms;
    }
    
    intermediates.ln1_out = autograd::rms_norm(input, rms1_gamma_, config_.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 1: RMSNorm1 DONE\n");
    
    // [RMSNORM_EQUATION] Rule 21 diagnostic - MUST verify math matches expectation
    // Equation: y[i] = x[i] * gamma[i] / sqrt(mean(x²) + eps)
    // Expected: per_row_rms(y) = input_rms * gamma_rms / sqrt(input_rms² + eps)
    // Note: When input_rms >> sqrt(eps), this approaches gamma_rms ≈ 1.0
    //       When input_rms ≈ sqrt(eps), epsilon contributes significantly to denominator
    if (g_issue77_fwd_diag_enabled && g_issue77_fwd_layer_count < 24) {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        
        // Get gamma stats
        std::vector<float> h_gamma(d_model);
        cudaMemcpy(h_gamma.data(), rms1_gamma_.data, d_model * sizeof(float), cudaMemcpyDeviceToHost);
        float gamma_min = h_gamma[0], gamma_max = h_gamma[0];
        double gamma_sum_sq = 0.0;
        for (int c = 0; c < d_model; c++) {
            gamma_min = std::min(gamma_min, h_gamma[c]);
            gamma_max = std::max(gamma_max, h_gamma[c]);
            gamma_sum_sq += h_gamma[c] * h_gamma[c];
        }
        float gamma_rms = sqrtf(gamma_sum_sq / d_model);
        
        const int sample_rows = std::min(10, total_tokens);
        std::vector<float> h_output_rows(sample_rows * d_model);
        cudaMemcpy(h_output_rows.data(), intermediates.ln1_out.data, h_output_rows.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        float min_row_rms = FLT_MAX, max_row_rms = 0.0f;
        double sum_row_rms = 0.0;
        float min_val = FLT_MAX, max_val = -FLT_MAX;
        for (int r = 0; r < sample_rows; r++) {
            double row_sum_sq = 0.0;
            for (int c = 0; c < d_model; c++) {
                float val = h_output_rows[r * d_model + c];
                row_sum_sq += val * val;
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
            float row_rms = sqrtf(row_sum_sq / d_model);
            min_row_rms = std::min(min_row_rms, row_rms);
            max_row_rms = std::max(max_row_rms, row_rms);
            sum_row_rms += row_rms;
        }
        float avg_row_rms = sum_row_rms / sample_rows;
        
        // [RMSNORM_EQUATION] format per Rule 21
        // Issue #105 FIX: Correct expected value formula accounting for epsilon contribution
        // Formula: y = x * gamma / sqrt(mean(x²) + eps)
        // Therefore: output_rms = input_rms * gamma_rms / sqrt(input_rms² + eps)
        // When input_rms² >> eps: output_rms ≈ gamma_rms (the old assumption)
        // When input_rms² ≈ eps: output_rms ≈ input_rms * gamma_rms / sqrt(2*eps) < gamma_rms
        const float input_rms_sq = g_diag_input_avg_rms * g_diag_input_avg_rms;
        const float eps = config_.rms_epsilon;
        const float expected_output_rms = g_diag_input_avg_rms * gamma_rms / sqrtf(input_rms_sq + eps);
        const float eps_contribution_pct = 100.0f * eps / (input_rms_sq + eps);
        
        if (isEquationLoggingEnabled()) {
            fprintf(stderr, "[RMSNORM_EQUATION] layer=%d: y = x * gamma / sqrt(mean(x²) + eps)\n", layer_idx);
            fprintf(stderr, "  INPUT x: shape=[%d, %d], per_row_rms=%.6f (rms²=%.2e), eps=%.2e\n", 
                    total_tokens, d_model, g_diag_input_avg_rms, input_rms_sq, eps);
            fprintf(stderr, "  GAMMA: min=%.6f max=%.6f rms=%.6f\n", gamma_min, gamma_max, gamma_rms);
            fprintf(stderr, "  EPSILON CONTRIBUTION: %.1f%% of denominator (eps / (input_rms² + eps))\n", eps_contribution_pct);
            fprintf(stderr, "  EXPECTED output_rms = input_rms * gamma_rms / sqrt(input_rms² + eps) = %.6f\n", expected_output_rms);
            fprintf(stderr, "  ACTUAL output_rms: min=%.6f max=%.6f avg=%.6f\n", min_row_rms, max_row_rms, avg_row_rms);
            
            // Structured logging via centralized EquationLogger
            EQ_LOG_HOST(
                "RMSNORM_EQUATION",
                "y = x * gamma / sqrt(mean(x^2) + eps)",
                "input_rms=" + std::to_string(g_diag_input_avg_rms) + " gamma_rms=" + std::to_string(gamma_rms) + " eps=" + std::to_string(eps),
                "eps_contrib=" + std::to_string(eps_contribution_pct) + "%",
                "expected_rms=" + std::to_string(expected_output_rms),
                "actual_rms=" + std::to_string(avg_row_rms) + " min=" + std::to_string(min_row_rms) + " max=" + std::to_string(max_row_rms),
                0, layer_idx, 0, GRIM::EquationPhase::RMSNORM);
            
            if (std::abs(avg_row_rms - expected_output_rms) > 0.01f) {
                float ratio = avg_row_rms / expected_output_rms;
                fprintf(stderr, "  [ANOMALY] output_rms=%.4f != expected=%.4f (ratio=%.2fx) - CHECK KERNEL!\n",
                        avg_row_rms, expected_output_rms, ratio);
            } else if (eps_contribution_pct > 5.0f) {
                fprintf(stderr, "  [INFO] Epsilon is %.1f%% of denominator - consider scaling inputs larger (e.g., sqrt(d_model) embedding scale)\n",
                        eps_contribution_pct);
            }
        }
    }
    
    // ISSUE #77 DIAGNOSTIC: Log ln1_out statistics during forward pass
    // This is the activation that gets cached and used for W_qkv weight gradient
    logLn1OutStats(g_issue77_fwd_layer_count % 12, intermediates.ln1_out.data, 
                   total_tokens, d_model, stream);
    
    if (qkv_debug >= 3) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:ln1_out", intermediates.ln1_out, stream, always_log);
        logTensorNonFinite("AutogradQKV:W_qkv", W_qkv_, stream, always_log);
        logTensorNonFinite("AutogradQKV:b_qkv", b_qkv_, stream, always_log);
    }
    
    //--------------------------------------------------
    // 2. QKV Projection: ln1_out @ W_qkv^T + b_qkv
    //    W_qkv is [total_qkv_dim, d_model] so we compute ln1_out @ W_qkv^T
    // Issue #56: Store in intermediates to keep autograd graph alive
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) {
        fprintf(stderr, "[EncoderFwd] Step 2: QKV matmul...\n");
        fprintf(stderr, "[EncoderFwd] Step 2: ln1_out.data=%p shape=[%d,%d] W_qkv.data=%p shape=[%d,%d]\n",
                (void*)intermediates.ln1_out.data, intermediates.ln1_out.shape.flat.rows, intermediates.ln1_out.shape.flat.cols,
                (void*)W_qkv_.data, W_qkv_.shape.flat.rows, W_qkv_.shape.flat.cols);
        fflush(stderr);
    }
    // Use transpose_b=true since W_qkv is [qkv_dim, d_model] and we need [tokens, d_model] @ [d_model, qkv_dim]
    intermediates.qkv_out = autograd::matmul(intermediates.ln1_out, W_qkv_, stream, nullptr, nullptr, true);
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:qkv_out_prebias", intermediates.qkv_out, stream, always_log);
    }
    
    // ========================================================================
    // [QKV_EQUATION] DIAGNOSTIC (Rule 21) - Equation-based logging
    // Formula: qkv_out = ln1_out @ W_qkv^T + b_qkv
    // ========================================================================
    {
        const int qkv_dim_local = config_.d_model + 2 * config_.kvDim();
        const int d_model_local = config_.d_model;
        const int kv_dim_local = config_.kvDim();
        const int head_dim_local = d_model_local / config_.num_heads;
        
        // Sample first N tokens for statistics (to avoid huge GPU->CPU transfers)
        const int n_sample = std::min(64, total_tokens);
        
        // Allocate host buffers
        std::vector<float> h_ln1_sample(static_cast<size_t>(n_sample) * d_model_local);
        std::vector<float> h_wqkv_sample(static_cast<size_t>(qkv_dim_local) * 64);  // First 64 cols of W_qkv
        std::vector<float> h_qkv_sample(static_cast<size_t>(n_sample) * qkv_dim_local);
        std::vector<float> h_bias_sample(qkv_dim_local);
        
        // Copy data from GPU
        cudaMemcpyAsync(h_ln1_sample.data(), intermediates.ln1_out.data,
                        h_ln1_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_wqkv_sample.data(), W_qkv_.data,
                        h_wqkv_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_qkv_sample.data(), intermediates.qkv_out.data,
                        h_qkv_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (b_qkv_.data) {
            cudaMemcpyAsync(h_bias_sample.data(), b_qkv_.data,
                            h_bias_sample.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        
        // Compute ln1_out statistics
        float ln1_min = FLT_MAX, ln1_max = -FLT_MAX;
        double ln1_sum_sq = 0.0;
        double ln1_row_norm_sum = 0.0;
        for (int t = 0; t < n_sample; ++t) {
            double row_sum_sq = 0.0;
            for (int d = 0; d < d_model_local; ++d) {
                float v = h_ln1_sample[t * d_model_local + d];
                ln1_min = fminf(ln1_min, v);
                ln1_max = fmaxf(ln1_max, v);
                ln1_sum_sq += v * v;
                row_sum_sq += v * v;
            }
            ln1_row_norm_sum += sqrtf(static_cast<float>(row_sum_sq));
        }
        float ln1_rms = sqrtf(static_cast<float>(ln1_sum_sq / (n_sample * d_model_local)));
        float ln1_row_norm_mean = static_cast<float>(ln1_row_norm_sum / n_sample);
        
        // Compute W_qkv statistics (first 64 columns)
        float wqkv_min = FLT_MAX, wqkv_max = -FLT_MAX;
        double wqkv_sum_sq = 0.0;
        double wqkv_row_norm_sum = 0.0;
        const int w_sample_cols = 64;
        for (int row = 0; row < qkv_dim_local; ++row) {
            double row_sum_sq = 0.0;
            for (int col = 0; col < w_sample_cols; ++col) {
                float v = h_wqkv_sample[row * w_sample_cols + col];
                wqkv_min = fminf(wqkv_min, v);
                wqkv_max = fmaxf(wqkv_max, v);
                wqkv_sum_sq += v * v;
                row_sum_sq += v * v;
            }
            // Scale row norm to full d_model (assuming similar variance)
            wqkv_row_norm_sum += sqrtf(static_cast<float>(row_sum_sq * d_model_local / w_sample_cols));
        }
        float wqkv_rms = sqrtf(static_cast<float>(wqkv_sum_sq / (qkv_dim_local * w_sample_cols)));
        float wqkv_row_norm_mean = static_cast<float>(wqkv_row_norm_sum / qkv_dim_local);
        
        // Compute qkv_out statistics (Q portion only - first d_model columns)
        float qkv_min = FLT_MAX, qkv_max = -FLT_MAX;
        double qkv_sum_sq = 0.0;
        double qkv_row_norm_sum = 0.0;
        for (int t = 0; t < n_sample; ++t) {
            double row_sum_sq = 0.0;
            for (int d = 0; d < d_model_local; ++d) {  // Q portion only
                float v = h_qkv_sample[t * qkv_dim_local + d];
                qkv_min = fminf(qkv_min, v);
                qkv_max = fmaxf(qkv_max, v);
                qkv_sum_sq += v * v;
                row_sum_sq += v * v;
            }
            qkv_row_norm_sum += sqrtf(static_cast<float>(row_sum_sq));
        }
        float qkv_rms = sqrtf(static_cast<float>(qkv_sum_sq / (n_sample * d_model_local)));
        float qkv_row_norm_mean = static_cast<float>(qkv_row_norm_sum / n_sample);
        
        // Compute bias statistics
        float bias_min = 0.0f, bias_max = 0.0f, bias_rms = 0.0f;
        if (b_qkv_.data) {
            double bias_sum_sq = 0.0;
            for (int i = 0; i < qkv_dim_local; ++i) {
                float v = h_bias_sample[i];
                bias_min = fminf(bias_min, v);
                bias_max = fmaxf(bias_max, v);
                bias_sum_sq += v * v;
            }
            bias_rms = sqrtf(static_cast<float>(bias_sum_sq / qkv_dim_local));
        }
        
        // Compute EXPECTED qkv_out magnitude
        // GEMM: Y = X @ W^T where X is [N, d_model], W is [qkv_dim, d_model]
        // For random orthogonal vectors: E[||Y_row||] = ||X_row|| * ||W_row|| / sqrt(d_model)
        float expected_qkv_row_norm = ln1_row_norm_mean * wqkv_row_norm_mean / sqrtf(static_cast<float>(d_model_local));
        float expected_qkv_elem_rms = expected_qkv_row_norm / sqrtf(static_cast<float>(d_model_local));
        
        // What Q row norm SHOULD be for healthy attention scores
        // Attention score = Q_row · K_row / sqrt(head_dim) should be O(1)
        // So Q_row_norm and K_row_norm should each be ~sqrt(head_dim) = 8.0
        float target_qkv_row_norm = sqrtf(static_cast<float>(head_dim_local));
        
        // Get layer index
        const int layer_idx_local = (g_issue77_fwd_layer_count > 0) 
            ? ((g_issue77_fwd_layer_count - 1) % 12) : 0;
        
        // Print [QKV_EQUATION] diagnostic
        if (isEquationLoggingEnabled()) {
            fprintf(stderr, "\n[QKV_EQUATION] ENCODER_LAYER_%d: qkv_out = ln1_out @ W_qkv^T + b_qkv\n", layer_idx_local);
            fprintf(stderr, "  ln1_out (sample %d tokens): shape=[%d,%d] min=%.10f max=%.10f rms=%.10f\n",
                    n_sample, n_sample, d_model_local, ln1_min, ln1_max, ln1_rms);
            fprintf(stderr, "  ln1_out row_norms: mean=%.10f\n", ln1_row_norm_mean);
            fprintf(stderr, "  W_qkv (sample 64 cols): shape=[%d,%d] min=%.10f max=%.10f rms=%.10f\n",
                    qkv_dim_local, d_model_local, wqkv_min, wqkv_max, wqkv_rms);
            fprintf(stderr, "  W_qkv row_norms (scaled to d_model): mean=%.10f\n", wqkv_row_norm_mean);
            if (b_qkv_.data) {
                fprintf(stderr, "  b_qkv: shape=[%d] min=%.10f max=%.10f rms=%.10f\n",
                        qkv_dim_local, bias_min, bias_max, bias_rms);
            } else {
                fprintf(stderr, "  b_qkv: [nullptr]\n");
            }
            fprintf(stderr, "  EXPECTED qkv_row_norm = ln1_row_norm * wqkv_row_norm / sqrt(d_model)\n");
            fprintf(stderr, "                       = %.4f * %.4f / sqrt(%d) = %.4f\n",
                    ln1_row_norm_mean, wqkv_row_norm_mean, d_model_local, expected_qkv_row_norm);
            fprintf(stderr, "  ACTUAL qkv_out (Q portion): min=%.10f max=%.10f rms=%.10f\n",
                    qkv_min, qkv_max, qkv_rms);
            fprintf(stderr, "  ACTUAL qkv_row_norms (Q portion): mean=%.10f\n", qkv_row_norm_mean);
            fprintf(stderr, "  TARGET qkv_row_norm for healthy attention: ~%.1f (sqrt(head_dim=%d))\n",
                    target_qkv_row_norm, head_dim_local);
            
            // Structured logging via centralized EquationLogger
            EQ_LOG_HOST(
                "QKV_PROJECTION_EQUATION",
                "qkv_out = ln1_out @ W_qkv^T + b_qkv",
                "ln1_rms=" + std::to_string(ln1_rms) + " ln1_row_norm=" + std::to_string(ln1_row_norm_mean) + " wqkv_rms=" + std::to_string(wqkv_rms),
                "qkv_rms=" + std::to_string(qkv_rms) + " qkv_row_norm=" + std::to_string(qkv_row_norm_mean),
                "expected_row_norm=" + std::to_string(expected_qkv_row_norm) + " target=" + std::to_string(target_qkv_row_norm),
                "actual_row_norm=" + std::to_string(qkv_row_norm_mean) + " inflation=" + std::to_string(qkv_row_norm_mean / target_qkv_row_norm),
                0, layer_idx_local, 0, GRIM::EquationPhase::QKV_PROJECTION);
            
            // Anomaly detection
            float inflation_ratio = qkv_row_norm_mean / target_qkv_row_norm;
            if (inflation_ratio > 5.0f) {
                fprintf(stderr, "  *** ANOMALY: qkv_row_norm=%.2f is %.1fx larger than target=%.1f! ***\n",
                        qkv_row_norm_mean, inflation_ratio, target_qkv_row_norm);
                fprintf(stderr, "      This will cause attention score explosion (score ~ %.1f instead of ~1)\n",
                        inflation_ratio * inflation_ratio);
            }
            if (ln1_row_norm_mean > 50.0f) {
                fprintf(stderr, "  *** ANOMALY: ln1_out row_norm=%.2f >> expected ~27.7 (sqrt(d_model)*rms~1) ***\n",
                        ln1_row_norm_mean);
                fprintf(stderr, "      Input to QKV projection is already too large!\n");
            }
            if (wqkv_row_norm_mean > 5.0f) {
                fprintf(stderr, "  *** ANOMALY: W_qkv row_norm=%.2f >> expected ~1.0 (Xavier init) ***\n",
                        wqkv_row_norm_mean);
                fprintf(stderr, "      Weight magnitudes are too large!\n");
            }
            fprintf(stderr, "\n");
        }
    }
    
    // ISSUE #93 DIAGNOSTIC: Compare actual QKV output vs expected GEMM math
    // This helps identify if the explosion is in cuBLAS, weights, or input
    // NOTE: layer_idx is (g_issue77_fwd_layer_count - 1) % 12 because logLn1OutStats
    // already incremented the counter for this layer earlier in forward()
    const int qkv_dim = config_.d_model + 2 * config_.kvDim();
    const int actual_layer_idx = (g_issue77_fwd_layer_count > 0) 
        ? ((g_issue77_fwd_layer_count - 1) % 12)  // Correct for early increment
        : 0;
    DebugQKVExpectations(
        actual_layer_idx,
        intermediates.ln1_out.data,       // input: [total_tokens, d_model]
        W_qkv_.data,                      // weight: [qkv_dim, d_model]
        intermediates.qkv_out.data,       // output: [total_tokens, qkv_dim] PRE-BIAS
        b_qkv_.data,                      // bias: [qkv_dim] (can be nullptr)
        total_tokens,
        config_.d_model,
        qkv_dim,
        config_.kvDim(),                  // kv_dim for Q/K/V split
        stream
    );
    
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: QKV matmul DONE, adding bias...\n");
    // ISSUE #97 FIX: Use autograd::broadcast_add for proper gradient tracking
    // Previously: launchFFNBiasAdd bypassed autograd, so b_qkv never received gradients
    // Now: autograd::broadcast_add creates BiasAddGradFn which computes grad_bias = sum(grad_output)
    intermediates.qkv_out = autograd::broadcast_add(intermediates.qkv_out, b_qkv_, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 2: QKV bias DONE\n");
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:qkv_out", intermediates.qkv_out, stream, always_log);
    }
    
    //--------------------------------------------------
    // 3. Split QKV and reshape to BHSD for attention
    //    qkv_out is [total_tokens, d_model + 2*kv_dim]
    //    Q: [total_tokens, 0:d_model]
    //    K: [total_tokens, d_model:d_model+kv_dim]  
    //    V: [total_tokens, d_model+kv_dim:end]
     //
    // ISSUE #61 FIX: Use autograd::split_and_reshape_qkv() to maintain gradient chain
    // Previous code used Tensor::empty() + cudaMemcpy2D which broke autograd (Q/K/V had no grad_fn)
    // This caused W_qkv gradients to be ZERO since ScaledDotProductAttentionGradFn couldn't
    // continue the chain through Q/K/V with null grad_fn.
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3: Split QKV with autograd tracking...\n");
    
    // ISSUE #61: This properly tracks gradients through the split/reshape operation
    auto [Q_bhsd_tmp, K_bhsd_tmp, V_bhsd_tmp] = autograd::split_and_reshape_qkv(
        intermediates.qkv_out,
        batch_size, seq_len, num_heads, num_kv_heads, head_dim,
        stream);
    intermediates.Q_bhsd = std::move(Q_bhsd_tmp);
    intermediates.K_bhsd = std::move(K_bhsd_tmp);
    intermediates.V_bhsd = std::move(V_bhsd_tmp);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3: Split QKV DONE (autograd tracked)\n");
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradQKV:Q_bhsd", intermediates.Q_bhsd, stream, always_log);
        logTensorNonFinite("AutogradQKV:K_bhsd", intermediates.K_bhsd, stream, always_log);
        logTensorNonFinite("AutogradQKV:V_bhsd", intermediates.V_bhsd, stream, always_log);
    }
    
    //--------------------------------------------------
    // 3b. Apply RoPE rotation to Q and K
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3c: RoPE rotation...\n");
    if (config_.pos_encoding && config_.pos_encoding->valid && 
        config_.pos_encoding->rope_inv_freq != nullptr && config_.pos_encoding->rotary_dim > 0) {
        // ISSUE #119 FIX: Use autograd::rope_rotation() instead of raw kernel call
        // The raw PBM::launchRoPERotationGQA() bypassed autograd - dQ/dK gradients
        // were never inverse-rotated in backward pass, causing gradient corruption.
        autograd::rope_rotation(
            intermediates.Q_bhsd, intermediates.K_bhsd,  // Tensor refs, not .data
            config_.pos_encoding->rope_inv_freq,
            batch_size, num_heads, num_kv_heads, seq_len, head_dim,
            config_.pos_encoding->rotary_dim, stream);
    } else {
        throw std::runtime_error("EncodingLayer::forward: RoPE not initialized");
    }
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 3c: RoPE DONE\n");
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradSDPA:Q_rope", intermediates.Q_bhsd, stream, always_log);
        logTensorNonFinite("AutogradSDPA:K_rope", intermediates.K_bhsd, stream, always_log);
        logTensorNonFinite("AutogradSDPA:V_rope", intermediates.V_bhsd, stream, always_log);
    }
    
    //--------------------------------------------------
    // 4. Flash Attention: Q, K, V -> attn_out
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 4: Flash Attention...\n");
    
    // RULE 20: Fail loud if pos_encoding is NULL - GRIM requires hybrid PBM
    if (!config_.pos_encoding) {
        throw std::runtime_error(
            "EncodingLayer::forward: config_.pos_encoding is NULL - "
            "GRIM requires PBM (ALiBi+RoPE) for positional encoding");
    }
    if (!config_.pos_encoding->alibi_slopes) {
        throw std::runtime_error(
            "EncodingLayer::forward: config_.pos_encoding->alibi_slopes is NULL - "
            "PBM ALiBi slopes not initialized");
    }
    
    // Issue #56: Store attention output in intermediates
    intermediates.attn_out_bhsd = autograd::scaled_dot_product_attention(
        intermediates.Q_bhsd, intermediates.K_bhsd, intermediates.V_bhsd, 
        config_.pos_encoding->alibi_slopes, 0.0f, stream, nullptr);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 4: Flash Attention DONE\n");
    if (qkv_debug > 0) {
        const bool always_log = (qkv_debug >= 2);
        logTensorNonFinite("AutogradSDPA:attn_out_bhsd", intermediates.attn_out_bhsd, stream, always_log);
    }
    
    // ISSUE #93 DIAGNOSTIC: Log Flash Attention output
    // This tells us if attention itself produces large values
    {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        logForwardStageStats("flash_attn_out", layer_idx, intermediates.attn_out_bhsd.data, batch_size * num_heads * seq_len * head_dim, stream);
    }
    
    //--------------------------------------------------
    // 5. Reshape attention output: BHSD -> [tokens, d_model]
    // ISSUE #62 FIX: Use autograd::reshape_bhsd_to_flat() to maintain gradient chain
    // Previous code used Tensor::empty() + launchReshapeFromBHSD which broke autograd
    // (attn_out had no grad_fn, causing W_o gradients to not flow through attention backward)
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 5: Reshape from BHSD with autograd tracking...\n");
    intermediates.attn_out = autograd::reshape_bhsd_to_flat(
        intermediates.attn_out_bhsd, batch_size, seq_len, num_heads, head_dim, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 5: Reshape DONE (autograd tracked)\n");
    
    //--------------------------------------------------
    // 6. Output projection: attn_out @ W_o^T + b_o
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 6: Output projection...\n");
    // W_o is [d_model, d_model], so W_o^T is also [d_model, d_model]
    // Use transpose_b=true to compute attn_out @ W_o^T
    intermediates.proj_out = autograd::matmul(intermediates.attn_out, W_o_, stream, nullptr, nullptr, true);
    // ISSUE #97 FIX: Use autograd::broadcast_add for proper gradient tracking on b_o
    intermediates.proj_out = autograd::broadcast_add(intermediates.proj_out, b_o_, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 6: Output projection DONE\n");
    
    // ISSUE #93 DIAGNOSTIC: Log attention-related intermediate values
    // This helps trace WHERE the explosion originates
    {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        logForwardStageStats("attn_out_flat", layer_idx, intermediates.attn_out.data, total_tokens * d_model, stream);
        logForwardStageStats("proj_out_W_o", layer_idx, intermediates.proj_out.data, total_tokens * d_model, stream);
    }
    
    //--------------------------------------------------
    // 7. Residual1: input + proj_out -> residual1
    // Issue #56: Store in intermediates
    // Issue #109: Apply LayerScale to proj_out before residual addition
    // Issue #118: Apply centering to remove common direction before residual add
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1...\n");
    
    // Issue #109: LayerScale gating for attention sublayer
    Tensor scaled_proj;
    const Tensor& proj_for_residual = (config_.use_layer_scale && layer_scale1_.data)
        ? (scaled_proj = autograd::layer_scale(intermediates.proj_out, layer_scale1_, stream), scaled_proj)
        : intermediates.proj_out;
    
    // ========================================================================
    // ISSUE #126 FIX: Center the COMBINED residual output, not just the layer contribution!
    //
    // OLD (Issue #118 - WRONG):
    //   centered_attn = center_columns(proj_for_residual)
    //   residual1 = add(input, centered_attn)
    //   Problem: `input` still carries correlation from previous layers!
    //   Result: mean_t(residual1[:,d]) = mean_t(input[:,d]) ≠ 0 → correlation preserved!
    //
    // NEW (Issue #126 - CORRECT):
    //   raw_residual1 = add(input, proj_for_residual)
    //   residual1 = center_columns(raw_residual1)
    //   Result: mean_t(residual1[:,d]) = 0 → accumulated correlation REMOVED!
    //
    // Math proof:
    //   center(A + B) = A + B - mean(A+B) = center(A) + center(B)
    //   If we only center B (layer output), the mean(A) term stays in the result!
    //   By centering the combined output, we remove mean(A) + mean(B) = mean(A+B).
    // ========================================================================
    Tensor raw_residual1 = autograd::add(input, proj_for_residual, stream);
    intermediates.residual1 = autograd::center_columns(raw_residual1, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 7: Residual1 DONE\n");
    
    // ISSUE #93 DIAGNOSTIC: Log residual1 = input + proj_out
    {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        logForwardStageStats("residual1", layer_idx, intermediates.residual1.data, total_tokens * d_model, stream);
    }
    
    //--------------------------------------------------
    // 8. RMSNorm2: residual1 -> ln2_out
    // Issue #56: Store in intermediates
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2...\n");
    intermediates.ln2_out = autograd::rms_norm(intermediates.residual1, rms2_gamma_, config_.rms_epsilon, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 8: RMSNorm2 DONE\n");
    
    //--------------------------------------------------
    // 9. FFN: ln2_out -> ffn_out (already using autograd)
    // Issue #56: FFN also stores its intermediates in this same ForwardIntermediates
    // (ffn_linear1_out, ffn_gelu_out are written by FFN forward)
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN...\n");
    intermediates.ffn_out = ffn_->forward(intermediates.ln2_out, intermediates);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 9: FFN DONE\n");
    
    // ISSUE #93 DIAGNOSTIC: Log FFN output  
    {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        logForwardStageStats("ffn_out", layer_idx, intermediates.ffn_out.data, total_tokens * d_model, stream);
    }
    
    //--------------------------------------------------
    // 10. Residual2: residual1 + ffn_out -> output
    // Issue #56: The final output IS stored in intermediates too
    // for consistency, but we also return it
    // Issue #109: Apply LayerScale to ffn_out before residual addition
    // Issue #118: Apply centering to remove common direction before residual add
    //--------------------------------------------------
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2...\n");
    
    // Issue #109: LayerScale gating for FFN sublayer
    Tensor scaled_ffn;
    const Tensor& ffn_for_residual = (config_.use_layer_scale && layer_scale2_.data)
        ? (scaled_ffn = autograd::layer_scale(intermediates.ffn_out, layer_scale2_, stream), scaled_ffn)
        : intermediates.ffn_out;
    
    // ========================================================================
    // ISSUE #126 FIX: Center the COMBINED residual output, not just the layer contribution!
    //
    // OLD (Issue #118 - WRONG):
    //   centered_ffn = center_columns(ffn_for_residual)
    //   output = add(residual1, centered_ffn)
    //   Problem: `residual1` still carries correlation (RMSNorm can shift angles)!
    //
    // NEW (Issue #126 - CORRECT):
    //   raw_output = add(residual1, ffn_for_residual)
    //   output = center_columns(raw_output)
    //   Result: mean_t(output[:,d]) = 0 → accumulated correlation REMOVED!
    // ========================================================================
    Tensor raw_output = autograd::add(intermediates.residual1, ffn_for_residual, stream);
    intermediates.output = autograd::center_columns(raw_output, stream);
    if constexpr (kEnableEncoderStepLogs) fprintf(stderr, "[EncoderFwd] Step 10: Residual2 DONE - layer COMPLETE\n");
    
    // ISSUE #93 DIAGNOSTIC: Log final layer output
    {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        logForwardStageStats("layer_output", layer_idx, intermediates.output.data, total_tokens * d_model, stream);
    };
    
    // ═══════════════════════════════════════════════════════════════════════════
    // RULE 21 DIAGNOSTIC: Per-Layer Cosine Similarity (correlation tracking)
    //
    // EQUATION: output = residual1 + LayerScale(ffn_out) = input + LS1*attn_out + LS2*ffn_out
    //           residual_fraction = (1 - LS_init)^2 ≈ (1 - 0.1)^2 = 0.81 (81% residual passthrough!)
    //           
    // LayerScale with init=0.1 means: output ≈ 0.9*input + 0.1*sublayer_output
    // After 2 residual connections: output ≈ 0.81*input + 0.19*layers
    // This PRESERVES input correlation through layers when LS_init is small!
    //
    // We track: avg_cos(h_i, h_j) for layer output
    //           delta_cos = (layer_cos - input_cos) to see if attention DIVERSIFIES
    // ═══════════════════════════════════════════════════════════════════════════
    {
        const int layer_idx = g_issue77_fwd_layer_count % 12;
        
        // Only log every 4th layer and first/last layers to reduce overhead
        if (layer_idx == 0 || layer_idx == 5 || layer_idx == 11) {
            cudaStreamSynchronize(stream);  // Ensure data is ready
            
            // Copy layer output to host for analysis
            const int output_size = total_tokens * d_model;
            std::vector<float> h_output(output_size);
            cudaMemcpy(h_output.data(), intermediates.output.data, output_size * sizeof(float), cudaMemcpyDeviceToHost);
            
            // Compute row norms
            std::vector<float> row_norms(total_tokens);
            for (int t = 0; t < total_tokens; t++) {
                double norm_sq = 0.0;
                for (int d = 0; d < d_model; d++) {
                    float v = h_output[t * d_model + d];
                    norm_sq += v * v;
                }
                row_norms[t] = sqrtf(norm_sq);
            }
            
            // Sample pairwise cosine similarity
            const int sample_pairs = std::min(30, total_tokens / 2);
            const int stride = std::max(1, total_tokens / sample_pairs);
            double cos_sum = 0.0;
            int num_pairs = 0;
            
            for (int i = 0; i < total_tokens && num_pairs < sample_pairs; i += stride) {
                int j = (i + total_tokens / 2) % total_tokens;
                if (i == j || row_norms[i] < 1e-8f || row_norms[j] < 1e-8f) continue;
                
                double dot = 0.0;
                for (int d = 0; d < d_model; d++) {
                    dot += h_output[i * d_model + d] * h_output[j * d_model + d];
                }
                cos_sum += dot / (row_norms[i] * row_norms[j]);
                num_pairs++;
            }
            
            const double avg_cos = (num_pairs > 0) ? cos_sum / num_pairs : 0.0;
            const float norm_min = *std::min_element(row_norms.begin(), row_norms.end());
            const float norm_max = *std::max_element(row_norms.begin(), row_norms.end());
            
            // Log LayerScale values if available
            float ls1_val = 0.0f, ls2_val = 0.0f;
            if (config_.use_layer_scale && layer_scale1_.data) {
                cudaMemcpy(&ls1_val, layer_scale1_.data, sizeof(float), cudaMemcpyDeviceToHost);
            }
            if (config_.use_layer_scale && layer_scale2_.data) {
                cudaMemcpy(&ls2_val, layer_scale2_.data, sizeof(float), cudaMemcpyDeviceToHost);
            }
            
            // Compute expected residual fraction: (1-ls1)*(1-ls2) = fraction of input that passes through
            const double residual_fraction = (1.0 - ls1_val) * (1.0 - ls2_val);
            
            if (isEquationLoggingEnabled()) {
                fprintf(stderr, "[LAYER_%d_COSINE_EQUATION] output = input + LS1*attn + LS2*ffn\n", layer_idx);
                fprintf(stderr, "  OUTPUT h_L%d: shape=[%d, %d] row_norm_range=[%.4f, %.4f]\n",
                        layer_idx, total_tokens, d_model, norm_min, norm_max);
                fprintf(stderr, "  LAYERSCALE: LS1=%.4f LS2=%.4f -> residual_fraction=(1-LS1)*(1-LS2)=%.4f (%.1f%% passthrough)\n",
                        ls1_val, ls2_val, residual_fraction, residual_fraction * 100.0);
                fprintf(stderr, "  ACTUAL avg_cos=%.6f (pairs=%d)\n", avg_cos, num_pairs);
                
                // Structured logging via centralized EquationLogger
                EQ_LOG_HOST(
                    "LAYER_COSINE_EQUATION",
                    "output = input + LS1*attn + LS2*ffn",
                    "LS1=" + std::to_string(ls1_val) + " LS2=" + std::to_string(ls2_val),
                    "residual_frac=" + std::to_string(residual_fraction) + " passthrough=" + std::to_string(residual_fraction * 100.0) + "%",
                    "expected_cos<0.5",
                    "actual_cos=" + std::to_string(avg_cos) + " pairs=" + std::to_string(num_pairs),
                    0, layer_idx, 0, GRIM::EquationPhase::RESIDUAL_ADD);
                
                if (avg_cos > 0.8) {
                    fprintf(stderr, "  [ANOMALY] Layer %d avg_cos=%.4f still HIGH after attention!\n", layer_idx, avg_cos);
                    if (residual_fraction > 0.7) {
                        fprintf(stderr, "  [ANOMALY] LayerScale residual_fraction=%.2f means %.0f%% of input passes through unchanged\n",
                                residual_fraction, residual_fraction * 100.0);
                        fprintf(stderr, "  [ANOMALY] High input correlation propagates because LS_init=%.2f is too small\n", ls1_val);
                        fprintf(stderr, "  [ANOMALY] RECOMMENDED: Increase layer_scale_init from %.2f to 0.5+ or add positional signal to residual\n", ls1_val);
                    }
                }
            }
        }
    }
    
    // Return a non-owning view of the output
    // The actual Tensor lives in intermediates and stays alive until backward completes
    Tensor result = Tensor::from_ptr(
        intermediates.output.data,
        intermediates.output.shape,
        false,  // doesn't own data - intermediates.output owns it
        true    // requires_grad
    );
    result.is_leaf = false;
    result.grad_fn = intermediates.output.grad_fn;
    result.owns_grad_fn = false;  // Borrowed, intermediates.output owns it
    result.stream = stream;
    
    return result;
}

//======================================================//
// Issue #37 DIAGNOSTIC: Public wrappers for alignment tracking
//======================================================//
void setEncoderW277Reference(const float* lm_weights, int vocab_size, int d_model, cudaStream_t stream) {
    setW277Reference(lm_weights, vocab_size, d_model, stream);
}

void resetEncoderDiagCount() {
    resetLayerDiagCount();
}

} // namespace GRIM

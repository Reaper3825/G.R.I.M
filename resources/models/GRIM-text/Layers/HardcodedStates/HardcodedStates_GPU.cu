#include "HardcodedStates_GPU.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>  // for std::partial_sort
#include <curand_kernel.h>

namespace GRIM {

// ═════════════════════════════════════════════════════════════════════════════
// CUDA Kernels for Pattern Generation
// ═════════════════════════════════════════════════════════════════════════════

/**
 * Generate random normal distribution with mean=0, stddev=1/sqrt(d_model)
 * Tests Issue #37 (hidden state centering) - should remain centered after generation
 */
__global__ void generateRandomCenteredKernel(
    float* output,           // [total_tokens, d_model]
    int total_tokens,
    int d_model,
    unsigned long long seed
) {
    const int token_idx = blockIdx.x;
    const int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= total_tokens || dim_idx >= d_model) return;
    
    const int global_idx = token_idx * d_model + dim_idx;
    
    // Initialize cuRAND state
    curandState state;
    curand_init(seed, global_idx, 0, &state);
    
    // Generate random normal with zero mean, variance = 1/d_model
    const float stddev = rsqrtf(static_cast<float>(d_model));
    float value = curand_normal(&state) * stddev;
    
    output[global_idx] = value;
}

/**
 * Center the generated random values to ensure exact zero mean
 */
__global__ void centerHiddenStatesKernel(
    float* hidden_states,    // [total_tokens, d_model] - will be modified in-place
    int total_tokens,
    int d_model
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    float* token_hidden = hidden_states + token_idx * d_model;
    
    // Compute mean
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();
    
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_sum += token_hidden[i];
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Block-level accumulation
    if ((threadIdx.x % 32) == 0) {
        atomicAdd(&s_sum, local_sum);
    }
    __syncthreads();
    
    const float mean = s_sum / static_cast<float>(d_model);
    
    // Subtract mean
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        token_hidden[i] -= mean;
    }
}

/**
 * Generate pattern orthogonal to W[277]
 * Expected: logit[277] ≈ 0 (dot product of orthogonal vectors is zero)
 */
__global__ void generateOrthogonalW277Kernel(
    float* output,              // [total_tokens, d_model]
    const float* w277,          // [d_model] - weight row for token 277
    int total_tokens,
    int d_model,
    unsigned long long seed
) {
    const int token_idx = blockIdx.x;
    const int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= total_tokens || dim_idx >= d_model) return;
    
    const int global_idx = token_idx * d_model + dim_idx;
    
    // Generate random vector
    curandState state;
    curand_init(seed, global_idx, 0, &state);
    float random_val = curand_normal(&state);
    
    // Gram-Schmidt orthogonalization: v_ortho = v - (v·w)/(w·w) * w
    // We'll do this in two passes: first generate random, then orthogonalize
    output[global_idx] = random_val;
}

/**
 * Second pass: Orthogonalize against W[277]
 */
__global__ void orthogonalizeKernel(
    float* hidden_states,       // [total_tokens, d_model]
    const float* w277,          // [d_model]
    int total_tokens,
    int d_model
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;
    
    float* token_hidden = hidden_states + token_idx * d_model;
    
    // Compute dot products: h·w and w·w
    __shared__ float s_dot_hw;
    __shared__ float s_dot_ww;
    if (threadIdx.x == 0) {
        s_dot_hw = 0.0f;
        s_dot_ww = 0.0f;
    }
    __syncthreads();
    
    float local_hw = 0.0f;
    float local_ww = 0.0f;
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        local_hw += token_hidden[i] * w277[i];
        local_ww += w277[i] * w277[i];
    }
    
    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        local_hw += __shfl_down_sync(0xffffffff, local_hw, offset);
        local_ww += __shfl_down_sync(0xffffffff, local_ww, offset);
    }
    
    if ((threadIdx.x % 32) == 0) {
        atomicAdd(&s_dot_hw, local_hw);
        atomicAdd(&s_dot_ww, local_ww);
    }
    __syncthreads();
    
    // Orthogonalize: h' = h - (h·w)/(w·w) * w
    const float scale = s_dot_hw / (s_dot_ww + 1e-8f);
    for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
        token_hidden[i] -= scale * w277[i];
    }
}

/**
 * Generate pattern aligned with W[277]
 * Expected: logit[277] >> other logits (high confidence SPACE prediction)
 */
__global__ void generateAlignedW277Kernel(
    float* output,              // [total_tokens, d_model]
    const float* w277,          // [d_model]
    int total_tokens,
    int d_model
) {
    const int token_idx = blockIdx.x;
    const int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= total_tokens || dim_idx >= d_model) return;
    
    const int global_idx = token_idx * d_model + dim_idx;
    
    // Normalize W[277] to unit length
    // We'll do this on CPU and pass normalized version, but for now just copy
    output[global_idx] = w277[dim_idx];
}

/**
 * Generate constant uniform pattern: all dimensions = 1/sqrt(d_model)
 * Tests Issue #40: logit[v] = sum(W[v,:]) / sqrt(d_model)
 * If any token has systematically higher row sum, it will have higher logit
 */
__global__ void generateConstantUniformKernel(
    float* output,              // [total_tokens, d_model]
    int total_tokens,
    int d_model
) {
    const int token_idx = blockIdx.x;
    const int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= total_tokens || dim_idx >= d_model) return;
    
    const int global_idx = token_idx * d_model + dim_idx;
    const float constant_val = rsqrtf(static_cast<float>(d_model));
    
    output[global_idx] = constant_val;
}

/**
 * Generate zero-mean sine wave pattern
 * Tests centering robustness with structured non-random data
 */
__global__ void generateZeroMeanSineKernel(
    float* output,              // [total_tokens, d_model]
    int total_tokens,
    int d_model,
    int batch_idx               // For phase variation
) {
    const int token_idx = blockIdx.x;
    const int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_idx >= total_tokens || dim_idx >= d_model) return;
    
    const int global_idx = token_idx * d_model + dim_idx;
    
    // Sine wave with frequency based on dimension index
    const float frequency = 2.0f * 3.14159f * (dim_idx + 1) / d_model;
    const float phase = batch_idx * 0.1f;
    const float amplitude = rsqrtf(static_cast<float>(d_model));
    
    output[global_idx] = amplitude * sinf(frequency * token_idx + phase);
}

// ═════════════════════════════════════════════════════════════════════════════
// Host Functions
// ═════════════════════════════════════════════════════════════════════════════

void generateHardcodedStates(
    float* output,
    const float* lm_head_weights,
    HardcodedPattern pattern,
    int total_tokens,
    int d_model,
    int vocab_size,
    int batch_idx,
    cudaStream_t stream
) {
    if (pattern == HardcodedPattern::DISABLED) {
        return; // No-op
    }
    
    const unsigned long long seed = 42 + batch_idx;  // Reproducible per batch
    
    // Compute grid dimensions for 2D launch (tokens x dimensions)
    dim3 grid_2d(total_tokens, (d_model + 255) / 256);
    dim3 block_2d(256);
    
    // Compute W[SPACE] pointer (SPACE = first unigram token = UNIGRAM_VOCAB_OFFSET)
    const int kSpaceToken = Tokenizer::UNIGRAM_VOCAB_OFFSET;
    const float* w277 = nullptr;
    if (lm_head_weights && kSpaceToken < vocab_size) {
        w277 = lm_head_weights + static_cast<size_t>(kSpaceToken) * d_model;
    }
    
    switch (pattern) {
        case HardcodedPattern::RANDOM_CENTERED:
            generateRandomCenteredKernel<<<grid_2d, block_2d, 0, stream>>>(
                output, total_tokens, d_model, seed);
            // Ensure exact zero mean
            centerHiddenStatesKernel<<<total_tokens, 256, 0, stream>>>(
                output, total_tokens, d_model);
            break;
            
        case HardcodedPattern::ORTHOGONAL_W277:
            if (w277) {
                generateOrthogonalW277Kernel<<<grid_2d, block_2d, 0, stream>>>(
                    output, w277, total_tokens, d_model, seed);
                orthogonalizeKernel<<<total_tokens, 256, 0, stream>>>(
                    output, w277, total_tokens, d_model);
                centerHiddenStatesKernel<<<total_tokens, 256, 0, stream>>>(
                    output, total_tokens, d_model);
            }
            break;
            
        case HardcodedPattern::ALIGNED_W277:
            if (w277) {
                generateAlignedW277Kernel<<<grid_2d, block_2d, 0, stream>>>(
                    output, w277, total_tokens, d_model);
            }
            break;
            
        case HardcodedPattern::CONSTANT_UNIFORM:
            generateConstantUniformKernel<<<grid_2d, block_2d, 0, stream>>>(
                output, total_tokens, d_model);
            break;
            
        case HardcodedPattern::ZERO_MEAN_SINE:
            generateZeroMeanSineKernel<<<grid_2d, block_2d, 0, stream>>>(
                output, total_tokens, d_model, batch_idx);
            break;
            
        default:
            break;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[HardcodedStates] CUDA error: %s\n", cudaGetErrorString(err));
    }
}

void logHardcodedStateDiagnostics(
    const float* hidden_states,
    const float* lm_head_weights,
    const float* logits,
    HardcodedPattern pattern,
    int total_tokens,
    int d_model,
    int vocab_size,
    int batch_idx,
    cudaStream_t stream
) {
    constexpr int kToken277 = 277;
    
    // Copy first token's hidden state to host
    std::vector<float> h_sample(d_model);
    cudaMemcpyAsync(h_sample.data(), hidden_states, d_model * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    
    // Copy W[277] to host
    std::vector<float> h_w277(d_model);
    if (lm_head_weights && kToken277 < vocab_size) {
        const float* w277 = lm_head_weights + static_cast<size_t>(kToken277) * d_model;
        cudaMemcpyAsync(h_w277.data(), w277, d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }
    
    // Copy first token's logits to host
    std::vector<float> h_logits(vocab_size);
    cudaMemcpyAsync(h_logits.data(), logits, vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    
    cudaStreamSynchronize(stream);
    
    // Compute statistics
    double h_mean = 0.0, h_var = 0.0, h_rms_sq = 0.0;
    double w_rms_sq = 0.0, dot_hw = 0.0;
    
    for (int i = 0; i < d_model; ++i) {
        h_mean += h_sample[i];
        h_rms_sq += h_sample[i] * h_sample[i];
        w_rms_sq += h_w277[i] * h_w277[i];
        dot_hw += h_sample[i] * h_w277[i];
    }
    h_mean /= d_model;
    double h_rms = std::sqrt(h_rms_sq / d_model);
    double w_rms = std::sqrt(w_rms_sq / d_model);
    
    for (int i = 0; i < d_model; ++i) {
        double diff = h_sample[i] - h_mean;
        h_var += diff * diff;
    }
    h_var /= d_model;
    
    const double cosine_hw = dot_hw / (h_rms * w_rms * d_model + 1e-8);
    
    // Find top-5 logits
    std::vector<std::pair<float, int>> logit_pairs;
    for (int v = 0; v < vocab_size; ++v) {
        logit_pairs.emplace_back(h_logits[v], v);
    }
    std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + 5, logit_pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    const float logit_277 = (kToken277 < vocab_size) ? h_logits[kToken277] : 0.0f;
    
    // Pattern name
    const char* pattern_names[] = {
        "DISABLED", "random_centered", "orthogonal_w277", "aligned_w277",
        "constant_uniform", "zero_mean_sine"
    };
    const char* pattern_name = pattern_names[static_cast<int>(pattern)];
    
    fprintf(stderr, "\n");
    fprintf(stderr, "╔═══════════════════════════════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║ HARDCODED HIDDEN STATE DIAGNOSTIC (Issue #42)                             ║\n");
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ Batch: %d | Pattern: %-25s                         ║\n", batch_idx, pattern_name);
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ Hidden State Statistics (first token):                                    ║\n");
    fprintf(stderr, "║   Mean:     %+.6f (should be ~0 for centered patterns)                ║\n", h_mean);
    fprintf(stderr, "║   Variance: %.6f  (should be ~1/%d = %.6f)                      ║\n", 
            h_var, d_model, 1.0 / d_model);
    fprintf(stderr, "║   RMS:      %.6f                                                        ║\n", h_rms);
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ Alignment with W[277]:                                                    ║\n");
    fprintf(stderr, "║   h·W[277]:     %+.6f                                                    ║\n", dot_hw);
    fprintf(stderr, "║   rms(W[277]):  %.6f                                                      ║\n", w_rms);
    fprintf(stderr, "║   cosine(h,W):  %+.6f (orthogonal≈0, aligned≈1)                       ║\n", cosine_hw);
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ Resulting Logits:                                                         ║\n");
    fprintf(stderr, "║   logit[277]:   %+.6f (SPACE token)                                     ║\n", logit_277);
    fprintf(stderr, "║   Top-5 predictions:                                                      ║\n");
    for (int i = 0; i < 5; ++i) {
        const bool is_277 = (logit_pairs[i].second == kToken277);
        fprintf(stderr, "║     %d. Token %5d: %+.6f %s                                     ║\n",
                i + 1, logit_pairs[i].second, logit_pairs[i].first,
                is_277 ? "<-- SPACE!" : "");
    }
    fprintf(stderr, "╠═══════════════════════════════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║ Expected Behavior:                                                        ║\n");
    
    switch (pattern) {
        case HardcodedPattern::RANDOM_CENTERED:
            fprintf(stderr, "║   - Mean should be exactly 0.0 (centered)                                  ║\n");
            fprintf(stderr, "║   - Logits should be roughly uniform (no strong prediction)                ║\n");
            fprintf(stderr, "║   - If collapse still occurs → gradient/optimizer bug                      ║\n");
            break;
        case HardcodedPattern::ORTHOGONAL_W277:
            fprintf(stderr, "║   - cosine(h,W) should be ~0.0 (orthogonal)                                ║\n");
            fprintf(stderr, "║   - logit[277] should be ~0.0 (dot product = 0)                            ║\n");
            fprintf(stderr, "║   - If logit[277] is high → LM head computation bug                        ║\n");
            break;
        case HardcodedPattern::ALIGNED_W277:
            fprintf(stderr, "║   - cosine(h,W) should be ~1.0 (aligned)                                   ║\n");
            fprintf(stderr, "║   - logit[277] should be HIGHEST (strong SPACE prediction)                 ║\n");
            fprintf(stderr, "║   - This simulates encoder outputting W[277]-aligned states                ║\n");
            break;
        case HardcodedPattern::CONSTANT_UNIFORM:
            fprintf(stderr, "║   - logit[v] = sum(W[v,:]) / sqrt(d_model)                                 ║\n");
            fprintf(stderr, "║   - If logit[277] is systematically higher → Issue #40 row sum bias       ║\n");
            break;
        case HardcodedPattern::ZERO_MEAN_SINE:
            fprintf(stderr, "║   - Mean should be ~0.0 (sine wave is symmetric)                           ║\n");
            fprintf(stderr, "║   - Tests centering robustness with structured data                        ║\n");
            break;
        default:
            break;
    }
    
    fprintf(stderr, "╚═══════════════════════════════════════════════════════════════════════════╝\n");
    fprintf(stderr, "\n");
}

} // namespace GRIM

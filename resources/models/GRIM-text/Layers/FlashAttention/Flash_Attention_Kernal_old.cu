// Flash_Attention_Kernal.cu - Flash Attention 2 Forward + Backward Implementation
// Based on "Flash Attention 2: Faster Attention with Better Parallelism and Work Partitioning"

// DEBUG: Disabled - use runtime ctx.enable_grad_checks instead
// #define FLASH_ATTN_DEBUG 1

//
// DESIGN DECISIONS:
//   - Fixed 32×32 block sizes for both forward and backward (fits 48KB shared memory)
//   - Statistics (row_max, row_sum) in SHARED memory, not thread-local
//   - HEAD_DIM as template parameter (32 or 64) - MUST match exactly, no fallback
//   - 256 threads per block
//   - GQA divisibility enforced: num_heads % num_kv_heads == 0
//
// Shared memory budget at 32×32×64:
//   Q_smem[32][64]     = 8192 bytes
//   K_smem[32][64]     = 8192 bytes
//   V_smem[32][64]     = 8192 bytes
//   S_smem[32][32]     = 4096 bytes
//   row_max[32]        = 128 bytes
//   row_sum[32]        = 128 bytes
//   O_smem[32][64]     = 8192 bytes (forward only)
//   dO_smem[32][64]    = 8192 bytes (backward only)
//   dQ_smem[32][64]    = 8192 bytes (backward only)
//   dp_sum[32]         = 128 bytes  (backward only)
//   Forward total:  ~37KB ✓
//   Backward total: ~45KB ✓ (max occupancy: 1-2 blocks/SM)
//
// KNOWN OPTIMIZATION OPPORTUNITIES (priority order):
//   1. Q·K recomputation (4x across passes) - PRIMARY TARGET if compute-bound
//      • Fuse row_max + row_sum passes
//      • Cache scores in registers/shared memory
//      • Warp-level primitives for reduction
//   2. AtomicAdd for grad_K/grad_V - non-deterministic, order-dependent
//      • Use deterministic accumulation (sort by warp ID)
//      • Or accept non-determinism for speed
//   3. Debug printf inside kernel - inhibits compiler optimization
//      • Wrap in #ifdef FLASH_DEBUG
//      • Remove from production builds
//   4. Shared memory pressure at 45KB - limits occupancy
//      • Any new feature (dropout, FP16) will exceed 48KB limit
//      • Consider register tiling for smaller blocks
//   5. Alpha gradient accumulation per-block - scale depends on tiling
//      • Currently correct but non-obvious
//      • Document scaling behavior
//   6. You scale grad_alpha_k by 1 / heads_per_kv_group
// You do not scale grad_alpha_q


#include "Flash_Attention_Kernal.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <cmath>
#include <cassert>
#include <numeric>  // For std::accumulate

namespace GRIM {

// Use centralized constants from HyperParameters
constexpr int FLASH_BLOCK_Q = HyperParameters::FLASH_ATTN_BLOCK_Q;
constexpr int FLASH_BLOCK_KV = HyperParameters::FLASH_ATTN_BLOCK_KV;
constexpr int FLASH_NUM_THREADS = HyperParameters::FLASH_ATTN_NUM_THREADS;

// ============================================================================
// PROPER FLOAT ATOMICS - CAS loop for correct float comparison
// ============================================================================
// CRITICAL: atomicMax(reinterpret_cast<int*>) is WRONG for negative floats!
// IEEE 754 sign-magnitude vs two's complement means -FLT_MAX > -0.5 as ints.
// These helpers use CAS loops with proper float comparison.
// ============================================================================
__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_float = __int_as_float(expected);
        if (value <= old_float) return old_float;  // value is not greater, no update
        old = atomicCAS(addr_as_int, expected, __float_as_int(value));
    } while (expected != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = reinterpret_cast<int*>(addr);
    int old = *addr_as_int;
    int expected;
    do {
        expected = old;
        float old_float = __int_as_float(expected);
        if (value >= old_float) return old_float;  // value is not smaller, no update
        old = atomicCAS(addr_as_int, expected, __float_as_int(value));
    } while (expected != old);
    return __int_as_float(old);
}

// ============================================================================
// ENTROPY NORMALIZATION KERNEL
// ============================================================================
// After forward pass, entropy_output contains SUM of block-average entropies.
// This kernel normalizes by the number of Q blocks to get true mean entropy.
// entropy_output[batch * num_heads + head] /= num_q_blocks
// ============================================================================
__global__ void normalizeEntropyKernel(
    float* __restrict__ entropy_output,   // [batch_size * num_heads]
    int batch_size,
    int num_heads,
    int num_q_blocks                      // ceil(seq_len / BLOCK_Q)
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_entries = batch_size * num_heads;
    
    if (idx < total_entries && num_q_blocks > 0) {
        entropy_output[idx] /= static_cast<float>(num_q_blocks);
    }
}

// ============================================================================
// DIAGNOSTIC KERNEL - Compute attention stats for debugging plateau
// ============================================================================
__global__ void computeAttentionDiagnosticsKernel(
    const float* __restrict__ Q,           // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ K,           // [batch, num_kv_heads, seq_len, head_dim]
    const float* __restrict__ grad_Q,      // [batch, num_heads, seq_len, head_dim] (optional)
    const float* __restrict__ grad_K,      // [batch, num_kv_heads, seq_len, head_dim] (optional)
    const float* __restrict__ grad_V,      // [batch, num_kv_heads, seq_len, head_dim] (optional)
    float* __restrict__ diag_output,       // [9]: max_prob, min_prob, mean_entropy, qk_max, qk_min, gq/gk/gv_norm, pos_count
    int batch_idx,
    int head_idx,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float softmax_temperature
) {
    __shared__ float max_prob;         // EXCLUDES position 0 (trivial 1.0 case)
    __shared__ float min_prob;
    __shared__ float entropy_sum;      // Sum of per-position entropies (will divide by count)
    __shared__ float qk_max;
    __shared__ float qk_min;
    __shared__ float grad_q_sq_sum;
    __shared__ float grad_k_sq_sum;
    __shared__ float grad_v_sq_sum;
    __shared__ int position_count;     // Number of positions sampled (for mean entropy)
    
    // DECISIVE TESTS 3-5: Violation counters
    __shared__ int causal_violations;
    __shared__ int entropy_bound_violations;
    __shared__ int prob_sum_violations;
    __shared__ float max_entropy_seen;
    __shared__ float max_entropy_allowed;
    __shared__ float max_prob_deviation;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    if (tid == 0) {
        max_prob = 0.0f;
        min_prob = 1.0f;
        entropy_sum = 0.0f;
        qk_max = -INFINITY;  // IEEE 754 proper infinity (not C89 shim)
        qk_min = INFINITY;
        grad_q_sq_sum = 0.0f;
        grad_k_sq_sum = 0.0f;
        grad_v_sq_sum = 0.0f;
        position_count = 0;
        
        // Initialize verification counters
        causal_violations = 0;
        entropy_bound_violations = 0;
        prob_sum_violations = 0;
        max_entropy_seen = 0.0f;
        max_entropy_allowed = 0.0f;
        max_prob_deviation = 0.0f;
    }
    __syncthreads();
    
    // GQA mapping
    const int heads_per_kv_group = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / heads_per_kv_group;
    
    // Compute Q offset
    const int q_offset = batch_idx * num_heads * seq_len * head_dim + 
                         head_idx * seq_len * head_dim;
    const int kv_offset = batch_idx * num_kv_heads * seq_len * head_dim + 
                          kv_head_idx * seq_len * head_dim;
    
    const float scale = rsqrtf(static_cast<float>(head_dim)) / fmaxf(softmax_temperature, 0.1f);
    
    // FIXED: Sample ACROSS THE FULL SEQUENCE with striding, not just first 32 positions
    // This gives representative statistics instead of being dominated by edge cases
    constexpr int MAX_SAMPLES = 64;
    const int stride = (seq_len > MAX_SAMPLES) ? (seq_len / MAX_SAMPLES) : 1;  // Evenly spaced samples
    const int actual_samples = (seq_len + stride - 1) / stride;  // Ceiling division
    
    float local_max_prob = 0.0f;
    float local_min_prob = 1.0f;
    float local_entropy = 0.0f;
    float local_qk_max = -INFINITY;
    float local_qk_min = INFINITY;
    int local_pos_count = 0;
    
    // Local tracking for verification tests
    float local_max_prob_deviation = 0.0f;
    float local_max_entropy = 0.0f;
    float local_max_entropy_allowed = 0.0f;
    
    // Each thread handles strided Q positions across the FULL sequence
    for (int sample_idx = tid; sample_idx < actual_samples; sample_idx += num_threads) {
        const int qi = sample_idx * stride;  // Actual position in sequence
        if (qi >= seq_len) continue;
        
        // SKIP position 0 for max_prob (it's always 1.0 due to causal mask - trivial case)
        const bool include_in_max_prob = (qi > 0);
        
        // Compute entropy over FULL causal sequence (no truncation!)
        // Three-pass algorithm to avoid storing scores array:
        // Pass 1: Find max_score for numerical stability
        // Pass 2: Compute sum_exp for softmax denominator
        // Pass 3: Compute entropy using stable softmax
        
        const int attend_len = qi + 1;  // Causal: attend to positions [0, qi]
        
        // PASS 1: Find max score for numerical stability
        float max_score = -INFINITY;
        for (int ki = 0; ki < attend_len; ++ki) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += Q[q_offset + qi * head_dim + d] * K[kv_offset + ki * head_dim + d];
            }
            dot *= scale;
            if (dot > local_qk_max) local_qk_max = dot;
            if (dot < local_qk_min) local_qk_min = dot;
            if (dot > max_score) max_score = dot;
        }
        
        // PASS 2: Compute softmax denominator
        float sum_exp = 0.0f;
        for (int ki = 0; ki < attend_len; ++ki) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += Q[q_offset + qi * head_dim + d] * K[kv_offset + ki * head_dim + d];
            }
            dot *= scale;
            sum_exp += expf(dot - max_score);
        }
        
        // PASS 3: Compute entropy and probability statistics
        float pos_entropy = 0.0f;  // Entropy for THIS position (in BITS)
        float pos_max_prob = 0.0f;
        float pos_min_prob = 1.0f;
        float prob_sum = 0.0f;  // TEST 5: Verify sum(p) == 1
        for (int ki = 0; ki < attend_len; ++ki) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += Q[q_offset + qi * head_dim + d] * K[kv_offset + ki * head_dim + d];
            }
            dot *= scale;
            float p = expf(dot - max_score) / fmaxf(sum_exp, 1e-10f);
            prob_sum += p;
            if (p > pos_max_prob) pos_max_prob = p;
            if (p < pos_min_prob && p > 1e-10f) pos_min_prob = p;
            if (p > 1e-10f) pos_entropy -= p * log2f(p);  // Use log2 for BITS
        }
        
        // =====================================================
        // DECISIVE TESTS 3-5 (per position)
        // =====================================================
        
        // TEST 3: Causal mask check - attend_len MUST equal qi + 1 for causal attention
        if (attend_len != qi + 1) {
            atomicAdd(&causal_violations, 1);
        }
        
        // TEST 4: Entropy upper bound - H <= log2(attend_len) 
        const float max_entropy = log2f(static_cast<float>(attend_len));
        if (pos_entropy > max_entropy + 1e-3f) {
            atomicAdd(&entropy_bound_violations, 1);
        }
        
        // TEST 5: Probability sum - must be ~1.0
        const float prob_deviation = fabsf(prob_sum - 1.0f);
        if (prob_deviation > 1e-3f) {
            atomicAdd(&prob_sum_violations, 1);
        }
        
        // Track worst case for reporting
        if (prob_deviation > local_max_prob_deviation) {
            local_max_prob_deviation = prob_deviation;
        }
        if (pos_entropy > local_max_entropy) {
            local_max_entropy = pos_entropy;
            local_max_entropy_allowed = max_entropy;
        }
        
        // Accumulate per-position entropy (will average later)
        local_entropy += pos_entropy;
        local_pos_count++;
        
        // Only include qi > 0 in max_prob (position 0 is trivially 1.0)
        if (include_in_max_prob && pos_max_prob > local_max_prob) {
            local_max_prob = pos_max_prob;
        }
        if (pos_min_prob < local_min_prob) {
            local_min_prob = pos_min_prob;
        }
    }
    
    // Gradient norms (if provided)
    float local_gq_sq = 0.0f, local_gk_sq = 0.0f, local_gv_sq = 0.0f;
    if (grad_Q) {
        for (int i = tid; i < seq_len * head_dim; i += num_threads) {
            float g = grad_Q[q_offset + i];
            local_gq_sq += g * g;
        }
    }
    if (grad_K) {
        for (int i = tid; i < seq_len * head_dim; i += num_threads) {
            float g = grad_K[kv_offset + i];
            local_gk_sq += g * g;
        }
    }
    if (grad_V) {
        for (int i = tid; i < seq_len * head_dim; i += num_threads) {
            float g = grad_V[kv_offset + i];
            local_gv_sq += g * g;
        }
    }
    
    // Reduce using proper float atomics for QK min/max
    // CRITICAL: max_prob/min_prob are POSITIVE so int-based atomics work fine
    // But QK can be NEGATIVE, so we MUST use proper float comparison
    atomicMax(reinterpret_cast<int*>(&max_prob), __float_as_int(local_max_prob));
    atomicMin(reinterpret_cast<int*>(&min_prob), __float_as_int(local_min_prob));
    atomicAdd(&entropy_sum, local_entropy);
    atomicAdd(&position_count, local_pos_count);
    // Use CAS-loop float atomics for QK (can be negative!)
    atomicMaxFloat(&qk_max, local_qk_max);
    atomicMinFloat(&qk_min, local_qk_min);
    atomicAdd(&grad_q_sq_sum, local_gq_sq);
    atomicAdd(&grad_k_sq_sum, local_gk_sq);
    atomicAdd(&grad_v_sq_sum, local_gv_sq);
    
    // Reduce verification test results
    atomicMaxFloat(&max_prob_deviation, local_max_prob_deviation);
    atomicMaxFloat(&max_entropy_seen, local_max_entropy);
    atomicMaxFloat(&max_entropy_allowed, local_max_entropy_allowed);
    __syncthreads();
    
    // Write output - compute MEAN entropy per position
    // Extended output: [0-8] = original, [9-14] = verification test results
    if (tid == 0) {
        diag_output[0] = max_prob;  // EXCLUDES position 0 (trivial 1.0)
        diag_output[1] = min_prob;
        diag_output[2] = (position_count > 0) ? (entropy_sum / position_count) : 0.0f;  // MEAN entropy in bits
        diag_output[3] = qk_max;
        diag_output[4] = qk_min;
        diag_output[5] = sqrtf(grad_q_sq_sum);
        diag_output[6] = sqrtf(grad_k_sq_sum);
        diag_output[7] = sqrtf(grad_v_sq_sum);
        diag_output[8] = static_cast<float>(position_count);  // How many positions sampled
        
        // DECISIVE TEST RESULTS (indices 9-14)
        diag_output[9] = static_cast<float>(causal_violations);       // Test 3
        diag_output[10] = max_entropy_seen;                           // Test 4 - worst case
        diag_output[11] = max_entropy_allowed;                        // Test 4 - bound
        diag_output[12] = static_cast<float>(entropy_bound_violations);  // Test 4 - count
        diag_output[13] = max_prob_deviation;                         // Test 5 - worst case
        diag_output[14] = static_cast<float>(prob_sum_violations);    // Test 5 - count
    }
}

// DELETED (Rule 20): Broken diagnostic with MAX_ATTEND=128 truncation
// Use entropy_output in forward pass for accurate entropy over full sequences
/*
void collectAttentionDiagnostics(
    const float* Q,
    const float* K,
    const float* grad_Q,
    const float* grad_K,
    const float* grad_V,
    const FlashAttentionConfig& config,
    cudaStream_t stream
) {
    AttentionDiagnostics& diag = getAttentionDiagnostics();
    if (!diag.enabled) return;
    
    // Filter by layer/head
    if (diag.dump_layer >= 0 && diag.current_layer != diag.dump_layer) return;
    if (diag.dump_head >= 0 && diag.dump_head >= config.num_heads) return;
    
    const int head_to_check = (diag.dump_head >= 0) ? diag.dump_head : 0;
    
    // Allocate output buffer (15 floats: 9 original + 6 verification tests)
    float* d_diag_output;
    cudaMalloc(&d_diag_output, 15 * sizeof(float));
    cudaMemset(d_diag_output, 0, 15 * sizeof(float));  // Zero-init for safety
    
    computeAttentionDiagnosticsKernel<<<1, 256, 0, stream>>>(
        Q, K, grad_Q, grad_K, grad_V, d_diag_output,
        diag.dump_batch, head_to_check,
        config.num_heads, config.num_kv_heads,
        config.seq_len, config.head_dim,
        config.softmax_temperature
    );
    
    // Copy back to host
    float h_diag[15];
    cudaMemcpyAsync(h_diag, d_diag_output, 15 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    diag.max_attn_prob = h_diag[0];     // EXCLUDES position 0 (trivial case)
    diag.min_attn_prob = h_diag[1];
    diag.mean_entropy = h_diag[2];      // MEAN per-position entropy in BITS
    diag.qk_dot_max = h_diag[3];
    diag.qk_dot_min = h_diag[4];
    diag.grad_q_norm = h_diag[5];
    diag.grad_k_norm = h_diag[6];
    diag.grad_v_norm = h_diag[7];
    diag.positions_sampled = static_cast<int>(h_diag[8]);
    
    // DECISIVE TEST RESULTS (propagate to verification state)
    const int causal_violations = static_cast<int>(h_diag[9]);
    const float max_entropy_seen = h_diag[10];
    const float max_entropy_allowed = h_diag[11];
    const int entropy_violations = static_cast<int>(h_diag[12]);
    const float max_prob_deviation = h_diag[13];
    const int prob_violations = static_cast<int>(h_diag[14]);
    
    /* REMOVED: Old RoPE verification system
    auto& vstate = PBM::RoPEVerification::getVerificationState();
    vstate.causal_violations += causal_violations;
    vstate.entropy_violations += entropy_violations;
    vstate.prob_sum_violations += prob_violations;
    if (max_entropy_seen > vstate.max_entropy_seen) {
        vstate.max_entropy_seen = max_entropy_seen;
    }
    if (max_entropy_allowed > vstate.max_entropy_allowed) {
        vstate.max_entropy_allowed = max_entropy_allowed;
    }
    if (max_prob_deviation > vstate.max_prob_sum_deviation) {
        vstate.max_prob_sum_deviation = max_prob_deviation;
    }
    
    cudaFree(d_diag_output);
    
    // Auto-print
    diag.print();
}
*/  // END DELETED DIAGNOSTIC (entire collectAttentionDiagnostics function)

// ============================================================================
// K-TENSOR TRACE KERNEL - Compute K statistics for debugging
// ============================================================================
__global__ void computeKTensorStatsKernel(
    const float* __restrict__ K,
    float* __restrict__ stats_output,  // [6]: min, max, sum, sum_sq, nan_count, inf_count
    int total_elements
) {
    __shared__ float s_min;
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_sum_sq;
    __shared__ int s_nan_count;
    __shared__ int s_inf_count;
    
    if (threadIdx.x == 0) {
        s_min = INFINITY;
        s_max = -INFINITY;
        s_sum = 0.0f;
        s_sum_sq = 0.0f;
        s_nan_count = 0;
        s_inf_count = 0;
    }
    __syncthreads();
    
    float local_min = INFINITY;
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int local_nan = 0;
    int local_inf = 0;
    
    // Each thread processes multiple elements
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        float val = K[i];
        
        if (isnan(val)) {
            local_nan++;
        } else if (isinf(val)) {
            local_inf++;
        } else {
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }
    
    // Reduce within block - K values CAN be negative, so use proper float atomics
    atomicMinFloat(&s_min, local_min);
    atomicMaxFloat(&s_max, local_max);
    atomicAdd(&s_sum, local_sum);
    atomicAdd(&s_sum_sq, local_sum_sq);
    atomicAdd(&s_nan_count, local_nan);
    atomicAdd(&s_inf_count, local_inf);
    
    __syncthreads();
    
    if (threadIdx.x == 0) {
        stats_output[0] = s_min;
        stats_output[1] = s_max;
        stats_output[2] = s_sum;
        stats_output[3] = s_sum_sq;
        stats_output[4] = static_cast<float>(s_nan_count);
        stats_output[5] = static_cast<float>(s_inf_count);
    }
}

// Host function to trace K tensor - logs stats and CRASHES on NaN/Inf
void traceKTensor(const float* K, 
                  int total_elements,
                  const char* operation_name,
                  int layer_idx,
                  cudaStream_t stream) {
    KTensorTrace& trace = getKTensorTrace();
    if (!trace.enabled) return;
    
    // Allocate stats buffer
    float* d_stats;
    cudaMalloc(&d_stats, 6 * sizeof(float));
    
    // Launch kernel with single block
    computeKTensorStatsKernel<<<1, 256, 0, stream>>>(K, d_stats, total_elements);
    
    // Copy back to host
    float h_stats[6];
    cudaMemcpyAsync(h_stats, d_stats, 6 * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_stats);
    
    trace.k_min = h_stats[0];
    trace.k_max = h_stats[1];
    float sum = h_stats[2];
    float sum_sq = h_stats[3];
    trace.k_nan_count = static_cast<int>(h_stats[4]);
    trace.k_inf_count = static_cast<int>(h_stats[5]);
    
    // Compute mean and std
    int valid_count = total_elements - trace.k_nan_count - trace.k_inf_count;
    trace.k_mean = (valid_count > 0) ? (sum / valid_count) : 0.0f;
    float variance = (valid_count > 0) ? ((sum_sq / valid_count) - (trace.k_mean * trace.k_mean)) : 0.0f;
    trace.k_std = sqrtf(fmaxf(0.0f, variance));
    
    // Check for -FLT_MAX (indicates all masked or broken computation)
    constexpr float INVALID_THRESHOLD = -1e30f;
    bool has_flt_max = (trace.k_min < INVALID_THRESHOLD) || (trace.k_max < INVALID_THRESHOLD);
    
    // Print trace
    printf("[K-TRACE] step=%d layer=%d op='%s' n=%d: min=%.6e max=%.6e mean=%.6e std=%.6e nan=%d inf=%d\n",
           trace.current_step, layer_idx, operation_name, total_elements,
           trace.k_min, trace.k_max, trace.k_mean, trace.k_std,
           trace.k_nan_count, trace.k_inf_count);
    
    // Rule 20: FAIL-LOUD on bad K
    if (trace.k_nan_count > 0) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔══════════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ FATAL: NaN DETECTED IN K TENSOR                                 ║\n");
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ Step:      %d                                                    \n", trace.current_step);
        fprintf(stderr, "║ Layer:     %d                                                    \n", layer_idx);
        fprintf(stderr, "║ Operation: %s                                                    \n", operation_name);
        fprintf(stderr, "║ NaN count: %d / %d                                               \n", trace.k_nan_count, total_elements);
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ CAUSE: K tensor corrupted at '%s'                               ║\n", operation_name);
        fprintf(stderr, "║ Check: W_k weights, input to projection, bias                   ║\n");
        fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
        std::abort();
    }
    
    if (trace.k_inf_count > 0) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔══════════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ FATAL: Inf DETECTED IN K TENSOR                                 ║\n");
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ Step:      %d                                                    \n", trace.current_step);
        fprintf(stderr, "║ Layer:     %d                                                    \n", layer_idx);
        fprintf(stderr, "║ Operation: %s                                                    \n", operation_name);
        fprintf(stderr, "║ Inf count: %d / %d                                               \n", trace.k_inf_count, total_elements);
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ CAUSE: K tensor overflow at '%s'                                ║\n", operation_name);
        fprintf(stderr, "║ Check: Learning rate, weight initialization scale               ║\n");
        fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
        std::abort();
    }
    
    if (has_flt_max) {
        fprintf(stderr, "\n");
        fprintf(stderr, "╔══════════════════════════════════════════════════════════════════╗\n");
        fprintf(stderr, "║ FATAL: -FLT_MAX DETECTED IN K TENSOR                            ║\n");
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ Step:      %d                                                    \n", trace.current_step);
        fprintf(stderr, "║ Layer:     %d                                                    \n", layer_idx);
        fprintf(stderr, "║ Operation: %s                                                    \n", operation_name);
        fprintf(stderr, "║ K min:     %.6e                                                  \n", trace.k_min);
        fprintf(stderr, "║ K max:     %.6e                                                  \n", trace.k_max);
        fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
        fprintf(stderr, "║ CAUSE: K tensor contains -FLT_MAX at '%s'                       ║\n", operation_name);
        fprintf(stderr, "║ This indicates all keys were masked or computation failed       ║\n");
        fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
        fprintf(stderr, "\n");
        std::abort();
    }
}

// ============================================================================
// FORWARD KERNEL - Fixed with shared memory statistics and QK-Normalization
// Supports Grouped Query Attention (GQA): num_kv_heads <= num_heads
// When GQA is active, multiple Q heads share the same K,V head
// ============================================================================
template<int BLOCK_SIZE_Q, int BLOCK_SIZE_KV, int HEAD_DIM>
__launch_bounds__(256, 2)
__global__ void flashAttentionForwardKernel(
    const float* __restrict__ Q,      // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ K,      // [batch, num_kv_heads, seq_len, head_dim] - GQA: fewer KV heads
    const float* __restrict__ V,      // [batch, num_kv_heads, seq_len, head_dim] - GQA: fewer KV heads
    float* __restrict__ output,       // [batch, num_heads, seq_len, head_dim]
    float* __restrict__ entropy_output, // [batch, num_heads] - optional entropy output (or nullptr)
    PBM::PBMSpec pos_encoding,
    int batch_size,
    int num_heads,
    int num_kv_heads,                 // GQA: number of KV heads (num_kv_heads <= num_heads)
    int seq_len,
    int head_dim,
    bool use_alibi,
    float alibi_scale,
    float softmax_temperature,
    bool qk_norm_enabled,
    float qk_norm_scale,
    const float* __restrict__ alpha_q,  // [num_heads] learnable Q scale (or nullptr)
    const float* __restrict__ alpha_k   // [num_kv_heads] learnable K scale (or nullptr)
) {
    // Shared memory for data tiles
    __shared__ float Q_smem[BLOCK_SIZE_Q][HEAD_DIM];
    __shared__ float K_smem[BLOCK_SIZE_KV][HEAD_DIM];
    __shared__ float V_smem[BLOCK_SIZE_KV][HEAD_DIM];
    __shared__ float S_smem[BLOCK_SIZE_Q][BLOCK_SIZE_KV];
    __shared__ float O_smem[BLOCK_SIZE_Q][HEAD_DIM];
    
    // CRITICAL: Statistics in SHARED memory, not thread-local!
    __shared__ float row_max_smem[BLOCK_SIZE_Q];
    __shared__ float row_sum_smem[BLOCK_SIZE_Q];
    
    // Entropy accumulation: H = -sum(P * log2(P)) using online formula
    // Online entropy: H_i = log2(row_sum_i) + row_max_i / (row_sum_i * ln(2))
    // We accumulate: entropy_sum_smem = sum_over_q(H_q) for this block
    __shared__ float row_entropy_smem[BLOCK_SIZE_Q];  // Per-row entropy accumulation
    
    // QK-Norm: Store L2 norms for Q vectors (K norms computed per-block)
    __shared__ float Q_norm_smem[BLOCK_SIZE_Q];
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;      // Q head index [0, num_heads)
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // GQA: Compute which KV head this Q head maps to
    // heads_per_kv_group = num_heads / num_kv_heads (e.g., 12/4 = 3)
    // kv_head_idx = head_idx / heads_per_kv_group
    const int heads_per_kv_group = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / heads_per_kv_group;
    
    // Q and output use full num_heads indexing
    const int q_offset = batch_idx * num_heads * seq_len * head_dim + 
                         head_idx * seq_len * head_dim;
    
    // K and V use reduced num_kv_heads indexing (GQA)
    const int kv_offset = batch_idx * num_kv_heads * seq_len * head_dim + 
                          kv_head_idx * seq_len * head_dim;
    
    const float* Q_head = Q + q_offset;
    const float* K_head = K + kv_offset;
    const float* V_head = V + kv_offset;
    float* O_head = output + q_offset;
    
    const int q_start = q_block_idx * BLOCK_SIZE_Q;
    const int q_end = min(q_start + BLOCK_SIZE_Q, seq_len);
    const int q_size = q_end - q_start;
    
    // ALiBi setup - PBM is always hybrid (ALiBi + RoPE)
    const bool apply_alibi = use_alibi && pos_encoding.alibi_slopes != nullptr;
    
    // Hard assertion: if apply_alibi is true, slopes MUST be configured correctly
    assert((!apply_alibi) || (pos_encoding.num_heads > 0 && head_idx < pos_encoding.num_heads));
    
    const float alibi_slope = apply_alibi ? 
        pos_encoding.alibi_slopes[head_idx] * alibi_scale : 0.0f;
    
    // Scale: when QK-norm is enabled, use the explicit scale factor
    // When disabled, use traditional 1/sqrt(d) / temperature
    const float base_scale = qk_norm_enabled ? qk_norm_scale : rsqrtf(static_cast<float>(head_dim));
    const float scale = base_scale / fmaxf(softmax_temperature, 0.1f);
    const int num_kv_blocks = (seq_len + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;
    
    // Initialize shared memory and load Q
    for (int i = tid; i < BLOCK_SIZE_Q * HEAD_DIM; i += num_threads) {
        const int q_row = i / HEAD_DIM;
        const int q_col = i % HEAD_DIM;
        if (q_row < q_size) {
            Q_smem[q_row][q_col] = Q_head[(q_start + q_row) * head_dim + q_col];
        } else {
            Q_smem[q_row][q_col] = 0.0f;
        }
        O_smem[q_row][q_col] = 0.0f;
    }
    
    for (int i = tid; i < BLOCK_SIZE_Q; i += num_threads) {
        row_max_smem[i] = -INFINITY;  // IEEE 754 identity for max()
        row_sum_smem[i] = 0.0f;
        row_entropy_smem[i] = 0.0f;   // Entropy accumulation
        Q_norm_smem[i] = 0.0f;
    }
    __syncthreads();
    
    // QK-Normalization: Compute L2 norm of each Q vector
    if (qk_norm_enabled) {
        for (int q_idx = tid; q_idx < q_size; q_idx += num_threads) {
            float norm_sq = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; ++d) {
                norm_sq += Q_smem[q_idx][d] * Q_smem[q_idx][d];
            }
            // Clamp inv_norm to prevent gradient explosion (max 50x amplification)
            // Must match K norm clamping for symmetric gradient behavior
            Q_norm_smem[q_idx] = fminf(rsqrtf(norm_sq + 1e-6f), 50.0f);
        }
        __syncthreads();
    }
    
    // K norm storage for QK-normalization (per KV block)
    __shared__ float K_norm_smem[BLOCK_SIZE_KV];
    
    // Iterate over KV blocks
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const int kv_start = kv_block_idx * BLOCK_SIZE_KV;
        const int kv_end = min(kv_start + BLOCK_SIZE_KV, seq_len);
        const int kv_size = kv_end - kv_start;
        
        // Load K, V blocks
        for (int i = tid; i < BLOCK_SIZE_KV * HEAD_DIM; i += num_threads) {
            const int kv_row = i / HEAD_DIM;
            const int kv_col = i % HEAD_DIM;
            if (kv_row < kv_size) {
                const int global_idx = (kv_start + kv_row) * head_dim + kv_col;
                K_smem[kv_row][kv_col] = K_head[global_idx];
                V_smem[kv_row][kv_col] = V_head[global_idx];
            } else {
                K_smem[kv_row][kv_col] = 0.0f;
                V_smem[kv_row][kv_col] = 0.0f;
            }
        }
        
        // Initialize K norms
        for (int i = tid; i < BLOCK_SIZE_KV; i += num_threads) {
            K_norm_smem[i] = 1.0f;  // Default to 1.0 (no normalization)
        }
        __syncthreads();
        
        // QK-Normalization: Compute L2 norm of each K vector in this block
        if (qk_norm_enabled) {
            for (int kv_idx = tid; kv_idx < kv_size; kv_idx += num_threads) {
                float norm_sq = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    norm_sq += K_smem[kv_idx][d] * K_smem[kv_idx][d];
                }
                // Clamp inv_norm to prevent gradient explosion (max 50x amplification)
                K_norm_smem[kv_idx] = fminf(rsqrtf(norm_sq + 1e-6f), 50.0f);  // Store inverse norm
            }
            __syncthreads();
        }
        
        // Compute attention scores S = (Q/||Q||) @ (K/||K||)^T * scale
        // With QK-norm: score = (Q · K) * (1/||Q||) * (1/||K||) * scale
        for (int idx = tid; idx < q_size * kv_size; idx += num_threads) {
            const int q_idx = idx / kv_size;
            const int kv_idx = idx % kv_size;
            
            float score = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q_smem[q_idx][d] * K_smem[kv_idx][d];
            }
            
            // Apply QK-normalization with learnable per-head scales (nGPT-style)
            // q̂ = alpha_q * (q / ||q||), k̂ = alpha_k * (k / ||k||)
            // score = q̂ · k̂ = alpha_q * alpha_k * (q/||q||) · (k/||k||)
            if (qk_norm_enabled) {
                float q_scale = Q_norm_smem[q_idx];  // 1/||q||
                float k_scale = K_norm_smem[kv_idx]; // 1/||k||
                
                // Apply learnable scales if provided
                // GQA: alpha_q uses head_idx, alpha_k uses kv_head_idx
                if (alpha_q != nullptr) {
                    q_scale *= alpha_q[head_idx];
                }
                if (alpha_k != nullptr) {
                    k_scale *= alpha_k[kv_head_idx];  // GQA: index by KV head
                }
                
                score *= q_scale * k_scale;
            }
            
            score *= scale;
            
            if (apply_alibi) {
                score += alibi_slope * static_cast<float>((kv_start + kv_idx) - (q_start + q_idx));
            }
            
            // Causal mask
            if ((kv_start + kv_idx) > (q_start + q_idx)) {
                score = -INFINITY;  // exp(-inf) = 0 always (not exp(-FLT_MAX) which can be non-zero)
            }
            
            S_smem[q_idx][kv_idx] = score;
        }
        __syncthreads();
        
        // Online softmax: find block max, update global max, rescale
        for (int q_idx = tid; q_idx < q_size; q_idx += num_threads) {
            // Find max in this KV block
            float block_max = -INFINITY;
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                block_max = fmaxf(block_max, S_smem[q_idx][kv_idx]);
            }
            
            const float old_max = row_max_smem[q_idx];
            const float new_max = fmaxf(old_max, block_max);
            const float old_scale = __expf(old_max - new_max);
            
            // Rescale previous sum and output
            const float old_row_sum = row_sum_smem[q_idx];
            row_sum_smem[q_idx] *= old_scale;
            for (int d = 0; d < HEAD_DIM; ++d) {
                O_smem[q_idx][d] *= old_scale;
            }
            row_max_smem[q_idx] = new_max;
            
            // Compute softmax probs and accumulate output + entropy
            float block_sum = 0.0f;
            float block_weighted_logp = 0.0f;  // sum(exp(s-max) * (s-max)) for entropy
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                const float score = S_smem[q_idx][kv_idx];
                const float prob = __expf(score - new_max);
                S_smem[q_idx][kv_idx] = prob;  // Store prob for V accumulation
                block_sum += prob;
                // For entropy: we need sum(p_normalized * log2(p_normalized))
                // We accumulate sum(exp(s - max) * (s - max)) = sum(p_unnorm * (s - max))
                // Final entropy: H = log2(sum) - row_entropy_smem / (sum * ln(2))
                if (prob > 1e-10f) {
                    block_weighted_logp += prob * (score - new_max);
                }
            }
            row_sum_smem[q_idx] += block_sum;
            
            // Rescale entropy when max changes: need to transform sum(p * (s - old_max)) to sum(p' * (s - new_max))
            // where p' = p * old_scale. The math works out to:
            // sum(p' * (s - new_max)) = old_scale * sum(p * (s - old_max)) + log(old_scale) * old_row_sum * old_scale
            // The second term accounts for the max shift in the (s - max) factor
            const float max_shift_correction = (old_max != new_max && old_row_sum > 0.0f) 
                ? logf(old_scale) * old_row_sum * old_scale : 0.0f;
            row_entropy_smem[q_idx] = row_entropy_smem[q_idx] * old_scale + max_shift_correction + block_weighted_logp;
            
            // O += P @ V
            for (int d = 0; d < HEAD_DIM; ++d) {
                float weighted = 0.0f;
                for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                    weighted += S_smem[q_idx][kv_idx] * V_smem[kv_idx][d];
                }
                O_smem[q_idx][d] += weighted;
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write output
    for (int i = tid; i < q_size * HEAD_DIM; i += num_threads) {
        const int q_idx = i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        const float sum = row_sum_smem[q_idx];
        O_head[(q_start + q_idx) * head_dim + d] = (sum > 0.0f) ? O_smem[q_idx][d] / sum : 0.0f;
    }
    
    // Compute and write entropy if requested
    // Entropy H = -sum(p * log2(p)) where p = exp(s - max) / sum
    // = -sum(p * (s - max - log(sum))) / ln(2)
    // = (log(sum) * sum - sum(exp(s-max) * (s-max))) / (sum * ln(2))
    // = log2(sum) - row_entropy_smem / (sum * ln(2))
    if (entropy_output != nullptr) {
        __shared__ float block_entropy_sum;
        __shared__ int block_q_count;
        if (tid == 0) {
            block_entropy_sum = 0.0f;
            block_q_count = 0;
        }
        __syncthreads();
        
        // Compute per-row entropy and accumulate
        for (int q_idx = tid; q_idx < q_size; q_idx += num_threads) {
            const float sum = row_sum_smem[q_idx];
            if (sum > 1e-10f) {
                // row_entropy_smem = sum(p_unnorm * log(p_unnorm)) = sum(exp(s-max) * (s-max))
                // H = log2(sum) - row_entropy_smem / (sum * ln(2))
                const float log2_sum = log2f(sum);
                const float weighted_term = row_entropy_smem[q_idx] / (sum * 0.6931471805599453f);
                const float entropy_bits = log2_sum - weighted_term;
                atomicAdd(&block_entropy_sum, fmaxf(0.0f, entropy_bits));  // Clamp to non-negative
                atomicAdd(&block_q_count, 1);
            }
        }
        __syncthreads();
        
        // Write mean entropy for this (batch, head) - use atomicAdd for multi-block accumulation
        if (tid == 0 && block_q_count > 0) {
            const int entropy_idx = batch_idx * num_heads + head_idx;
            atomicAdd(&entropy_output[entropy_idx], block_entropy_sum / static_cast<float>(block_q_count));
        }
    }
}

// Host function to launch Flash Attention forward pass
void flashAttentionForward(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const PBM::PBMSpec& pos_encoding,
    const FlashAttentionConfig& config
) {
    // ========== VALIDATION CHECKS ==========
    // 1. Head dimension must be exactly 32 or 64 - template unrolls require exact match
    if (config.head_dim != 32 && config.head_dim != 64) {
        printf("[FlashAttn Forward] FATAL: head_dim=%d not supported. Must be exactly 32 or 64.\n", 
               config.head_dim);
        printf("[FlashAttn Forward] Template unrolling reads HEAD_DIM elements - mismatched dims corrupt memory.\n");
        return;  // Hard fail - do not silently corrupt output
    }
    
    // 2. GQA divisibility check - num_heads must be evenly divisible by num_kv_heads
    if (config.num_heads % config.num_kv_heads != 0) {
        printf("[FlashAttn Forward] FATAL: num_heads=%d not divisible by num_kv_heads=%d\n",
               config.num_heads, config.num_kv_heads);
        printf("[FlashAttn Forward] GQA requires num_heads %% num_kv_heads == 0\n");
        return;  // Hard fail - incorrect KV head mapping corrupts output
    }
    
    const int num_q_blocks = (config.seq_len + FLASH_BLOCK_Q - 1) / FLASH_BLOCK_Q;
    
    // Grid: x=Q blocks, y=Q heads (not KV heads!), z=batch
    dim3 grid(num_q_blocks, config.num_heads, config.batch_size);
    
    // FAIL LOUD: Stream MUST be provided - no silent fallback to default stream
    if (config.stream == nullptr) {
        printf("[FlashAttn Forward] FATAL: config.stream is NULL - caller MUST provide valid CUDA stream\n");
        return;
    }
    cudaStream_t stream = config.stream;
    
    // Zero entropy output if provided (multiple blocks will atomicAdd to it)
    if (config.entropy_output != nullptr) {
        cudaMemsetAsync(config.entropy_output, 0, 
                        config.batch_size * config.num_heads * sizeof(float), stream);
    }
    
    // Dispatch based on head_dim only - block sizes are fixed
    if (config.head_dim == 64) {
        flashAttentionForwardKernel<FLASH_BLOCK_Q, FLASH_BLOCK_KV, 64><<<grid, FLASH_NUM_THREADS, 0, stream>>>(
            Q, K, V, output, config.entropy_output, pos_encoding,
            config.batch_size, config.num_heads, config.num_kv_heads, config.seq_len, config.head_dim,
            config.use_alibi, config.alibi_scale, config.softmax_temperature,
            config.qk_norm_enabled, config.qk_norm_scale,
            config.alpha_q, config.alpha_k
        );
    } else if (config.head_dim == 32) {
        flashAttentionForwardKernel<FLASH_BLOCK_Q, FLASH_BLOCK_KV, 32><<<grid, FLASH_NUM_THREADS, 0, stream>>>(
            Q, K, V, output, config.entropy_output, pos_encoding,
            config.batch_size, config.num_heads, config.num_kv_heads, config.seq_len, config.head_dim,
            config.use_alibi, config.alibi_scale, config.softmax_temperature,
            config.qk_norm_enabled, config.qk_norm_scale,
            config.alpha_q, config.alpha_k
        );
    } else {
        // Should never reach here due to validation above, but defensive check
        printf("[FlashAttn Forward] FATAL: Unsupported head_dim=%d (validation should have caught this)\n", 
               config.head_dim);
        return;
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FlashAttn Forward] CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // Normalize entropy output by number of Q blocks (each block atomicAdds its average)
    if (config.entropy_output != nullptr) {
        const int total_entries = config.batch_size * config.num_heads;
        const int threads_per_block = 256;
        const int entropy_norm_blocks = (total_entries + threads_per_block - 1) / threads_per_block;
        
        // CRITICAL: Sync stream before reading entropy buffer (kernel launched on stream)
        // BUG FIX Issue #25: cudaMemcpy was reading zeros because kernel hadn't finished writing
        cudaStreamSynchronize(stream);
        
        // DEBUG: Check raw entropy before normalization
        std::vector<float> h_entropy_raw(total_entries);
        cudaMemcpy(h_entropy_raw.data(), config.entropy_output, 
                   total_entries * sizeof(float), cudaMemcpyDeviceToHost);
        float raw_sum = std::accumulate(h_entropy_raw.begin(), h_entropy_raw.end(), 0.0f);
        printf("[EntropyDebug] seq_len=%d num_q_blocks=%d batch=%d heads=%d raw_sum=%.4f\n",
               config.seq_len, num_q_blocks, config.batch_size, config.num_heads, raw_sum);
        
        normalizeEntropyKernel<<<entropy_norm_blocks, threads_per_block, 0, stream>>>(
            config.entropy_output, config.batch_size, config.num_heads, num_q_blocks);
    }
}

size_t getFlashAttentionWorkspaceSize(const FlashAttentionConfig& config) {
    // Flash Attention 2 implementation doesn't require external workspace
    // All temporary buffers are in shared memory (48KB per block)
    return 0;
}

void flashAttentionForwardWithWorkspace(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const PBM::PBMSpec& pos_encoding,
    void* workspace,
    const FlashAttentionConfig& config
) {
    flashAttentionForward(Q, K, V, output, pos_encoding, config);
}

// ============================================================================
// BACKWARD KERNEL - Shared memory statistics, proper accumulation
// Supports Grouped Query Attention (GQA): num_kv_heads <= num_heads
// CRITICAL: In GQA backward, grad_K and grad_V must be atomically accumulated
// because multiple Q heads contribute gradients to the same KV head
// ============================================================================
template<int BLOCK_SIZE_Q, int BLOCK_SIZE_KV, int HEAD_DIM>
__launch_bounds__(256, 2)
__global__ void flashAttentionBackwardKernel(
    const float* __restrict__ Q,           // [batch, num_heads, seq_len, head_dim]
    const float* __restrict__ K,           // [batch, num_kv_heads, seq_len, head_dim] - GQA: fewer KV heads
    const float* __restrict__ V,           // [batch, num_kv_heads, seq_len, head_dim] - GQA: fewer KV heads
    const float* __restrict__ output,      // [batch, num_heads, seq_len, head_dim] (unused, recomputed)
    const float* __restrict__ grad_output, // [batch, num_heads, seq_len, head_dim]
    float* __restrict__ grad_Q,            // [batch, num_heads, seq_len, head_dim]
    float* __restrict__ grad_K,            // [batch, num_kv_heads, seq_len, head_dim] - GQA: accumulated
    float* __restrict__ grad_V,            // [batch, num_kv_heads, seq_len, head_dim] - GQA: accumulated
    PBM::PBMSpec pos_encoding,
    int batch_size,
    int num_heads,
    int num_kv_heads,                      // GQA: number of KV heads (num_kv_heads <= num_heads)
    int seq_len,
    int head_dim,
    bool use_alibi,
    float alibi_scale,
    float softmax_temperature,
    bool qk_norm_enabled,
    float qk_norm_scale,
    const float* __restrict__ alpha_q,     // [num_heads] learnable Q scale (or nullptr)
    const float* __restrict__ alpha_k,     // [num_kv_heads] learnable K scale (or nullptr)
    float* __restrict__ grad_alpha_q,      // [num_heads] output gradient (or nullptr)
    float* __restrict__ grad_alpha_k       // [num_kv_heads] output gradient (or nullptr)
) {
    // Shared memory - fits in 48KB at 32×32×64
    __shared__ float Q_smem[BLOCK_SIZE_Q][HEAD_DIM];
    __shared__ float K_smem[BLOCK_SIZE_KV][HEAD_DIM];
    __shared__ float V_smem[BLOCK_SIZE_KV][HEAD_DIM];
    __shared__ float dO_smem[BLOCK_SIZE_Q][HEAD_DIM];
    __shared__ float dQ_smem[BLOCK_SIZE_Q][HEAD_DIM];
    __shared__ float S_smem[BLOCK_SIZE_Q][BLOCK_SIZE_KV];  // Reused for scores/P/dS
    
    // Statistics in SHARED memory
    __shared__ float row_max_smem[BLOCK_SIZE_Q];
    __shared__ float row_sum_smem[BLOCK_SIZE_Q];
    __shared__ float dp_sum_smem[BLOCK_SIZE_Q];
    
    // QK-Norm: Store L2 inverse norms
    __shared__ float Q_norm_smem[BLOCK_SIZE_Q];
    __shared__ float K_norm_smem[BLOCK_SIZE_KV];
    
    // Shared memory for learnable alpha gradient accumulation
    __shared__ float local_grad_alpha_q;
    __shared__ float local_grad_alpha_k;
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;      // Q head index [0, num_heads)
    const int q_block_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // GQA: Compute which KV head this Q head maps to
    const int heads_per_kv_group = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / heads_per_kv_group;
    
    // Load learnable alpha values (default to 1.0 if not provided)
    // GQA: alpha_q uses head_idx, alpha_k uses kv_head_idx
    const float alpha_q_val = (alpha_q != nullptr) ? alpha_q[head_idx] : 1.0f;
    const float alpha_k_val = (alpha_k != nullptr) ? alpha_k[kv_head_idx] : 1.0f;
    
    // Initialize alpha gradient accumulators
    if (tid == 0) {
        local_grad_alpha_q = 0.0f;
        local_grad_alpha_k = 0.0f;
    }
    __syncthreads();
    
    // Q and grad_Q use full num_heads indexing
    const int q_offset = batch_idx * num_heads * seq_len * head_dim + 
                         head_idx * seq_len * head_dim;
    
    // K, V, grad_K, grad_V use reduced num_kv_heads indexing (GQA)
    const int kv_offset = batch_idx * num_kv_heads * seq_len * head_dim + 
                          kv_head_idx * seq_len * head_dim;
    
    const float* Q_head = Q + q_offset;
    const float* K_head = K + kv_offset;
    const float* V_head = V + kv_offset;
    const float* dO_head = grad_output + q_offset;
    float* dQ_head = grad_Q + q_offset;
    float* dK_head = grad_K + kv_offset;  // GQA: Multiple Q heads write to same dK via atomicAdd
    float* dV_head = grad_V + kv_offset;  // GQA: Multiple Q heads write to same dV via atomicAdd
    
    const int q_start = q_block_idx * BLOCK_SIZE_Q;
    const int q_end = min(q_start + BLOCK_SIZE_Q, seq_len);
    const int q_size = q_end - q_start;
    
    // ALiBi setup - PBM is always hybrid (ALiBi + RoPE)
    const bool apply_alibi = use_alibi && pos_encoding.alibi_slopes != nullptr;
    
    // Hard assertion: if apply_alibi is true, slopes MUST be configured correctly
    assert((!apply_alibi) || (pos_encoding.num_heads > 0 && head_idx < pos_encoding.num_heads));
    
    const float alibi_slope = apply_alibi ? 
        pos_encoding.alibi_slopes[head_idx] * alibi_scale : 0.0f;
    
    // Scale: when QK-norm is enabled, use the explicit scale factor for FORWARD score computation
    // When disabled, use traditional 1/sqrt(d) / temperature
    // IMPORTANT: Backward pass uses the SAME scale for chain rule consistency
    const float base_scale = qk_norm_enabled ? qk_norm_scale : rsqrtf(static_cast<float>(head_dim));
    const float scale = base_scale / fmaxf(softmax_temperature, 0.1f);
    
    // GQA gradient scaling factor: when multiple Q heads share the same KV head,
    // their gradient contributions are accumulated via atomicAdd. To maintain
    // proper gradient magnitude, we scale each contribution by 1/heads_per_kv_group.
    const float gqa_grad_scale = 1.0f / static_cast<float>(heads_per_kv_group);
    
    const int num_kv_blocks = (seq_len + BLOCK_SIZE_KV - 1) / BLOCK_SIZE_KV;
    
    // Load Q, dO and initialize accumulators
    for (int i = tid; i < BLOCK_SIZE_Q * HEAD_DIM; i += num_threads) {
        const int q_row = i / HEAD_DIM;
        const int q_col = i % HEAD_DIM;
        if (q_row < q_size) {
            const int global_idx = (q_start + q_row) * head_dim + q_col;
            Q_smem[q_row][q_col] = Q_head[global_idx];
            dO_smem[q_row][q_col] = dO_head[global_idx];
        } else {
            Q_smem[q_row][q_col] = 0.0f;
            dO_smem[q_row][q_col] = 0.0f;
        }
        dQ_smem[q_row][q_col] = 0.0f;
    }
    
    for (int i = tid; i < BLOCK_SIZE_Q; i += num_threads) {
        row_max_smem[i] = -INFINITY;  // IEEE 754 identity for max()
        row_sum_smem[i] = 0.0f;
        dp_sum_smem[i] = 0.0f;
        Q_norm_smem[i] = 1.0f;  // Default: no normalization
    }
    __syncthreads();
    
    // QK-Normalization: Compute L2 norm of each Q vector
    if (qk_norm_enabled) {
        for (int q_idx = tid; q_idx < q_size; q_idx += num_threads) {
            float norm_sq = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; ++d) {
                norm_sq += Q_smem[q_idx][d] * Q_smem[q_idx][d];
            }
            // Clamp inv_norm to prevent gradient explosion (max 50x amplification)
            // Must match K norm clamping for symmetric gradient behavior
            Q_norm_smem[q_idx] = fminf(rsqrtf(norm_sq + 1e-6f), 50.0f);
        }
        __syncthreads();
    }
    
    // ==================== Pass 1: Compute softmax statistics ====================
    // PERFORMANCE NOTE: Q·K is recomputed 4 times across passes (row_max, row_sum, dp_sum, gradients)
    // This is FlashAttention v2's memory-vs-compute tradeoff:
    //   ✓ Saves ~48KB shared memory (no S matrix storage)
    //   ✓ Enables long sequences without OOM
    //   ✗ 4x redundant dot product computation
    //   ✗ No warp-level fusion (unlike Triton implementation)
    // OPTIMIZATION TARGET: If training is compute-bound (not memory-bound), fuse passes or cache scores
    // Find row_max across all KV blocks
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const int kv_start = kv_block_idx * BLOCK_SIZE_KV;
        const int kv_end = min(kv_start + BLOCK_SIZE_KV, seq_len);
        const int kv_size = kv_end - kv_start;
        
        for (int i = tid; i < BLOCK_SIZE_KV * HEAD_DIM; i += num_threads) {
            const int kv_row = i / HEAD_DIM;
            const int kv_col = i % HEAD_DIM;
            K_smem[kv_row][kv_col] = (kv_row < kv_size) ? 
                K_head[(kv_start + kv_row) * head_dim + kv_col] : 0.0f;
        }
        
        // Initialize K norms
        for (int i = tid; i < BLOCK_SIZE_KV; i += num_threads) {
            K_norm_smem[i] = 1.0f;  // Default: no normalization
        }
        __syncthreads();
        
        // QK-Normalization: Compute L2 norm of each K vector in this block
        if (qk_norm_enabled) {
            for (int kv_idx = tid; kv_idx < kv_size; kv_idx += num_threads) {
                float norm_sq = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    norm_sq += K_smem[kv_idx][d] * K_smem[kv_idx][d];
                }
                // Clamp inv_norm to prevent gradient explosion (max 50x amplification)
                // Must match forward pass K norm clamping
                K_norm_smem[kv_idx] = fminf(rsqrtf(norm_sq + 1e-6f), 50.0f);
            }
            __syncthreads();
        }
        
        for (int q_idx = tid; q_idx < q_size; q_idx += num_threads) {
            float local_max = row_max_smem[q_idx];
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                float score = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem[q_idx][d] * K_smem[kv_idx][d];
                }
                // Apply QK-normalization with learnable per-head scales (nGPT-style)
                if (qk_norm_enabled) {
                    score *= alpha_q_val * Q_norm_smem[q_idx] * alpha_k_val * K_norm_smem[kv_idx];
                }
                score *= scale;
                if (apply_alibi) {
                    score += alibi_slope * static_cast<float>((kv_start + kv_idx) - (q_start + q_idx));
                }
                if ((kv_start + kv_idx) > (q_start + q_idx)) {
                    score = -INFINITY;  // Pass 0: max computation
                }
                local_max = fmaxf(local_max, score);
            }
            row_max_smem[q_idx] = local_max;
        }
        __syncthreads();
    }
    
    // Compute row_sum
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const int kv_start = kv_block_idx * BLOCK_SIZE_KV;
        const int kv_end = min(kv_start + BLOCK_SIZE_KV, seq_len);
        const int kv_size = kv_end - kv_start;
        
        for (int i = tid; i < BLOCK_SIZE_KV * HEAD_DIM; i += num_threads) {
            const int kv_row = i / HEAD_DIM;
            const int kv_col = i % HEAD_DIM;
            K_smem[kv_row][kv_col] = (kv_row < kv_size) ? 
                K_head[(kv_start + kv_row) * head_dim + kv_col] : 0.0f;
        }
        
        // Initialize K norms
        for (int i = tid; i < BLOCK_SIZE_KV; i += num_threads) {
            K_norm_smem[i] = 1.0f;
        }
        __syncthreads();
        
        // QK-Normalization: Compute K norms
        if (qk_norm_enabled) {
            for (int kv_idx = tid; kv_idx < kv_size; kv_idx += num_threads) {
                float norm_sq = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    norm_sq += K_smem[kv_idx][d] * K_smem[kv_idx][d];
                }
                // Clamp inv_norm to prevent gradient explosion (max 50x amplification)
                // Must match forward pass K norm clamping
                K_norm_smem[kv_idx] = fminf(rsqrtf(norm_sq + 1e-6f), 50.0f);
            }
            __syncthreads();
        }
        
        for (int q_idx = tid; q_idx < q_size; q_idx += num_threads) {
            float local_sum = 0.0f;
            const float max_val = row_max_smem[q_idx];
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                float score = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem[q_idx][d] * K_smem[kv_idx][d];
                }
                // Apply QK-normalization with learnable per-head scales (nGPT-style)
                if (qk_norm_enabled) {
                    score *= alpha_q_val * Q_norm_smem[q_idx] * alpha_k_val * K_norm_smem[kv_idx];
                }
                score *= scale;
                if (apply_alibi) {
                    score += alibi_slope * static_cast<float>((kv_start + kv_idx) - (q_start + q_idx));
                }
                if ((kv_start + kv_idx) > (q_start + q_idx)) {
                    score = -INFINITY;  // Pass 1: sum computation
                }
                local_sum += __expf(score - max_val);
            }
            row_sum_smem[q_idx] += local_sum;
        }
        __syncthreads();
    }
    
    // ==================== Pass 2: Compute dp_sum = sum(dP * P) ====================
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const int kv_start = kv_block_idx * BLOCK_SIZE_KV;
        const int kv_end = min(kv_start + BLOCK_SIZE_KV, seq_len);
        const int kv_size = kv_end - kv_start;
        
        for (int i = tid; i < BLOCK_SIZE_KV * HEAD_DIM; i += num_threads) {
            const int kv_row = i / HEAD_DIM;
            const int kv_col = i % HEAD_DIM;
            if (kv_row < kv_size) {
                const int global_idx = (kv_start + kv_row) * head_dim + kv_col;
                K_smem[kv_row][kv_col] = K_head[global_idx];
                V_smem[kv_row][kv_col] = V_head[global_idx];
            }
        }
        
        // Initialize K norms
        for (int i = tid; i < BLOCK_SIZE_KV; i += num_threads) {
            K_norm_smem[i] = 1.0f;
        }
        __syncthreads();
        
        // QK-Normalization: Compute K norms
        if (qk_norm_enabled) {
            for (int kv_idx = tid; kv_idx < kv_size; kv_idx += num_threads) {
                float norm_sq = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    norm_sq += K_smem[kv_idx][d] * K_smem[kv_idx][d];
                }
                // Clamp inv_norm to prevent gradient explosion
                K_norm_smem[kv_idx] = fminf(rsqrtf(norm_sq + 1e-6f), 50.0f);
            }
            __syncthreads();
        }
        
        for (int q_idx = tid; q_idx < q_size; q_idx += num_threads) {
            const float max_val = row_max_smem[q_idx];
            const float sum_val = fmaxf(row_sum_smem[q_idx], 1e-10f);  // Prevent division by zero
            #ifdef FLASH_ATTN_DEBUG
            if (sum_val <= 0.0f || !isfinite(sum_val)) {
                printf("[FlashAttn] BAD sum_val q=%d val=%f\n", q_idx, sum_val);
            }
            #endif
            float local_dp_sum = 0.0f;
            
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                float score = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    score += Q_smem[q_idx][d] * K_smem[kv_idx][d];
                }
                // Apply QK-normalization with learnable per-head scales (nGPT-style)
                if (qk_norm_enabled) {
                    score *= alpha_q_val * Q_norm_smem[q_idx] * alpha_k_val * K_norm_smem[kv_idx];
                }
                score *= scale;
                if (apply_alibi) {
                    score += alibi_slope * static_cast<float>((kv_start + kv_idx) - (q_start + q_idx));
                }
                if ((kv_start + kv_idx) > (q_start + q_idx)) {
                    score = -INFINITY;  // Pass 2: dp_sum computation
                }
                const float P = __expf(score - max_val) / sum_val;
                
                // Guard: If P == 0, skip dP computation (contributes nothing to dp_sum)
                if (P == 0.0f) {
                    continue;
                }
                
                // dP = dO @ V^T
                float dP = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dP += dO_smem[q_idx][d] * V_smem[kv_idx][d];
                }
                local_dp_sum += dP * P;
            }
            // Clamp dp_sum to prevent overflow from numerical instability
            local_dp_sum = fmaxf(fminf(local_dp_sum, 1e4f), -1e4f);
            dp_sum_smem[q_idx] += local_dp_sum;
        }
        __syncthreads();
    }
    
    // ==================== Pass 3: Compute gradients ====================
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const int kv_start = kv_block_idx * BLOCK_SIZE_KV;
        const int kv_end = min(kv_start + BLOCK_SIZE_KV, seq_len);
        const int kv_size = kv_end - kv_start;
        
        for (int i = tid; i < BLOCK_SIZE_KV * HEAD_DIM; i += num_threads) {
            const int kv_row = i / HEAD_DIM;
            const int kv_col = i % HEAD_DIM;
            if (kv_row < kv_size) {
                const int global_idx = (kv_start + kv_row) * head_dim + kv_col;
                K_smem[kv_row][kv_col] = K_head[global_idx];
                V_smem[kv_row][kv_col] = V_head[global_idx];
            }
        }
        
        // Initialize K norms
        for (int i = tid; i < BLOCK_SIZE_KV; i += num_threads) {
            K_norm_smem[i] = 1.0f;
        }
        __syncthreads();
        
        // QK-Normalization: Compute K norms
        if (qk_norm_enabled) {
            for (int kv_idx = tid; kv_idx < kv_size; kv_idx += num_threads) {
                float norm_sq = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; ++d) {
                    norm_sq += K_smem[kv_idx][d] * K_smem[kv_idx][d];
                }
                // Clamp inv_norm to prevent gradient explosion
                K_norm_smem[kv_idx] = fminf(rsqrtf(norm_sq + 1e-6f), 50.0f);
            }
            __syncthreads();
        }
        
        // Step 1: Compute P and store in S_smem (compute Q·K score ONCE per position)
        for (int idx = tid; idx < q_size * kv_size; idx += num_threads) {
            const int q_idx = idx / kv_size;
            const int kv_idx = idx % kv_size;
            
            const float max_val = row_max_smem[q_idx];
            const float sum_val = fmaxf(row_sum_smem[q_idx], 1e-10f);  // Prevent division by zero
            #ifdef FLASH_ATTN_DEBUG
            if (sum_val <= 0.0f || !isfinite(sum_val)) {
                printf("[FlashAttn] BAD sum_val q=%d val=%f\n", q_idx, sum_val);
            }
            #endif
            
            float score = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += Q_smem[q_idx][d] * K_smem[kv_idx][d];
            }
            // Apply QK-normalization with learnable per-head scales (nGPT-style)
            if (qk_norm_enabled) {
                score *= alpha_q_val * Q_norm_smem[q_idx] * alpha_k_val * K_norm_smem[kv_idx];
            }
            score *= scale;
            if (apply_alibi) {
                score += alibi_slope * static_cast<float>((kv_start + kv_idx) - (q_start + q_idx));
            }
            if ((kv_start + kv_idx) > (q_start + q_idx)) {
                score = -INFINITY;  // Pass 3: dK/dQ computation
            }
            
            // Store P in S_smem for reuse
            S_smem[q_idx][kv_idx] = __expf(score - max_val) / sum_val;
        }
        __syncthreads();
        
        // Step 2: dV += P^T @ dO (use cached P from S_smem - no score recomputation!)
        // GQA: Scale by 1/heads_per_kv_group since multiple Q heads accumulate to same KV head
        for (int i = tid; i < kv_size * HEAD_DIM; i += num_threads) {
            const int kv_idx = i / HEAD_DIM;
            const int d = i % HEAD_DIM;
            float dv = 0.0f;
            for (int q_idx = 0; q_idx < q_size; ++q_idx) {
                dv += S_smem[q_idx][kv_idx] * dO_smem[q_idx][d];
            }
            // Apply GQA gradient scaling before atomic accumulation
            dv *= gqa_grad_scale;
            if (dv != 0.0f) {
                atomicAdd(&dV_head[(kv_start + kv_idx) * head_dim + d], dv);
            }
        }
        __syncthreads();
        
        // Step 3: Convert P to dS = P * (dP - dp_sum) * scale (reuse S_smem)
        // DECISION: Always use forward scale for chain rule consistency
        // The backward gradient must use the same scale that was used in forward softmax(QK^T * scale)
        // - QK-norm mode: scale = qk_norm_scale / temperature
        // - Standard mode: scale = 1/sqrt(d) / temperature  
        // Using a different "backward_scale" breaks the chain rule: d/dx softmax(x*s) = s * softmax_jacobian
        const float dS_scale = scale;  // Always match forward pass
        
        for (int idx = tid; idx < q_size * kv_size; idx += num_threads) {
            const int q_idx = idx / kv_size;
            const int kv_idx = idx % kv_size;
            
            const float P = S_smem[q_idx][kv_idx];
            
            // Guard: If P == 0, dS = 0 (softmax Jacobian has zero row) → skip computation
            if (P == 0.0f) {
                S_smem[q_idx][kv_idx] = 0.0f;
                continue;
            }
            
            const float dp_sum = dp_sum_smem[q_idx];
            
            // dP = dO @ V^T
            float dP = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; ++d) {
                dP += dO_smem[q_idx][d] * V_smem[kv_idx][d];
            }
            
            // dS = P * (dP - dp_sum) * dS_scale
            S_smem[q_idx][kv_idx] = P * (dP - dp_sum) * dS_scale;
        }
        __syncthreads();
        
        // DEBUG: Check dS values after Step 3
        #ifdef FLASH_ATTN_DEBUG
        // if (tid == 0 && batch_idx == 0 && head_idx == 0 && q_block_idx == 0 && kv_block_idx == 0) {
        //     float ds_sum = 0.0f;
        //     float ds_max = 0.0f;
        //     for (int i = 0; i < q_size; i++) {
        //         for (int j = 0; j < kv_size; j++) {
        //             float val = fabsf(S_smem[i][j]);
        //             ds_sum += val;
        //             ds_max = fmaxf(ds_max, val);
        //         }
        //     }
        //     printf("[FlashAttn DEBUG] Step3 dS: sum=%e max=%e dp_sum[0]=%e P[0][0]=%e scale=%e\n", 
        //            ds_sum, ds_max, dp_sum_smem[0], S_smem[0][0], dS_scale);
        // }
        // __syncthreads();
        #endif
        
        // Step 4: ACCUMULATE dQ contributions across KV tiles
        // When QK-norm enabled: accumulate g_q̂ = dS @ k̂ (Jacobian applied AFTER all tiles)
        // Note: k̂ = α_k * k/||k||, so g_q̂ = α_k * dS @ (k/||k||)
        // When QK-norm disabled: accumulate dQ = dS @ K directly
        // NOTE: dQ does NOT need GQA scaling because Q heads are independent (not shared)
        // Only dV and dK need scaling since they accumulate to shared KV heads
        for (int i = tid; i < q_size * HEAD_DIM; i += num_threads) {
            const int q_idx = i / HEAD_DIM;
            const int d = i % HEAD_DIM;
            
            float dq_contrib = 0.0f;
            for (int kv_idx = 0; kv_idx < kv_size; ++kv_idx) {
                float k_val = K_smem[kv_idx][d];
                if (qk_norm_enabled) {
                    // Use normalized K: k̂ = α_k * K/||K||
                    // Include α_k factor since g_q̂ = dS @ k̂
                    k_val *= alpha_k_val * K_norm_smem[kv_idx];
                }
                dq_contrib += S_smem[q_idx][kv_idx] * k_val;
            }
            // ACCUMULATE (not overwrite!) - dQ_smem holds g_q̂ when QK-norm enabled
            dQ_smem[q_idx][d] += dq_contrib;
        }
        __syncthreads();
        
        // DEBUG: Check dQ_smem after Step 4
        #ifdef FLASH_ATTN_DEBUG
        // if (tid == 0 && batch_idx == 0 && head_idx == 0 && q_block_idx == 0 && kv_block_idx == 0) {
        //     float dq_sum = 0.0f;
        //     float dq_max = 0.0f;
        //     for (int i = 0; i < q_size; i++) {
        //         for (int d = 0; d < HEAD_DIM; d++) {
        //             float val = fabsf(dQ_smem[i][d]);
        //             dq_sum += val;
        //             dq_max = fmaxf(dq_max, val);
        //         }
        //     }
        //     printf("[FlashAttn DEBUG] Step4 dQ_smem: sum=%e max=%e K_smem[0][0]=%e\n", 
        //            dq_sum, dq_max, K_smem[0][0]);
        // }
        // __syncthreads();
        #endif
        
        // Step 5: Compute dK with proper L2-normalization Jacobian
        // Forward: k̂ = k/||k||, score = scale * (q̂ · k̂)
        // Backward through normalization: g_k = (1/||k||)(g_k̂ - k̂(k̂ᵀ·g_k̂))
        // This projects out the component along k̂
        
        // First, compute g_k̂ = dSᵀ @ Q̂ for all K positions in this tile
        // Need shared memory for intermediate g_khat storage
        __shared__ float dK_temp_smem[BLOCK_SIZE_KV][HEAD_DIM];
        
        // Initialize temp storage
        for (int i = tid; i < kv_size * HEAD_DIM; i += num_threads) {
            const int kv_idx = i / HEAD_DIM;
            const int d = i % HEAD_DIM;
            dK_temp_smem[kv_idx][d] = 0.0f;
        }
        __syncthreads();
        
        for (int i = tid; i < kv_size * HEAD_DIM; i += num_threads) {
            const int kv_idx = i / HEAD_DIM;
            const int d = i % HEAD_DIM;
            
            if (qk_norm_enabled) {
                // Step 5a: Compute g_k̂ = dSᵀ @ q̂ (gradient wrt normalized K)
                // Note: q̂ = α_q * q/||q||, so g_k̂ = α_q * dSᵀ @ (q/||q||)
                float g_khat = 0.0f;
                for (int q_idx = 0; q_idx < q_size; ++q_idx) {
                    // q̂ = α_q * Q/||Q||
                    // Include α_q factor since g_k̂ = dSᵀ @ q̂
                    float q_hat = alpha_q_val * Q_smem[q_idx][d] * Q_norm_smem[q_idx];
                    g_khat += S_smem[q_idx][kv_idx] * q_hat;
                }
                dK_temp_smem[kv_idx][d] = g_khat;  // Temporarily store g_k̂
            } else {
                // Standard attention: dK = dSᵀ @ Q
                // GQA: Scale by 1/heads_per_kv_group since multiple Q heads accumulate to same KV head
                float dk = 0.0f;
                for (int q_idx = 0; q_idx < q_size; ++q_idx) {
                    dk += S_smem[q_idx][kv_idx] * Q_smem[q_idx][d];
                }
                dk *= gqa_grad_scale;  // Apply GQA gradient scaling
                if (dk != 0.0f) {
                    atomicAdd(&dK_head[(kv_start + kv_idx) * head_dim + d], dk);
                }
            }
        }
        __syncthreads();
        
        // Step 5b-c: Apply normalization Jacobian projection (QK-norm only)
        // g_k = (g_k̂ - k̂(k̂ᵀ·g_k̂)) * inv_norm
        if (qk_norm_enabled) {
            for (int kv_idx = tid; kv_idx < kv_size; kv_idx += num_threads) {
                const float inv_norm = K_norm_smem[kv_idx];
                
                // Compute k̂ᵀ · g_k̂ (dot product)
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float k_hat = K_smem[kv_idx][d] * inv_norm;
                    dot += k_hat * dK_temp_smem[kv_idx][d];  // dK_temp holds g_k̂
                }
                
                // Accumulate alpha_k gradient: d_alpha_k = sum((k/||k||) · g_k̂) = sum(k̂ · g_k̂) / α_k
                // Since k̂ = α_k * (k/||k||), we have ∂k̂/∂α_k = k/||k|| = k̂/α_k
                // So: ∂L/∂α_k = g_k̂ · (k̂/α_k) = (k̂ · g_k̂) / α_k
                // GQA: Scale by 1/heads_per_kv_group since multiple Q heads contribute
                atomicAdd(&local_grad_alpha_k, (dot / alpha_k_val) * gqa_grad_scale);
                
                // Apply projection and atomic add: g_k = (g_k̂ - k̂ * dot) * inv_norm
                // GQA: Scale by 1/heads_per_kv_group since multiple Q heads accumulate to same KV head
                for (int d = 0; d < HEAD_DIM; ++d) {
                    float g_khat = dK_temp_smem[kv_idx][d];
                    float k_hat = K_smem[kv_idx][d] * inv_norm;
                    float g_k = (g_khat - k_hat * dot) * inv_norm * gqa_grad_scale;
                    if (g_k != 0.0f) {
                        atomicAdd(&dK_head[(kv_start + kv_idx) * head_dim + d], g_k);
                    }
                }
            }
            __syncthreads();
        };
    }
    
    // ==================== Apply QK-norm Jacobian to dQ AFTER all KV tiles ====================
    // dQ_smem currently holds g_q̂ (gradient wrt normalized Q, summed over all K positions)
    // Apply Jacobian: g_q = (g_q̂ - q̂(q̂ᵀ·g_q̂)) / ||q||
    if (qk_norm_enabled) {
        for (int q_idx = tid; q_idx < q_size; q_idx += num_threads) {
            const float inv_norm = Q_norm_smem[q_idx];
            
            // Compute q̂ᵀ · g_q̂ (dot product across all dimensions)
            float dot = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                float q_hat = Q_smem[q_idx][d] * inv_norm;
                dot += q_hat * dQ_smem[q_idx][d];  // dQ_smem holds accumulated g_q̂
            }
            
            // Accumulate alpha_q gradient: d_alpha_q = sum((q/||q||) · g_q̂) = sum(q̂ · g_q̂) / α_q
            // Since q̂ = α_q * (q/||q||), we have ∂q̂/∂α_q = q/||q|| = q̂/α_q
            // So: ∂L/∂α_q = g_q̂ · (q̂/α_q) = (q̂ · g_q̂) / α_q
            // GQA: Scale by 1/heads_per_kv_group since multiple Q heads contribute
            atomicAdd(&local_grad_alpha_q, (dot / alpha_q_val) * gqa_grad_scale);
            
            // Apply projection: g_q = (g_q̂ - q̂ * dot) * inv_norm
            for (int d = 0; d < HEAD_DIM; ++d) {
                float g_qhat = dQ_smem[q_idx][d];
                float q_hat = Q_smem[q_idx][d] * inv_norm;
                float g_q = (g_qhat - q_hat * dot) * inv_norm;
                dQ_smem[q_idx][d] = g_q;  // Now contains actual dQ
            }
        }
        __syncthreads();
    }
    
    // Write dQ from shared to global
    for (int i = tid; i < q_size * HEAD_DIM; i += num_threads) {
        const int q_idx = i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        const int global_idx = (q_start + q_idx) * head_dim + d;
        
        // Bounds check - skip invalid indices silently
        if (global_idx >= seq_len * head_dim) {
            continue;
        }
        
        dQ_head[global_idx] = dQ_smem[q_idx][d];
    }

    // ==================== Write learnable alpha gradients to global memory ====================
    // Each block accumulates partial gradients in shared memory, then atomically adds to global
    // GQA: grad_alpha_q uses head_idx, grad_alpha_k uses kv_head_idx
    __syncthreads();
    if (tid == 0 && qk_norm_enabled) {
        if (grad_alpha_q != nullptr) {
            atomicAdd(&grad_alpha_q[head_idx], local_grad_alpha_q);
        }
        if (grad_alpha_k != nullptr) {
            atomicAdd(&grad_alpha_k[kv_head_idx], local_grad_alpha_k);  // GQA: index by KV head
        }
    }
}

// Host function for Flash Attention backward pass
// Supports Grouped Query Attention (GQA): K,V have num_kv_heads, Q has num_heads
void flashAttentionBackward(
    const float* Q,
    const float* K,
    const float* V,
    const float* output,
    const float* grad_output,
    float* grad_Q,
    float* grad_K,
    float* grad_V,
    const PBM::PBMSpec& pos_encoding,
    const FlashAttentionConfig& config
) {
    // ========== VALIDATION CHECKS ==========
    // 0. Basic dimension validation - zero or negative dims cause "invalid argument" kernel launch errors
    if (config.batch_size <= 0) {
        printf("[FlashAttn Backward] FATAL: batch_size=%d must be positive\n", config.batch_size);
        return;
    }
    if (config.seq_len <= 0) {
        printf("[FlashAttn Backward] FATAL: seq_len=%d must be positive\n", config.seq_len);
        return;
    }
    if (config.num_heads <= 0) {
        printf("[FlashAttn Backward] FATAL: num_heads=%d must be positive\n", config.num_heads);
        return;
    }
    
    // 1. Head dimension must be exactly 32 or 64 - template unrolls require exact match
    if (config.head_dim != 32 && config.head_dim != 64) {
        printf("[FlashAttn Backward] FATAL: head_dim=%d not supported. Must be exactly 32 or 64.\n", 
               config.head_dim);
        printf("[FlashAttn Backward] Template unrolling reads HEAD_DIM elements - mismatched dims corrupt memory.\n");
        return;  // Hard fail - do not silently corrupt gradients
    }
    
    // 2. GQA divisibility check - num_heads must be evenly divisible by num_kv_heads
    if (config.num_heads % config.num_kv_heads != 0) {
        printf("[FlashAttn Backward] FATAL: num_heads=%d not divisible by num_kv_heads=%d\n",
               config.num_heads, config.num_kv_heads);
        printf("[FlashAttn Backward] GQA requires num_heads %% num_kv_heads == 0\n");
        return;  // Hard fail - incorrect KV head mapping corrupts gradients
    }
    
    // 3. Pointer validation - null pointers cause "invalid argument" errors
    if (!Q || !K || !V || !output || !grad_output || !grad_Q || !grad_K || !grad_V) {
        printf("[FlashAttn Backward] FATAL: NULL pointer detected\n");
        printf("  Q=%p K=%p V=%p output=%p grad_output=%p grad_Q=%p grad_K=%p grad_V=%p\n",
               Q, K, V, output, grad_output, grad_Q, grad_K, grad_V);
        return;
    }
    
    // Q and grad_Q use full num_heads
    const size_t q_elements = config.batch_size * config.num_heads * 
                              config.seq_len * config.head_dim;
    // K, V, grad_K, grad_V use num_kv_heads (GQA)
    const size_t kv_elements = config.batch_size * config.num_kv_heads * 
                               config.seq_len * config.head_dim;
    
    // FAIL LOUD: Stream MUST be provided - no silent fallback to default stream
    if (config.stream == nullptr) {
        printf("[FlashAttn Backward] FATAL: config.stream is NULL - caller MUST provide valid CUDA stream\n");
        return;
    }
    cudaStream_t stream = config.stream;
    
    // Zero gradients on the correct stream
    // GQA: grad_Q has num_heads, grad_K/grad_V have num_kv_heads
    cudaError_t err;
    err = cudaMemsetAsync(grad_Q, 0, q_elements * sizeof(float), stream);
    if (err != cudaSuccess) {
        printf("[FlashAttn Host] ERROR: cudaMemsetAsync(grad_Q) failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemsetAsync(grad_K, 0, kv_elements * sizeof(float), stream);
    if (err != cudaSuccess) {
        printf("[FlashAttn Host] ERROR: cudaMemsetAsync(grad_K) failed: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemsetAsync(grad_V, 0, kv_elements * sizeof(float), stream);
    if (err != cudaSuccess) {
        printf("[FlashAttn Host] ERROR: cudaMemsetAsync(grad_V) failed: %s\n", cudaGetErrorString(err));
    }
    
    // Zero alpha gradients if provided
    // GQA: alpha_q has num_heads, alpha_k has num_kv_heads
    if (config.grad_alpha_q != nullptr) {
        err = cudaMemsetAsync(config.grad_alpha_q, 0, config.num_heads * sizeof(float), stream);
        if (err != cudaSuccess) {
            printf("[FlashAttn Host] ERROR: cudaMemsetAsync(grad_alpha_q) failed: %s\n", cudaGetErrorString(err));
        }
    }
    if (config.grad_alpha_k != nullptr) {
        err = cudaMemsetAsync(config.grad_alpha_k, 0, config.num_kv_heads * sizeof(float), stream);
        if (err != cudaSuccess) {
            printf("[FlashAttn Host] ERROR: cudaMemsetAsync(grad_alpha_k) failed: %s\n", cudaGetErrorString(err));
        }
    }
    
    const int num_q_blocks = (config.seq_len + FLASH_BLOCK_Q - 1) / FLASH_BLOCK_Q;
    
    // Grid: x=Q blocks, y=Q heads (kernel handles KV head mapping internally), z=batch
    dim3 grid(num_q_blocks, config.num_heads, config.batch_size);
    
    // Validate grid dimensions - zero dims cause "invalid argument"
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) {
        printf("[FlashAttn Backward] FATAL: Invalid grid dimensions (%d, %d, %d)\n", 
               grid.x, grid.y, grid.z);
        printf("  num_q_blocks=%d, num_heads=%d, batch_size=%d, seq_len=%d\n",
               num_q_blocks, config.num_heads, config.batch_size, config.seq_len);
        return;
    }
    
    // Dispatch based on head_dim only - block sizes are fixed
    if (config.head_dim == 64) {
        flashAttentionBackwardKernel<FLASH_BLOCK_Q, FLASH_BLOCK_KV, 64><<<grid, FLASH_NUM_THREADS, 0, stream>>>(
            Q, K, V, output, grad_output, grad_Q, grad_K, grad_V, pos_encoding,
            config.batch_size, config.num_heads, config.num_kv_heads, config.seq_len, config.head_dim,
            config.use_alibi, config.alibi_scale, config.softmax_temperature,
            config.qk_norm_enabled, config.qk_norm_scale,
            config.alpha_q, config.alpha_k, config.grad_alpha_q, config.grad_alpha_k
        );
    } else if (config.head_dim == 32) {
        flashAttentionBackwardKernel<FLASH_BLOCK_Q, FLASH_BLOCK_KV, 32><<<grid, FLASH_NUM_THREADS, 0, stream>>>(
            Q, K, V, output, grad_output, grad_Q, grad_K, grad_V, pos_encoding,
            config.batch_size, config.num_heads, config.num_kv_heads, config.seq_len, config.head_dim,
            config.use_alibi, config.alibi_scale, config.softmax_temperature,
            config.qk_norm_enabled, config.qk_norm_scale,
            config.alpha_q, config.alpha_k, config.grad_alpha_q, config.grad_alpha_k
        );
    } else {
        // Should never reach here due to validation above, but defensive check
        printf("[FlashAttn Backward] FATAL: Unsupported head_dim=%d (validation should have caught this)\n", 
               config.head_dim);
        return;
    }
    
    // Check for kernel launch errors (synchronous)
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[FlashAttn Backward] CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }
    
    // DELETED: collectAttentionDiagnostics (Rule 20 - broken diagnostic with MAX_ATTEND=128 truncation)
    // Use config.entropy_output in forward pass for accurate entropy over full sequences
    
    // NOTE: Removed cudaDeviceSynchronize() - errors propagate to next forward pass
    // Use runtime ctx.enable_grad_checks for debugging, not compile-time macros
}

void flashAttentionBackwardWithWorkspace(
    const float* Q,
    const float* K,
    const float* V,
    const float* output,
    const float* grad_output,
    float* grad_Q,
    float* grad_K,
    float* grad_V,
    const PBM::PBMSpec& pos_encoding,
    void* workspace,
    const FlashAttentionConfig& config
) {
    flashAttentionBackward(Q, K, V, output, grad_output, grad_Q, grad_K, grad_V, 
                          pos_encoding, config);
}

} // namespace GRIM

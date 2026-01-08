#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <atomic>
#include <cfloat>
#include <cstdlib>
#include <cmath>

#include "../../Shared/PBM/PositionalBiasMethod.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM {

//======================================================//
// Attention Diagnostics - Rule 20: No backwards compat
// Dumps attention stats every step when enabled
// FAIL-LOUD on -FLT_MAX or NaN detection
//======================================================//
struct AttentionDiagnostics {
    bool enabled = true;           // Master switch
    int dump_layer = -1;            // Which layer to dump (-1 = all)
    int dump_head = -1;             // Which head to dump (-1 = all, 0 = first only for perf)
    int dump_batch = 0;             // Which batch item to dump
    int current_layer = 0;          // Set by caller before forward/backward
    int current_step = 0;           // Set by caller (training batch index)
    
    // Output stats (written by kernel, read by host)
    float max_attn_prob = 0.0f;     // Max attention probability EXCLUDING pos 0 (1.0 = collapsed)
    float min_attn_prob = 0.0f;     // Min attention probability (excluding padding)
    float mean_entropy = 0.0f;      // Mean entropy per position in BITS (not summed!)
    float qk_dot_max = 0.0f;        // Max Q·K dot product (before softmax)
    float qk_dot_min = 0.0f;        // Min Q·K dot product
    float grad_q_norm = 0.0f;       // |grad_Q| from backward
    float grad_k_norm = 0.0f;       // |grad_K| from backward
    float grad_v_norm = 0.0f;       // |grad_V| from backward
    int positions_sampled = 0;      // Number of positions actually sampled (for transparency)
    
    void print() const {
        if (!enabled) return;
        
        // Rule 20: FAIL-LOUD on invalid QK values
        // -FLT_MAX threshold: anything below -1e30 is suspicious
        constexpr float INVALID_THRESHOLD = -1e30f;
        
        if (qk_dot_min < INVALID_THRESHOLD || qk_dot_max < INVALID_THRESHOLD) {
            fprintf(stderr, "\n");
            fprintf(stderr, "╔══════════════════════════════════════════════════════════════════╗\n");
            fprintf(stderr, "║ FATAL: -FLT_MAX DETECTED IN QK DOT PRODUCTS                      ║\n");
            fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
            fprintf(stderr, "║ Step:  %d                                                        \n", current_step);
            fprintf(stderr, "║ Layer: %d                                                        \n", current_layer);
            fprintf(stderr, "║ QK min: %.6e                                                     \n", qk_dot_min);
            fprintf(stderr, "║ QK max: %.6e                                                     \n", qk_dot_max);
            fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
            fprintf(stderr, "║ DIAGNOSIS:                                                       ║\n");
            fprintf(stderr, "║ - If qk_max = -FLT_MAX: ALL queries see only masked positions   ║\n");
            fprintf(stderr, "║   => Causal mask is masking ALL keys for some queries           ║\n");
            fprintf(stderr, "║   => Check: is seq_len correct? Q/K memory layout correct?      ║\n");
            fprintf(stderr, "║ - If qk_min = -FLT_MAX but max is normal: Only some positions   ║\n");
            fprintf(stderr, "║   => This is expected from causal masking (future positions)    ║\n");
            fprintf(stderr, "║ - If both are -FLT_MAX: K tensor may be all zeros or NaN        ║\n");
            fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
            fprintf(stderr, "\n");
            std::abort();
        }
        
        // Also check for NaN
        if (std::isnan(qk_dot_min) || std::isnan(qk_dot_max) || 
            std::isnan(grad_q_norm) || std::isnan(grad_k_norm) || std::isnan(grad_v_norm)) {
            fprintf(stderr, "\n");
            fprintf(stderr, "╔══════════════════════════════════════════════════════════════════╗\n");
            fprintf(stderr, "║ FATAL: NaN DETECTED IN ATTENTION DIAGNOSTICS                    ║\n");
            fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
            fprintf(stderr, "║ Step:  %d                                                        \n", current_step);
            fprintf(stderr, "║ Layer: %d                                                        \n", current_layer);
            fprintf(stderr, "║ QK: [%.6e, %.6e]                                                 \n", qk_dot_min, qk_dot_max);
            fprintf(stderr, "║ Grads: Q=%.6e K=%.6e V=%.6e                                      \n", grad_q_norm, grad_k_norm, grad_v_norm);
            fprintf(stderr, "╠══════════════════════════════════════════════════════════════════╣\n");
            fprintf(stderr, "║ DIAGNOSIS: NaN propagated from earlier computation              ║\n");
            fprintf(stderr, "║ - Check W_qkv initialization (should be Xavier/He scaled)       ║\n");
            fprintf(stderr, "║ - Check input embeddings for NaN                                ║\n");
            fprintf(stderr, "║ - Check LayerNorm for division by zero                          ║\n");
            fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
            fprintf(stderr, "\n");
            std::abort();
        }
        
        // Format: max_prob EXCLUDES pos 0 (trivial 1.0), mean_entropy is PER-POSITION in bits
        printf("[AttnDiag] step=%d layer=%d (n=%d): "
               "probs=[%.4f,%.4f] H=%.2f bits/pos qk=[%.2f,%.2f] "
               "grads=[Q:%.6f K:%.6f V:%.6f]\n",
               current_step, current_layer, positions_sampled,
               min_attn_prob, max_attn_prob, mean_entropy,
               qk_dot_min, qk_dot_max,
               grad_q_norm, grad_k_norm, grad_v_norm);
    }
};

// Global diagnostics instance (Rule 22: centralized, but GRIM-text specific per Rule 23)
inline AttentionDiagnostics& getAttentionDiagnostics() {
    static AttentionDiagnostics instance;
    return instance;
}

//======================================================//
// K-Tensor Trace System - Rule 20: Fail-loud on bad K
// Logs K statistics after each operation that modifies K
//======================================================//
struct KTensorTrace {
    bool enabled = true;              // Master switch
    int current_step = 0;              // Training batch index
    int current_layer = 0;             // Encoder layer index
    
    // Stats computed by traceK()
    float k_min = 0.0f;
    float k_max = 0.0f;
    float k_mean = 0.0f;
    float k_std = 0.0f;
    int k_nan_count = 0;
    int k_inf_count = 0;
    
    void enable() { enabled = true; }
    void disable() { enabled = false; }
};

inline KTensorTrace& getKTensorTrace() {
    static KTensorTrace instance;
    return instance;
}

// Forward declaration - implemented in Flash_Attention_Kernal.cu
// Computes and logs K tensor statistics, crashes if NaN/Inf detected
void traceKTensor(const float* K, 
                  int total_elements,
                  const char* operation_name,
                  int layer_idx,
                  cudaStream_t stream);

struct FlashAttentionConfig {
    int batch_size;
    int num_heads;           // Number of query heads
    int num_kv_heads;        // Number of key/value heads (GQA: num_kv_heads < num_heads)
    int seq_len;
    int head_dim;
    int block_size_q;
    int block_size_kv;
    bool use_alibi;
    float alibi_scale;
    float softmax_temperature;  // Temperature for attention softmax (>1 reduces saturation)
    bool qk_norm_enabled;       // Enable QK-normalization
    float qk_norm_scale;        // Scale factor after normalization (typically sqrt(head_dim) or learned)
    
    // Learnable per-head QK-norm scales (nGPT-style)
    // When non-null: q̂ = alpha_q[head] * (q / ||q||), k̂ = alpha_k[head] * (k / ||k||)
    // Arrays of size [num_heads], one scale per attention head
    // Set to nullptr to use fixed qk_norm_scale for all heads
    const float* alpha_q;       // [num_heads] - learnable Q scale per head (forward only)
    const float* alpha_k;       // [num_kv_heads] - learnable K scale per KV head (forward only)
    float* grad_alpha_q;        // [num_heads] - gradient output for alpha_q (backward only)
    float* grad_alpha_k;        // [num_kv_heads] - gradient output for alpha_k (backward only)
    
    // Entropy output (optional diagnostic) - computed using actual online softmax values
    // When non-null, writes per-head entropy in bits: [batch_size, num_heads]
    // Entropy computed as: H = -sum(P * log2(P)) averaged over all Q positions
    float* entropy_output;      // [batch_size * num_heads] - optional entropy output
    
    cudaStream_t stream;

    FlashAttentionConfig()
        : batch_size(1), num_heads(8), num_kv_heads(8), seq_len(128), head_dim(64),
          block_size_q(HyperParameters::FLASH_ATTN_BLOCK_Q),    // Use centralized constant
          block_size_kv(HyperParameters::FLASH_ATTN_BLOCK_KV),  // Use centralized constant
          use_alibi(false), alibi_scale(1.0f), 
          softmax_temperature(HyperParameters::SOFTMAX_TEMPERATURE),  // Use centralized constant
          qk_norm_enabled(HyperParameters::QK_NORMALIZATION_ENABLED),
          qk_norm_scale(HyperParameters::QK_NORM_SCALE),
          alpha_q(nullptr), alpha_k(nullptr),
          grad_alpha_q(nullptr), grad_alpha_k(nullptr),
          entropy_output(nullptr),
          stream(nullptr) {}
    
    // Helper: Get the number of Q heads that share each KV head
    int getHeadsPerKVGroup() const {
        return (num_kv_heads > 0) ? (num_heads / num_kv_heads) : num_heads;
    }
    
    // Helper: Check if GQA is active (fewer KV heads than Q heads)
    bool isGQA() const {
        return num_kv_heads < num_heads;
    }
    
    // Helper: Get the KV head index for a given Q head
    int getKVHeadIndex(int q_head) const {
        return q_head / getHeadsPerKVGroup();
    }
};

void flashAttentionForward(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const PBM::PBMSpec& pos_encoding,
    const FlashAttentionConfig& config
);

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
);

void computeOptimalBlockSizes(
    int seq_len,
    int head_dim,
    int& block_size_q,
    int& block_size_kv
);

size_t getFlashAttentionWorkspaceSize(const FlashAttentionConfig& config);

void flashAttentionForwardWithWorkspace(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    const PBM::PBMSpec& pos_encoding,
    void* workspace,
    const FlashAttentionConfig& config
);

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
);

} // namespace GRIM


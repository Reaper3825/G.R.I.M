#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace GRIM {

//======================================================//
// Attention Diagnostics (legacy stats - optional)
//======================================================//
struct AttentionDiagnostics {
    bool enabled = true;
    int dump_layer = -1;
    int dump_head = -1;
    int dump_batch = 0;
    int current_layer = 0;
    int current_step = 0;

    float max_attn_prob = 0.0f;
    float min_attn_prob = 0.0f;
    float mean_entropy = 0.0f;
    float qk_dot_max = 0.0f;
    float qk_dot_min = 0.0f;
    float grad_q_norm = 0.0f;
    float grad_k_norm = 0.0f;
    float grad_v_norm = 0.0f;
    int positions_sampled = 0;

    void print() const {
        if (!enabled) return;

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
            fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
            fprintf(stderr, "\n");
            std::abort();
        }

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
            fprintf(stderr, "╚══════════════════════════════════════════════════════════════════╝\n");
            fprintf(stderr, "\n");
            std::abort();
        }

        printf("[AttnDiag] step=%d layer=%d (n=%d): "
               "probs=[%.4f,%.4f] H=%.2f bits/pos qk=[%.2f,%.2f] "
               "grads=[Q:%.6f K:%.6f V:%.6f]\n",
               current_step, current_layer, positions_sampled,
               min_attn_prob, max_attn_prob, mean_entropy,
               qk_dot_min, qk_dot_max,
               grad_q_norm, grad_k_norm, grad_v_norm);
    }
};

inline AttentionDiagnostics& getAttentionDiagnostics() {
    static AttentionDiagnostics instance;
    return instance;
}

//======================================================//
// K-Tensor Trace (fail-loud on NaN/Inf/-FLT_MAX)
//======================================================//
struct KTensorTrace {
    bool enabled = true;
    int current_step = 0;
    int current_layer = 0;

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

void traceKTensor(const float* K,
                  int total_elements,
                  const char* operation_name,
                  int layer_idx,
                  cudaStream_t stream);

} // namespace GRIM

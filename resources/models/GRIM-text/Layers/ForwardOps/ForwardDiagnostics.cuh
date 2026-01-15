#pragma once
/**
 * ForwardDiagnostics.cuh
 * 
 * Diagnostic utilities to trace values through the forward pass.
 * Enables logging of buffer statistics (mean, var, min, max, NaN count)
 * to debug training plateau issues.
 * 
 * EXPECTED VALUES (after proper initialization):
 * - Embeddings: mean≈0, var≈1/d_model (Xavier init)
 * - After RMSNorm: mean≈0, rms≈1 
 * - Encoder output: mean≈0, var varies by layer depth
 * - Logits: mean≈0 (if well-initialized), range [-10, 10] typical
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cfloat>
#include <string>
#include "../../Shared/LogRecorder/LogRecorder.hpp"

namespace GRIM {
namespace Forward {

// Enable/disable all forward diagnostics (compile-time toggle)
#ifndef FORWARD_DIAGNOSTICS_ENABLED
#define FORWARD_DIAGNOSTICS_ENABLED 1
#endif

// Struct to hold buffer statistics
struct BufferStats {
    float mean;
    float variance;
    float min_val;
    float max_val;
    int nan_count;
    int inf_count;
    size_t num_elements;
    
    void print(const char* name) const {
        fprintf(stderr, "[ForwardDiag] %s: elements=%zu mean=%.6f var=%.6f min=%.4f max=%.4f nan=%d inf=%d\n",
                name, num_elements, mean, variance, min_val, max_val, nan_count, inf_count);
    }
    
    bool isHealthy() const {
        return nan_count == 0 && inf_count == 0 && 
               !std::isnan(mean) && !std::isinf(mean) &&
               !std::isnan(variance) && !std::isinf(variance);
    }
};

// Declaration only - defined in ForwardDiagnostics.cu
BufferStats computeBufferStats(
    const float* d_buffer,
    size_t num_elements,
    cudaStream_t stream = 0
);

// Convenience function with expected value checking
inline void logBufferWithExpected(
    const char* name,
    const float* d_buffer,
    size_t num_elements,
    float expected_mean,
    float expected_var,
    float expected_min,
    float expected_max,
    cudaStream_t stream = 0
) {
#if FORWARD_DIAGNOSTICS_ENABLED
    BufferStats stats = computeBufferStats(d_buffer, num_elements, stream);
    
    fprintf(stderr, "[ForwardDiag] %s:\n", name);
    fprintf(stderr, "  Actual:   mean=%.6f var=%.6f min=%.4f max=%.4f (nan=%d inf=%d)\n",
            stats.mean, stats.variance, stats.min_val, stats.max_val, stats.nan_count, stats.inf_count);
    fprintf(stderr, "  Expected: mean~%.2f var~%.4f range=[%.2f, %.2f]\n",
            expected_mean, expected_var, expected_min, expected_max);
    
    // Flag anomalies
    if (!stats.isHealthy()) {
        fprintf(stderr, "  [X] UNHEALTHY: Contains NaN/Inf!\n");
    }
    if (std::abs(stats.mean - expected_mean) > 1.0f) {
        fprintf(stderr, "  [!] Mean deviation: %.4f (expected ~%.2f)\n", stats.mean, expected_mean);
    }
    if (stats.variance > expected_var * 10.0f) {
        fprintf(stderr, "  [!] High variance: %.6f (expected ~%.4f)\n", stats.variance, expected_var);
    }
    if (stats.min_val < expected_min || stats.max_val > expected_max) {
        fprintf(stderr, "  [!] Range exceeded: [%.4f, %.4f] vs expected [%.2f, %.2f]\n",
                stats.min_val, stats.max_val, expected_min, expected_max);
    }
#endif
}

// Quick diagnostic macro for easy insertion
#define FWD_DIAG_BUFFER(name, buffer, elements, stream) \
    do { \
        if (FORWARD_DIAGNOSTICS_ENABLED && (buffer) && (elements) > 0) { \
            auto _stats = GRIM::Forward::computeBufferStats((buffer), (elements), (stream)); \
            _stats.print(name); \
        } \
    } while(0)

#define FWD_DIAG_BUFFER_EXPECTED(name, buffer, elements, exp_mean, exp_var, exp_min, exp_max, stream) \
    do { \
        if (FORWARD_DIAGNOSTICS_ENABLED && (buffer) && (elements) > 0) { \
            GRIM::Forward::logBufferWithExpected(name, (buffer), (elements), \
                (exp_mean), (exp_var), (exp_min), (exp_max), (stream)); \
        } \
    } while(0)

// ═══════════════════════════════════════════════════════════════════════════
// TOKEN 277 ALIGNMENT DIAGNOSTIC (Issue #37)
// Tracks how hidden states align with W[277] (SPACE token) throughout forward pass
// Key insight: logit[277] = hidden_state @ W[277]^T
// If hidden states learn to align with W[277] direction, logit_277 increases
// even if W[277] norm decreases
// ═══════════════════════════════════════════════════════════════════════════

struct Token277AlignmentStats {
    float dot_product_mean;      // Mean of hidden @ W[277]^T across all tokens
    float dot_product_max;       // Max alignment (worst case)
    float cosine_sim_mean;       // Mean cosine similarity (normalized alignment)
    float cosine_sim_max;        // Max cosine similarity
    float hidden_norm_mean;      // Mean hidden state norm
    float w277_norm;             // Norm of W[277] row
    int num_tokens;
    
    void print(const char* stage_name) const {
        char buf[512];
        snprintf(buf, sizeof(buf), 
                "[Token277Align] %s: dot_mean=%.4f dot_max=%.4f cos_mean=%.4f cos_max=%.4f h_norm=%.4f w277_norm=%.4f",
                stage_name, dot_product_mean, dot_product_max, 
                cosine_sim_mean, cosine_sim_max,
                hidden_norm_mean, w277_norm);
        GRIM::Logging::EmitModuleInfo("ForwardDiagnostics", std::string(buf));
    }
};

// Compute alignment between hidden states and W[277]
// hidden: [total_tokens, d_model] - row-major hidden states
// w277: [d_model] - the W[277] weight row from LM head (token 277 embedding)
Token277AlignmentStats computeToken277Alignment(
    const float* d_hidden,
    const float* d_w277,
    int total_tokens,
    int d_model,
    cudaStream_t stream
);

// Macro for easy insertion at each forward pass stage
#define FWD_DIAG_TOKEN277_ALIGNMENT(stage_name, hidden, w277, tokens, d_model, stream) \
    do { \
        GRIM::Logging::EmitModuleInfo("ForwardDiagnostics", "[Token277Align] MACRO_REACHED stage=" + std::string(stage_name) + \
                " enabled=" + std::to_string(FORWARD_DIAGNOSTICS_ENABLED) + \
                " hidden=" + std::to_string((unsigned long long)(void*)(hidden)) + \
                " w277=" + std::to_string((unsigned long long)(void*)(w277)) + \
                " tokens=" + std::to_string(tokens)); \
        if (FORWARD_DIAGNOSTICS_ENABLED && (hidden) && (w277) && (tokens) > 0) { \
            auto _align = GRIM::Forward::computeToken277Alignment((hidden), (w277), (tokens), (d_model), (stream)); \
            _align.print(stage_name); \
        } \
    } while(0)

} // namespace Forward
} // namespace GRIM

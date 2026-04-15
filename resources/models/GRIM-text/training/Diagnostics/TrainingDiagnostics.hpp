//======================================================//
//  TrainingDiagnostics.hpp
//  Diagnostic structs and functions extracted from Phase2_TrainingLoop.cu
//
//  Contains: WeightSample, EmbGradEquationDiag, Token277Diagnostic, UpdateTrace
//
//  Author: Extracted Feb 2026
//======================================================//

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

// Forward declarations — only pointers/references used in signatures
namespace GRIM {
    class EmbeddingLayer;
    class LMHeadLayer;
    struct TrainingState;
    struct ParameterGroup;
    enum class ParamGroupType : uint8_t;
}

namespace GRIM::Diagnostics {

//======================================================//
//  WeightSample — small weight snapshot for optimizer diagnostics
//======================================================//

constexpr int kWeightSampleSize = 10;

struct WeightSample {
    bool valid = false;
    float values[kWeightSampleSize] = {0.0f};
    float rms = 0.0f;
};

WeightSample sampleWeightStats(const GRIM::LMHeadLayer* lm_head, const GRIM::TrainingState& ts, bool sync_for_host = false);
std::string formatWeightSample(const WeightSample& sample);
float computeUpdateRms(const WeightSample& before, const WeightSample& after);

//======================================================//
//  EmbGradEquationDiag — Embedding Gradient Spike Diagnostic (Issue #141)
//  Rule 21: [EMB_GRAD_EQUATION] format
//======================================================//

struct EmbGradEquationDiag {
    bool valid = false;
    
    // Overall gradient buffer statistics
    float grad_rms = 0.0f;             // RMS = sqrt(sum_sq / count) — matches training's norm metric
    float mean_row_rms = 0.0f;         // mean(rms(grad_W[v])) across active vocab rows
    float max_row_rms = 0.0f;          // max(rms(grad_W[v]))
    int max_row_token = -1;            // Token ID with max row RMS
    float spike_ratio = 0.0f;          // max_row_rms / mean_row_rms (>10 = spike)
    
    // Top-5 gradient row RMS (identifies which tokens drive spikes)
    static constexpr int kTopK = 5;
    int top_tokens[5] = {};
    float top_rms[5] = {};
    
    // Scatter density (atomicAdd concentration)
    int num_active_rows = 0;           // Rows with non-zero gradient
    int total_vocab = 0;               // Total vocab size
    float active_ratio = 0.0f;         // active / total (lower = sparser → more atomicAdd per row)
    
    // Token frequency in batch (tokens that appear most often get most atomicAdd calls)
    int most_frequent_token = -1;
    int most_frequent_count = 0;
    
    // Per-batch delta tracking
    float prev_emb_rms = 0.0f;        // Previous batch emb_lm_tied rms (from gradient metrics)
    float curr_emb_rms = 0.0f;        // Current batch emb_lm_tied rms
    float emb_rms_delta = 0.0f;       // curr - prev (positive = growing spike)
};

EmbGradEquationDiag computeEmbGradEquation(
    const GRIM::EmbeddingLayer* embedding_layer,
    const int* d_token_ids,     // GPU-resident batch token IDs [total_tokens]
    int total_tokens,
    int d_model,
    int vocab_size,
    float prev_emb_rms,
    float curr_emb_rms,
    cudaStream_t stream
);
std::string formatEmbGradEquation(const EmbGradEquationDiag& diag, int batch_idx);

//======================================================//
//  Token277Diagnostic — Mode Collapse Token Tracking (Issue #36+)
//======================================================//

struct Token277Diagnostic {
    bool valid = false;
    int tracked_token = -1;          // Which token we're tracking (argmax of predictions)
    
    // Gradient info for the tracked embedding row
    float grad_row_rms = 0.0f;       // RMS of gradient for tracked row
    float grad_row_sum = 0.0f;       // Sum of gradient for tracked row (sign indicates direction)
    float grad_row_mean = 0.0f;      // Mean of gradient elements
    
    // Weight info for the tracked embedding row
    float weight_row_rms = 0.0f;     // RMS of embedding row
    float weight_row_mean = 0.0f;    // Mean of embedding row
    
    // LM head output projection analysis (for weight-tied case)
    // When logits = x @ W_emb^T, row N of W_emb affects ALL output logits
    // A positive update to row N increases dot product for ANY x
    
    // Target statistics in current batch
    int target_count = 0;            // How many times tracked token was the target
    int total_valid_targets = 0;     // Total valid (non-padding) targets
    float target_ratio = 0.0f;       // Ratio of tracked token in targets
};

Token277Diagnostic computeToken277Diagnostic(
    const GRIM::LMHeadLayer* lm_head,
    const GRIM::EmbeddingLayer* embedding_layer,
    const std::vector<int>& targets,  // Batch targets (flattened)
    int d_model,
    int tracked_token,  // Which token to track (argmax from predictions)
    cudaStream_t stream
);
std::string formatToken277Diagnostic(const Token277Diagnostic& diag, int batch_idx);


//======================================================//
//  UpdateTrace — Per-component Adam update magnitude (Issue #150)
//  Rule 21: [UPDATE_TRACE_EQUATION] format
//
//  Measures actual Adam update RMS per component type by sampling
//  moment buffers and computing: update ≈ lr × m_hat / (√v_hat + ε)
//  Then computes update_rms / param_rms to measure effective learning
//  rate relative to parameter scale.
//
//  This answers: "Does Adam normalize the 18× gradient gap between
//  attention and FFN into similar update magnitudes?"
//======================================================//

/// Number of elements to sample per component type for update estimation.
/// 64 gives <1% RMS error vs full-buffer computation (CLT: 1/√64 = 12.5%).
constexpr int kUpdateTraceSampleSize = 64;

/// Number of component types (matches ParamGroupType::COUNT)
constexpr int kNumComponentTypes = 6;

struct UpdateTraceMetrics {
    bool valid = false;

    // Per component type (indexed by ParamGroupType enum value 0..5):
    //   EMBEDDING=0, LM_HEAD=1, ATTENTION=2, FFN=3, RMSNORM=4, SCRATCHBLOCK=5
    float update_rms[kNumComponentTypes]     = {};  // RMS of Adam update per element
    float param_rms[kNumComponentTypes]      = {};  // RMS of parameter values
    float update_over_param[kNumComponentTypes] = {};  // update_rms / param_rms
    int   element_count[kNumComponentTypes]  = {};  // Elements sampled per type
    bool  has_data[kNumComponentTypes]       = {};  // Whether this type had groups

    // Names for logging
    static const char* typeName(int type_idx);
};

/// Compute per-component update_rms by sampling Adam moment buffers.
/// Call AFTER launchAdamWStep() on diagnostic-sync batches.
/// Requires cudaStreamSynchronize() internally for small D2H copies.
UpdateTraceMetrics computePerComponentUpdateTrace(
    const std::vector<GRIM::ParameterGroup>& groups,
    float learning_rate,
    int optimizer_step,       // step AFTER increment (1-based iteration count)
    cudaStream_t stream
);

/// Format as [UpdateTrace] log line
std::string formatUpdateTrace(const UpdateTraceMetrics& m, int batch_idx, bool tied);

} // namespace GRIM::Diagnostics

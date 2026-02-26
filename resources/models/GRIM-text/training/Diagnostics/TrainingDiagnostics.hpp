//======================================================//
//  TrainingDiagnostics.hpp
//  Diagnostic structs and functions extracted from Phase2_TrainingLoop.cu
//
//  Contains: WeightSample, EmbGradEquationDiag, Token277Diagnostic,
//            HiddenState277Analysis, FeedbackLoopDiagnostic, PC1CausalityTest
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
//  HiddenState277Analysis — Hidden State Analysis for tracked token (Issue #37)
//======================================================//

struct HiddenState277Analysis {
    bool valid = false;
    int tracked_token = -1;           // Which token we're tracking
    
    // Hidden state statistics
    float hidden_mean = 0.0f;         // Mean of all hidden states
    float hidden_rms_mean = 0.0f;     // Mean RMS per position
    float hidden_std = 0.0f;          // Std dev of hidden values
    
    // Split by target type
    float hidden_tracked_rms = 0.0f;  // Mean RMS at positions targeting tracked token
    float hidden_other_rms = 0.0f;    // Mean RMS at positions targeting other
    
    // Gradient logits for tracked token
    float grad_tracked_at_tracked_targets = 0.0f;  // Sum of grad_logits[t,tok] where target=tok
    float grad_tracked_at_other_targets = 0.0f;    // Sum of grad_logits[t,tok] where target≠tok
    
    // Issue #137: Per-dimension grad_W[tok] decomposition by target type
    float grad_w_rms_from_tracked = 0.0f;
    float grad_w_rms_from_other = 0.0f;
    float grad_w_sum_from_tracked = 0.0f;
    float grad_w_sum_from_other = 0.0f;
    float grad_w_cosine = 0.0f;
    
    // Counts
    int count_tracked_targets = 0;
    int count_other_targets = 0;
};

HiddenState277Analysis computeHiddenState277Analysis(
    const GRIM::TrainingState& ts,
    const std::vector<int>& targets,  // Batch targets (flattened)
    int d_model,
    int vocab_size,
    int tracked_token,   // Which token to track (argmax from predictions)
    bool use_centering,  // Issue #115: Added to read correct buffer
    cudaStream_t stream
);
std::string formatHiddenState277Analysis(const HiddenState277Analysis& a, int batch_idx);

//======================================================//
//  FeedbackLoopDiagnostic — Issue #114: FEEDBACK_LOOP_EQUATION (Rule 21)
//
//  logit[argmax[t]] = ||h[t]|| × ||W[argmax[t]]|| × cos(h[t], W[argmax[t]])
//======================================================//

struct FeedbackLoopDiagnostic {
    bool valid = false;
    int tracked_token = -1;  // Dynamically detected collapse token
    
    // ==========================================
    // Core Equation Components (Issue #114)
    // logit[T] = h_rms × W_rms × cos(h, W[T]) × d_model
    // ==========================================
    float hidden_rms_mean = 0.0f;     // h_rms averaged over positions
    float weight_tracked_rms = 0.0f;  // W_rms[T]
    float cosine_h_w_tracked_mean = 0.0f;  // cos(h, W[T]) averaged over positions
    
    // Predicted logit from decomposition (should match actual)
    float predicted_logit_tracked = 0.0f;
    float actual_logit_tracked_mean = 0.0f;
    float decomposition_error_pct = 0.0f;  // |predicted - actual| / actual * 100
    
    // ==========================================
    // Growth Rates (Issue #114 Anomaly Tracking)
    // These track d(metric)/d(batch) to detect explosion
    // ==========================================
    float hidden_rms_growth_pct = 0.0f;    // (current - previous) / previous * 100
    float weight_rms_growth_pct = 0.0f;
    float cosine_growth_pct = 0.0f;
    float logit_growth_pct = 0.0f;
    
    // ==========================================
    // Cosine Similarity Collapse Detection
    // avg_cos(h_i, h_j) measures hidden state correlation
    // Expected: ~1/sqrt(768) ≈ 0.036 for orthogonal vectors
    // Anomaly: avg_cos > 0.5 indicates collapse
    // ==========================================
    float avg_hidden_cosine = 0.0f;  // avg cos(h_i, h_j) pairwise
    int hidden_cosine_samples = 0;   // Number of pairs sampled
    float max_hidden_cosine = 0.0f;  // Max cos(h_i, h_j) across sampled pairs
    float min_hidden_cosine = 0.0f;  // Min cos(h_i, h_j) across sampled pairs
    int max_hidden_cosine_i = -1;    // Position index for max cosine (t_i)
    int max_hidden_cosine_j = -1;    // Position index for max cosine (t_j)
    int min_hidden_cosine_i = -1;    // Position index for min cosine (t_i)
    int min_hidden_cosine_j = -1;    // Position index for min cosine (t_j)
    static constexpr int kTopCosinePairs = 5;
    float top_cosines[5] = {};        // Top cosine values (descending)
    int top_cosine_i[5] = {};         // Position indices for top cosines (t_i)
    int top_cosine_j[5] = {};         // Position indices for top cosines (t_j)
    float bottom_cosines[5] = {};     // Bottom cosine values (ascending)
    int bottom_cosine_i[5] = {};      // Position indices for bottom cosines (t_i)
    int bottom_cosine_j[5] = {};      // Position indices for bottom cosines (t_j)
    
    // ==========================================
    // Weight Paradox Detection (Issue #114)
    // AdamW: W_new = W - lr * grad
    // grad · W > 0 → optimizer decreases ||W|| (gradient points WITH weight)
    // grad · W < 0 → optimizer increases ||W|| (gradient points AGAINST weight)
    // Paradox: grad · W > 0 (should shrink) BUT ||W[T]|| actually GREW
    // ==========================================
    float grad_dot_w_tracked = 0.0f;      // Dot product: grad_W[T] · W[T]
    bool weight_paradox = false;     // True if grad·W>0 but weight grew
    
    // ==========================================
    // Per-component logit contribution breakdown
    // ==========================================
    float contribution_from_h_rms = 0.0f;     // Factor: h_rms / h_rms_prev
    float contribution_from_w_rms = 0.0f;     // Factor: W_rms / W_rms_prev
    float contribution_from_cosine = 0.0f;    // Factor: cos / cos_prev
    
    // ==========================================
    // CRITICAL FIX: Split logit[T] by target class
    // Compares APPLES TO APPLES, not self-consistent math
    // ==========================================
    float logit_tracked_when_target_is_tracked = 0.0f;     // At positions where target=T (should be HIGH)
    float logit_tracked_when_target_not_tracked = 0.0f;    // At positions where target≠T (should be LOW)
    int count_tracked_targets = 0;                      // Number of positions with target=T
    int count_non_tracked_targets = 0;                  // Number of positions with target≠T
    
    // Statistics
    int valid_position_count = 0;
    
    // NOTE: Cross-batch tracking state (prev_hidden_rms, prev_weight_rms, etc.)
    // is stored as file-level statics in TrainingDiagnostics.cu, NOT as struct members.
};

FeedbackLoopDiagnostic computeFeedbackLoopDiagnostic(
    const GRIM::LMHeadLayer* lm_head,
    const GRIM::TrainingState& ts,
    const std::vector<int>& targets,
    int d_model,
    int vocab_size,
    int batch_idx,
    int tracked_token,   // Dynamically detected collapse token
    bool use_centering,  // Issue #115: Read correct buffer (centered vs raw)
    cudaStream_t stream
);
std::string formatFeedbackLoopDiagnostic(const FeedbackLoopDiagnostic& d, int batch_idx);

//======================================================//
//  PC1CausalityTest — Issue #134: Principal Component Causality
//  Tests if PC1 of hidden states is the root cause of mode collapse
//======================================================//

struct PC1CausalityTest {
    bool valid = false;
    int tracked_token = -1;  // Dynamically detected collapse token

    // PC1 estimation
    float pc1_variance_explained = 0.0f;     // λ_1 / trace(H^T H) — fraction of total variance
    float pc1_rms = 0.0f;                    // Should be ~1.0 (RMS-normalized vector)
    int   power_iterations = 0;              // Number of iterations used

    // Before projection (original hidden states)
    float before_logit_tracked_mean = 0.0f;  // Mean logit[T] across all non-pad positions
    float before_entropy_mean = 0.0f;        // Mean softmax entropy (bits)
    float before_top1_mass_mean = 0.0f;      // Mean max(softmax) — top-1 probability
    float before_top5_mass_mean = 0.0f;      // Mean sum(top-5 softmax)
    float before_avg_cos = 0.0f;             // avg|cos(h_i, h_j)| before projection

    // After projection (PC1 removed)
    float after_logit_tracked_mean = 0.0f;
    float after_entropy_mean = 0.0f;
    float after_top1_mass_mean = 0.0f;
    float after_top5_mass_mean = 0.0f;
    float after_avg_cos = 0.0f;             // avg|cos(h'_i, h'_j)| after projection

    // Deltas (after - before)
    float delta_logit_tracked = 0.0f;        // Negative = deflation (GOOD)
    float delta_entropy = 0.0f;              // Positive = more discriminative (GOOD)
    float delta_top1_mass = 0.0f;            // Depends: better discrimination could raise OR lower
    float delta_avg_cos = 0.0f;              // Negative = decorrelation (GOOD)

    // PC1 alignment with W[T]
    float cos_pc1_w_tracked = 0.0f;          // cos(g, W[T]) — how aligned is PC1 with the tracked weight row

    // Projection invariant checks (should both be ≈0 if projection is correct)
    float pc1_ortho_err = 0.0f;              // max |h'_t · g| / ||g|| — should be ~0 (orthogonality)
    float pc1_pythag_err = 0.0f;             // max |( ||h'||² + c²/D - ||h||² ) / ||h||²| — Pythagorean check

    int n_positions = 0;                     // Valid (non-pad) positions analyzed
};

PC1CausalityTest computePC1CausalityTest(
    const GRIM::LMHeadLayer* lm_head,
    const GRIM::TrainingState& ts,
    const std::vector<int>& targets,
    int d_model,
    int vocab_size,
    int tracked_token,   // Dynamically detected collapse token
    bool use_centering,
    cudaStream_t stream
);
std::string formatPC1CausalityTest(const PC1CausalityTest& r, int batch_idx);

//======================================================//
//  HiddenSpectrum — Top-k singular value spectrum of hidden states
//  Answers: "What is the rank structure of the correlation?"
//
//  Uses deflated power iteration to compute top-k eigenvalues of H^T H
//  (equivalently, top-k squared singular values of H).
//
//  Key metrics:
//    r_k = Σ_{i=1}^{k} σ_i² / Σ_i σ_i²   (cumulative variance ratio)
//    erank = exp(-Σ p_i ln p_i)             (effective rank from entropy)
//    prank = (Σ σ_i²)² / Σ σ_i⁴            (participation ratio)
//    gap_ij = σ_i / σ_j                     (spectral gap)
//======================================================//

/// Maximum number of singular values to compute via deflation
constexpr int kSpectrumTopK = 20;

struct HiddenSpectrum {
    bool valid = false;
    int n_positions = 0;       // Valid (non-pad) positions
    int d_model = 0;

    // Top-k squared singular values (eigenvalues of H^T H), descending
    float lambda[kSpectrumTopK] = {};
    float trace = 0.0f;        // Σ_i σ_i² = trace(H^T H) = Σ_t ||h_t||²

    // Cumulative variance ratios: r_k = Σ_{i=1}^{k} λ_i / trace
    float r1 = 0.0f;           // PC1 alone
    float r2 = 0.0f;           // Top 2
    float r5 = 0.0f;           // Top 5
    float r10 = 0.0f;          // Top 10
    float r20 = 0.0f;          // Top 20

    // Effective rank measures
    float erank = 0.0f;        // exp(-Σ p_i ln p_i) using top-k normalized spectrum
    float prank = 0.0f;        // (Σ λ_i)² / Σ λ_i²  (participation ratio)

    // Spectral gaps
    float gap12 = 0.0f;        // λ_1 / λ_2  (how dominant is top component)
    float gap5_10 = 0.0f;      // λ_5 / λ_10 (steepness of spectrum tail)
};

HiddenSpectrum computeHiddenSpectrum(
    const GRIM::TrainingState& ts,
    const std::vector<int>& targets,
    int d_model,
    bool use_centering,
    cudaStream_t stream
);
std::string formatHiddenSpectrum(const HiddenSpectrum& s, int batch_idx);

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

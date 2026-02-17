//======================================================//
//  TrainingDiagnostics.cu
//  Implementation of diagnostic functions extracted from Phase2_TrainingLoop.cu
//
//  Contains: WeightSample, EmbGradEquationDiag, Token277Diagnostic,
//            HiddenState277Analysis, FeedbackLoopDiagnostic, PC1CausalityTest
//
//  Author: Extracted Feb 2026
//======================================================//

#include "TrainingDiagnostics.hpp"

#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/UnigramByte/Byte.hpp"

#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <unordered_map>

namespace GRIM::Diagnostics {

//======================================================//
//  FeedbackLoopDiagnostic cross-batch tracking state
//  (Moved from static struct members to file-level statics)
//======================================================//
namespace {
    float s_prev_hidden_norm = 0.0f;
    float s_prev_weight_norm = 0.0f;
    float s_prev_cosine = 0.0f;
    float s_prev_logit = 0.0f;
    int   s_prev_batch_idx = -1;
} // anonymous namespace

//======================================================//
//  WeightSample
//======================================================//

WeightSample sampleWeightStats(const GRIM::LMHeadLayer* lm_head, const GRIM::TrainingState& ts, bool sync_for_host) {
    WeightSample sample{};
    if (!lm_head || !lm_head->weights().data) {
        return sample;
    }

    if (!sync_for_host) {
        return sample;
    }

    // Only sync when explicitly requested - hot path must not block
    cudaDeviceSynchronize();

    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    cudaMemcpyAsync(sample.values, lm_head->weights().data,
                    kWeightSampleSize * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    // Need a sync point to read the values, but only for this small copy
    cudaStreamSynchronize(stream);  // Sync primary stream only, not all streams
    
    float sum_sq = 0.0f;
    for (int i = 0; i < kWeightSampleSize; ++i) {
        sum_sq += sample.values[i] * sample.values[i];
    }
    sample.rms = std::sqrt(sum_sq / kWeightSampleSize);
    sample.valid = true;
    return sample;
}

std::string formatWeightSample(const WeightSample& sample) {
    if (!sample.valid) {
        return "lm_head_weights=nullptr";
    }
    
    std::ostringstream oss;
    oss << "lm_w[0:10]=[";
    for (int i = 0; i < kWeightSampleSize; ++i) {
        oss << std::fixed << std::setprecision(6) << sample.values[i];
        if (i + 1 < kWeightSampleSize) oss << ",";
    }
    oss << "] rms=" << std::scientific << std::setprecision(4) << sample.rms;
    return oss.str();
}

float computeUpdateRms(const WeightSample& before, const WeightSample& after) {
    if (!before.valid || !after.valid) {
        return 0.0f;
    }
    
    float sum_sq = 0.0f;
    for (int i = 0; i < kWeightSampleSize; ++i) {
        const float delta = after.values[i] - before.values[i];
        sum_sq += delta * delta;
    }
    return std::sqrt(sum_sq / kWeightSampleSize);
}

//======================================================//
//  EmbGradEquationDiag
//======================================================//

EmbGradEquationDiag computeEmbGradEquation(
    const GRIM::EmbeddingLayer* embedding_layer,
    const int* d_token_ids,     // GPU-resident batch token IDs [total_tokens]
    int total_tokens,
    int d_model,
    int vocab_size,
    float prev_emb_norm,
    float curr_emb_norm,
    cudaStream_t stream
) {
    EmbGradEquationDiag diag{};
    diag.total_vocab = vocab_size;
    diag.prev_emb_norm = prev_emb_norm;
    diag.curr_emb_norm = curr_emb_norm;
    diag.emb_norm_delta = curr_emb_norm - prev_emb_norm;
    
    // Get the gradient buffer (tied weights → embedding tokenWeights().grad_data)
    if (!embedding_layer) throw std::runtime_error("embedding_layer is NULL in computeEmbGradEquation");
    const float* grad_ptr = embedding_layer->tokenWeights().grad_data();
    if (!grad_ptr) return diag;
    
    // Copy token IDs to host for frequency analysis
    std::vector<int> h_token_ids(total_tokens);
    cudaMemcpy(h_token_ids.data(), d_token_ids, total_tokens * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Compute token frequency in this batch
    std::unordered_map<int, int> token_freq;
    for (int t = 0; t < total_tokens; ++t) {
        int tok = h_token_ids[t];
        if (tok >= 0 && tok < vocab_size) {
            token_freq[tok]++;
        }
    }
    
    // Find most frequent token
    for (auto& [tok, count] : token_freq) {
        if (count > diag.most_frequent_count) {
            diag.most_frequent_count = count;
            diag.most_frequent_token = tok;
        }
    }
    
    // Sample per-row norms from the gradient buffer
    // Full vocab scan is too expensive (50K × 768 = 38M floats = 153MB D2H)
    // Strategy: read ALL rows but compute norms on GPU or use sampled approach
    // For diagnostic frequency (every 10 batches), full D2H is acceptable (~20ms)
    const size_t grad_size = static_cast<size_t>(vocab_size) * d_model;
    std::vector<float> h_grad(grad_size);
    cudaMemcpy(h_grad.data(), grad_ptr, grad_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compute per-row norms
    std::vector<float> row_norms(vocab_size, 0.0f);
    double total_norm_sq = 0.0;
    for (int v = 0; v < vocab_size; ++v) {
        const float* row = h_grad.data() + static_cast<size_t>(v) * d_model;
        double row_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            row_sq += static_cast<double>(row[d]) * row[d];
        }
        row_norms[v] = static_cast<float>(std::sqrt(row_sq));
        total_norm_sq += row_sq;
        if (row_norms[v] > 1e-10f) {
            diag.num_active_rows++;
        }
    }
    
    diag.total_norm = static_cast<float>(std::sqrt(total_norm_sq));
    diag.active_ratio = static_cast<float>(diag.num_active_rows) / vocab_size;
    
    // Find top-K row norms
    // Use partial sort for top-K
    std::vector<std::pair<float, int>> norm_idx(vocab_size);
    for (int v = 0; v < vocab_size; ++v) {
        norm_idx[v] = {row_norms[v], v};
    }
    std::partial_sort(norm_idx.begin(), norm_idx.begin() + EmbGradEquationDiag::kTopK, 
                      norm_idx.end(), [](auto& a, auto& b) { return a.first > b.first; });
    
    for (int k = 0; k < EmbGradEquationDiag::kTopK; ++k) {
        diag.top_tokens[k] = norm_idx[k].second;
        diag.top_norms[k] = norm_idx[k].first;
    }
    
    diag.max_row_norm = norm_idx[0].first;
    diag.max_row_token = norm_idx[0].second;
    
    // Compute mean row norm (over active rows only)
    if (diag.num_active_rows > 0) {
        double sum_norms = 0.0;
        for (int v = 0; v < vocab_size; ++v) {
            if (row_norms[v] > 1e-10f) {
                sum_norms += row_norms[v];
            }
        }
        diag.mean_row_norm = static_cast<float>(sum_norms / diag.num_active_rows);
    }
    
    diag.spike_ratio = (diag.mean_row_norm > 1e-10f) ? (diag.max_row_norm / diag.mean_row_norm) : 0.0f;
    
    diag.valid = true;
    return diag;
}

std::string formatEmbGradEquation(const EmbGradEquationDiag& diag, int batch_idx) {
    if (!diag.valid) {
        return "[EMB_GRAD_EQUATION] batch=" + std::to_string(batch_idx + 1) + " INVALID (no grad data)";
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    // Rule 21 format: equation, inputs, expected, actual, anomaly
    oss << "[EMB_GRAD_EQUATION] TIED_WEIGHT_GRAD: grad_W = grad_lm + pcgrad(grad_emb)\n";
    oss << "  EQUATION: grad_lm[v] = centered^T @ grad_logits[:,v] (dense matmul)\n";
    oss << "            grad_emb[tok] += grad_encoder[t] * emb_scale (sparse atomicAdd)\n";
    oss << "            grad_final = grad_lm + orthogonal(grad_emb, grad_lm) (PCGrad)\n";
    oss << "  GRADIENT BUFFER: ||grad_W||_F=" << diag.total_norm
        << " active_rows=" << diag.num_active_rows << "/" << diag.total_vocab
        << " active_ratio=" << std::setprecision(4) << diag.active_ratio << "\n";
    oss << std::setprecision(6);
    oss << "  ROW NORMS: mean=" << diag.mean_row_norm
        << " max=" << diag.max_row_norm << " (tok=" << diag.max_row_token << ")"
        << " spike_ratio=" << std::setprecision(2) << diag.spike_ratio << "x\n";
    oss << std::setprecision(6);
    oss << "  TOP-5 ROWS: ";
    for (int k = 0; k < EmbGradEquationDiag::kTopK; ++k) {
        if (k > 0) oss << ", ";
        oss << "tok" << diag.top_tokens[k] << "=" << diag.top_norms[k];
    }
    oss << "\n";
    oss << "  SCATTER DENSITY: most_frequent=tok" << diag.most_frequent_token
        << " (count=" << diag.most_frequent_count << ")\n";
    oss << "  BATCH TREND: prev_emb_norm=" << diag.prev_emb_norm
        << " curr_emb_norm=" << diag.curr_emb_norm
        << " delta=" << std::showpos << diag.emb_norm_delta << std::noshowpos << "\n";
    
    // Anomaly detection
    if (diag.spike_ratio > 10.0f) {
        oss << "  [ANOMALY] SPIKE_RATIO=" << std::setprecision(1) << diag.spike_ratio
            << "x > 10x — single token row dominates gradient. "
            << "Likely cause: frequent token (tok" << diag.max_row_token 
            << ") with high atomicAdd contention OR concentrated LM head gradient.\n";
    }
    if (diag.emb_norm_delta > diag.prev_emb_norm * 0.5f && diag.prev_emb_norm > 0.01f) {
        oss << "  [ANOMALY] EMB_NORM_SPIKE: delta=" << diag.emb_norm_delta 
            << " > 50% of prev=" << diag.prev_emb_norm
            << " — non-deterministic gradient magnitude jump.\n";
    }
    if (diag.active_ratio < 0.1f) {
        oss << "  [ANOMALY] SPARSE_GRADS: only " << std::setprecision(1) << (diag.active_ratio * 100.0f)
            << "% of vocab rows have non-zero grads — high atomicAdd contention on active rows.\n";
    }
    
    return oss.str();
}

//======================================================//
//  Token277Diagnostic
//======================================================//

Token277Diagnostic computeToken277Diagnostic(
    const GRIM::LMHeadLayer* lm_head,
    const GRIM::EmbeddingLayer* embedding_layer,
    const std::vector<int>& targets,  // Batch targets (flattened)
    int d_model,
    int tracked_token,  // Which token to track (argmax from predictions)
    cudaStream_t stream
) {
    Token277Diagnostic diag{};
    diag.tracked_token = tracked_token;
    if (tracked_token < 0) return diag;
    
    // Count how many times tracked token appears as a TARGET in this batch
    const int PAD_ID = GRIM::Tokenizer::PAD_TOKEN_ID;
    for (int target : targets) {
        if (target >= 0 && target != PAD_ID) {
            diag.total_valid_targets++;
            if (target == tracked_token) {
                diag.target_count++;
            }
        }
    }
    diag.target_ratio = (diag.total_valid_targets > 0) 
        ? static_cast<float>(diag.target_count) / diag.total_valid_targets 
        : 0.0f;
    
    // Get gradient and weight info for tracked row
    const size_t row_offset = static_cast<size_t>(tracked_token) * d_model;
    const size_t row_bytes = d_model * sizeof(float);
    
    std::vector<float> grad_row(d_model, 0.0f);
    std::vector<float> weight_row(d_model, 0.0f);
    
    // Copy gradient row (if embedding grad buffer is aliased to lm_head grad buffer for tied weights)
    // The gradient buffer contains BOTH embedding backward + LM head backward contributions
    if (!embedding_layer) throw std::runtime_error("embedding_layer is NULL in computeToken277Diagnostic");
    const float* emb_grads_ptr = embedding_layer->tokenWeights().grad_data();
    if (emb_grads_ptr) {
        cudaMemcpyAsync(grad_row.data(), emb_grads_ptr + row_offset,
                        row_bytes, cudaMemcpyDeviceToHost, stream);
    }
    
    // Copy weight row from actual embeddings
    if (lm_head && lm_head->weights().data) {
        cudaMemcpyAsync(weight_row.data(), lm_head->weights().data + row_offset,
                        row_bytes, cudaMemcpyDeviceToHost, stream);
    }
    
    cudaStreamSynchronize(stream);
    
    // Compute gradient statistics
    double grad_sum = 0.0, grad_sum_sq = 0.0;
    for (int i = 0; i < d_model; ++i) {
        grad_sum += grad_row[i];
        grad_sum_sq += static_cast<double>(grad_row[i]) * grad_row[i];
    }
    diag.grad_row_sum = static_cast<float>(grad_sum);
    diag.grad_row_mean = static_cast<float>(grad_sum / d_model);
    diag.grad_row_norm = static_cast<float>(std::sqrt(grad_sum_sq));
    
    // Compute weight statistics
    double weight_sum = 0.0, weight_sum_sq = 0.0;
    for (int i = 0; i < d_model; ++i) {
        weight_sum += weight_row[i];
        weight_sum_sq += static_cast<double>(weight_row[i]) * weight_row[i];
    }
    diag.weight_row_norm = static_cast<float>(std::sqrt(weight_sum_sq));
    diag.weight_row_mean = static_cast<float>(weight_sum / d_model);
    
    diag.valid = true;
    return diag;
}

std::string formatToken277Diagnostic(const Token277Diagnostic& diag, int batch_idx) {
    if (!diag.valid) {
        return "[CollapseTokenTrace] batch=" + std::to_string(batch_idx + 1) + " INVALID";
    }
    
    const int tok = diag.tracked_token;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    // Rule 21 Equation-Based Diagnostic Format
    oss << "[WEIGHT_GRADIENT_EQUATION] W_UPDATE[" << tok << "]: W_new[" << tok << "] = W[" << tok << "] - lr × grad_W[" << tok << "] / sqrt(v + eps)\n";
    oss << "  GRAD_W[" << tok << "]: ||grad||=" << diag.grad_row_norm
        << " sum=" << diag.grad_row_sum
        << " mean=" << diag.grad_row_mean << "\n";
    oss << "  WEIGHT[" << tok << "]: ||W[" << tok << "]||=" << diag.weight_row_norm
        << " mean=" << diag.weight_row_mean << "\n";
    oss << "  TARGET_DISTRIBUTION: collapse_token_count=" << diag.target_count
        << "/" << diag.total_valid_targets
        << " ratio=" << std::setprecision(4) << (diag.target_ratio * 100.0f) << "%\n";
    
    // Issue #114: Predict weight change direction
    // AdamW: W_new = W - lr × m_hat / (sqrt(v_hat) + eps)
    // where m_hat ≈ grad (after bias correction)
    //   grad_sum > 0 → W_new = W - (+value) → W DECREASES
    //   grad_sum < 0 → W_new = W - (-value) → W INCREASES
    
    if (diag.grad_row_sum > 0.0f) {
        oss << "  [PREDICTION] W[" << tok << "] DECREASE: grad_sum=" << diag.grad_row_sum << " > 0 → ||W|| decreases\n";
    } else if (diag.grad_row_sum < 0.0f) {
        oss << "  [PREDICTION] W[" << tok << "] INCREASE: grad_sum=" << diag.grad_row_sum << " < 0 → ||W|| increases\n";
    } else {
        oss << "  [PREDICTION] W[" << tok << "] STABLE: grad_sum=" << diag.grad_row_sum << " ≈ 0\n";
    }
    
    return oss.str();
}

//======================================================//
//  HiddenState277Analysis
//======================================================//

HiddenState277Analysis computeHiddenState277Analysis(
    const GRIM::TrainingState& ts,
    const std::vector<int>& targets,  // Batch targets (flattened)
    int d_model,
    int vocab_size,
    int tracked_token,   // Which token to track (argmax from predictions)
    bool use_centering,  // Issue #115: Added to read correct buffer
    cudaStream_t stream
) {
    HiddenState277Analysis analysis{};
    analysis.tracked_token = tracked_token;
    if (tracked_token < 0) return analysis;
    
    const int total_tokens = static_cast<int>(targets.size());
    // Issue #115: Check the buffer we'll actually use (centered or raw)
    const bool have_hidden_data = use_centering 
        ? (ts.centering_scratch_tensor.data != nullptr)
        : (ts.cached_encoder_output.data != nullptr);
    if (total_tokens == 0 || !have_hidden_data || !ts.grad_logits_tensor.data) {
        return analysis;
    }
    
    // Copy encoder output (hidden states) and grad_logits to host
    const size_t hidden_size = static_cast<size_t>(total_tokens) * d_model;
    const size_t grad_logits_size = static_cast<size_t>(total_tokens) * vocab_size;
    
    std::vector<float> h_hidden(hidden_size);
    std::vector<float> h_grad_logits(grad_logits_size);
    
    // ============================================================================
    // ISSUE #115 FIX: Diagnostic was reading WRONG buffer!
    //
    // BUG: ts.cached_encoder_output is written BEFORE centering in AutogradTraining.cu:947
    //      The actual LM head uses centered_scratch (via lm_input_ptr) at line 1055
    //
    // RESULT: Diagnostic showed hidden_mean ≠ 0 even when centering was ACTIVE!
    //         This led to FALSE conclusions that centering wasn't working.
    //
    // FIX: Read from centered_scratch when centering is enabled, else cached_encoder_output
    // ============================================================================
    const float* hidden_source = use_centering && ts.centering_scratch_tensor.data
        ? ts.centering_scratch_tensor.data  // CENTERED: actual LM head input
        : ts.cached_encoder_output.data;    // UNCENTERED: raw encoder output
    
    cudaMemcpyAsync(h_hidden.data(), hidden_source, 
                    hidden_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_grad_logits.data(), ts.grad_logits_tensor.data,
                    grad_logits_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Issue #137: Per-dimension accumulation of grad_W[T] split by target type
    std::vector<double> grad_w_from_tracked(d_model, 0.0);
    std::vector<double> grad_w_from_other(d_model, 0.0);
    
    // Analyze per-position
    double sum_hidden = 0.0, sum_hidden_sq = 0.0;
    double sum_norm = 0.0;
    double sum_hidden_tracked = 0.0, sum_norm_tracked = 0.0;
    double sum_hidden_other = 0.0, sum_norm_other = 0.0;
    double sum_grad_tracked_at_tracked = 0.0, sum_grad_tracked_at_other = 0.0;
    
    const int PAD_ID = GRIM::Tokenizer::PAD_TOKEN_ID;
    
    for (int t = 0; t < total_tokens; ++t) {
        const int target = targets[t];
        if (target < 0 || target == PAD_ID) continue;
        
        // Hidden state at position t
        const float* hidden_t = &h_hidden[static_cast<size_t>(t) * d_model];
        
        // Compute hidden norm and stats
        double norm_sq = 0.0;
        for (int i = 0; i < d_model; ++i) {
            sum_hidden += hidden_t[i];
            sum_hidden_sq += hidden_t[i] * hidden_t[i];
            norm_sq += hidden_t[i] * hidden_t[i];
        }
        double norm = std::sqrt(norm_sq);
        sum_norm += norm;
        
        // Grad_logits[t, T] (T = tracked collapse token)
        const float grad_tracked_t = h_grad_logits[static_cast<size_t>(t) * vocab_size + tracked_token];
        
        if (target == tracked_token) {
            analysis.count_tracked_targets++;
            sum_norm_tracked += norm;
            sum_grad_tracked_at_tracked += grad_tracked_t;
            // Accumulate per-dimension: grad_W[T,i] += h[t,i] * g[t,T]
            for (int i = 0; i < d_model; ++i) {
                grad_w_from_tracked[i] += hidden_t[i] * grad_tracked_t;
            }
        } else {
            analysis.count_other_targets++;
            sum_norm_other += norm;
            sum_grad_tracked_at_other += grad_tracked_t;
            for (int i = 0; i < d_model; ++i) {
                grad_w_from_other[i] += hidden_t[i] * grad_tracked_t;
            }
        }
    }
    
    const int total_valid = analysis.count_tracked_targets + analysis.count_other_targets;
    if (total_valid == 0) return analysis;
    
    analysis.valid = true;
    analysis.hidden_mean = static_cast<float>(sum_hidden / (total_valid * d_model));
    analysis.hidden_norm_mean = static_cast<float>(sum_norm / total_valid);
    analysis.hidden_std = static_cast<float>(std::sqrt(
        sum_hidden_sq / (total_valid * d_model) - analysis.hidden_mean * analysis.hidden_mean));
    
    if (analysis.count_tracked_targets > 0) {
        analysis.hidden_tracked_norm = static_cast<float>(sum_norm_tracked / analysis.count_tracked_targets);
    }
    if (analysis.count_other_targets > 0) {
        analysis.hidden_other_norm = static_cast<float>(sum_norm_other / analysis.count_other_targets);
    }
    
    analysis.grad_tracked_at_tracked_targets = static_cast<float>(sum_grad_tracked_at_tracked);
    analysis.grad_tracked_at_other_targets = static_cast<float>(sum_grad_tracked_at_other);
    
    // Issue #137: Compute per-dimension gradient norms, sums, and cosine between the two sources
    double norm_sq_tracked = 0.0, norm_sq_other = 0.0, dot = 0.0;
    double sum_tracked = 0.0, sum_other = 0.0;
    for (int i = 0; i < d_model; ++i) {
        norm_sq_tracked += grad_w_from_tracked[i] * grad_w_from_tracked[i];
        norm_sq_other += grad_w_from_other[i] * grad_w_from_other[i];
        dot += grad_w_from_tracked[i] * grad_w_from_other[i];
        sum_tracked += grad_w_from_tracked[i];
        sum_other += grad_w_from_other[i];
    }
    analysis.grad_w_norm_from_tracked = static_cast<float>(std::sqrt(norm_sq_tracked));
    analysis.grad_w_norm_from_other = static_cast<float>(std::sqrt(norm_sq_other));
    analysis.grad_w_sum_from_tracked = static_cast<float>(sum_tracked);
    analysis.grad_w_sum_from_other = static_cast<float>(sum_other);
    const double denom = std::sqrt(norm_sq_tracked) * std::sqrt(norm_sq_other);
    analysis.grad_w_cosine = (denom > 1e-12) ? static_cast<float>(dot / denom) : 0.0f;
    
    return analysis;
}

std::string formatHiddenState277Analysis(const HiddenState277Analysis& a, int batch_idx) {
    if (!a.valid) {
        return "[HiddenStateCollapse] batch=" + std::to_string(batch_idx + 1) + " INVALID (no cached encoder output)";
    }
    
    const int tok = a.tracked_token;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    // Rule 21 Equation-Based Diagnostic Format
    oss << "[HIDDEN_STATE_EQUATION] GRAD_W[" << tok << "]: grad_W[" << tok << ",i] = Σ_t (hidden[t,i] × grad_logits[t," << tok << "])\n";
    oss << "  HIDDEN STATES (encoder output): mean=" << a.hidden_mean
        << " ||h||_mean=" << a.hidden_norm_mean
        << " std=" << a.hidden_std << "\n";
    
    // Breakdown by target type (Issue #37 root cause: hidden_sum × grad contribution)
    oss << "  AT_TRACKED_TARGETS (n=" << a.count_tracked_targets << "): ||h||=" << a.hidden_tracked_norm << "\n";
    oss << "  AT_OTHER_TARGETS (n=" << a.count_other_targets << "): ||h||=" << a.hidden_other_norm << "\n";
    
    // Gradient at tracked column
    oss << "  GRAD_LOGITS[" << tok << "]: at_tracked_targets=" << a.grad_tracked_at_tracked_targets 
        << " (p_t - 1, always negative), at_other_targets=" << a.grad_tracked_at_other_targets 
        << " (p_v/N, always positive for pure CE)\n";
    
    // Issue #137: Per-dimension gradient decomposition
    oss << "  GRAD_W[" << tok << "] PER-DIMENSION DECOMPOSITION:\n";
    oss << "    from_tracked_targets: ||g||=" << a.grad_w_norm_from_tracked << "\n";
    oss << "    from_other_targets: ||g||=" << a.grad_w_norm_from_other << "\n";
    float ratio = (a.grad_w_norm_from_other > 1e-12f) 
        ? a.grad_w_norm_from_tracked / a.grad_w_norm_from_other : 0.0f;
    oss << "    ||g_tracked||/||g_other||=" << std::setprecision(1) << ratio << "x"
        << std::setprecision(6) << " cos(g_tracked, g_other)=" << a.grad_w_cosine 
        << " (negative=opposing, positive=reinforcing)\n";
    
    if (std::abs(a.hidden_mean) > 0.001f) {
        oss << "  [ANOMALY] NON_ZERO_HIDDEN_MEAN: hidden_mean=" << a.hidden_mean 
            << " should be ~0 for RMSNorm output. This biases weight gradients!\n";
    }
    
    return oss.str();
}

//======================================================//
//  FeedbackLoopDiagnostic
//======================================================//

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
) {
    FeedbackLoopDiagnostic diag{};
    diag.tracked_token = tracked_token;
    
    if (tracked_token < 0 || tracked_token >= vocab_size) return diag;
    
    const int total_tokens = static_cast<int>(targets.size());
    // Issue #115: Check the buffer we'll actually use (centered or raw)
    const bool have_hidden_data = use_centering 
        ? (ts.centering_scratch_tensor.data != nullptr)
        : (ts.cached_encoder_output.data != nullptr);
    if (total_tokens == 0 || !have_hidden_data || !lm_head || !lm_head->weights().data) {
        return diag;
    }
    
    // ==========================================
    // D2H Copy: Hidden states, W[T], logits, gradients
    // ==========================================
    const size_t hidden_size = static_cast<size_t>(total_tokens) * d_model;
    std::vector<float> h_hidden(hidden_size);
    std::vector<float> w277(d_model);
    std::vector<float> h_logits(static_cast<size_t>(total_tokens) * vocab_size);
    std::vector<float> grad_w277(d_model);
    
    // ============================================================================
    // ISSUE #115 FIX: Same bug as computeHiddenState277Analysis()
    // Read from centered buffer when centering is active, otherwise raw encoder output
    // ============================================================================
    const float* hidden_source = use_centering && ts.centering_scratch_tensor.data
        ? ts.centering_scratch_tensor.data  // CENTERED: actual LM head input
        : ts.cached_encoder_output.data;    // UNCENTERED: raw encoder output
    
    cudaMemcpyAsync(h_hidden.data(), hidden_source,
                    hidden_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(w277.data(), lm_head->weights().data + static_cast<size_t>(tracked_token) * d_model,
                    d_model * sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // Get logits from autograd intermediates (alive until processBatch clears them)
    if (ts.autograd_intermediates.hasLogits()) {
        cudaMemcpyAsync(h_logits.data(), ts.autograd_intermediates.logits_tensor.data,
                        static_cast<size_t>(total_tokens) * vocab_size * sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
    }
    
    // Get gradient of W[277] row if available
    if (lm_head->weights().grad_data()) {
        cudaMemcpyAsync(grad_w277.data(), 
                        lm_head->weights().grad_data() + static_cast<size_t>(tracked_token) * d_model,
                        d_model * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }
    
    cudaStreamSynchronize(stream);
    
    // ==========================================
    // Compute ||W[T]||
    // Equation: ||W[T]|| = sqrt(Σ_d W[T,d]²)
    // ==========================================
    double w277_sq = 0.0;
    for (int d = 0; d < d_model; ++d) {
        w277_sq += static_cast<double>(w277[d]) * w277[d];
    }
    diag.weight_tracked_norm = static_cast<float>(std::sqrt(w277_sq));
    
    // ==========================================
    // Compute grad_W[T] · W[T] for weight paradox detection
    // Math: If grad · W > 0, optimizer will DECREASE ||W|| (gradient aligns with weight)
    //       If grad · W < 0, optimizer will INCREASE ||W|| (gradient opposes weight)
    // ==========================================
    double grad_dot_w = 0.0;
    for (int d = 0; d < d_model; ++d) {
        grad_dot_w += static_cast<double>(grad_w277[d]) * w277[d];
    }
    diag.grad_dot_w_tracked = static_cast<float>(grad_dot_w);
    
    // ==========================================
    // Per-position analysis
    // ==========================================
    const int PAD_ID = GRIM::Tokenizer::PAD_TOKEN_ID;
    double sum_h_norm = 0.0;
    double sum_cosine = 0.0;
    double sum_actual_logit = 0.0;
    double sum_predicted_logit = 0.0;
    double sum_logit_tracked_for_tracked_targets = 0.0;    // logit[T] when target IS T
    double sum_logit_tracked_for_non_tracked_targets = 0.0; // logit[T] when target is NOT T
    int count_tracked_targets_local = 0;
    int count_non_tracked_targets_local = 0;
    int valid_count = 0;
    
    // For hidden cosine sampling (pairwise)
    std::vector<std::pair<int, double>> position_norms;  // (position_idx, norm)
    
    for (int t = 0; t < total_tokens; ++t) {
        if (targets[t] < 0 || targets[t] == PAD_ID) continue;
        
        const float* h_t = &h_hidden[static_cast<size_t>(t) * d_model];
        
        // ==========================================
        // Compute ||h[t]||
        // Equation: ||h[t]|| = sqrt(Σ_d h[t,d]²)
        // ==========================================
        double h_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            h_sq += static_cast<double>(h_t[d]) * h_t[d];
        }
        double h_norm = std::sqrt(h_sq);
        sum_h_norm += h_norm;
        position_norms.emplace_back(t, h_norm);
        
        // ==========================================
        // Find argmax at this position (what token the model predicts here)
        // ==========================================
        int argmax_t = 0;
        float max_logit = h_logits[static_cast<size_t>(t) * vocab_size];
        for (int v = 1; v < vocab_size; ++v) {
            float logit_v = h_logits[static_cast<size_t>(t) * vocab_size + v];
            if (logit_v > max_logit) {
                max_logit = logit_v;
                argmax_t = v;
            }
        }
        
        // Get W[argmax_t] row (predicted token's weight vector)
        std::vector<float> w_argmax(d_model);
        cudaMemcpyAsync(w_argmax.data(), 
                       lm_head->weights().data + static_cast<size_t>(argmax_t) * d_model,
                       d_model * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Compute W[argmax_t] norm
        double w_argmax_norm_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            w_argmax_norm_sq += static_cast<double>(w_argmax[d]) * w_argmax[d];
        }
        double w_argmax_norm = std::sqrt(w_argmax_norm_sq);
        
        // ==========================================
        // Compute cos(h[t], W[argmax[t]])
        // Equation: cos(h,W) = (h · W) / (||h|| × ||W||)
        // ==========================================
        double dot_hw = 0.0;
        for (int d = 0; d < d_model; ++d) {
            dot_hw += static_cast<double>(h_t[d]) * w_argmax[d];
        }
        double cosine = (h_norm > 1e-8 && w_argmax_norm > 1e-8) 
                        ? dot_hw / (h_norm * w_argmax_norm) : 0.0;
        sum_cosine += cosine;
        
        // ==========================================
        // Compute predicted vs actual logit[argmax[t]]
        // Equation: logit[argmax] = ||h|| × ||W[argmax]|| × cos(h,W[argmax])
        // Alternatively: logit[argmax] = h · W[argmax]^T (direct dot product)
        // ==========================================
        double predicted_logit = h_norm * w_argmax_norm * cosine;
        sum_predicted_logit += predicted_logit;
        
        // Actual logit from forward pass (should equal max_logit we found above)
        float actual = max_logit;
        sum_actual_logit += actual;
        
        // ==========================================
        // CRITICAL: Split by whether target IS T or NOT
        // This shows if model discriminates between correct vs incorrect token
        // ==========================================
        if (targets[t] == tracked_token) {
            sum_logit_tracked_for_tracked_targets += actual;
            count_tracked_targets_local++;
        } else {
            sum_logit_tracked_for_non_tracked_targets += actual;
            count_non_tracked_targets_local++;
        }
        
        valid_count++;
    }
    
    if (valid_count == 0) return diag;
    
    diag.valid = true;
    diag.valid_position_count = valid_count;
    
    // ==========================================
    // Compute means for the feedback loop equation
    // ==========================================
    diag.hidden_norm_mean = static_cast<float>(sum_h_norm / valid_count);
    diag.cosine_h_w_tracked_mean = static_cast<float>(sum_cosine / valid_count);
    diag.predicted_logit_tracked = static_cast<float>(sum_predicted_logit / valid_count);
    diag.actual_logit_tracked_mean = static_cast<float>(sum_actual_logit / valid_count);
    
    // ==========================================
    // CRITICAL: Store split logits
    // ==========================================
    diag.count_tracked_targets = count_tracked_targets_local;
    diag.count_non_tracked_targets = count_non_tracked_targets_local;
    
    if (count_tracked_targets_local > 0) {
        diag.logit_tracked_when_target_is_tracked = static_cast<float>(sum_logit_tracked_for_tracked_targets / count_tracked_targets_local);
    }
    if (count_non_tracked_targets_local > 0) {
        diag.logit_tracked_when_target_not_tracked = static_cast<float>(sum_logit_tracked_for_non_tracked_targets / count_non_tracked_targets_local);
    }
    
    // ==========================================
    // Decomposition error check
    // If ||h|| × ||W|| × cos ≠ actual logit, something is wrong
    // ==========================================
    if (std::abs(diag.actual_logit_tracked_mean) > 1e-8) {
        diag.decomposition_error_pct = std::abs(diag.predicted_logit_tracked - diag.actual_logit_tracked_mean) 
                                       / std::abs(diag.actual_logit_tracked_mean) * 100.0f;
    }
    
    // ==========================================
    // Hidden Cosine Collapse Detection
    // Sample pairwise cosine similarities between hidden states
    // ==========================================
    constexpr int kMaxCosineSamples = 100;
    double sum_pairwise_cos = 0.0;
    int pair_count = 0;
    
    for (size_t i = 0; i < position_norms.size() && pair_count < kMaxCosineSamples; ++i) {
        for (size_t j = i + 1; j < position_norms.size() && pair_count < kMaxCosineSamples; ++j) {
            int t_i = position_norms[i].first;
            int t_j = position_norms[j].first;
            double norm_i = position_norms[i].second;
            double norm_j = position_norms[j].second;
            
            if (norm_i < 1e-8 || norm_j < 1e-8) continue;
            
            // Compute h[i] · h[j]
            const float* h_i = &h_hidden[static_cast<size_t>(t_i) * d_model];
            const float* h_j = &h_hidden[static_cast<size_t>(t_j) * d_model];
            double dot_ij = 0.0;
            for (int d = 0; d < d_model; ++d) {
                dot_ij += static_cast<double>(h_i[d]) * h_j[d];
            }
            
            double cos_ij = dot_ij / (norm_i * norm_j);
            sum_pairwise_cos += std::abs(cos_ij);  // Take abs for avg magnitude
            pair_count++;
        }
    }
    
    if (pair_count > 0) {
        diag.avg_hidden_cosine = static_cast<float>(sum_pairwise_cos / pair_count);
        diag.hidden_cosine_samples = pair_count;
    }
    
    // ==========================================
    // Growth Rate Calculation (Issue #114)
    // Track explosion via d(metric)/d(batch)
    // ==========================================
    if (s_prev_batch_idx >= 0 && 
        batch_idx > s_prev_batch_idx) {
        
        if (s_prev_hidden_norm > 1e-8) {
            diag.hidden_norm_growth_pct = 
                (diag.hidden_norm_mean - s_prev_hidden_norm) 
                / s_prev_hidden_norm * 100.0f;
            diag.contribution_from_h_norm = 
                diag.hidden_norm_mean / s_prev_hidden_norm;
        }
        
        if (s_prev_weight_norm > 1e-8) {
            diag.weight_norm_growth_pct = 
                (diag.weight_tracked_norm - s_prev_weight_norm)
                / s_prev_weight_norm * 100.0f;
            diag.contribution_from_w_norm = 
                diag.weight_tracked_norm / s_prev_weight_norm;
            
            // Weight paradox detection: grad·W > 0 means optimizer will decrease ||W||
            // If ||W|| actually GREW despite this, that's a paradox
            if (diag.grad_dot_w_tracked > 0 && diag.weight_norm_growth_pct > 0.1f) {
                diag.weight_paradox = true;
            }
        }
        
        if (std::abs(s_prev_cosine) > 1e-8) {
            diag.cosine_growth_pct = 
                (diag.cosine_h_w_tracked_mean - s_prev_cosine)
                / std::abs(s_prev_cosine) * 100.0f;
            diag.contribution_from_cosine = 
                diag.cosine_h_w_tracked_mean / s_prev_cosine;
        }
        
        if (std::abs(s_prev_logit) > 1e-8) {
            diag.logit_growth_pct = 
                (diag.actual_logit_tracked_mean - s_prev_logit)
                / std::abs(s_prev_logit) * 100.0f;
        }
    }
    
    // Update static storage for next batch
    s_prev_hidden_norm = diag.hidden_norm_mean;
    s_prev_weight_norm = diag.weight_tracked_norm;
    s_prev_cosine = diag.cosine_h_w_tracked_mean;
    s_prev_logit = diag.actual_logit_tracked_mean;
    s_prev_batch_idx = batch_idx;
    
    return diag;
}

std::string formatFeedbackLoopDiagnostic(const FeedbackLoopDiagnostic& d, int batch_idx) {
    if (!d.valid) {
        return "[FEEDBACK_LOOP_EQUATION] batch=" + std::to_string(batch_idx + 1) + " INVALID (missing data)";
    }
    
    const int tok = d.tracked_token;
    const std::string tok_str = std::to_string(tok);
    std::ostringstream oss;
    oss << std::fixed;
    
    // ==========================================
    // Rule 21: Equation-Based Diagnostic Format
    // NOTE: Cosine is now computed per-position with argmax[t], not a global tracked token
    // ==========================================
    oss << "[FEEDBACK_LOOP_EQUATION] ARGMAX_ALIGNMENT: logit[argmax[t]] = ||h[t]|| × ||W[argmax[t]]|| × cos(h[t], W[argmax[t]])\n";
    
    oss << std::setprecision(6);
    oss << "  INPUT h (encoder output): n_positions=" << d.valid_position_count 
        << " ||h||_mean=" << d.hidden_norm_mean << "\n";
    oss << "  ALIGNMENT: cos(h[t], W[argmax[t]])_mean=" << d.cosine_h_w_tracked_mean 
        << " (" << (d.cosine_h_w_tracked_mean > 0.5f ? "HIGH alignment" : "normal") << ")\n";
    
    oss << std::setprecision(4);
    oss << "  TRACKED TOKEN " << tok << " (most common argmax):\n";
    oss << "    ||W[" << tok << "]||=" << d.weight_tracked_norm << "\n";
    oss << "    When target=" << tok << ": logit[" << tok << "]_mean=" << d.logit_tracked_when_target_is_tracked 
        << " (n=" << d.count_tracked_targets << ")\n";
    oss << "    When target≠" << tok << ": logit[" << tok << "]_mean=" << d.logit_tracked_when_target_not_tracked 
        << " (n=" << d.count_non_tracked_targets << ")\n";
    oss << "    DELTA = " << (d.logit_tracked_when_target_is_tracked - d.logit_tracked_when_target_not_tracked)
        << " (should be POSITIVE and LARGE for good discrimination)\n";
    
    // ==========================================
    // Growth Rates - Issue #114 Anomaly Detection
    // ==========================================
    oss << "  GROWTH_RATES: ||h||=" << std::showpos << d.hidden_norm_growth_pct << "% "
        << "||W||=" << d.weight_norm_growth_pct << "% "
        << "cos=" << d.cosine_growth_pct << "% "
        << "logit=" << d.logit_growth_pct << "%" << std::noshowpos << "\n";
    
    // ==========================================
    // Anomaly Flags
    // ==========================================
    std::vector<std::string> anomalies;
    
    // Hidden norm explosion (Issue #114: +111% observed)
    if (d.hidden_norm_growth_pct > 5.0f) {
        anomalies.push_back("HIDDEN_NORM_EXPLOSION(+" + std::to_string(static_cast<int>(d.hidden_norm_growth_pct)) + "%)");
    }
    
    // Cosine collapse (Issue #114: avg_cos → 0.84)
    constexpr float kExpectedRandomCosine = 0.036f;  // 1/sqrt(768)
    if (d.avg_hidden_cosine > 0.5f) {
        std::ostringstream anom;
        anom << "COSINE_COLLAPSE(avg_cos=" << std::setprecision(3) << d.avg_hidden_cosine 
             << " vs expected=" << kExpectedRandomCosine << ")";
        anomalies.push_back(anom.str());
    }
    
    // Weight paradox (Issue #114: grad·W > 0 means optimizer wants ||W|| to decrease, but it grew)
    if (d.weight_paradox) {
        anomalies.push_back("WEIGHT_PARADOX(grad·W=" + std::to_string(d.grad_dot_w_tracked) + 
                           ">0 should_shrink but ||W|| grew by " + std::to_string(d.weight_norm_growth_pct) + "%)");
    }
    
    // Alignment explosion (Issue #114: cos 0.05 → 0.44)
    if (d.cosine_h_w_tracked_mean > 0.3f) {
        std::ostringstream anom;
        anom << "ALIGNMENT_EXPLOSION(cos=" << std::setprecision(3) << d.cosine_h_w_tracked_mean << ")";
        anomalies.push_back(anom.str());
    }
    
    // Logit explosion
    if (d.actual_logit_tracked_mean > 3.0f) {
        anomalies.push_back("LOGIT_EXPLOSION(logit_" + tok_str + "=" + std::to_string(d.actual_logit_tracked_mean) + ")");
    }
    
    if (!anomalies.empty()) {
        oss << "  [ANOMALIES] ";
        for (size_t i = 0; i < anomalies.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << anomalies[i];
        }
        oss << "\n";
    }
    
    // ==========================================
    // Hidden State Correlation Analysis
    // ==========================================
    if (d.hidden_cosine_samples > 0) {
        oss << "  HIDDEN_CORRELATION: avg|cos(h_i,h_j)|=" << std::setprecision(4) << d.avg_hidden_cosine
            << " (sampled " << d.hidden_cosine_samples << " pairs, expected~" << kExpectedRandomCosine << ")\n";
    }
    
    // ==========================================
    // Per-Component Contribution (if growth data available)
    // ==========================================
    if (d.contribution_from_h_norm > 0 || d.contribution_from_w_norm > 0 || d.contribution_from_cosine != 0) {
        oss << "  CONTRIBUTION_FACTORS: h_norm=" << std::setprecision(3) << d.contribution_from_h_norm << "x"
            << " w_norm=" << d.contribution_from_w_norm << "x"
            << " cosine=" << d.contribution_from_cosine << "x"
            << " PRODUCT=" << (d.contribution_from_h_norm * d.contribution_from_w_norm * d.contribution_from_cosine) << "x\n";
    }
    
    return oss.str();
}

//======================================================//
//  PC1CausalityTest
//======================================================//

PC1CausalityTest computePC1CausalityTest(
    const GRIM::LMHeadLayer* lm_head,
    const GRIM::TrainingState& ts,
    const std::vector<int>& targets,
    int d_model,
    int vocab_size,
    int tracked_token,   // Dynamically detected collapse token
    bool use_centering,
    cudaStream_t stream
) {
    PC1CausalityTest result{};
    result.tracked_token = tracked_token;
    
    if (tracked_token < 0 || tracked_token >= vocab_size) return result;

    const int total_tokens = static_cast<int>(targets.size());
    const bool have_hidden = use_centering
        ? (ts.centering_scratch_tensor.data != nullptr)
        : (ts.cached_encoder_output.data != nullptr);

    if (total_tokens == 0 || !have_hidden || !lm_head || !lm_head->weights().data) {
        return result;
    }

    // ==========================================
    // D2H Copy: Hidden states + W[277] row + cached logits
    // Forward pass already computed logits = h @ W^T (stored in
    // autograd_intermediates.logits_tensor). Use those for BEFORE stats.
    //
    // AFTER stats use the analytical delta from linearity of projection:
    //   logit_after[t,v] = logit_before[t,v] - (h_t · g)(g · W[v])
    // So we need g · W[v] for all v, computed as gW = W @ g [V-vector].
    // This eliminates any CPU matmul for AFTER logits.
    // ==========================================
    const size_t hidden_size = static_cast<size_t>(total_tokens) * d_model;
    const size_t logits_size = static_cast<size_t>(total_tokens) * vocab_size;
    // Full weight matrix needed to compute gW[v] = g · W[v] for all v
    const size_t weight_size = static_cast<size_t>(vocab_size) * d_model;

    std::vector<float> h_hidden(hidden_size);
    std::vector<float> h_weights(weight_size);
    std::vector<float> h_cached_logits;  // Forward-pass logits (BEFORE)

    const float* hidden_source = use_centering && ts.centering_scratch_tensor.data
        ? ts.centering_scratch_tensor.data
        : ts.cached_encoder_output.data;

    const bool have_cached_logits = ts.autograd_intermediates.hasLogits();
    if (!have_cached_logits) {
        // No logits in intermediates = can't do the test without recomputing (pointless)
        return result;
    }
    h_cached_logits.resize(logits_size);
    cudaMemcpyAsync(h_cached_logits.data(), ts.autograd_intermediates.logits_tensor.data,
                    logits_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaMemcpyAsync(h_hidden.data(), hidden_source,
                    hidden_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_weights.data(), lm_head->weights().data,
                    weight_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // ==========================================
    // Identify valid (non-PAD) positions
    // ==========================================
    constexpr int PAD_ID = GRIM::Tokenizer::PAD_TOKEN_ID;
    std::vector<int> valid_positions;
    valid_positions.reserve(total_tokens);
    for (int t = 0; t < total_tokens; ++t) {
        if (targets[t] != PAD_ID) {
            valid_positions.push_back(t);
        }
    }
    const int N = static_cast<int>(valid_positions.size());
    if (N < 2) return result;
    result.n_positions = N;

    // ==========================================
    // Step 1: Estimate PC1 via Power Iteration
    // v_{k+1} = H^T (H v_k) / || H^T (H v_k) ||
    // where H is [N x d_model] matrix of valid hidden states
    // ==========================================
    const int D = d_model;
    std::vector<float> v(D);

    // Initialize v with deterministic pseudo-random values (seeded by N for reproducibility)
    {
        uint32_t seed = static_cast<uint32_t>(N * 7919 + D * 104729);
        for (int d = 0; d < D; ++d) {
            seed = seed * 1103515245u + 12345u;
            v[d] = static_cast<float>(static_cast<int>(seed >> 16) & 0x7FFF) / 32768.0f - 0.5f;
        }
        // Normalize
        float norm = 0.0f;
        for (int d = 0; d < D; ++d) norm += v[d] * v[d];
        norm = std::sqrt(norm);
        if (norm < 1e-12f) { v[0] = 1.0f; norm = 1.0f; }
        for (int d = 0; d < D; ++d) v[d] /= norm;
    }

    constexpr int kPowerIters = 5;
    float eigenvalue = 0.0f;

    for (int iter = 0; iter < kPowerIters; ++iter) {
        // Compute Hv = H @ v   [N-vector]
        std::vector<float> Hv(N, 0.0f);
        for (int i = 0; i < N; ++i) {
            const int t = valid_positions[i];
            const float* h_t = &h_hidden[static_cast<size_t>(t) * D];
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) dot += h_t[d] * v[d];
            Hv[i] = dot;
        }

        // Compute w = H^T @ Hv  [D-vector]  (this is H^T H v)
        std::vector<float> w(D, 0.0f);
        for (int i = 0; i < N; ++i) {
            const int t = valid_positions[i];
            const float* h_t = &h_hidden[static_cast<size_t>(t) * D];
            for (int d = 0; d < D; ++d) {
                w[d] += h_t[d] * Hv[i];
            }
        }

        // Eigenvalue estimate = ||w|| (since v is unit, w = λv approximately)
        float w_norm = 0.0f;
        for (int d = 0; d < D; ++d) w_norm += w[d] * w[d];
        w_norm = std::sqrt(w_norm);
        eigenvalue = w_norm;

        // Normalize: v = w / ||w||
        if (w_norm < 1e-12f) break;
        for (int d = 0; d < D; ++d) v[d] = w[d] / w_norm;
    }

    result.power_iterations = kPowerIters;

    // Normalize PC1 direction (should already be unit but enforce)
    {
        float norm = 0.0f;
        for (int d = 0; d < D; ++d) norm += v[d] * v[d];
        norm = std::sqrt(norm);
        if (norm > 1e-12f) {
            for (int d = 0; d < D; ++d) v[d] /= norm;
        }
        result.pc1_norm = norm;
    }

    // ==========================================
    // Step 2: Compute variance explained by PC1
    // variance_explained = λ_1 / trace(H^T H)
    // trace(H^T H) = Σ_t ||h_t||^2
    // ==========================================
    float trace = 0.0f;
    for (int i = 0; i < N; ++i) {
        const int t = valid_positions[i];
        const float* h_t = &h_hidden[static_cast<size_t>(t) * D];
        for (int d = 0; d < D; ++d) trace += h_t[d] * h_t[d];
    }
    result.pc1_variance_explained = (trace > 1e-12f) ? (eigenvalue / trace) : 0.0f;

    // ==========================================
    // Step 3: cos(PC1, W[277])
    // ==========================================
    {
        const float* w277 = &h_weights[static_cast<size_t>(tracked_token) * D];
        float dot = 0.0f, w_norm = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += v[d] * w277[d];
            w_norm += w277[d] * w277[d];
        }
        w_norm = std::sqrt(w_norm);
        result.cos_pc1_w_tracked = (w_norm > 1e-12f) ? (dot / w_norm) : 0.0f;
    }

    // ==========================================
    // Step 4: Compute PC1 coefficients per position
    // coeff_t = h_t · g  (projection of h_t onto PC1)
    // No need to actually project — we use the analytical delta instead.
    // ==========================================
    std::vector<float> pc1_coeffs(N);
    for (int i = 0; i < N; ++i) {
        const int t = valid_positions[i];
        const float* h_t = &h_hidden[static_cast<size_t>(t) * D];
        float coeff = 0.0f;
        for (int d = 0; d < D; ++d) coeff += h_t[d] * v[d];
        pc1_coeffs[i] = coeff;
    }

    // ==========================================
    // Step 5: Compute gW[v] = g · W[v] for all vocab tokens
    // This is the per-token logit shift from removing PC1:
    //   Δlogit[t,v] = -coeff_t × gW[v]
    //   logit_after[t,v] = logit_before[t,v] + Δlogit[t,v]
    //
    // No CPU matmul needed — just one dot product per vocab row.
    // ==========================================
    std::vector<float> gW(vocab_size);
    for (int j = 0; j < vocab_size; ++j) {
        const float* w_j = &h_weights[static_cast<size_t>(j) * D];
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += w_j[d] * v[d];
        gW[j] = dot;
    }

    // ==========================================
    // Step 6: BEFORE/AFTER logit[277] from cached logits + analytical delta
    // BEFORE: cached_logits[t, 277]  (from actual forward pass)
    // AFTER:  cached_logits[t, 277] - coeff_t × gW[277]
    // ==========================================
    const float gW_277 = gW[tracked_token];

    float sum_logit277_before = 0.0f;
    float sum_logit277_after = 0.0f;
    for (int i = 0; i < N; ++i) {
        const int t = valid_positions[i];
        const float before = h_cached_logits[static_cast<size_t>(t) * vocab_size + tracked_token];
        sum_logit277_before += before;
        sum_logit277_after += before - pc1_coeffs[i] * gW_277;
    }
    result.before_logit_tracked_mean = sum_logit277_before / N;
    result.after_logit_tracked_mean = sum_logit277_after / N;

    // ==========================================
    // Step 7: BEFORE/AFTER entropy + top-k mass at sampled positions
    // AFTER logits = cached logits - coeff_t × gW  (element-wise)
    // ==========================================

    // Helper: softmax entropy from a logit vector
    auto computeSoftmaxStats = [&](const float* logits_row, int V,
                                    float& out_logit_277, float& out_entropy,
                                    float& out_top1_mass, float& out_top5_mass) {
        float max_logit = logits_row[0];
        for (int j = 1; j < V; ++j) {
            if (logits_row[j] > max_logit) max_logit = logits_row[j];
        }

        float sum_exp = 0.0f;
        float top5[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < V; ++j) {
            float e = std::exp(logits_row[j] - max_logit);
            sum_exp += e;
            if (e > top5[4]) {
                top5[4] = e;
                for (int k = 3; k >= 0; --k) {
                    if (top5[k+1] > top5[k]) std::swap(top5[k], top5[k+1]);
                    else break;
                }
            }
        }

        float inv_sum = 1.0f / sum_exp;
        out_logit_277 = logits_row[tracked_token];

        out_entropy = 0.0f;
        for (int j = 0; j < V; ++j) {
            float p = std::exp(logits_row[j] - max_logit) * inv_sum;
            if (p > 1e-12f) {
                out_entropy -= p * std::log2(p);
            }
        }

        out_top1_mass = top5[0] * inv_sum;
        float top5_sum = 0.0f;
        for (int k = 0; k < 5; ++k) top5_sum += top5[k];
        out_top5_mass = top5_sum * inv_sum;
    };

    constexpr int kMaxEntropyPositions = 64;
    const int n_entropy = std::min(N, kMaxEntropyPositions);
    const int entropy_stride = std::max(1, N / n_entropy);

    float sum_ent_before = 0.0f, sum_top1_before = 0.0f, sum_top5_before = 0.0f;
    float sum_ent_after = 0.0f, sum_top1_after = 0.0f, sum_top5_after = 0.0f;
    int entropy_count = 0;

    std::vector<float> logits_after_row(vocab_size);

    for (int si = 0; si < n_entropy; ++si) {
        const int idx = std::min(si * entropy_stride, N - 1);
        const int t = valid_positions[idx];

        // BEFORE: cached forward-pass logits (actual output of the model)
        const float* logits_before_row = &h_cached_logits[static_cast<size_t>(t) * vocab_size];

        // AFTER: cached logits + analytical delta from PC1 removal
        // logit_after[v] = logit_before[v] - coeff_t × gW[v]
        const float coeff = pc1_coeffs[idx];
        for (int j = 0; j < vocab_size; ++j) {
            logits_after_row[j] = logits_before_row[j] - coeff * gW[j];
        }

        float l277_b, ent_b, t1_b, t5_b;
        float l277_a, ent_a, t1_a, t5_a;
        computeSoftmaxStats(logits_before_row, vocab_size, l277_b, ent_b, t1_b, t5_b);
        computeSoftmaxStats(logits_after_row.data(), vocab_size, l277_a, ent_a, t1_a, t5_a);

        sum_ent_before += ent_b;  sum_top1_before += t1_b; sum_top5_before += t5_b;
        sum_ent_after += ent_a;   sum_top1_after += t1_a;  sum_top5_after += t5_a;
        ++entropy_count;
    }

    if (entropy_count > 0) {
        const float inv = 1.0f / entropy_count;
        result.before_entropy_mean = sum_ent_before * inv;
        result.before_top1_mass_mean = sum_top1_before * inv;
        result.before_top5_mass_mean = sum_top5_before * inv;
        result.after_entropy_mean = sum_ent_after * inv;
        result.after_top1_mass_mean = sum_top1_after * inv;
        result.after_top5_mass_mean = sum_top5_after * inv;
    }

    // ==========================================
    // Step 8: Pairwise cosine BEFORE vs AFTER projection
    // AFTER uses analytical projection: h'_t = h_t - coeff_t × g
    // cos(h'_i, h'_j) computed directly without materializing h_projected
    // ==========================================
    constexpr int kCosineSamples = 50;
    {
        const int step = std::max(1, N / kCosineSamples);

        // BEFORE: pairwise cosine on original hidden states
        float sum_cos_before = 0.0f;
        int count_before = 0;
        for (int i = 0; i < N && count_before < kCosineSamples; i += step) {
            const int j = (i + step / 2) % N;
            if (i == j) continue;
            const int t_i = valid_positions[i];
            const int t_j = valid_positions[j];
            const float* a = &h_hidden[static_cast<size_t>(t_i) * D];
            const float* b = &h_hidden[static_cast<size_t>(t_j) * D];
            float dot = 0.0f, na = 0.0f, nb = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += a[d] * b[d];
                na += a[d] * a[d];
                nb += b[d] * b[d];
            }
            float denom = std::sqrt(na) * std::sqrt(nb);
            if (denom > 1e-12f) {
                sum_cos_before += std::abs(dot / denom);
                ++count_before;
            }
        }
        result.before_avg_cos = (count_before > 0) ? (sum_cos_before / count_before) : 0.0f;

        // AFTER: pairwise cosine on projected hidden states (analytical)
        // h'_t = h_t - c_t × g, so:
        //   h'_i · h'_j = (h_i - c_i g) · (h_j - c_j g)
        //               = h_i·h_j - c_i(g·h_j) - c_j(g·h_i) + c_i c_j
        //               = h_i·h_j - c_i c_j - c_j c_i + c_i c_j
        //               = h_i·h_j - c_i c_j
        //   ||h'_t||^2  = ||h_t||^2 - c_t^2
        float sum_cos_after = 0.0f;
        int count_after = 0;
        for (int i = 0; i < N && count_after < kCosineSamples; i += step) {
            const int j = (i + step / 2) % N;
            if (i == j) continue;
            const int t_i = valid_positions[i];
            const int t_j = valid_positions[j];
            const float* a = &h_hidden[static_cast<size_t>(t_i) * D];
            const float* b = &h_hidden[static_cast<size_t>(t_j) * D];
            float dot_ab = 0.0f, na = 0.0f, nb = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot_ab += a[d] * b[d];
                na += a[d] * a[d];
                nb += b[d] * b[d];
            }
            const float c_i = pc1_coeffs[i];
            const float c_j = pc1_coeffs[j];
            const float proj_dot = dot_ab - c_i * c_j;
            const float proj_na = na - c_i * c_i;
            const float proj_nb = nb - c_j * c_j;
            float denom = std::sqrt(std::max(0.0f, proj_na)) * std::sqrt(std::max(0.0f, proj_nb));
            if (denom > 1e-12f) {
                sum_cos_after += std::abs(proj_dot / denom);
                ++count_after;
            }
        }
        result.after_avg_cos = (count_after > 0) ? (sum_cos_after / count_after) : 0.0f;
    }

    // ==========================================
    // Deltas
    // ==========================================
    result.delta_logit_tracked = result.after_logit_tracked_mean - result.before_logit_tracked_mean;
    result.delta_entropy = result.after_entropy_mean - result.before_entropy_mean;
    result.delta_top1_mass = result.after_top1_mass_mean - result.before_top1_mass_mean;
    result.delta_avg_cos = result.after_avg_cos - result.before_avg_cos;

    result.valid = true;
    return result;
}

std::string formatPC1CausalityTest(const PC1CausalityTest& r, int batch_idx) {
    if (!r.valid) {
        return "[PC1_CAUSALITY_TEST] batch=" + std::to_string(batch_idx + 1) + " INVALID (missing data)";
    }

    const int tok = r.tracked_token;
    std::ostringstream oss;
    oss << std::fixed;

    // ==========================================
    // Rule 21: Equation-Based Diagnostic Format
    // ==========================================
    oss << "[PC1_CAUSALITY_TEST] HIDDEN_STATE_GEOMETRY: h'_t = h_t - (h_t · g) g  (g = PC1 unit vector)\n";

    oss << std::setprecision(6);
    oss << "  PC1 ESTIMATION (" << r.power_iterations << " power iterations):\n";
    oss << "    variance_explained = λ_1/trace(H^T H) = " << r.pc1_variance_explained
        << " (" << std::setprecision(1) << (r.pc1_variance_explained * 100.0f) << "% of total)\n";
    oss << std::setprecision(6);
    oss << "    ||g|| = " << r.pc1_norm << " (should be 1.0)\n";
    oss << "    cos(PC1, W[" << tok << "]) = " << r.cos_pc1_w_tracked << "\n";

    oss << "  BEFORE PROJECTION (original hidden states, n=" << r.n_positions << "):\n";
    oss << std::setprecision(4);
    oss << "    logit[" << tok << "]_mean = " << r.before_logit_tracked_mean << "\n";
    oss << "    entropy_mean = " << r.before_entropy_mean << " bits\n";
    oss << "    top1_mass_mean = " << r.before_top1_mass_mean << "\n";
    oss << "    top5_mass_mean = " << r.before_top5_mass_mean << "\n";
    oss << "    avg|cos(h_i,h_j)| = " << r.before_avg_cos << "\n";

    oss << "  AFTER PROJECTION (PC1 removed):\n";
    oss << "    logit[" << tok << "]_mean = " << r.after_logit_tracked_mean << "\n";
    oss << "    entropy_mean = " << r.after_entropy_mean << " bits\n";
    oss << "    top1_mass_mean = " << r.after_top1_mass_mean << "\n";
    oss << "    top5_mass_mean = " << r.after_top5_mass_mean << "\n";
    oss << "    avg|cos(h_i,h_j)| = " << r.after_avg_cos << "\n";

    oss << "  DELTAS (after - before):\n";
    oss << std::showpos;
    oss << "    Δlogit[" << tok << "] = " << r.delta_logit_tracked
        << (r.delta_logit_tracked < 0 ? " (DEFLATED ✓)" : " (INFLATED ✗)") << "\n";
    oss << "    Δentropy = " << r.delta_entropy
        << (r.delta_entropy > 0 ? " bits (MORE DISCRIMINATIVE ✓)" : " bits (LESS DISCRIMINATIVE ✗)") << "\n";
    oss << "    Δtop1_mass = " << r.delta_top1_mass << "\n";
    oss << "    Δavg_cos = " << r.delta_avg_cos
        << (r.delta_avg_cos < 0 ? " (DECORRELATED ✓)" : " (STILL CORRELATED ✗)") << "\n";
    oss << std::noshowpos;

    // ==========================================
    // VERDICT
    // ==========================================
    const bool logit_deflated = r.delta_logit_tracked < -0.01f;
    const bool entropy_improved = r.delta_entropy > 0.01f;
    const bool decorrelated = r.delta_avg_cos < -0.01f;

    if (logit_deflated && entropy_improved && decorrelated) {
        oss << "  [VERDICT] PC1 IS THE MECHANISM: Token " << tok << " deflates, entropy improves, correlation drops.\n";
        oss << "           → Implement project_out_pc1 as autograd operation for production fix.\n";
    } else if (logit_deflated && entropy_improved) {
        oss << "  [VERDICT] PC1 IS LIKELY THE MECHANISM: Token " << tok << " deflates and entropy improves.\n";
        oss << "           avg_cos improvement: " << std::setprecision(4) << r.delta_avg_cos << "\n";
    } else if (logit_deflated) {
        oss << "  [VERDICT] PC1 PARTIALLY EXPLAINS collapse: logit deflates but entropy unchanged.\n";
        oss << "           Higher-rank structure may also contribute.\n";
    } else {
        oss << "  [VERDICT] PC1 IS NOT THE PRIMARY MECHANISM: Token " << tok << " did NOT deflate.\n";
        oss << "           Investigate: rank-2+ components, weight norm, or loss landscape.\n";
    }

    return oss.str();
}

} // namespace GRIM::Diagnostics

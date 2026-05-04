//======================================================//
//  TrainingDiagnostics.cu
//  Implementation of diagnostic functions extracted from Phase2_TrainingLoop.cu
//
//  Contains: WeightSample, EmbGradEquationDiag, Token277Diagnostic, UpdateTrace
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
#include <unordered_set>
#include <cstdint>
#include <array>

namespace GRIM::Diagnostics {

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

    cudaStream_t stream = ts.stream_ctrl.getPrimaryStream();
    cudaMemcpyAsync(sample.values, lm_head->weights().data,
                    kWeightSampleSize * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    // Sync primary stream only (not full device) so we can read the values
    cudaStreamSynchronize(stream);
    
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
    float prev_emb_rms,
    float curr_emb_rms,
    cudaStream_t stream
) {
    // SYNC DIAGNOSTIC CONTRACT: the cudaMemcpy calls below are blocking D2H
    // copies by design. This routine is only valid on the gated diagnostics
    // path after the caller has accepted sync-diagnostic cost.
    EmbGradEquationDiag diag{};
    diag.total_vocab = vocab_size;
    diag.prev_emb_rms = prev_emb_rms;
    diag.curr_emb_rms = curr_emb_rms;
    diag.emb_rms_delta = curr_emb_rms - prev_emb_rms;
    
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
    
    // Compute per-row RMS (matches training's norm metric)
    std::vector<float> row_rms_vals(vocab_size, 0.0f);
    double total_sum_sq = 0.0;
    size_t total_count = 0;
    for (int v = 0; v < vocab_size; ++v) {
        const float* row = h_grad.data() + static_cast<size_t>(v) * d_model;
        double row_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            row_sq += static_cast<double>(row[d]) * row[d];
        }
        row_rms_vals[v] = static_cast<float>(std::sqrt(row_sq / d_model));  // RMS per row
        total_sum_sq += row_sq;
        total_count += d_model;
        if (row_rms_vals[v] > 1e-10f) {
            diag.num_active_rows++;
        }
    }
    
    diag.grad_rms = (total_count > 0) ? static_cast<float>(std::sqrt(total_sum_sq / total_count)) : 0.0f;
    diag.active_ratio = static_cast<float>(diag.num_active_rows) / vocab_size;
    
    // Find top-K row RMS values
    std::vector<std::pair<float, int>> rms_idx(vocab_size);
    for (int v = 0; v < vocab_size; ++v) {
        rms_idx[v] = {row_rms_vals[v], v};
    }
    std::partial_sort(rms_idx.begin(), rms_idx.begin() + EmbGradEquationDiag::kTopK, 
                      rms_idx.end(), [](auto& a, auto& b) { return a.first > b.first; });
    
    for (int k = 0; k < EmbGradEquationDiag::kTopK; ++k) {
        diag.top_tokens[k] = rms_idx[k].second;
        diag.top_rms[k] = rms_idx[k].first;
    }
    
    diag.max_row_rms = rms_idx[0].first;
    diag.max_row_token = rms_idx[0].second;
    
    // Compute mean row RMS (over active rows only)
    if (diag.num_active_rows > 0) {
        double sum_rms = 0.0;
        for (int v = 0; v < vocab_size; ++v) {
            if (row_rms_vals[v] > 1e-10f) {
                sum_rms += row_rms_vals[v];
            }
        }
        diag.mean_row_rms = static_cast<float>(sum_rms / diag.num_active_rows);
    }
    
    diag.spike_ratio = (diag.mean_row_rms > 1e-10f) ? (diag.max_row_rms / diag.mean_row_rms) : 0.0f;
    
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
    oss << "[EMB_GRAD_EQUATION] WEIGHT_GRAD: grad_W = grad_lm + grad_emb (direct accumulation)\n";
    oss << "  EQUATION: grad_lm[v] = centered^T @ grad_logits[:,v] (dense matmul)\n";
    oss << "            grad_emb[tok] += grad_encoder[t] * emb_scale (sparse atomicAdd)\n";
    oss << "            grad_final = grad_lm + grad_emb (same buffer when tied, separate when untied)\n";
    oss << "  GRADIENT BUFFER: rms=" << diag.grad_rms
        << " active_rows=" << diag.num_active_rows << "/" << diag.total_vocab
        << " active_ratio=" << std::setprecision(4) << diag.active_ratio << "\n";
    oss << std::setprecision(6);
    oss << "  ROW RMS: mean=" << diag.mean_row_rms
        << " max=" << diag.max_row_rms << " (tok=" << diag.max_row_token << ")"
        << " spike_ratio=" << std::setprecision(2) << diag.spike_ratio << "x\n";
    oss << std::setprecision(6);
    oss << "  TOP-5 ROWS: ";
    for (int k = 0; k < EmbGradEquationDiag::kTopK; ++k) {
        if (k > 0) oss << ", ";
        oss << "tok" << diag.top_tokens[k] << "=" << diag.top_rms[k];
    }
    oss << "\n";
    oss << "  SCATTER DENSITY: most_frequent=tok" << diag.most_frequent_token
        << " (count=" << diag.most_frequent_count << ")\n";
    oss << "  BATCH TREND: prev_emb_rms=" << diag.prev_emb_rms
        << " curr_emb_rms=" << diag.curr_emb_rms
        << " delta=" << std::showpos << diag.emb_rms_delta << std::noshowpos << "\n";
    
    // Anomaly detection
    if (diag.spike_ratio > 10.0f) {
        oss << "  [ANOMALY] SPIKE_RATIO=" << std::setprecision(1) << diag.spike_ratio
            << "x > 10x — single token row dominates gradient. "
            << "Likely cause: frequent token (tok" << diag.max_row_token 
            << ") with high atomicAdd contention OR concentrated LM head gradient.\n";
    }
    if (diag.emb_rms_delta > diag.prev_emb_rms * 0.5f && diag.prev_emb_rms > 0.01f) {
        oss << "  [ANOMALY] EMB_RMS_SPIKE: delta=" << diag.emb_rms_delta 
            << " > 50% of prev=" << diag.prev_emb_rms
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
    diag.grad_row_rms = static_cast<float>(std::sqrt(grad_sum_sq / d_model));
    
    // Compute weight statistics
    double weight_sum = 0.0, weight_sum_sq = 0.0;
    for (int i = 0; i < d_model; ++i) {
        weight_sum += weight_row[i];
        weight_sum_sq += static_cast<double>(weight_row[i]) * weight_row[i];
    }
    diag.weight_row_rms = static_cast<float>(std::sqrt(weight_sum_sq / d_model));
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
    oss << "  GRAD_W[" << tok << "]: rms=" << diag.grad_row_rms
        << " sum=" << diag.grad_row_sum
        << " mean=" << diag.grad_row_mean << "\n";
    oss << "  WEIGHT[" << tok << "]: rms=" << diag.weight_row_rms
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
//  UpdateTrace — Per-component Adam update magnitude (Issue #150)
//======================================================//

const char* UpdateTraceMetrics::typeName(int type_idx) {
    switch (type_idx) {
        case 0: return "emb";
        case 1: return "lm";
        case 2: return "attn";
        case 3: return "ffn";
        case 4: return "rmsnorm";
        case 5: return "sb";
        default: return "?";
    }
}

UpdateTraceMetrics computePerComponentUpdateTrace(
    const std::vector<GRIM::ParameterGroup>& groups,
    float learning_rate,
    int optimizer_step,
    cudaStream_t stream
) {
    UpdateTraceMetrics result{};
    if (groups.empty() || optimizer_step <= 0) return result;
    if (!stream) throw std::runtime_error("[computePerComponentUpdateTrace] stream is NULL");

    // Bias corrections for Adam moment estimates
    const float bc1 = 1.0f - std::pow(0.9f, static_cast<float>(optimizer_step));
    const float bc2 = 1.0f - std::pow(0.999f, static_cast<float>(optimizer_step));
    if (bc1 <= 0.0f || bc2 <= 0.0f) return result;
    const float inv_bc1 = 1.0f / bc1;
    const float inv_bc2 = 1.0f / bc2;

    // For each component type, find the FIRST group with valid data and sample it.
    // We sample m_state, v_state, and params from that group.
    bool any_sampled = false;

    // Track which types we've already sampled (one group per type is sufficient)
    bool type_sampled[kNumComponentTypes] = {};

    for (const auto& group : groups) {
        const int ti = static_cast<int>(group.type);
        if (ti < 0 || ti >= kNumComponentTypes) continue;
        if (type_sampled[ti]) continue;
        if (!group.weights() || !group.m_state() || !group.v_state() || group.size() == 0) continue;

        const int total = static_cast<int>(group.size());
        const int n = std::min(total, kUpdateTraceSampleSize);

        // STRIDED sampling: spread across the full buffer to avoid dead rows.
        // E.g., embedding [2533×768] — consecutive sampling hits only token 0
        // (UNK, never trained). Stride = total/n ensures we touch diverse rows.
        const int stride = (total > n) ? (total / n) : 1;

        // Small host buffers (stack-allocated, no heap)
        float h_params[kUpdateTraceSampleSize];
        float h_m[kUpdateTraceSampleSize];
        float h_v[kUpdateTraceSampleSize];

        // Copy strided elements from GPU to host (one element at a time for stride>1)
        if (stride <= 1) {
            // Consecutive — single copy is fine
            cudaMemcpyAsync(h_params, group.weights(), n * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_m, group.m_state(), n * sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_v, group.v_state(), n * sizeof(float), cudaMemcpyDeviceToHost, stream);
        } else {
            // Strided — copy individual elements
            for (int s = 0; s < n; ++s) {
                const int idx = s * stride;
                cudaMemcpyAsync(&h_params[s], group.weights() + idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(&h_m[s],      group.m_state() + idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(&h_v[s],      group.v_state() + idx, sizeof(float), cudaMemcpyDeviceToHost, stream);
            }
        }
        cudaStreamSynchronize(stream);

        // Compute Adam update RMS and param RMS from samples
        float update_sq_sum = 0.0f;
        float param_sq_sum = 0.0f;

        for (int i = 0; i < n; ++i) {
            const float m_hat = h_m[i] * inv_bc1;
            const float v_hat = h_v[i] * inv_bc2;
            // Adam update magnitude (without weight decay contribution)
            const float adam_update = learning_rate * m_hat / (std::sqrt(std::abs(v_hat)) + 1e-8f);
            update_sq_sum += adam_update * adam_update;
            param_sq_sum += h_params[i] * h_params[i];
        }

        result.update_rms[ti] = std::sqrt(update_sq_sum / static_cast<float>(n));
        result.param_rms[ti] = std::sqrt(param_sq_sum / static_cast<float>(n));
        result.update_over_param[ti] = (result.param_rms[ti] > 1e-12f)
            ? (result.update_rms[ti] / result.param_rms[ti]) : 0.0f;
        result.element_count[ti] = n;
        result.has_data[ti] = true;
        type_sampled[ti] = true;
        any_sampled = true;
    }

    result.valid = any_sampled;
    return result;
}

std::string formatUpdateTrace(const UpdateTraceMetrics& m, int batch_idx, bool tied) {
    if (!m.valid) return "";

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(4);

    // Line 1: Per-component update_rms
    oss << "[UpdateTrace] COMPONENTS(upd_rms) batch=" << batch_idx;
    if (tied) {
        // When tied, EMBEDDING bucket is empty; LM_HEAD contains both
        if (m.has_data[1]) oss << " emb_lm_tied=" << m.update_rms[1];
    } else {
        if (m.has_data[0]) oss << " emb=" << m.update_rms[0];
        if (m.has_data[1]) oss << " lm=" << m.update_rms[1];
    }
    if (m.has_data[2]) oss << " attn=" << m.update_rms[2];
    if (m.has_data[3]) oss << " ffn=" << m.update_rms[3];
    if (m.has_data[4]) oss << " rmsnorm=" << m.update_rms[4];
    if (m.has_data[5]) oss << " sb=" << m.update_rms[5];

    // Line 2: Per-component update_rms / param_rms (effective relative LR)
    oss << "\n[UpdateTrace] COMPONENTS(upd/param) batch=" << batch_idx;
    if (tied) {
        if (m.has_data[1]) oss << " emb_lm_tied=" << m.update_over_param[1];
    } else {
        if (m.has_data[0]) oss << " emb=" << m.update_over_param[0];
        if (m.has_data[1]) oss << " lm=" << m.update_over_param[1];
    }
    if (m.has_data[2]) oss << " attn=" << m.update_over_param[2];
    if (m.has_data[3]) oss << " ffn=" << m.update_over_param[3];
    if (m.has_data[4]) oss << " rmsnorm=" << m.update_over_param[4];
    if (m.has_data[5]) oss << " sb=" << m.update_over_param[5];

    // Line 3: Ratios relative to FFN (the "reference" component)
    if (m.has_data[3] && m.update_rms[3] > 1e-15f) {
        oss << "\n[UpdateTrace] RATIOS(vs_ffn) batch=" << batch_idx;
        for (int t = 0; t < kNumComponentTypes; ++t) {
            if (!m.has_data[t] || t == 3) continue;
            if (tied && t == 0) continue;  // Skip empty embedding bucket when tied
            const float ratio = m.update_rms[t] / m.update_rms[3];
            oss << " " << UpdateTraceMetrics::typeName(t) << "=" << std::fixed << std::setprecision(2) << ratio << "x";
        }
        oss << std::scientific << std::setprecision(4);
    }

    return oss.str();
}

} // namespace GRIM::Diagnostics

//======================================================//
//  RhoDiagnostic.cu
//  Per-Layer Hidden State Correlation Diagnostic
//======================================================//

#include "RhoDiagnostic.hpp"

#include "../Phases/Phase1_Startup.hpp"
#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"

#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <cuda_runtime.h>

using GRIMText::Training::Internal::formatScalar;

namespace GRIM::Diagnostics {

void computeRhoDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx)
{
    const auto& ts = ctx.model->getTrainingState();
    const auto& ai = ts.autograd_intermediates;
    const int num_layers = static_cast<int>(ai.encoder_layer_outputs.size());
    const int d_model = ctx.model->getConfig().d_model;
    const int max_seq_len = ts.cached_seq_len;
    const int rect_positions = ts.cached_batch_size * max_seq_len;

    if (num_layers <= 0 || rect_positions < 2) return;

    // ── Geometry invariant guards ──
    // The hidden-state rectangular buffer is [cached_batch_size * cached_seq_len, d_model].
    // payload dimensions must fit inside that allocation or we index past the copy.
    if (payload.batch_size > ts.cached_batch_size) {
        throw std::runtime_error(
            "[RhoDiagnostic] payload.batch_size (" + std::to_string(payload.batch_size) +
            ") > ts.cached_batch_size (" + std::to_string(ts.cached_batch_size) +
            ") at " + __FILE__ + ":" + std::to_string(__LINE__));
    }
    if (payload.max_seq_len != max_seq_len) {
        throw std::runtime_error(
            "[RhoDiagnostic] payload.max_seq_len (" + std::to_string(payload.max_seq_len) +
            ") != ts.cached_seq_len (" + std::to_string(max_seq_len) +
            ") — layout mismatch, at " + __FILE__ + ":" + std::to_string(__LINE__));
    }
    for (int s = 0; s < payload.batch_size; ++s) {
        if (payload.seq_lengths[s] > max_seq_len) {
            throw std::runtime_error(
                "[RhoDiagnostic] seq_lengths[" + std::to_string(s) + "] = " +
                std::to_string(payload.seq_lengths[s]) + " > max_seq_len (" +
                std::to_string(max_seq_len) + ") at " + __FILE__ + ":" + std::to_string(__LINE__));
        }
    }

    // Build valid position indices from payload.seq_lengths.
    // Hidden states are stored in [batch_size * max_seq_len, d_model] rectangular layout.
    // Only positions [s * max_seq_len .. s * max_seq_len + seq_lengths[s] - 1] are real tokens;
    // the rest are padding / stale cached positions that would contaminate ρ and RMS.
    std::vector<int> valid_positions;
    {
        int total_valid = 0;
        for (int s = 0; s < payload.batch_size; ++s) {
            total_valid += payload.seq_lengths[s];
        }
        valid_positions.reserve(total_valid);
        for (int s = 0; s < payload.batch_size; ++s) {
            const int flat_start = s * max_seq_len;
            const int len = payload.seq_lengths[s];
            for (int t = 0; t < len; ++t) {
                valid_positions.push_back(flat_start + t);
            }
        }
    }
    const int num_valid = static_cast<int>(valid_positions.size());
    if (num_valid < 2) return;

    // Helper: compute avg |cos| from a stratified sample of valid position pairs.
    // Copies the full rectangular buffer once, then indexes only valid rows.
    // Stratified sampling: pick up to MAX_SAMPLE positions uniformly, then exhaust
    // all C(sample,2) pairs.  This keeps cost O(sample^2 * d_model) per layer
    // instead of O(num_valid^2 * d_model), which would dominate wall-time when
    // num_valid is large (e.g. 3000+ tokens → ~4.5M pairs × 768 dims per layer).
    static constexpr int MAX_RHO_SAMPLE = 128;

    // Raw components returned alongside ρ and h_rms so we can trace WHY ρ moves.
    struct RhoRaw {
        float rho;            // avg|cos(h_i, h_j)|
        float avg_rms;        // mean(rms(h[t]))
        float avg_abs_dot;    // mean|dot(h_i, h_j)|  — numerator signal
        float avg_norm_prod;  // mean(‖h_i‖·‖h_j‖·d) — denominator
        float rms_min;        // min per-position rms  — collapse detector
        float rms_max;        // max per-position rms  — explosion detector
    };

    auto compute_rho = [&](const float* device_ptr) -> RhoRaw {
        if (!device_ptr) return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        const size_t bytes = static_cast<size_t>(rect_positions) * d_model * sizeof(float);
        std::vector<float> h(rect_positions * d_model);
        cudaError_t err = cudaMemcpy(h.data(), device_ptr, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("[RhoDiagnostic] cudaMemcpy failed: ") + cudaGetErrorString(err) +
                " at " + __FILE__ + ":" + std::to_string(__LINE__));
        }

        // Build sample indices: if num_valid <= MAX_RHO_SAMPLE, use all;
        // otherwise pick MAX_RHO_SAMPLE positions via stride-based uniform sampling
        // (deterministic per batch, no RNG dependency).
        std::vector<int> sample_indices;
        if (num_valid <= MAX_RHO_SAMPLE) {
            sample_indices.resize(num_valid);
            for (int i = 0; i < num_valid; ++i) sample_indices[i] = i;
        } else {
            sample_indices.reserve(MAX_RHO_SAMPLE);
            // Stride-based uniform selection: pick every (num_valid / MAX_RHO_SAMPLE)-th position
            const double stride = static_cast<double>(num_valid) / MAX_RHO_SAMPLE;
            for (int k = 0; k < MAX_RHO_SAMPLE; ++k) {
                sample_indices.push_back(static_cast<int>(k * stride));
            }
        }
        const int n_sample = static_cast<int>(sample_indices.size());

        // Pre-compute row RMS for ALL valid positions (needed for avg_rms),
        // but pairwise cos only uses sample_indices.
        std::vector<float> rms_vals(num_valid);
        double rms_sum = 0.0;
        for (int vi = 0; vi < num_valid; ++vi) {
            const int row = valid_positions[vi];
            double sq = 0.0;
            for (int d = 0; d < d_model; ++d) {
                float v = h[row * d_model + d];
                sq += static_cast<double>(v) * v;
            }
            rms_vals[vi] = static_cast<float>(std::sqrt(sq / d_model));
            rms_sum += rms_vals[vi];
        }
        float avg_rms = static_cast<float>(rms_sum / num_valid);
        float rms_min_val = *std::min_element(rms_vals.begin(), rms_vals.end());
        float rms_max_val = *std::max_element(rms_vals.begin(), rms_vals.end());

        // Compute pairwise |cos| over sampled position pairs
        // Also accumulate raw |dot| and norm products separately.
        // cos = |dot| / (rms_i * rms_j * d_model)
        double cos_acc = 0.0;
        double dot_acc = 0.0;       // Σ|dot(h_i, h_j)|
        double norm_prod_acc = 0.0; // Σ(rms_i * rms_j * d_model)
        int n_pairs = 0;
        for (int si = 0; si < n_sample; ++si) {
            const int vi = sample_indices[si];
            if (rms_vals[vi] < 1e-8f) continue;
            const int ri = valid_positions[vi];
            for (int sj = si + 1; sj < n_sample; ++sj) {
                const int vj = sample_indices[sj];
                if (rms_vals[vj] < 1e-8f) continue;
                const int rj = valid_positions[vj];
                double dot = 0.0;
                for (int d = 0; d < d_model; ++d) {
                    dot += static_cast<double>(h[ri * d_model + d]) * h[rj * d_model + d];
                }
                double abs_dot = std::abs(dot);
                double denom = static_cast<double>(rms_vals[vi]) * rms_vals[vj] * d_model;
                cos_acc += abs_dot / denom;
                dot_acc += abs_dot;
                norm_prod_acc += denom;
                ++n_pairs;
            }
        }
        float rho = (n_pairs > 0) ? static_cast<float>(cos_acc / n_pairs) : 0.0f;
        float avg_abs_dot = (n_pairs > 0) ? static_cast<float>(dot_acc / n_pairs) : 0.0f;
        float avg_norm_prod = (n_pairs > 0) ? static_cast<float>(norm_prod_acc / n_pairs) : 0.0f;
        return {rho, avg_rms, avg_abs_dot, avg_norm_prod, rms_min_val, rms_max_val};
    };

    // Collect ρ for each layer + embedding.
    // Track the real layer identity so labels stay correct even if tensors are missing.
    struct LayerRho {
        int layer_id;         // -1 = embedding, 0..N-1 = encoder layer
        int delta_vs_id;      // layer_id of predecessor used for Δρ, or -2 if none
        float rho;            // avg |cos(h_i, h_j)|
        float rms;            // avg rms(h[t])
        float delta_rho;      // ρ(this) - ρ(delta_vs_id)
        float avg_abs_dot;    // raw numerator: mean|dot(h_i, h_j)|
        float avg_norm_prod;  // raw denominator: mean(‖h_i‖·‖h_j‖·d)
        float rms_min;        // min per-position rms
        float rms_max;        // max per-position rms
    };
    std::vector<LayerRho> layer_rhos;
    layer_rhos.reserve(num_layers + 1);  // +1 for embedding

    // Embedding (input to layer 0)
    if (ai.embedding_tensor.data) {
        auto raw = compute_rho(ai.embedding_tensor.data);
        layer_rhos.push_back({-1, -2, raw.rho, raw.avg_rms, 0.0f,
                              raw.avg_abs_dot, raw.avg_norm_prod, raw.rms_min, raw.rms_max});
    }

    // Each encoder layer output
    for (int l = 0; l < num_layers; ++l) {
        if (ai.encoder_layer_outputs[l].data) {
            auto raw = compute_rho(ai.encoder_layer_outputs[l].data);
            float delta = 0.0f;
            int vs_id = -2;
            if (!layer_rhos.empty()) {
                delta = raw.rho - layer_rhos.back().rho;
                vs_id = layer_rhos.back().layer_id;
            }
            layer_rhos.push_back({l, vs_id, raw.rho, raw.avg_rms, delta,
                                  raw.avg_abs_dot, raw.avg_norm_prod, raw.rms_min, raw.rms_max});
        }
    }

    // LM head input (post-centering when centering is enabled).
    // cached_encoder_output is overwritten with centered data after LM head forward.
    // Using layer_id = num_layers to distinguish from raw encoder layers.
    if (ts.cached_encoder_output.data) {
        auto raw = compute_rho(ts.cached_encoder_output.data);
        float delta = 0.0f;
        int vs_id = -2;
        if (!layer_rhos.empty()) {
            delta = raw.rho - layer_rhos.back().rho;
            vs_id = layer_rhos.back().layer_id;
        }
        layer_rhos.push_back({num_layers, vs_id, raw.rho, raw.avg_rms, delta,
                              raw.avg_abs_dot, raw.avg_norm_prod, raw.rms_min, raw.rms_max});
    }

    // Build the equation log
    std::ostringstream rho_eq;
    rho_eq << std::fixed << std::setprecision(4);
    rho_eq << "[RHO_BUILDUP_EQUATION] ρ(l) = avg|cos(h_i^l, h_j^l)|, "
           << "Δρ = ρ(l) - ρ(prev_collected)\n";
    rho_eq << "  ARCH: h^l = h^{l-1} + LS*Attn(RMSNorm(h^{l-1})) "
           << "+ LS*FFN(RMSNorm(...))\n";

    // Compact per-layer table
    rho_eq << "  LAYER  ρ(l)    Δρ(vs)    h_rms_avg  interpretation\n";
    rho_eq << "  ─────  ──────  ────────  ─────────  ──────────────\n";

    float max_delta = 0.0f;
    int max_delta_layer = -1;  // real layer index from layer_id, not vector position
    float rho_growth = 0.0f;   // total ρ change first→final

    for (size_t i = 0; i < layer_rhos.size(); ++i) {
        const auto& lr = layer_rhos[i];

        // Label from real layer identity, not compacted vector index
        std::string label;
        if (lr.layer_id == -1) {
            label = "emb  ";
        } else if (lr.layer_id == num_layers) {
            label = "lm_in";
        } else {
            label = "L" + std::to_string(lr.layer_id);
            if (lr.layer_id < 10) label += "   ";
            else label += "  ";
        }

        rho_eq << "  " << label << "  "
               << std::setw(6) << lr.rho << "  ";

        // Δρ with honest predecessor label
        if (lr.delta_vs_id == -2) {
            rho_eq << "  —       ";
        } else {
            // Show what the delta is actually against
            std::string vs_tag;
            if (lr.delta_vs_id == -1) vs_tag = "e";
            else vs_tag = std::to_string(lr.delta_vs_id);
            // Format: "+0.0312(v3)" — the (vN) shows which layer Δ is relative to
            std::ostringstream delta_cell;
            delta_cell << std::fixed << std::setprecision(4)
                       << std::showpos << lr.delta_rho << std::noshowpos
                       << "(v" << vs_tag << ")";
            rho_eq << std::setw(10) << delta_cell.str() << "  ";
        }

        rho_eq << std::setw(9) << lr.rms << "  ";

        // Interpretation
        const bool is_encoder = (lr.layer_id >= 0 && lr.layer_id < num_layers);
        if (lr.rho > 0.8f) {
            rho_eq << "[ANOMALY] COLLAPSE RISK";
        } else if (lr.rho > 0.5f) {
            rho_eq << "[WARNING] HIGH CORRELATION";
        } else if (is_encoder && lr.delta_rho > 0.05f) {
            rho_eq << "[ANOMALY] LAYER AMPLIFIES ρ";
        } else if (is_encoder && lr.delta_rho < -0.05f) {
            rho_eq << "decorrelation";
        } else {
            rho_eq << "healthy";
        }
        rho_eq << "\n";

        // Track worst amplifier (encoder layers only)
        if (is_encoder && lr.delta_rho > max_delta) {
            max_delta = lr.delta_rho;
            max_delta_layer = lr.layer_id;
        }
    }

    // Summary line — use actual layer identities, not vector position assumptions.
    // "first" and "last" are whatever layers were actually collected; label them honestly.
    if (layer_rhos.size() >= 2) {
        const auto& first_lr = layer_rhos.front();
        const auto& last_lr  = layer_rhos.back();

        float rho_first = first_lr.rho;
        float rho_final = last_lr.rho;
        rho_growth = rho_final - rho_first;
        float h_rms_growth_ratio = last_lr.rms / std::max(first_lr.rms, 1e-8f);

        // Honest labels for what we actually measured
        auto make_label = [&](int id) -> std::string {
            if (id == -1) return "emb";
            if (id == num_layers) return "lm_in";
            return "L" + std::to_string(id);
        };
        std::string first_label = make_label(first_lr.layer_id);
        std::string last_label  = make_label(last_lr.layer_id);

        // Write rho observations directly into the telemetry observation array
        // (streams 5-8 persist between diagnostic intervals via last_obs[])
        ctx.telemetry.last_obs[5] = rho_final;          // RHO_FINAL
        ctx.telemetry.last_obs[6] = rho_growth;         // RHO_GROWTH
        ctx.telemetry.last_obs[7] = max_delta;          // RHO_WORST_DELTA
        ctx.telemetry.last_obs[8] = h_rms_growth_ratio; // H_RMS_GROWTH

        // Raw decomposition of final layer's ρ — trace WHY correlation moves
        ctx.telemetry.last_obs[31] = last_lr.avg_abs_dot;    // RHO_RAW_AVG_ABS_DOT
        ctx.telemetry.last_obs[32] = last_lr.avg_norm_prod;  // RHO_RAW_AVG_NORM_PROD
        ctx.telemetry.last_obs[33] = last_lr.rms_min;        // RHO_RAW_H_RMS_MIN
        ctx.telemetry.last_obs[34] = last_lr.rms_max;        // RHO_RAW_H_RMS_MAX
        // Per-position rms bifurcation — the proximate driver of ρ spikes when
        // a subset of positions is stripped to near-zero (Apr 2026 finding).
        // ρ = mean(|dot|/(rms_i*rms_j*d)); when rms_min collapses, the per-pair
        // denominator goes to ~0 and ρ reads as noise/0 ⇒ spuriously high ρ.
        // Healthy training: spread ≈ 1.0–1.5x.  >2x = early warning.
        const float rms_spread = last_lr.rms_max
                               / std::max(last_lr.rms_min, 1e-8f);
        ctx.telemetry.last_obs[38] = rms_spread;             // RHO_RAW_RMS_SPREAD

        rho_eq << "  SUMMARY: ρ(" << first_label << ")=" << rho_first
               << " → ρ(" << last_label << ")=" << rho_final
               << " growth=" << std::showpos << rho_growth << std::noshowpos
               << " h_rms_growth=" << h_rms_growth_ratio
               << "x\n";
        rho_eq << "  RAW(" << last_label << "): |dot|=" << last_lr.avg_abs_dot
               << " denom=" << last_lr.avg_norm_prod
               << " rms[min..max]=[" << last_lr.rms_min
               << ".." << last_lr.rms_max
               << "] spread=" << rms_spread << "x";
        if (rms_spread > 4.0f) {
            rho_eq << "  [ANOMALY] RMS BIFURCATION: positions are being stripped to"
                      " near-zero — ρ numerator/denominator collapse";
        } else if (rms_spread > 2.0f) {
            rho_eq << "  [WARNING] rms spread >2x — watch denominator";
        }
        rho_eq << "\n";

        if (max_delta_layer >= 0 && max_delta > 0.02f) {
            // Find the entry to report what it was measured against
            std::string vs_label = "?";
            for (const auto& lr : layer_rhos) {
                if (lr.layer_id == max_delta_layer) {
                    vs_label = make_label(lr.delta_vs_id);
                    break;
                }
            }
            rho_eq << "  WORST AMPLIFIER: L" << max_delta_layer
                   << " (Δρ=" << std::showpos << max_delta << std::noshowpos
                   << " vs " << vs_label << ")\n";
        }

        if (rho_final > 0.8f) {
            rho_eq << "  [ANOMALY] MODE COLLAPSE IMMINENT: ρ(" << last_label << ")=" << rho_final
                   << " → rank-1 hidden states → winner-take-all logits\n";
        }
    }

    // Top 10 most frequent input tokens in this batch
    {
        std::unordered_map<int, int> tok_freq;
        for (int s = 0; s < payload.batch_size; ++s) {
            const int flat_start = s * max_seq_len;
            const int len = payload.seq_lengths[s];
            for (int t = 0; t < len; ++t) {
                ++tok_freq[payload.input_ids[flat_start + t]];
            }
        }
        std::vector<std::pair<int, int>> freq_sorted(tok_freq.begin(), tok_freq.end());
        std::sort(freq_sorted.begin(), freq_sorted.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        const size_t top_n = std::min(freq_sorted.size(), size_t(10));
        rho_eq << "  TOP-10 INPUT TOKENS:";
        for (size_t i = 0; i < top_n; ++i) {
            const int tid = freq_sorted[i].first;
            const int cnt = freq_sorted[i].second;
            std::string decoded = ctx.tokenizer.decode({tid});
            // Escape newlines/tabs for single-line display
            for (auto& c : decoded) {
                if (c == '\n') c = ' ';
                else if (c == '\t') c = ' ';
                else if (c == '\r') c = ' ';
            }
            if (decoded.size() > 20) decoded = decoded.substr(0, 20) + "…";
            rho_eq << " [" << tid << " \"" << decoded << "\" ×" << cnt << "]";
        }
        rho_eq << "\n";
    }

    ctx.logging.logger->log(rho_eq.str());
    EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Telemetry, GRIM::Logging::LogPhase::DIAGNOSTICS, -1, "RHO_BUILDUP_EQUATION", rho_eq.str().c_str());
}

} // namespace GRIM::Diagnostics

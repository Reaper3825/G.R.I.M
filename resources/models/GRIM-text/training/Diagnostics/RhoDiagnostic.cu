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
#include "../../Shared/Telemetry/TelemetryLattice_GPU.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"

#include <vector>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <cuda_runtime.h>

namespace GRIM::Diagnostics {

namespace {

std::string decodeAggregateTokenForDisplay(const GRIM::Tokenizer::UniByte& tokenizer, int token_id) {
    const GRIM::Tokenizer::TokenLayout layout = tokenizer.tokenLayout();
    if (layout.isAtom(token_id)) {
        return std::string("<") +
               GRIM::Tokenizer::atomTypeName(GRIM::Tokenizer::tokenIdToAtomType(token_id)) +
               ">";
    }
    return tokenizer.decode(GRIM::Tokenizer::DecodeRequest({token_id}));
}

void sanitizeSingleLineTokenDisplay(std::string& decoded) {
    for (auto& c : decoded) {
        if (c == '\n') c = ' ';
        else if (c == '\t') c = ' ';
        else if (c == '\r') c = ' ';
    }
    if (decoded.size() > 20) decoded = decoded.substr(0, 20) + "…";
}

} // namespace

void computeRhoDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Forward::ModelForwardOutputs& ai,
    int batch_idx)
{
    const int num_layers = static_cast<int>(ai.encoder_layer_outputs.size());
    const int d_model = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "d_model");
    const int max_seq_len = payload.max_seq_len;
    const int rect_positions = payload.total_tokens;

    if (num_layers <= 0 || rect_positions < 2) return;
    if (!ctx.tokenizer) {
        throw std::runtime_error("RhoDiagnostic requires initialized ctx.tokenizer");
    }
    const auto& tokenizer = *ctx.tokenizer;

    // ── Geometry invariant guards ──
    // Hidden-state snapshots are laid out using the Phase1-authored payload rectangle:
    // [payload.batch_size * payload.max_seq_len, d_model]. Do not rediscover
    // current-step geometry from TrainingState.
    if (payload.total_tokens != payload.batch_size * payload.max_seq_len) {
        throw std::runtime_error(
            "[RhoDiagnostic] payload.total_tokens (" + std::to_string(payload.total_tokens) +
            ") != payload.batch_size * payload.max_seq_len (" +
            std::to_string(payload.batch_size * payload.max_seq_len) +
            ") at " + __FILE__ + ":" + std::to_string(__LINE__));
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

    // Partition valid positions into atom-token vs non-atom-token subsets.
    // ScratchBlock injects a shared per-type vector into atom positions only
    // (first_type_only mode). If that injection is what drives ρ up, the
    // atom-only ρ will spike while non-atom ρ stays flat. Computing both on the
    // final collected layer isolates the injection's contribution directly.
    const GRIM::Tokenizer::TokenLayout token_layout = tokenizer.tokenLayout();
    std::vector<int> atom_positions;
    std::vector<int> nonatom_positions;
    atom_positions.reserve(num_valid);
    nonatom_positions.reserve(num_valid);
    for (int pos : valid_positions) {
        if (token_layout.isAtom(payload.input_ids[pos])) {
            atom_positions.push_back(pos);
        } else {
            nonatom_positions.push_back(pos);
        }
    }

    // Helper: compute avg |cos| from a stratified sample of valid position pairs.
    // Copies the full rectangular buffer once, then indexes only valid rows.
    // Stratified sampling: pick up to MAX_SAMPLE positions uniformly, then exhaust
    // all C(sample,2) pairs.  This keeps cost O(sample^2 * d_model) per layer
    // instead of O(num_valid^2 * d_model), which would dominate wall-time when
    // num_valid is large (e.g. 3000+ tokens → ~4.5M pairs × 768 dims per layer).
    static constexpr int MAX_RHO_SAMPLE = 12288;

    // Raw components returned alongside ρ and h_rms so we can trace WHY ρ moves.
    struct RhoRaw {
        float rho;            // avg|cos(h_i, h_j)|
        float avg_rms;        // mean(rms(h[t]))
        float avg_abs_dot;    // mean|dot(h_i, h_j)|  — numerator signal
        float avg_signed_dot; // mean dot(h_i, h_j)  — signed common-mode signal
        float avg_norm_prod;  // mean(‖h_i‖·‖h_j‖·d) — denominator
        float rms_min;        // min per-position rms  — collapse detector
        float rms_max;        // max per-position rms  — explosion detector
        float centered_avg_abs_dot; // mean|dot(h_i - μ, h_j - μ)| — mean-removed alignment
        float mean_rms;       // rms(μ), μ = mean_i h_i — hidden DC vector magnitude
    };

    auto compute_rho = [&](const float* device_ptr, const std::vector<int>& positions) -> RhoRaw {
        if (!device_ptr) return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const int n_pos = static_cast<int>(positions.size());
        if (n_pos < 2) return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        const size_t bytes = static_cast<size_t>(rect_positions) * d_model * sizeof(float);
        std::vector<float> h(rect_positions * d_model);
        cudaError_t err = cudaMemcpy(h.data(), device_ptr, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("[RhoDiagnostic] cudaMemcpy failed: ") + cudaGetErrorString(err) +
                " at " + __FILE__ + ":" + std::to_string(__LINE__));
        }

        // Build sample indices: if n_pos <= MAX_RHO_SAMPLE, use all;
        // otherwise pick MAX_RHO_SAMPLE positions via stride-based uniform sampling
        // (deterministic per batch, no RNG dependency).
        std::vector<int> sample_indices;
        if (n_pos <= MAX_RHO_SAMPLE) {
            sample_indices.resize(n_pos);
            for (int i = 0; i < n_pos; ++i) sample_indices[i] = i;
        } else {
            sample_indices.reserve(MAX_RHO_SAMPLE);
            // Stride-based uniform selection: pick every (n_pos / MAX_RHO_SAMPLE)-th position
            const double stride = static_cast<double>(n_pos) / MAX_RHO_SAMPLE;
            for (int k = 0; k < MAX_RHO_SAMPLE; ++k) {
                sample_indices.push_back(static_cast<int>(k * stride));
            }
        }
        const int n_sample = static_cast<int>(sample_indices.size());

        // Mean hidden vector for centered pairwise dots:
        //   μ_l = (1/N) Σ_i h_i^l
        //   mean_rms(l) = sqrt((1/d_model) Σ_d μ_{l,d}²)
        // Use ALL positions in this subset for μ, not only the sampled pair subset.
        std::vector<double> mean(d_model, 0.0);
        for (int vi = 0; vi < n_pos; ++vi) {
            const int row = positions[vi];
            for (int d = 0; d < d_model; ++d) {
                mean[d] += static_cast<double>(h[row * d_model + d]);
            }
        }
        double mean_sq = 0.0;
        for (int d = 0; d < d_model; ++d) {
            mean[d] /= static_cast<double>(n_pos);
            mean_sq += mean[d] * mean[d];
        }
        const float mean_rms = static_cast<float>(std::sqrt(mean_sq / d_model));

        // Pre-compute row RMS for ALL positions in this subset (needed for avg_rms),
        // but pairwise cos only uses sample_indices.
        std::vector<float> rms_vals(n_pos);
        double rms_sum = 0.0;
        for (int vi = 0; vi < n_pos; ++vi) {
            const int row = positions[vi];
            double sq = 0.0;
            for (int d = 0; d < d_model; ++d) {
                float v = h[row * d_model + d];
                sq += static_cast<double>(v) * v;
            }
            rms_vals[vi] = static_cast<float>(std::sqrt(sq / d_model));
            rms_sum += rms_vals[vi];
        }
        float avg_rms = static_cast<float>(rms_sum / n_pos);
        float rms_min_val = *std::min_element(rms_vals.begin(), rms_vals.end());
        float rms_max_val = *std::max_element(rms_vals.begin(), rms_vals.end());

        // Compute pairwise |cos| over sampled position pairs.
        // Raw dot metrics use P = C(n_sample, 2):
        //   avg_signed_dot = P^-1 Σ_{i<j}(h_i · h_j)
        //   centered_avg_abs_dot = P^-1 Σ_{i<j}|h̃_i · h̃_j|
        // Rho alone skips zero-denominator pairs because cos is undefined there.
        // cos = |dot| / (rms_i * rms_j * d_model)
        double cos_acc = 0.0;
        double dot_acc = 0.0;       // Σ|dot(h_i, h_j)|
        double signed_dot_acc = 0.0; // Σ dot(h_i, h_j)
        double centered_dot_acc = 0.0; // Σ|dot(h_i - μ, h_j - μ)|
        double norm_prod_acc = 0.0; // Σ(rms_i * rms_j * d_model)
        int n_pairs = 0;
        int n_cos_pairs = 0;
        for (int si = 0; si < n_sample; ++si) {
            const int vi = sample_indices[si];
            const int ri = positions[vi];
            for (int sj = si + 1; sj < n_sample; ++sj) {
                const int vj = sample_indices[sj];
                const int rj = positions[vj];
                double dot = 0.0;
                double centered_dot = 0.0;
                for (int d = 0; d < d_model; ++d) {
                    const double hi = static_cast<double>(h[ri * d_model + d]);
                    const double hj = static_cast<double>(h[rj * d_model + d]);
                    dot += hi * hj;
                    centered_dot += (hi - mean[d]) * (hj - mean[d]);
                }
                double abs_dot = std::abs(dot);
                double denom = static_cast<double>(rms_vals[vi]) * rms_vals[vj] * d_model;
                dot_acc += abs_dot;
                signed_dot_acc += dot;
                centered_dot_acc += std::abs(centered_dot);
                norm_prod_acc += denom;
                ++n_pairs;
                if (denom > 1e-8) {
                    cos_acc += abs_dot / denom;
                    ++n_cos_pairs;
                }
            }
        }
        float rho = 0.0f;
        if (n_cos_pairs > 0) {
            rho = static_cast<float>(cos_acc / n_cos_pairs);
        }
        float avg_abs_dot = 0.0f;
        float avg_signed_dot = 0.0f;
        float centered_avg_abs_dot = 0.0f;
        float avg_norm_prod = 0.0f;
        if (n_pairs > 0) {
            avg_abs_dot = static_cast<float>(dot_acc / n_pairs);
            avg_signed_dot = static_cast<float>(signed_dot_acc / n_pairs);
            centered_avg_abs_dot = static_cast<float>(centered_dot_acc / n_pairs);
            avg_norm_prod = static_cast<float>(norm_prod_acc / n_pairs);
        }
        return {rho, avg_rms, avg_abs_dot, avg_signed_dot, avg_norm_prod,
                rms_min_val, rms_max_val, centered_avg_abs_dot, mean_rms};
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
        float avg_signed_dot; // signed numerator: mean dot(h_i, h_j)
        float avg_norm_prod;  // raw denominator: mean(‖h_i‖·‖h_j‖·d)
        float rms_min;        // min per-position rms
        float rms_max;        // max per-position rms
        float centered_avg_abs_dot; // mean-centered pairwise numerator
        float mean_rms;       // rms of mean hidden vector μ
    };
    std::vector<LayerRho> layer_rhos;
    layer_rhos.reserve(num_layers + 1);  // +1 for embedding

    // Embedding (input to layer 0)
    const float* final_device_ptr = nullptr;
    if (ai.embedding_tensor.data) {
        auto raw = compute_rho(ai.embedding_tensor.data, valid_positions);
        final_device_ptr = ai.embedding_tensor.data;
        layer_rhos.push_back({-1, -2, raw.rho, raw.avg_rms, 0.0f,
                              raw.avg_abs_dot, raw.avg_signed_dot, raw.avg_norm_prod,
                              raw.rms_min, raw.rms_max,
                              raw.centered_avg_abs_dot, raw.mean_rms});
    }

    // Each encoder layer output
    for (int l = 0; l < num_layers; ++l) {
        if (ai.encoder_layer_outputs[l].data) {
            auto raw = compute_rho(ai.encoder_layer_outputs[l].data, valid_positions);
            final_device_ptr = ai.encoder_layer_outputs[l].data;
            float delta = 0.0f;
            int vs_id = -2;
            if (!layer_rhos.empty()) {
                delta = raw.rho - layer_rhos.back().rho;
                vs_id = layer_rhos.back().layer_id;
            }
            layer_rhos.push_back({l, vs_id, raw.rho, raw.avg_rms, delta,
                                  raw.avg_abs_dot, raw.avg_signed_dot, raw.avg_norm_prod,
                                  raw.rms_min, raw.rms_max,
                                  raw.centered_avg_abs_dot, raw.mean_rms});
        }
    }

    // LM-head input (post-centering when centering is enabled) stays live only
    // inside the current autograd boundary. Using layer_id = num_layers to
    // distinguish it from raw encoder layers.
    const Tensor* lm_head_input_tensor = ai.liveLmHeadInputOrNull();
    if (lm_head_input_tensor && lm_head_input_tensor->data) {
        const auto& live_shape = lm_head_input_tensor->shape.require("RhoDiagnostic lm_head_input_tensor");
        if (!live_shape.is_2d_layout()) {
            throw std::runtime_error("[RhoDiagnostic] LM-head input tensor must be a 2D buffer");
        }
        if (payload.total_tokens > live_shape.as_2d().rows) {
            throw std::runtime_error(
                "[RhoDiagnostic] payload.total_tokens (" + std::to_string(payload.total_tokens) +
                ") exceeds LM-head input rows (" + std::to_string(live_shape.as_2d().rows) +
                ") at " + __FILE__ + ":" + std::to_string(__LINE__));
        }
        auto raw = compute_rho(lm_head_input_tensor->data, valid_positions);
        final_device_ptr = lm_head_input_tensor->data;
        float delta = 0.0f;
        int vs_id = -2;
        if (!layer_rhos.empty()) {
            delta = raw.rho - layer_rhos.back().rho;
            vs_id = layer_rhos.back().layer_id;
        }
        layer_rhos.push_back({num_layers, vs_id, raw.rho, raw.avg_rms, delta,
                              raw.avg_abs_dot, raw.avg_signed_dot, raw.avg_norm_prod,
                              raw.rms_min, raw.rms_max,
                              raw.centered_avg_abs_dot, raw.mean_rms});
    }

    // Build the equation log
    std::ostringstream rho_eq;
    rho_eq << std::fixed << std::setprecision(4);
    rho_eq << "[RHO_BUILDUP_EQUATION] ρ(l) = avg|cos(h_i^l, h_j^l)|, "
           << "Δρ = ρ(l) - ρ(prev_collected)\n";
        rho_eq << "  RAW DOT: avg_signed_dot(l)=P^-1 Σ_{i<j}(h_i^l · h_j^l), P=C(n_sample,2)\n";
        rho_eq << "  CENTERED DOT: μ_l=(1/N)Σ_i h_i^l, h̃_i^l=h_i^l-μ_l, "
            << "centered_avg_abs_dot(l)=P^-1 Σ_{i<j}|h̃_i^l · h̃_j^l|, "
            << "mean_rms(l)=sqrt((1/d_model)Σ_d μ_{l,d}²)\n";
    rho_eq << "  ARCH: h^l = h^{l-1} + LS*Attn(RMSNorm(h^{l-1})) "
           << "+ LS*FFN(RMSNorm(...))\n";

    // Compact per-layer table
        rho_eq << "  LAYER  ρ(l)    Δρ(vs)    h_rms_avg  avg_signed_dot  centered_avg_abs_dot  mean_rms  interpretation\n";
        rho_eq << "  ─────  ──────  ────────  ─────────  ──────────────  ────────────────────  ────────  ──────────────\n";

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

        rho_eq << std::setw(9) << lr.rms << "  "
               << std::setw(14) << lr.avg_signed_dot << "  "
               << std::setw(20) << lr.centered_avg_abs_dot << "  "
               << std::setw(8) << lr.mean_rms << "  ";

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

        ctx.telemetry.last_obs[(int)GRIM::Telemetry::MetricStream::RHO_RAW_AVG_SIGNED_DOT]
            = last_lr.avg_signed_dot;
        ctx.telemetry.last_obs[(int)GRIM::Telemetry::MetricStream::RHO_CENTERED_AVG_ABS_DOT]
            = last_lr.centered_avg_abs_dot;
        ctx.telemetry.last_obs[(int)GRIM::Telemetry::MetricStream::RHO_MEAN_VECTOR_RMS]
            = last_lr.mean_rms;

        // Atom-only vs non-atom-only ρ on the final collected layer.
        // ScratchBlock injects a shared per-type vector into atom positions only
        // (first_type_only mode), so if that injection drives ρ up, atom-only ρ
        // spikes while non-atom ρ stays flat. This split isolates the cause.
        float rho_atom = 0.0f;
        float rho_nonatom = 0.0f;
        if (final_device_ptr) {
            if (atom_positions.size() >= 2) {
                rho_atom = compute_rho(final_device_ptr, atom_positions).rho;
            }
            if (nonatom_positions.size() >= 2) {
                rho_nonatom = compute_rho(final_device_ptr, nonatom_positions).rho;
            }
        }
        ctx.telemetry.last_obs[(int)GRIM::Telemetry::MetricStream::RHO_ATOM_ONLY]
            = rho_atom;
        ctx.telemetry.last_obs[(int)GRIM::Telemetry::MetricStream::RHO_NONATOM_ONLY]
            = rho_nonatom;

        rho_eq << "  SUMMARY: ρ(" << first_label << ")=" << rho_first
               << " → ρ(" << last_label << ")=" << rho_final
               << " growth=" << std::showpos << rho_growth << std::noshowpos
               << " h_rms_growth=" << h_rms_growth_ratio
               << "x\n";
        rho_eq << "  RAW(" << last_label << "): |dot|=" << last_lr.avg_abs_dot
                             << " signed_dot=" << last_lr.avg_signed_dot
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
         rho_eq << "  CENTERED(" << last_label << "): centered_avg_abs_dot="
             << last_lr.centered_avg_abs_dot
             << " mean_rms=" << last_lr.mean_rms << "\n";
         rho_eq << "  SPLIT(" << last_label << "): ρ_atom=" << rho_atom
                << " (n=" << atom_positions.size() << ")"
                << " ρ_nonatom=" << rho_nonatom
                << " (n=" << nonatom_positions.size() << ")";
         if (atom_positions.size() >= 2 && nonatom_positions.size() >= 2
             && rho_atom > rho_nonatom + 0.05f) {
             rho_eq << "  [SCRATCHBLOCK] atom-only ρ exceeds non-atom ρ — per-type "
                       "injection is concentrating hidden-state alignment";
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
            std::string decoded = decodeAggregateTokenForDisplay(tokenizer, tid);
            sanitizeSingleLineTokenDisplay(decoded);
            rho_eq << " [" << tid << " \"" << decoded << "\" ×" << cnt << "]";
        }
        rho_eq << "\n";
    }

    ctx.logging.logger->log(rho_eq.str());
    EQ_LOG(ctx.logging.tape.get(), GRIM::Logging::LogGroup::Telemetry, GRIM::Logging::LogPhase::DIAGNOSTICS, -1, "RHO_BUILDUP_EQUATION", rho_eq.str().c_str());
}

} // namespace GRIM::Diagnostics

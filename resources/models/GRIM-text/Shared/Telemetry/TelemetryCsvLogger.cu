/**
 * @file TelemetryCsvLogger.cu
 * @brief Per-step CSV dump of all measured telemetry state (global + local)
 * 
 * Reads TelemetryState from GPU for every (stream, level) pair and writes
 * one CSV row per pair per step.  Only measured/computed values — no constants.
 */

#include "TelemetryCsvLogger.hpp"
#include <stdexcept>
#include <cstdio>
#include <iomanip>
#include <sstream>

namespace GRIM::Telemetry {

//=============================================================================
// CONSTRUCTOR — open file + write header
//=============================================================================

TelemetryCsvLogger::TelemetryCsvLogger(const std::string& csv_path,
                                       const TelemetryLattice& lattice)
    : num_levels_(lattice.config().num_levels),
      num_streams_(lattice.config().num_streams)
{
    if (csv_path.empty()) {
        throw std::runtime_error("[TelemetryCsvLogger] FATAL: csv_path is empty");
    }
    file_.open(csv_path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("[TelemetryCsvLogger] FATAL: Cannot open " + csv_path);
    }
    writeHeader();
    fprintf(stdout, "[TelemetryCsvLogger] Opened %s  (%d levels × %d streams)\n",
            csv_path.c_str(), num_levels_, num_streams_);
}

TelemetryCsvLogger::~TelemetryCsvLogger() {
    if (file_.is_open()) {
        file_.flush();
        file_.close();
    }
}

//=============================================================================
// MOVE
//=============================================================================

TelemetryCsvLogger::TelemetryCsvLogger(TelemetryCsvLogger&& o) noexcept
    : file_(std::move(o.file_)),
      num_levels_(o.num_levels_),
      num_streams_(o.num_streams_)
{
    o.num_levels_ = 0;
    o.num_streams_ = 0;
}

TelemetryCsvLogger& TelemetryCsvLogger::operator=(TelemetryCsvLogger&& o) noexcept {
    if (this != &o) {
        if (file_.is_open()) file_.close();
        file_ = std::move(o.file_);
        num_levels_ = o.num_levels_;
        num_streams_ = o.num_streams_;
        o.num_levels_ = 0;
        o.num_streams_ = 0;
    }
    return *this;
}

//=============================================================================
// HEADER
//=============================================================================

void TelemetryCsvLogger::writeHeader() {
    file_ << "global_step"
          << ",stream_idx"
          << ",stream_name"
          << ",level"
          << ",stride"
          << ",raw_observation"

          // --- TelemetryState: fast magnitude ---
          << ",mu"
          << ",m2"
          << ",sigma"
          << ",sigma_tilde"

          // --- TelemetryState: slow anchor ---
          << ",mu_a"
          << ",sigma_a"
          << ",delta_mu"
          << ",delta_sigma"

          // --- TelemetryState: meta-volatility ---
          << ",v_sigma"
          << ",sigma_prev"

          // --- TelemetryState: trend ---
          << ",delta_bar"
          << ",p"
          << ",mu_prev"

          // --- TelemetryState: outliers ---
          << ",r_out"
          << ",ell_out"
          << ",mu_ex"

          // --- TelemetryState: adaptive threshold ---
          << ",k_out"
          << ",c_out"

          // --- TelemetryState: metadata ---
          << ",step_count"
          << "\n";
}

//=============================================================================
// LOG — one call per training step
//=============================================================================

void TelemetryCsvLogger::log(const TelemetryLattice& lattice,
                             const float* raw_obs,
                             uint32_t global_step)
{
    if (!file_.is_open()) return;

    // Stream names (matches MetricStream enum order, indices 0-38)
    static const char* stream_names[] = {
        "loss", "grad_norm_mean", "grad_norm_max", "learning_rate", "tokens_per_batch",
        "rho_final", "rho_growth", "rho_worst_delta", "h_rms_growth",
        "adam_bc2_v_convergence", "adam_signal_dominance", "adam_cumulative_disp",
        "adam_disruption_emb", "adam_inv_bc2_amp",
        "exec_grad_norm", "exec_grad_ratio", "exec_selection_entropy",
        "exec_op_entropy", "exec_div_clamp_rate", "exec_max_p_write", "exec_active_ratio",
        "eb_inject_gate", "eb_read_gate_mean", "eb_inject_weight_norm",
        "eb_read_weight_norm", "eb_loss_frac", "sb_atom_embed_rms",
        "pbm_alibi_slope_rms", "pbm_alibi_eff_bias_max", "pbm_rope_inv_freq_rms", "pbm_batch_max_seq_len",
        "rho_raw_avg_abs_dot", "rho_raw_avg_norm_prod", "rho_raw_h_rms_min", "rho_raw_h_rms_max",
        "rms_gamma_pre_attn_rms", "rms_gamma_pre_ffn_rms", "rms_gamma_final_rms",
        "rho_raw_rms_spread"
    };
    static constexpr int num_named_streams = sizeof(stream_names) / sizeof(stream_names[0]);

    for (int level = 0; level < num_levels_; ++level) {
        const uint32_t stride = 1u << level;

        // Only emit rows for levels that actually updated at this step
        if (level > 0 && (global_step % stride) != 0) {
            continue;
        }

        for (int s = 0; s < num_streams_; ++s) {
            TelemetryState state{};
            TelemetryError err = lattice.readState(level, s, &state);
            if (err != TelemetryError::OK) continue;
            if (state.initialized == 0) continue;

            const char* name = (s < num_named_streams) ? stream_names[s] : "unknown";
            const float obs = (raw_obs && s < num_streams_) ? raw_obs[s] : 0.0f;

            // Use fixed-precision for stability; scientific for very
            // small/large values would lose relative comparisons in plots.
            file_ << global_step
                  << "," << s
                  << "," << name
                  << "," << level
                  << "," << stride
                  << "," << std::setprecision(8) << obs

                  // fast magnitude
                  << "," << std::setprecision(8) << state.mu
                  << "," << std::setprecision(8) << state.m2
                  << "," << std::setprecision(8) << state.sigma
                  << "," << std::setprecision(8) << state.sigma_tilde

                  // slow anchor
                  << "," << std::setprecision(8) << state.mu_a
                  << "," << std::setprecision(8) << state.sigma_a
                  << "," << std::setprecision(8) << state.delta_mu
                  << "," << std::setprecision(8) << state.delta_sigma

                  // meta-volatility
                  << "," << std::setprecision(10) << state.v_sigma
                  << "," << std::setprecision(8) << state.sigma_prev

                  // trend
                  << "," << std::setprecision(8) << state.delta_bar
                  << "," << std::setprecision(8) << state.p
                  << "," << std::setprecision(8) << state.mu_prev

                  // outliers
                  << "," << std::setprecision(8) << state.r_out
                  << "," << std::setprecision(8) << state.ell_out
                  << "," << std::setprecision(8) << state.mu_ex

                  // adaptive threshold
                  << "," << std::setprecision(8) << state.k_out
                  << "," << std::setprecision(8) << state.c_out

                  // metadata
                  << "," << state.step_count
                  << "\n";
        }
    }

    // Flush every 50 steps to balance I/O vs. data safety
    if ((global_step % 50) == 0) {
        file_.flush();
    }
}

void TelemetryCsvLogger::flush() {
    if (file_.is_open()) {
        file_.flush();
    }
}

} // namespace GRIM::Telemetry

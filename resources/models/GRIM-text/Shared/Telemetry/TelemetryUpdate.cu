//======================================================//
//  TelemetryUpdate.cu
//  Centralized per-batch telemetry observation update
//======================================================//
//
//  Extracts all telemetry metric computation from Phase2_TrainingLoop.cu
//  into a single updateTelemetryObservations() call, following the same
//  pattern as GuessCacheTraining.cu.
//
//======================================================//

#include "TelemetryUpdate.hpp"
#include "../../training/Phases/Phase1_Startup.hpp"  // TrainingContext
#include "../../training/Phases/Phase2_TrainingLoop.hpp"  // BatchResult
#include "../../training/Diagnostics/DiagnosticInference.hpp"
#include "../../training/Diagnostics/MtpDiagnostic.hpp"
#include "../../Layers/GRIMTS/GuessCacheTraining.hpp"
#include "../TrainingState/TrainingState_GPU.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace GRIM::Telemetry {

//======================================================//
//  Internal helpers
//======================================================//

namespace {

std::string formatScalar(float value, int precision = 4) {
    std::ostringstream oss;
    if (std::isfinite(value)) {
        if (value == 0.0f) value = 0.0f; // kill -0.0
        oss << std::fixed << std::setprecision(precision) << value;
    } else if (std::isnan(value)) {
        oss << "nan";
    } else {
        oss << (value > 0 ? "inf" : "-inf");
    }
    return oss.str();
}

std::string formatMetric(std::string_view name, float value, int precision = 4) {
    return std::string(name) + "=" + formatScalar(value, precision);
}

/// Compute RMS of a host-side float vector. Returns 0 if empty.
float vectorRMS(const std::vector<float>& v) {
    if (v.empty()) return 0.0f;
    double sum_sq = 0.0;
    for (float x : v) sum_sq += static_cast<double>(x) * x;
    return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(v.size())));
}

/// Compute RMS of a GPU buffer via D2H copy. Returns 0 if nullptr/empty.
float gpuBufferRMS(const float* d_ptr, size_t count) {
    if (!d_ptr || count == 0) return 0.0f;
    std::vector<float> h_buf(count);
    cudaMemcpy(h_buf.data(), d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
    double sum_sq = 0.0;
    for (size_t i = 0; i < count; ++i) sum_sq += static_cast<double>(h_buf[i]) * h_buf[i];
    return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count)));
}

//------------------------------------------------------
// Streams 0-4: Core training metrics
//------------------------------------------------------
void populateCoreStreams(float* obs, const TelemetryBatchInput& input) {
    obs[0] = input.loss;
    obs[1] = input.preclip_grad_rms;
    obs[2] = input.preclip_grad_rms;
    obs[3] = input.learning_rate;
    obs[4] = static_cast<float>(input.total_tokens);
}

//------------------------------------------------------
// Streams 9-13: Adam warmup causation tracking
//------------------------------------------------------
void populateAdamCausationStreams(float* obs, const TelemetryBatchInput& input,
                                  float& adam_cumulative_disp) {
    constexpr float BETA2 = 0.999f;
    const int iteration = input.optimizer_step + 1;
    const float beta2_pow_t = std::pow(BETA2, static_cast<float>(iteration));
    const float bc2 = 1.0f - beta2_pow_t;
    const float inv_bc2 = 1.0f / bc2;
    const float signal_dominance = (beta2_pow_t > 1e-12f) ? (bc2 / beta2_pow_t) : 1e6f;

    if (input.should_step) {
        adam_cumulative_disp += input.learning_rate;
    }

    const float vocab_f = static_cast<float>(input.actual_vocab_size);
    const float d_model_f = static_cast<float>(input.d_model);
    const float xavier_emb_scale = std::sqrt(6.0f / (vocab_f + d_model_f));
    const float disruption_emb = adam_cumulative_disp / xavier_emb_scale;

    obs[9]  = bc2;
    obs[10] = signal_dominance;
    obs[11] = adam_cumulative_disp;
    obs[12] = disruption_emb;
    obs[13] = inv_bc2;
}

//------------------------------------------------------
// Streams 14-20: Execution Block health tracking
//------------------------------------------------------
void populateExecBlockHealthStreams(
    float* obs,
    const GRIM::GradNorm::GradMetrics& gm,
    float enc_rms_pre,
    const GRIM::TrainingState& training_state,
    const GRIM::Batching::BatchPayload* payload) {

    float exec_grad_norm = 0.0f;
    float exec_grad_ratio = 0.0f;
    float exec_selection_entropy = 0.0f;
    float exec_op_entropy = 0.0f;
    float exec_div_clamp_rate = 0.0f;
    float exec_max_p_write = 0.0f;
    float exec_active_ratio = 0.0f;

    if (gm.execution_block_count > 0) {
        exec_grad_norm = static_cast<float>(std::sqrt(gm.execution_block_sum_sq / static_cast<double>(gm.execution_block_count)));
        if (enc_rms_pre > 1e-12f) {
            exec_grad_ratio = exec_grad_norm / enc_rms_pre;
        }
    }

    const auto& ai = training_state.autograd_intermediates;
    if (!ai.exec_outputs_per_row.empty() && payload) {
        int active_rows = 0;
        int total_steps = 0;
        float sum_selection_entropy = 0.0f;
        float sum_op_entropy = 0.0f;
        int total_div_clamps = 0;
        float sum_max_p_write = 0.0f;

        const int B = static_cast<int>(ai.exec_outputs_per_row.size());
        for (int b = 0; b < B; ++b) {
            const bool row_active = !payload->execution_active.empty()
                && b < static_cast<int>(payload->execution_active.size())
                && payload->execution_active[b];
            if (!row_active) continue;
            active_rows++;

            for (const auto& step : ai.exec_outputs_per_row[b].steps) {
                const auto& m = step.metrics;
                sum_selection_entropy += (m.arg1_entropy + m.arg2_entropy + m.op_entropy + m.write_entropy) / 4.0f;
                sum_op_entropy += m.op_entropy;
                total_div_clamps += m.div_clamp_count;
                sum_max_p_write += m.max_p_write;
                total_steps++;
            }
        }

        if (total_steps > 0) {
            exec_selection_entropy = sum_selection_entropy / static_cast<float>(total_steps);
            exec_op_entropy = sum_op_entropy / static_cast<float>(total_steps);
            exec_div_clamp_rate = static_cast<float>(total_div_clamps) / static_cast<float>(total_steps);
            exec_max_p_write = sum_max_p_write / static_cast<float>(total_steps);
        }
        if (B > 0) {
            exec_active_ratio = static_cast<float>(active_rows) / static_cast<float>(B);
        }
    }

    obs[14] = exec_grad_norm;
    obs[15] = exec_grad_ratio;
    obs[16] = exec_selection_entropy;
    obs[17] = exec_op_entropy;
    obs[18] = exec_div_clamp_rate;
    obs[19] = exec_max_p_write;
    obs[20] = exec_active_ratio;
}

//------------------------------------------------------
// Streams 21-26: EB injection diagnostics
//------------------------------------------------------
void populateEBInjectionStreams(
    float* obs,
    const GRIM::TrainingState& training_state,
    const GRIM::Batching::BatchPayload* payload,
    GRIMText::Training::TrainingContext& ctx) {

    const auto& ai = training_state.autograd_intermediates;

    // Stream 21: EB_INJECT_GATE
    float inject_gate_mean = 0.0f;
    if (!ai.exec_outputs_per_row.empty() && payload) {
        float sum_gate = 0.0f;
        int gate_count = 0;
        const int B = static_cast<int>(ai.exec_outputs_per_row.size());
        for (int b = 0; b < B; ++b) {
            const bool row_active = !payload->execution_active.empty()
                && b < static_cast<int>(payload->execution_active.size())
                && payload->execution_active[b];
            if (!row_active) continue;
            for (const auto& step : ai.exec_outputs_per_row[b].steps) {
                sum_gate += step.metrics.inject_gate_value;
                gate_count++;
            }
        }
        if (gate_count > 0) {
            inject_gate_mean = sum_gate / static_cast<float>(gate_count);
        }
    }
    obs[21] = inject_gate_mean;

    // Stream 22: EB_READ_GATE_MEAN (Category 2 telemetry snapshot on TrainingState)
    obs[22] = training_state.h_read_gate_mean;

    // Streams 23-24: Gate weight norms
    float inject_w_rms = 0.0f;
    float read_w_rms = 0.0f;
    auto* eb = ctx.model->getExecutionBlockLayer();
    if (eb) {
        const auto& w_inj = eb->w_inject_gate();
        inject_w_rms = gpuBufferRMS(w_inj.data, w_inj.numel());

        const auto& w_read = eb->W_gate_read();
        read_w_rms = gpuBufferRMS(w_read.data, w_read.numel());
    }
    obs[23] = inject_w_rms;
    obs[24] = read_w_rms;

    // Stream 25: EB_LOSS_FRAC
    float eb_loss_frac = 0.0f;
    // Use input values passed through TelemetryBatchInput (not accessible here directly)
    // This is populated by the caller after populateEBInjectionStreams
    obs[25] = eb_loss_frac;

    // Stream 26: SB_ATOM_EMBED_RMS
    float atom_embed_rms = 0.0f;
    auto* sb = ctx.model->getScratchBlockLayer();
    if (sb) {
        const auto& ate = sb->atomTypeEmbeddings();
        atom_embed_rms = gpuBufferRMS(ate.data, ate.numel());
    }
    obs[26] = atom_embed_rms;
}

//------------------------------------------------------
// Streams 35-37: RMSNorm learned gamma tracking
//------------------------------------------------------
void populateRmsGammaStreams(float* obs, GRIMText::Training::TrainingContext& ctx) {
    auto& encoder = ctx.model->getGpuEncoder();
    const int num_layers = encoder.getNumLayers();

    // Mean RMS(γ₁) and RMS(γ₂) across all encoder layers
    float sum_gamma1_rms = 0.0f;
    float sum_gamma2_rms = 0.0f;
    for (int i = 0; i < num_layers; ++i) {
        auto* layer = encoder.getLayer(i);
        const auto& g1 = layer->rms1Gamma();
        const auto& g2 = layer->rms2Gamma();
        sum_gamma1_rms += gpuBufferRMS(g1.data, g1.numel());
        sum_gamma2_rms += gpuBufferRMS(g2.data, g2.numel());
    }
    obs[35] = (num_layers > 0) ? (sum_gamma1_rms / static_cast<float>(num_layers)) : 0.0f;
    obs[36] = (num_layers > 0) ? (sum_gamma2_rms / static_cast<float>(num_layers)) : 0.0f;

    // Final RMSNorm gamma (LM head)
    auto* lm_head = ctx.model->getLmHeadLayer();
    const auto& g_final = lm_head->finalRmsGamma();
    obs[37] = gpuBufferRMS(g_final.data, g_final.numel());
}

//------------------------------------------------------
// Streams 27-30: PBM positional encoding diagnostics
//------------------------------------------------------
void populatePBMStreams(float* obs, GRIMText::Training::TrainingContext& ctx, int max_seq_len) {
    const auto* alibi_bias = ctx.model->getEmbedderPtr()->getALiBiBias();
    if (!alibi_bias || !alibi_bias->isInitialized()) {
        throw std::runtime_error("PBM telemetry: ALiBi bias not initialized — model MUST have positional encoding");
    }
    const auto& pbm = alibi_bias->getPBMState();

    // Stream 27: PBM_ALIBI_SLOPE_RMS
    obs[27] = vectorRMS(pbm.alibi_slopes_host);

    // Stream 28: PBM_ALIBI_EFF_BIAS_MAX
    {
        float max_abs_slope = 0.0f;
        for (float s : pbm.alibi_slopes_host) {
            float a = std::fabs(s);
            if (a > max_abs_slope) max_abs_slope = a;
        }
        obs[28] = max_abs_slope * static_cast<float>(max_seq_len);
    }

    // Stream 29: PBM_ROPE_INV_FREQ_RMS
    obs[29] = vectorRMS(pbm.rope_inv_freq_host);

    // Stream 30: PBM_BATCH_MAX_SEQ_LEN
    obs[30] = static_cast<float>(max_seq_len);
}

} // anonymous namespace

//======================================================//
//  Public API: updateTelemetryObservations
//======================================================//

void updateTelemetryObservations(
    GRIMText::Training::TrainingContext& ctx,
    const TelemetryBatchInput& input,
    const GRIM::GradNorm::GradMetrics& gm,
    const GRIM::Batching::BatchPayload* payload) {

    if (!ctx.telemetry.lattice || !ctx.telemetry.enabled) return;

    // Guard against NaN/Inf BEFORE passing to telemetry (Rule 20: fail loud)
    if (!std::isfinite(input.preclip_grad_rms)) {
        throw std::runtime_error(
            "FATAL: Gradient RMS is " + std::string(std::isnan(input.preclip_grad_rms) ? "NaN" : "Inf") +
            " at batch " + std::to_string(input.batch_idx + 1) + " step " + std::to_string(input.global_step) +
            " loss=" + std::to_string(input.loss) +
            " - indicates gradient explosion or numerical instability in backward pass");
    }

    ctx.logging.logger->log("[TelemetryLattice] PRE-UPDATE batch=" + std::to_string(input.batch_idx + 1) +
                            " step=" + std::to_string(input.global_step) +
                            " grad_rms=" + formatScalar(input.preclip_grad_rms, 6));

    float* obs = ctx.telemetry.last_obs;
    auto& training_state = ctx.model->getTrainingState();

    // Streams 0-4: Core metrics
    populateCoreStreams(obs, input);

    // Streams 9-13: Adam causation
    populateAdamCausationStreams(obs, input, ctx.telemetry.adam_cumulative_disp);

    // Streams 14-20: Execution Block health
    populateExecBlockHealthStreams(obs, gm, input.enc_rms_pre, training_state, payload);

    // Streams 21-26: EB injection diagnostics
    populateEBInjectionStreams(obs, training_state, payload, ctx);

    // stream 25: EB_LOSS_FRAC (needs loss values from input)
    if (input.total_loss_value > 1e-12f) {
        obs[25] = input.aux_loss / input.total_loss_value;
    }

    // Streams 27-30: PBM diagnostics
    populatePBMStreams(obs, ctx, input.max_seq_len);

    // Streams 35-37: RMSNorm learned gamma tracking
    populateRmsGammaStreams(obs, ctx);

    // Run lattice update
    GRIM::Telemetry::TelemetryError tel_err = ctx.telemetry.lattice->update(obs, input.global_step);

    ctx.logging.logger->log("[TelemetryLattice] POST-UPDATE batch=" + std::to_string(input.batch_idx + 1) +
                            " step=" + std::to_string(input.global_step) +
                            " error_code=" + std::to_string(static_cast<int>(tel_err)));

    if (tel_err != GRIM::Telemetry::TelemetryError::OK) {
        throw std::runtime_error(
            std::string("FATAL Telemetry: ") +
            GRIM::Telemetry::getTelemetryErrorMessage(tel_err));
    }

    // CSV export
    if (ctx.telemetry.csv_logger) {
        ctx.telemetry.csv_logger->log(*ctx.telemetry.lattice, obs, input.global_step);
    }

    // First-batch CUDA sanity check
    if (input.batch_idx == 0) {
        cudaError_t e = cudaDeviceSynchronize();
        cudaError_t last = (e != cudaSuccess) ? e : cudaGetLastError();
        if (last != cudaSuccess) {
            ctx.logging.logger->log("[CUDA] first_batch AFTER telemetry update: " + std::string(cudaGetErrorString(last)));
            cudaGetLastError();
        } else {
            ctx.logging.logger->log("[CUDA] first_batch AFTER telemetry update: ok");
        }
    }
}

void logIntervalTelemetry(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::TrainingLoopState& state,
    const GRIMText::Training::BatchResult& batch_result) {

    const int interval = ctx.config.hyperparameters.log_interval;
    if (interval > 0 && ctx.global_step % interval == 0) {
        ctx.logging.logger->log("[Step " + std::to_string(ctx.global_step) + "] " +
                                formatMetric("loss", batch_result.loss) + " " +
                                formatMetric("lr", batch_result.learning_rate, 8));

        GRIM::Diagnostics::runMtpDiagnostic(ctx, batch_result);

        if (ctx.config.hyperparameters.guess_aux_enabled) {
            GRIMTS::Training::logGuessCacheTelemetry(
                ctx.guess_cache_state, ctx.global_step);
        }
    }

    GRIMText::Training::logDiagnosticSample(ctx, state);
}

//======================================================//
//  Public API: logTelemetrySummary
//======================================================//

void logTelemetrySummary(GRIMText::Training::TrainingContext& ctx) {
    if (!ctx.telemetry.lattice || !ctx.telemetry.enabled) return;

    ctx.logging.logger->log("========== TELEMETRY SUMMARY ==========");
    ctx.logging.logger->log("[Tel-Legend] Levels: L0=stride 1 fast telemetry; L2=stride 4 smoother telemetry; '(s=4)' means the L2 state updates every 4 observations.");
    ctx.logging.logger->log("[Tel-Legend] Common fields: μ=EMA baseline for the logged stream; σ̃=normalized volatility σ/(|μ|+ε); v_σ=volatility-of-volatility; Δ̄=EMA normalized slope/trend.");
    ctx.logging.logger->log("[Tel-Legend] Drift/outlier fields: p=directional bias (-1 falling, +1 rising); r_out=soft outlier frequency; δμ=mean drift vs slow anchor μ_a; δσ=volatility drift vs slow anchor σ_a.");

    // Level 0 (fast, every step)
    GRIM::Telemetry::TelemetryVector vec0_loss, vec0_grad;
    ctx.telemetry.lattice->readVector(0, (int)GRIM::Telemetry::MetricStream::LOSS, &vec0_loss);
    ctx.telemetry.lattice->readVector(0, (int)GRIM::Telemetry::MetricStream::GRAD_NORM_MEAN, &vec0_grad);

    std::ostringstream oss;
    oss << "[Tel-L0] LOSS: μ=" << vec0_loss.mu << " σ̃=" << vec0_loss.sigma_tilde
        << " v_σ=" << vec0_loss.v_sigma << " Δ̄=" << vec0_loss.delta_bar
        << " p=" << vec0_loss.p << " r_out=" << vec0_loss.r_out;
    ctx.logging.logger->log(oss.str());

    oss.str("");
    oss << "[Tel-L0] GRAD: μ=" << vec0_grad.mu << " σ̃=" << vec0_grad.sigma_tilde
        << " v_σ=" << vec0_grad.v_sigma << " δμ=" << vec0_grad.delta_mu
        << " δσ=" << vec0_grad.delta_sigma;
    ctx.logging.logger->log(oss.str());

    // Level 2 (medium scale, stride=4)
    GRIM::Telemetry::TelemetryVector vec2_loss;
    ctx.telemetry.lattice->readVector(2, (int)GRIM::Telemetry::MetricStream::LOSS, &vec2_loss);

    oss.str("");
    oss << "[Tel-L2] LOSS (s=4): μ=" << vec2_loss.mu << " σ̃=" << vec2_loss.sigma_tilde
        << " Δ̄=" << vec2_loss.delta_bar << " δμ=" << vec2_loss.delta_mu;
    ctx.logging.logger->log(oss.str());

    ctx.logging.logger->log("========================================");
}

} // namespace GRIM::Telemetry

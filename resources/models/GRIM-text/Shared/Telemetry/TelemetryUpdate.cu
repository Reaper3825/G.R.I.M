//======================================================//
//  TelemetryUpdate.cu
//  Centralized per-batch telemetry observation update
//======================================================//
//
//  Extracts all telemetry metric computation from Phase2_TrainingLoop.cu
//  into a single updateTelemetryObservations() call, following the same
//  ownership pattern used by the rest of the training runtime.
//
//======================================================//

#include "TelemetryUpdate.hpp"
#include "../../training/Phases/Phase1_Startup.hpp"  // TrainingContext
#include "../../training/Phases/Phase2_TrainingLoop.hpp"  // BatchResult
#include "../../training/Diagnostics/DiagnosticInference.hpp"
#include "../../training/Diagnostics/MtpDiagnostic.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
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

/// Compute RMS of a host-side float range. Returns 0 if empty.
template <typename Range>
float rangeRMS(const Range& values) {
    double sum_sq = 0.0;
    std::size_t count = 0;
    for (float x : values) {
        sum_sq += static_cast<double>(x) * x;
        ++count;
    }
    if (count == 0) return 0.0f;
    return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count)));
}

/// Compute RMS of a GPU buffer via D2H copy. Returns 0 if nullptr/empty.
float gpuBufferRMS(const float* d_ptr, size_t count, const char* label) {
    if (!d_ptr || count == 0) return 0.0f;
    std::vector<float> h_buf(count);
    const cudaError_t copy_err = cudaMemcpy(h_buf.data(), d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
    if (copy_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Telemetry::gpuBufferRMS failed for ") + label +
            " ptr=" + std::to_string(reinterpret_cast<std::uintptr_t>(d_ptr)) +
            " count=" + std::to_string(count) +
            " error=" + cudaGetErrorString(copy_err));
    }
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
    const TelemetryBatchInput& input) {

    float exec_grad_norm = 0.0f;
    float exec_grad_ratio = 0.0f;

    if (gm.execution_block_count > 0) {
        exec_grad_norm = static_cast<float>(std::sqrt(gm.execution_block_sum_sq / static_cast<double>(gm.execution_block_count)));
        if (input.enc_rms_pre > 1e-12f) {
            exec_grad_ratio = exec_grad_norm / input.enc_rms_pre;
        }
    }

    obs[14] = exec_grad_norm;
    obs[15] = exec_grad_ratio;
    obs[16] = input.exec_selection_entropy;
    obs[17] = input.exec_op_entropy;
    obs[18] = input.exec_div_clamp_rate;
    obs[19] = input.exec_max_p_write;
    obs[20] = input.exec_active_ratio;
}

//------------------------------------------------------
// Streams 21-26: EB injection diagnostics
//------------------------------------------------------
void populateEBInjectionStreams(
    float* obs,
    const GRIM::TrainingState& training_state,
    const TelemetryBatchInput& input,
    const GRIMText::Training::Startup::GpuModelState& gpu_model,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry) {

    // Stream 21: EB_INJECT_GATE
    obs[21] = input.inject_gate_mean;

    // Stream 22: EB_READ_GATE_MEAN (Category 2 telemetry snapshot on TrainingState)
    obs[22] = training_state.h_read_gate_mean;

    // Streams 23-24: Gate weight norms
    float inject_w_rms = 0.0f;
    float read_w_rms = 0.0f;
    auto* execution_block_parameters = parameter_registry.getExecutionBlockParameters();
    if (execution_block_parameters) {
        inject_w_rms = gpuBufferRMS(execution_block_parameters->w_inject_gate.data,
                                    execution_block_parameters->w_inject_gate.numel(),
                                    "execution_block_parameters.w_inject_gate");
        read_w_rms = gpuBufferRMS(execution_block_parameters->W_gate_read.data,
                                  execution_block_parameters->W_gate_read.numel(),
                                  "execution_block_parameters.W_gate_read");
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
    auto* sb = gpu_model.scratch_block_layer.get();
    if (sb) {
        const auto& ate = sb->atomTypeEmbeddings();
        atom_embed_rms = gpuBufferRMS(ate.data, ate.numel(), "scratch_block.atomTypeEmbeddings");
    }
    obs[26] = atom_embed_rms;
}

//------------------------------------------------------
// Streams 35-37: RMSNorm learned gamma tracking
//------------------------------------------------------
void populateRmsGammaStreams(
    float* obs,
    const GRIM::Config::AiConfigSnapshot& config,
    const GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry) {
    auto* gpu_encoder = gpu_model_state.gpu_encoder.get();
    if (!gpu_encoder) {
        throw std::runtime_error(
            "Telemetry::populateRmsGammaStreams: gpu_model_state.gpu_encoder is NULL");
    }
    const int num_layers = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "num_layers");

    // Mean RMS(γ₁) and RMS(γ₂) across all encoder layers
    float sum_gamma1_rms = 0.0f;
    float sum_gamma2_rms = 0.0f;
    for (int i = 0; i < num_layers; ++i) {
        auto* layer = gpu_encoder->getLayer(i);
        if (!layer) {
            throw std::runtime_error(
                "Telemetry::populateRmsGammaStreams: encoder layer " + std::to_string(i) + " is NULL");
        }
        const auto& g1 = layer->rms1Gamma();
        const auto& g2 = layer->rms2Gamma();
        sum_gamma1_rms += gpuBufferRMS(g1.data, g1.numel(), "encoder_layer.rms1Gamma");
        sum_gamma2_rms += gpuBufferRMS(g2.data, g2.numel(), "encoder_layer.rms2Gamma");
    }
    obs[35] = (num_layers > 0) ? (sum_gamma1_rms / static_cast<float>(num_layers)) : 0.0f;
    obs[36] = (num_layers > 0) ? (sum_gamma2_rms / static_cast<float>(num_layers)) : 0.0f;

    // Final RMSNorm gamma (LM head)
    auto* lm_head_parameters = parameter_registry.getLmHeadParameters();
    if (!lm_head_parameters) {
        throw std::runtime_error(
            "Telemetry::populateRmsGammaStreams: parameter_registry.lm_head_parameters is NULL");
    }
    const auto& g_final = lm_head_parameters->final_rms_gamma;
    obs[37] = gpuBufferRMS(g_final.data, g_final.numel(), "lm_head.final_rms_gamma");
}

//------------------------------------------------------
// Streams 27-30: PBM positional encoding diagnostics
//------------------------------------------------------
void populatePBMStreams(float* obs, GRIMText::Training::TrainingContext& ctx, int max_seq_len) {
    const auto pbm = GRIM::HyperParameters::pbmConstructionHP(ctx.config);

    // Stream 27: PBM_ALIBI_SLOPE_RMS
    obs[27] = rangeRMS(pbm.alibi_slopes);

    // Stream 28: PBM_ALIBI_EFF_BIAS_MAX
    {
        float max_abs_slope = 0.0f;
        for (float s : pbm.alibi_slopes) {
            float a = std::fabs(s);
            if (a > max_abs_slope) max_abs_slope = a;
        }
        obs[28] = max_abs_slope * static_cast<float>(max_seq_len);
    }

    // Stream 29: PBM_ROPE_INV_FREQ_RMS
    obs[29] = rangeRMS(pbm.rope_inv_freq);

    // Stream 30: PBM_BATCH_MAX_SEQ_LEN
    obs[30] = static_cast<float>(max_seq_len);
}

} // anonymous namespace

//======================================================//
//  Public API: updateTelemetryObservations
//======================================================//

void updateTelemetryObservations(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::TrainingState& training_state,
    const GRIMText::Training::Startup::GpuModelState& gpu_model,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const TelemetryBatchInput& input,
    const GRIM::GradNorm::GradMetrics& gm) {

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

    // Streams 0-4: Core metrics
    populateCoreStreams(obs, input);

    // Streams 9-13: Adam causation
    populateAdamCausationStreams(obs, input, ctx.telemetry.adam_cumulative_disp);

    // Streams 14-20: Execution Block health
    populateExecBlockHealthStreams(obs, gm, input);

    // Streams 21-26: EB injection diagnostics
    populateEBInjectionStreams(obs, training_state, input, gpu_model, parameter_registry);

    // stream 25: EB_LOSS_FRAC (needs loss values from input)
    if (input.total_loss_value > 1e-12f) {
        obs[25] = input.aux_loss / input.total_loss_value;
    }

    // Streams 27-30: PBM diagnostics
    populatePBMStreams(obs, ctx, input.max_seq_len);

    // Streams 35-37: RMSNorm learned gamma tracking
    populateRmsGammaStreams(obs, ctx.config, gpu_model, parameter_registry);

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

    const auto runtime_hp =
        GRIM::HyperParameters::trainingRuntimeControlHP(ctx.config);
    const int interval = runtime_hp.log_interval;
    if (interval > 0 && ctx.global_step % interval == 0) {
        ctx.logging.logger->log("[Step " + std::to_string(ctx.global_step) + "] " +
                                formatMetric("loss", batch_result.loss) + " " +
                                formatMetric("lr", batch_result.learning_rate, 8));

        GRIM::Diagnostics::runMtpDiagnostic(ctx, batch_result);
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

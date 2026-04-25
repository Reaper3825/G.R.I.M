#pragma once
/**
 * @file TelemetryControlConfig_FromHyperParams.hpp
 * @brief Build a TelemetryControlConfig from training hyperparameters.
 *
 * Decouples TelemetryControl construction from Phase1_Startup: the control
 * config is a pure function of (hyperparameters, token budget, max_seq_len).
 *
 * Rule 20: no hardcoded fallbacks here — every field is sourced directly
 * from the hyperparameters struct / call-site values. If a value is wrong,
 * fix the config / defaults in `control/ai_config_paths.hpp` (setDefaults),
 * not here.
 */

#include "TelemetryControl_GPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"

#include <cstdint>

namespace GRIM::Telemetry {

/**
 * @brief Assemble a TelemetryControlConfig from training hyperparameters.
 *
 * @param hp                 Training hyperparameters (telemetry_* fields).
 * @param reference_tokens   Model's actual token budget (max_tokens_per_batch).
 *                           Caller MUST validate non-zero before calling.
 * @param reference_seq_len  Configured max sequence length for this run.
 */
inline TelemetryControlConfig makeControlConfigFromHyperparameters(
    const ::GRIM::Config::TrainingHyperparameters& hp,
    uint32_t reference_tokens,
    uint32_t reference_seq_len)
{
    TelemetryControlConfig cfg;

    // Reference values for normalization
    cfg.reference_tokens  = static_cast<float>(reference_tokens);
    cfg.reference_seq_len = static_cast<float>(reference_seq_len);

    // Spike thresholds (diagnostic only — no interventions)
    cfg.spike_mild_threshold     = hp.telemetry_spike_mild_threshold;
    cfg.spike_moderate_threshold = hp.telemetry_spike_moderate_threshold;
    cfg.spike_severe_threshold   = hp.telemetry_spike_severe_threshold;

    // Accumulation bug detection (Rule 20: crash on zero gradients with non-zero loss)
    cfg.min_grad_for_nonzero_loss        = hp.telemetry_min_grad_for_nonzero_loss;
    cfg.loss_threshold_for_grad_check    = hp.telemetry_loss_threshold_for_grad_check;
    cfg.max_consecutive_zero_grad_steps  = hp.telemetry_max_consecutive_zero_grad_steps;

    // Monitoring config
    cfg.warmup_steps                    = hp.telemetry_warmup_steps;
    cfg.baseline_stabilization_steps    = hp.telemetry_baseline_stabilization_steps;
    cfg.verbose_logging                 = hp.telemetry_verbose_logging;
    cfg.fail_loud_on_accumulation_bug   = hp.telemetry_fail_loud_on_accumulation_bug;

    return cfg;
}

} // namespace GRIM::Telemetry

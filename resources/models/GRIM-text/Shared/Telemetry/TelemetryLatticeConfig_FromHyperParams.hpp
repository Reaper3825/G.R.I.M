#pragma once
/**
 * @file TelemetryLatticeConfig_FromHyperParams.hpp
 * @brief Build a TelemetryLattice LatticeConfig from training hyperparameters.
 *
 * Decouples TelemetryLattice construction from Phase1_Startup so the lattice
 * config is a pure function of the hyperparameters block + a CUDA stream.
 *
 * Rule 20: no hardcoded fallbacks here — every field is sourced directly
 * from the hyperparameters struct. If a value is wrong, fix the config /
 * defaults in `control/ai_config_paths.hpp` (setDefaults), not here.
 */

#include "TelemetryLattice_GPU.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM::Telemetry {

/**
 * @brief Assemble a LatticeConfig from training hyperparameters.
 *
 * @param hp     Training hyperparameters (provides telemetry_lattice_* fields).
 * @param stream CUDA stream the lattice kernels will run on.
 *
 * Stream layout for `num_streams` (must match the streams emitted by the
 * training loop):
 *   0-4   core
 *   5-8   rho
 *   9-13  adam
 *  14-20  exec block
 *  21-26  EB/SB injection
 *  27-30  PBM
 *  31-34  rho raw
 *  35-37  RMS gamma
 *  38     rho rms-spread
 *  39-44  h<->W alignment
 *  45-46  unigram-dir cosine
 *  47     lm_head_w_rms_rms
 *  48-54  init-time invariants
 *  55-57  rho signed/centered/mean-vector diagnostics
 */
inline LatticeConfig makeLatticeConfigFromHyperparameters(
    const ::GRIM::Config::TrainingHyperparameters& hp,
    cudaStream_t stream)
{
    LatticeConfig cfg;
    cfg.num_levels  = hp.telemetry_lattice_num_levels;
    cfg.num_streams = hp.telemetry_lattice_num_streams;
    cfg.hyperparams.beta_mu     = hp.telemetry_lattice_beta_mu;
    cfg.hyperparams.beta_a      = hp.telemetry_lattice_beta_a;
    cfg.hyperparams.beta_delta  = hp.telemetry_lattice_beta_delta;
    cfg.hyperparams.beta_r      = hp.telemetry_lattice_beta_r;
    cfg.hyperparams.beta_run    = hp.telemetry_lattice_beta_run;
    cfg.hyperparams.beta_v      = hp.telemetry_lattice_beta_v;
    cfg.hyperparams.k_out0      = hp.telemetry_lattice_k_out0;
    cfg.hyperparams.alpha_v     = hp.telemetry_lattice_alpha_v;
    cfg.hyperparams.epsilon     = hp.telemetry_lattice_epsilon;
    cfg.hyperparams.strict_mode = hp.telemetry_lattice_strict_mode;
    cfg.stream = stream;
    return cfg;
}

} // namespace GRIM::Telemetry

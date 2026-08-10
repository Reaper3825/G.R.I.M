#include "TelemetryInitInputs.hpp"

#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/LogRecorder/LogRecorder.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

void TelemetryReady(TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;
    const auto paths_hp =
        GRIM::HyperParameters::pathsHP(ctx.config);
    const auto lattice_hp =
        GRIM::HyperParameters::telemetryLatticeHP(ctx.config);
    const auto control_hp =
        GRIM::HyperParameters::telemetryControlHP(ctx.config);

    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry lattice...", 0);
    GRIM::Telemetry::LatticeConfig lattice_config;
    lattice_config.num_levels = lattice_hp.num_levels;
    lattice_config.num_streams = lattice_hp.num_streams;
    lattice_config.hyperparams.beta_mu = lattice_hp.beta_mu;
    lattice_config.hyperparams.beta_a = lattice_hp.beta_a;
    lattice_config.hyperparams.beta_delta = lattice_hp.beta_delta;
    lattice_config.hyperparams.beta_r = lattice_hp.beta_r;
    lattice_config.hyperparams.beta_run = lattice_hp.beta_run;
    lattice_config.hyperparams.beta_v = lattice_hp.beta_v;
    lattice_config.hyperparams.k_out0 = lattice_hp.k_out0;
    lattice_config.hyperparams.alpha_v = lattice_hp.alpha_v;
    lattice_config.hyperparams.epsilon = lattice_hp.epsilon;
    lattice_config.hyperparams.strict_mode = lattice_hp.strict_mode;
    lattice_config.stream = ctx.requireTrainingState("TelemetryReady").stream_ctrl.getPrimaryStream();
    ctx.telemetry.lattice = std::make_unique<GRIM::Telemetry::TelemetryLattice>(
        lattice_config);
    ctx.logging.logger->log("✓ Telemetry lattice initialized (GPU-resident); exact config values are listed by ConfigDump.");

    const std::string csv_path = paths_hp.log_dir + "/telemetry_" + ctx.logging.session_id + ".csv";
    ctx.telemetry.csv_logger = std::make_unique<GRIM::Telemetry::TelemetryCsvLogger>(
        csv_path, *ctx.telemetry.lattice);
    ctx.logging.logger->log("✓ Telemetry CSV logger: " + csv_path);

    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry control...", 0);
    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);
    const uint32_t token_budget = static_cast<uint32_t>(fixed_shape.max_tokens_per_batch);
    const uint32_t seq_cap = static_cast<uint32_t>(fixed_shape.max_seq_len);
    if (token_budget == 0 || seq_cap == 0) {
        throw std::runtime_error("FATAL: trainingFixedShapeHP is invalid (token_budget=" +
                                 std::to_string(token_budget) + " seq_cap=" + std::to_string(seq_cap) + ")");
    }
    ctx.telemetry.control_config.reference_tokens = static_cast<float>(token_budget);
    ctx.telemetry.control_config.reference_seq_len = static_cast<float>(seq_cap);
    ctx.telemetry.control_config.spike_mild_threshold = control_hp.spike_mild_threshold;
    ctx.telemetry.control_config.spike_moderate_threshold = control_hp.spike_moderate_threshold;
    ctx.telemetry.control_config.spike_severe_threshold = control_hp.spike_severe_threshold;
    ctx.telemetry.control_config.min_grad_for_nonzero_loss = control_hp.min_grad_for_nonzero_loss;
    ctx.telemetry.control_config.loss_threshold_for_grad_check = control_hp.loss_threshold_for_grad_check;
    ctx.telemetry.control_config.max_consecutive_zero_grad_steps = control_hp.max_consecutive_zero_grad_steps;
    ctx.telemetry.control_config.warmup_steps = control_hp.warmup_steps;
    ctx.telemetry.control_config.baseline_stabilization_steps = control_hp.baseline_stabilization_steps;
    ctx.telemetry.control_config.verbose_logging = control_hp.verbose_logging;
    ctx.telemetry.control_config.fail_loud_on_accumulation_bug = control_hp.fail_loud_on_accumulation_bug;
}

} // namespace GRIMText::Training


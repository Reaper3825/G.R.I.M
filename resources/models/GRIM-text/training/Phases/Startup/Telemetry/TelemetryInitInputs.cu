#include "TelemetryInitInputs.hpp"

#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../../../Shared/Telemetry/TelemetryControlConfig_FromHyperParams.hpp"
#include "../../../../Shared/Telemetry/TelemetryLatticeConfig_FromHyperParams.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

TelemetryInitInputs makeTelemetryInitInputs(const TrainingContext& ctx) {
    TelemetryInitInputs inputs;
    inputs.capacity = ctx.run_capacity;
    return inputs;
}

void TelemetryReady(TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry lattice...", 0);
    ctx.telemetry.lattice = std::make_unique<GRIM::Telemetry::TelemetryLattice>(
        GRIM::Telemetry::makeLatticeConfigFromHyperparameters(
            ctx.config.hyperparameters,
            ctx.model->getTrainingState().stream_ctrl.getPrimaryStream()));
    ctx.logging.logger->log("✓ Telemetry lattice initialized (GPU-resident); exact config values are listed by ConfigDump.");

    const std::string csv_path = ctx.config.paths.log_dir + "/telemetry_" + ctx.logging.session_id + ".csv";
    ctx.telemetry.csv_logger = std::make_unique<GRIM::Telemetry::TelemetryCsvLogger>(
        csv_path, *ctx.telemetry.lattice);
    ctx.logging.logger->log("✓ Telemetry CSV logger: " + csv_path);

    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing telemetry control...", 0);
    const uint32_t token_budget = ctx.run_capacity.max_tokens_per_batch;
    const uint32_t seq_cap = ctx.run_capacity.seq_cap;
    if (token_budget == 0 || seq_cap == 0) {
        throw std::runtime_error("FATAL: RunCapacity is invalid (token_budget=" +
                                 std::to_string(token_budget) + " seq_cap=" + std::to_string(seq_cap) + ")");
    }
    if (static_cast<uint32_t>(ctx.model->getConfig().max_tokens_per_batch) != token_budget) {
        throw std::runtime_error("FATAL: model max_tokens_per_batch does not match RunCapacity (model=" +
                                 std::to_string(ctx.model->getConfig().max_tokens_per_batch) +
                                 " stem=" + std::to_string(token_budget) + ")");
    }
    ctx.telemetry.control_config = GRIM::Telemetry::makeControlConfigFromHyperparameters(
        ctx.config.hyperparameters,
        token_budget,
        seq_cap);
}

} // namespace GRIMText::Training


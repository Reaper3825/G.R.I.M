//======================================================//
//  DiagnosticGates.cu
//  Definitions for the gates that need TrainingContext's
//  full layout. Lifted verbatim from the anonymous
//  namespace at the top of Phase2_TrainingLoop.cu.
//======================================================//

#include "DiagnosticGates.hpp"
#include "../Phases/Phase2_TrainingLoop.hpp"

namespace GRIM::Diagnostics {

bool shouldSyncDiagnostics(const GRIMText::Training::TrainingContext& ctx, std::size_t batch_idx) {
    // Skip expensive D2H syncs when equation logging disabled (avoids GPU pipeline drain)
    if (!ctx.logging.tape || !ctx.logging.tape->accepts(GRIM::Logging::LogLevel::Debug)) {
        return false;
    }
    const auto runtime_hp =
        GRIM::HyperParameters::trainingRuntimeControlHP(ctx.config);
    const int default_interval = runtime_hp.log_interval;
    const int interval = readEnvInt("GRIM_SYNC_INTERVAL", default_interval);
    if (interval <= 0) {
        return false;
    }
    return ((batch_idx + 1) % static_cast<std::size_t>(interval)) == 0;
}

bool shouldLogLogitTrace(const GRIMText::Training::TrainingContext& ctx, std::size_t batch_idx) {
    const auto runtime_hp =
        GRIM::HyperParameters::trainingRuntimeControlHP(ctx.config);
    if (!runtime_hp.logit_update_trace_enabled) {
        return false;
    }
    const int interval = readEnvInt("GRIM_LOGIT_TRACE_INTERVAL",
                                    runtime_hp.logit_update_trace_interval);
    if (interval <= 0) {
        return false;
    }
    return ((batch_idx + 1) % static_cast<std::size_t>(interval)) == 0;
}

bool shouldLogAtomStats(const GRIMText::Training::TrainingContext& ctx, int batch_idx) {
    const auto runtime_hp =
        GRIM::HyperParameters::trainingRuntimeControlHP(ctx.config);
    const int interval = runtime_hp.atom_stats_interval;
    if (interval <= 0) {
        return false;
    }
    return ((batch_idx + 1) % interval) == 0;
}

} // namespace GRIM::Diagnostics

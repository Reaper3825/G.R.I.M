#pragma once
//======================================================//
//  Startup/Logging.hpp
//  Logging subsystem extracted from Phase1_Startup.
//
//  Owns:
//    - LoggingContext lifetime and bootstrap log file
//    - TrainingLogger / MetricsCollector / StatusFileWriter
//    - BatchLogTape + sinks (text / equation CSV / stderr)
//    - Default logging-profile registration
//======================================================//

class TrainingLogger;

// AiConfigSnapshot lives behind Shared/HyperParameters/HyperParameters_GPU.hpp;
// forward-declare it here to avoid a circular include with Phase1_Startup.hpp.
namespace GRIM { namespace Config {
    struct AiConfigSnapshot;
} }

namespace GRIMText::Training {

struct LoggingContext;
struct TrainingContext;

void LoggingReady(TrainingContext& ctx);

namespace Internal {

/**
 * @brief Construct LoggingContext: session id, raw log path bootstrap,
 *        TrainingLogger, MetricsCollector, StatusFileWriter.
 *        Also registers default logging profiles on first call.
 */
LoggingContext initializeLogging(const ::GRIM::Config::AiConfigSnapshot& config);

/**
 * @brief Build the BatchLogTape and attach sinks (text / equation CSV /
 *        stderr) according to HyperparameterGroupings.hpp::tapeLogHP().
 *        Installs the global tape pointer for layer-level code.
 *
 * Must be called after initializeLogging (uses session_id and logger).
 */
void setupBatchLogTape(LoggingContext& logging,
                       const ::GRIM::Config::AiConfigSnapshot& config);

} // namespace Internal
} // namespace GRIMText::Training

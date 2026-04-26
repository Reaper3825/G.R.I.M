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

// PathConfig + StartupConfig live in Shared/HyperParameters/HyperParameters_GPU.hpp;
// forward-declare them here in their owning namespace to avoid a circular
// include with Phase1_Startup.hpp.
namespace GRIM { namespace HyperParameters {
    struct PathConfig;
    struct StartupConfig;
} }

namespace GRIMText::Training {

struct LoggingContext;
struct TrainingContext;

void LoggingReady(TrainingContext& ctx, int argc, char** argv);

namespace Internal {

/**
 * @brief Construct LoggingContext: session id, raw log path bootstrap,
 *        TrainingLogger, MetricsCollector, StatusFileWriter.
 *        Also registers default logging profiles on first call.
 */
LoggingContext initializeLogging(const ::GRIM::HyperParameters::PathConfig& paths);

/**
 * @brief Build the BatchLogTape and attach sinks (text / equation CSV /
 *        stderr) according to config.hyperparameters.tape_logging.
 *        Installs the global tape pointer for layer-level code.
 *
 * Must be called after initializeLogging (uses session_id and logger).
 */
void setupBatchLogTape(LoggingContext& logging,
                       const ::GRIM::HyperParameters::StartupConfig& config);

} // namespace Internal
} // namespace GRIMText::Training

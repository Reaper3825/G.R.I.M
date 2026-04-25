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

namespace GRIMText::Training {

struct PathConfig;
struct LoggingContext;
struct StartupConfig;

namespace Internal {

/**
 * @brief Construct LoggingContext: session id, raw log path bootstrap,
 *        TrainingLogger, MetricsCollector, StatusFileWriter.
 *        Also registers default logging profiles on first call.
 */
LoggingContext initializeLogging(const PathConfig& paths);

/**
 * @brief Build the BatchLogTape and attach sinks (text / equation CSV /
 *        stderr) according to config.hyperparameters.tape_logging.
 *        Installs the global tape pointer for layer-level code.
 *
 * Must be called after initializeLogging (uses session_id and logger).
 */
void setupBatchLogTape(LoggingContext& logging, const StartupConfig& config);

} // namespace Internal
} // namespace GRIMText::Training

//======================================================//
//  Startup/Logging.cu
//  Logging subsystem implementation.
//  Extracted from Phase1_Startup.cu.
//======================================================//

#include "Logging.hpp"
#include "../Phase1_Startup.hpp"

#include "../../../Shared/LogRecorder/LogRecorder.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>

namespace fs = std::filesystem;

namespace GRIMText::Training {
namespace Internal {

namespace {

void registerDefaultLoggingProfiles() {
    // Module log profile registration removed: API never implemented.
    // EmitModule*() functions remain available via LogRecorder.hpp.
}

} // anonymous namespace

LoggingContext initializeLogging(const PathConfig& paths) {
    registerDefaultLoggingProfiles();

    LoggingContext ctx;

    ctx.session_id = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    ctx.raw_log_path = paths.log_dir + "/training_" + ctx.session_id + ".log";

    // Bootstrap log file
    {
        std::ofstream bootstrap(ctx.raw_log_path, std::ios::app);
        if (bootstrap.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto tt = std::chrono::system_clock::to_time_t(now);
            bootstrap << "[BOOT] Phase1 logging bootstrap at "
                      << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S")
                      << std::endl;
        }
    }

    ctx.logger = std::make_unique<TrainingLogger>(paths.log_dir, ctx.session_id);
    ctx.metrics_collector = std::make_unique<MetricsCollector>();
    ctx.status_writer = std::make_unique<StatusFileWriter>(paths.status_path);

    return ctx;
}

void setupBatchLogTape(LoggingContext& logging, const StartupConfig& config) {
    const auto& tape_cfg = config.hyperparameters.tape_logging;

    auto tc = GRIM::Logging::parseTapeConfig(
        tape_cfg.default_level.c_str(),
        tape_cfg.equation_csv_enabled,
        tape_cfg.stderr_enabled,
        tape_cfg.initial_capacity);

    // Apply per-group overrides from ai_config.json
    for (const auto& [group_name, level_name] : tape_cfg.group_overrides) {
        GRIM::Logging::applyGroupOverride(tc, group_name.c_str(), level_name.c_str());
    }

    logging.tape = std::make_unique<GRIM::Logging::BatchLogTape>(tc);

    // Sinks
    std::string text_log_path =
        config.paths.log_dir + "/training_" + logging.session_id + "_tape.log";
    logging.text_sink = std::make_unique<GRIM::Logging::TextLogSink>(
        text_log_path.c_str(), /*also_stdout=*/false);
    logging.tape->addSink(logging.text_sink.get());

    if (tape_cfg.equation_csv_enabled) {
        std::string eq_csv_path = config.paths.log_dir + "/equation_log.csv";
        logging.equation_sink = std::make_unique<GRIM::Logging::CsvEquationSink>(
            eq_csv_path.c_str());
        logging.tape->addSink(logging.equation_sink.get());
    }

    if (tape_cfg.stderr_enabled) {
        logging.stderr_sink = std::make_unique<GRIM::Logging::StderrSink>();
        logging.tape->addSink(logging.stderr_sink.get());
    }

    // Set global tape pointer for layer-level code
    GRIM::Logging::setGlobalTape(logging.tape.get());

    logging.logger->log("BatchLogTape initialized; exact logging config values are listed by ConfigDump.");
}

} // namespace Internal

void LoggingReady(TrainingContext& ctx) {
    using GRIM::Logging::EmitModuleInfo;
    using GRIM::Logging::ModuleId;

    EmitModuleInfo(ModuleId::Training, "========================================", 0);
    EmitModuleInfo(ModuleId::Training, "  Phase 1: Startup & Initialization", 0);
    EmitModuleInfo(ModuleId::Training, "  GRIM-text GPU Training v3.0.0", 0);
    EmitModuleInfo(ModuleId::Training, "========================================", 0);

    if (ctx.config.hyperparameters.log_recorder.enabled) {
        GRIM::Logging::InitLogRecorder(ctx.config.paths.log_dir);

        const auto& layers = ctx.config.hyperparameters.log_recorder.layers;
        GRIM::Logging::ConfigureLayerLogging(
            ctx.config.hyperparameters.log_recorder.enabled,
            layers.embedding,
            layers.rms_norm,
            layers.attention,
            layers.feed_forward,
            layers.residual,
            layers.encoding,
            layers.serialization,
            layers.execution_block);
    } else {
        GRIM::Logging::ConfigureLayerLogging(
            false, false, false, false, false, false, false, false, false);
    }

    if (!ctx.config.paths.log_dir.empty()) {
        fs::create_directories(ctx.config.paths.log_dir);
    }
    if (!ctx.config.paths.status_path.empty()) {
        fs::path status_parent = fs::path(ctx.config.paths.status_path).parent_path();
        if (!status_parent.empty()) {
            fs::create_directories(status_parent);
        }
    }

    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing logging...", 0);
    ctx.logging = Internal::initializeLogging(ctx.config.paths);
    Internal::setupBatchLogTape(ctx.logging, ctx.config);
}

} // namespace GRIMText::Training

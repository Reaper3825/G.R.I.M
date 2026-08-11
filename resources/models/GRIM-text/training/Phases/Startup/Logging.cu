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
#include <sstream>
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

LoggingContext initializeLogging(const ::GRIM::Config::AiConfigSnapshot& config) {
    registerDefaultLoggingProfiles();

    LoggingContext ctx;
    const auto paths_hp = ::GRIM::HyperParameters::pathsHP(config);

    ctx.session_id = std::to_string(
        std::chrono::system_clock::now().time_since_epoch().count());
    ctx.raw_log_path = paths_hp.log_dir + "/training_" + ctx.session_id + ".log";

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

    ctx.logger = std::make_unique<TrainingLogger>(paths_hp.log_dir, ctx.session_id);
    ctx.metrics_collector = std::make_unique<MetricsCollector>();
    ctx.status_writer = std::make_unique<StatusFileWriter>(paths_hp.status_path);

    return ctx;
}

void setupBatchLogTape(
    LoggingContext& logging,
    const ::GRIM::Config::AiConfigSnapshot& config) {
    const auto tape_cfg = ::GRIM::HyperParameters::tapeLogHP(config);
    const auto paths_hp = ::GRIM::HyperParameters::pathsHP(config);

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
    logging.text_sink = std::make_unique<GRIM::Logging::TextLogSink>(
        logging.raw_log_path.c_str(), /*also_stdout=*/false);
    logging.tape->addSink(logging.text_sink.get());

    if (tape_cfg.equation_csv_enabled) {
        std::string eq_csv_path = paths_hp.log_dir + "/equation_log.csv";
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
    const auto recorder_cfg =
        ::GRIM::HyperParameters::logRecorderHP(ctx.config);
    const auto paths_hp =
        ::GRIM::HyperParameters::pathsHP(ctx.config);

    EmitModuleInfo(ModuleId::Training, "========================================", 0);
    EmitModuleInfo(ModuleId::Training, "  Phase 1: Startup & Initialization", 0);
    EmitModuleInfo(ModuleId::Training, "  GRIM-text GPU Training v3.0.0", 0);
    EmitModuleInfo(ModuleId::Training, "========================================", 0);

    if (recorder_cfg.enabled) {
        GRIM::Logging::InitLogRecorder(paths_hp.log_dir);

        const auto& layers = recorder_cfg.layers;
        GRIM::Logging::ConfigureLayerLogging(
            recorder_cfg.enabled,
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

    if (!paths_hp.log_dir.empty()) {
        fs::create_directories(paths_hp.log_dir);
    }
    if (!paths_hp.status_path.empty()) {
        fs::path status_parent = fs::path(paths_hp.status_path).parent_path();
        if (!status_parent.empty()) {
            fs::create_directories(status_parent);
        }
    }

    EmitModuleInfo(ModuleId::Training, "[Phase1] Initializing logging...", 0);
    ctx.logging = Internal::initializeLogging(ctx.config);
    Internal::setupBatchLogTape(ctx.logging, ctx.config);

    // loadCompiledModelConfig() returns this snapshot only after FlatBuffer,
    // schema, semantic, integrity, capability, and decoded-value validation.
    // Report that already-established result; validation remains owned by the
    // compiled-model loader.
    if (ctx.config.model_config) {
        const auto& compiled = *ctx.config.model_config;
        std::ostringstream message;
        message
            << "[GRIMCFG] verified artifact=\"" << compiled.source_path.string() << '"'
            << " schema_version=" << compiled.schema_version
            << " semantic_version=" << compiled.semantic_version
            << " model_compatibility_xxhash64=0x"
            << std::hex << std::setfill('0') << std::setw(16)
            << compiled.integrity.model_compatibility_xxhash64;
        ctx.logging.logger->log(message.str());
    }
}

} // namespace GRIMText::Training

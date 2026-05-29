//======================================================//
//  ConfigDump.hpp
//  Single-source hyperparameters dump used by Phase1
//  startup. Emits one log line per LanguageModelConfig root
//  field via a single internal loop, replacing scattered
//  EmitModuleInfo(...) / logger->log(...) calls.
//======================================================//

#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "DataStatsDump.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

namespace GRIMText { namespace Training {

// LogFn signature matches the startup/config dump emission pattern used by
// GRIM-text hyperparameter reporting.
using ConfigDumpLogFn = std::function<void(const std::string&)>;

//======================================================//
// Visual / formatting controls for the dump.
//
// Defaults are tuned for EmitModuleInfo (ModuleId::Training) output and
// a typical 100-col log line; override per-call when piping somewhere
// else (e.g. a wider terminal log).
//======================================================//
struct ConfigDumpOptions {
    std::string prefix          = "  ";   // Prepended to every emitted line.
    std::string banner_label    = "Hyperparameters";
    std::string banner_subtitle = "startup-final";  // Appended in parens after entry count.
    std::size_t banner_width    = 72;     // Width of the '=' rule.
    std::size_t section_width   = 72;     // Width of the section divider line.
    std::size_t name_col_width  = 0;      // 0 = auto-fit to longest name in the dump.
    bool        show_sections   = true;
    bool        show_footer     = true;
};

//======================================================//
// Primary API.
//
// Iterates every field in LanguageModelConfig (and the
// DerivedScheduleInfo, when supplied) and emits one log line per field
// via the supplied callback. MUST be called AFTER startup has finalized the
// root config and computed DerivedScheduleInfo so the dump reflects the
// actual authored root plus schedule evidence.
//
// If `data_stats` is non-null, dumpDataStats(...) is invoked inside
// the same banner block so vocab, sequence-count, and startup memory
// info is shown alongside the hyperparameters.
//======================================================//
void dumpAllHyperparameters(
    const GRIM::HyperParameters::LanguageModelConfig& hp,
    const GRIM::HyperParameters::DerivedScheduleInfo* derived,
    const DataStatsSnapshot* data_stats,
    const ConfigDumpOptions& opts,
    const ConfigDumpLogFn& log_fn);

void dumpAllHyperparameters(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const GRIM::HyperParameters::DerivedScheduleInfo* derived,
    const DataStatsSnapshot* data_stats,
    const ConfigDumpOptions& opts,
    const ConfigDumpLogFn& log_fn);

// Convenience overload using ConfigDumpOptions{} defaults.
inline void dumpAllHyperparameters(
    const GRIM::HyperParameters::LanguageModelConfig& hp,
    const GRIM::HyperParameters::DerivedScheduleInfo* derived,
    const DataStatsSnapshot* data_stats,
    const ConfigDumpLogFn& log_fn)
{
    dumpAllHyperparameters(hp, derived, data_stats, ConfigDumpOptions{}, log_fn);
}

inline void dumpAllHyperparameters(
    const GRIM::Config::AiConfigSnapshot& snapshot,
    const GRIM::HyperParameters::DerivedScheduleInfo* derived,
    const DataStatsSnapshot* data_stats,
    const ConfigDumpLogFn& log_fn)
{
    dumpAllHyperparameters(snapshot, derived, data_stats, ConfigDumpOptions{}, log_fn);
}

} } // namespace GRIMText::Training

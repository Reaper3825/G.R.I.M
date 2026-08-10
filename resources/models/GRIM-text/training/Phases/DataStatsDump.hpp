//======================================================//
//  DataStatsDump.hpp
//  Visual dump of startup data-side stats: source paths,
//  GRMT header vocab size, train/val sequence counts, and
//  host memory footprint.
//
//  Lives separately from ConfigDump (which only knows about
//  TrainingHyperparameters / DerivedScheduleInfo) but is
//  invoked from inside dumpAllHyperparameters when a
//  DataStatsSnapshot is supplied, so callers get one
//  unified visual block.
//======================================================//

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

namespace GRIMText { namespace Training {

//======================================================//
// Plain-data snapshot. Caller fills this once at Phase1
// startup (after tokenizer init + data load) and passes it
// in. No GPU pointers, no live references — safe to pass
// around and to copy into logs.
//======================================================//
struct DataStatsSnapshot {
    std::string data_path;
    std::string vocab_path;
    std::uint32_t actual_vocab_size    = 0;     // From GRMT header.
    std::size_t   train_sequence_count = 0;
    std::size_t   val_sequence_count   = 0;

};

//======================================================//
// Visual options. Mirrors ConfigDumpOptions semantics so
// the two blocks line up when emitted back-to-back.
//======================================================//
struct DataStatsDumpOptions {
    std::string prefix         = "  ";
    std::string banner_label   = "Data Stats";
    std::size_t banner_width   = 72;
    std::size_t name_col_width = 0;   // 0 = auto-fit
    bool        show_footer    = false;  // ConfigDump emits its own footer rule.
};

using DataStatsLogFn = std::function<void(const std::string&)>;

void dumpDataStats(
    const DataStatsSnapshot& snap,
    const DataStatsDumpOptions& opts,
    const DataStatsLogFn& log_fn);

} } // namespace GRIMText::Training

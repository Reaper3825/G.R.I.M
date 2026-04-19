#pragma once
//======================================================//
//  ILogSink.hpp — Output backend interface
//======================================================//
//
//  All log output goes through sinks. The BatchLogTape
//  dispatches flushed entries to registered sinks.
//
//  Sinks are responsible for:
//    - Filtering entries they care about (by group, tag, etc.)
//    - Formatting output (text, CSV, structured)
//    - Writing to their destination (file, stderr, GPU)
//
//  Sinks receive entries in phase-sorted order.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "../LogTypes.hpp"
#include <cstddef>

namespace GRIM::Logging {

class ILogSink {
public:
    virtual ~ILogSink() = default;

    /// Called once per flush with all entries for the batch,
    /// already sorted by phase. `count` may be 0 (empty batch).
    virtual void receive(const LogEntry* entries, size_t count, int global_step) = 0;

    /// Called during shutdown. Flush any buffered output.
    virtual void flush() = 0;

    /// Human-readable name for diagnostics (e.g. "TextLogSink", "CsvEquationSink").
    virtual const char* name() const = 0;
};

} // namespace GRIM::Logging

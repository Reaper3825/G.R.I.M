#pragma once
//======================================================//
//  StderrSink.hpp — Warning+ entries to stderr
//======================================================//
//
//  Catches Warning/Error/Fatal entries and writes them
//  to stderr for immediate console visibility, even when
//  stdout is redirected to a file.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "ILogSink.hpp"
#include <cstdio>

namespace GRIM::Logging {

class StderrSink : public ILogSink {
public:
    /// Minimum level to write. Default: Warning.
    explicit StderrSink(LogLevel min_level = LogLevel::Warning)
        : min_level_(min_level)
    {}

    ~StderrSink() override {
        flush();
    }

    void receive(const LogEntry* entries, size_t count, int /*global_step*/) override {
        if (!entries || count == 0) return;

        for (size_t i = 0; i < count; ++i) {
            const auto& e = entries[i];
            if (static_cast<uint8_t>(e.level) < static_cast<uint8_t>(min_level_)) continue;

            std::fprintf(stderr, "[%s][%s] s=%d %s: %s\n",
                logGroupToString(e.group),
                logLevelToString(e.level),
                e.global_step,
                e.tag,
                e.message);
        }
    }

    void flush() override {
        std::fflush(stderr);
    }

    const char* name() const override { return "StderrSink"; }

private:
    LogLevel min_level_;
};

} // namespace GRIM::Logging

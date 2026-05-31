#pragma once
//======================================================//
//  TextLogSink.hpp — Human-readable .log file output
//======================================================//
//
//  Writes structured entries to the training .log file.
//  Format: [TIMESTAMP][GROUP][LEVEL] tag: message
//
//  Replaces the scattered ctx.logging.logger->log() calls
//  for batch-scoped messages. Also writes to stdout.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "ILogSink.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIM::Logging {

class TextLogSink : public ILogSink {
public:
    /// Construct with path to the .log file. Opens in append mode.
    /// Throws on failure (Rule 20).
    explicit TextLogSink(const std::string& log_path, bool write_to_stdout = true)
        : write_to_stdout_(write_to_stdout)
    {
        file_.open(log_path, std::ios::app);
        if (!file_.is_open()) {
            throw std::runtime_error("TextLogSink: failed to open " + log_path);
        }
        file_ << std::fixed << std::setprecision(6);
    }

    ~TextLogSink() override {
        flush();
    }

    void receive(const LogEntry* entries, size_t count, int /*global_step*/) override {
        if (!entries || count == 0) return;

        for (size_t i = 0; i < count; ++i) {
            const auto& e = entries[i];
            writeEntry(e);
        }
    }

    void flush() override {
        if (file_.is_open()) {
            file_.flush();
        }
    }

    const char* name() const override { return "TextLogSink"; }

private:
    void writeEntry(const LogEntry& e) {
        // Timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        char ts_buf[32];
        std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&time_t));

        std::ostringstream line;
        line << '[' << ts_buf << ']'
             << '[' << logGroupToString(e.group) << ']'
             << '[' << logLevelToString(e.level) << ']'
             << " s=" << e.global_step
             << ' ' << e.tag << ": " << e.message;

        // Check for scalar metrics
        bool has_primary = !std::isnan(e.primary);
        bool has_secondary = !std::isnan(e.secondary);

        if (has_primary && has_secondary) {
            line << " [" << e.primary << ", " << e.secondary << ']';
        } else if (has_primary) {
            line << " [" << e.primary << ']';
        }

        const std::string rendered = line.str();

        if (file_.is_open()) {
            file_ << rendered << '\n';
        }
        if (write_to_stdout_) {
            std::cout << rendered << '\n';
        }
    }

    std::ofstream file_;
    bool write_to_stdout_;
};

} // namespace GRIM::Logging

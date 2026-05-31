#pragma once
//======================================================//
//  CsvEquationSink.hpp — Rule 21 equation CSV output
//======================================================//
//
//  Filters for entries with "EQUATION" in the tag and
//  writes them to a CSV file for structured analysis.
//
//  Columns: step,batch,layer,group,phase,tag,primary,secondary,message
//
//  Replaces the old EquationLogger sorted-flush mechanism.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "ILogSink.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIM::Logging {

class CsvEquationSink : public ILogSink {
public:
    /// Construct with path to the equation CSV file.
    /// Writes header on creation. Throws on failure (Rule 20).
    explicit CsvEquationSink(const std::string& csv_path)
    {
        file_.open(csv_path, std::ios::trunc);
        if (!file_.is_open()) {
            throw std::runtime_error("CsvEquationSink: failed to open " + csv_path);
        }
        // Write CSV header
        file_ << "step,batch,layer,group,phase,tag,primary,secondary,message\n";
    }

    ~CsvEquationSink() override {
        flush();
    }

    void receive(const LogEntry* entries, size_t count, int /*global_step*/) override {
        if (!entries || count == 0) return;

        for (size_t i = 0; i < count; ++i) {
            const auto& e = entries[i];

            // Only capture equation-tagged entries
            if (!isEquationEntry(e)) continue;

            writeEntry(e);
        }
    }

    void flush() override {
        if (file_.is_open()) {
            file_.flush();
        }
    }

    const char* name() const override { return "CsvEquationSink"; }

private:
    /// An equation entry has "EQUATION" somewhere in its tag.
    static bool isEquationEntry(const LogEntry& e) {
        return std::strstr(e.tag, "EQUATION") != nullptr;
    }

    void writeEntry(const LogEntry& e) {
        const std::string safe_msg = escapeCsv(e.message);

        if (file_.is_open()) {
            file_ << e.global_step
                  << ',' << e.batch_idx
                  << ',' << static_cast<int>(e.layer_idx)
                  << ',' << logGroupToString(e.group)
                  << ',' << logPhaseToString(e.phase)
                  << ',' << e.tag
                  << ',' << formatFloat(e.primary)
                  << ',' << formatFloat(e.secondary)
                  << ",\"" << safe_msg << "\"\n";
        }
    }

    /// Format float or empty string if not set
    static std::string formatFloat(float v) {
        if (std::isnan(v)) {
            return {};
        }
        std::ostringstream oss;
        oss << v;
        return oss.str();
    }

    /// Escape string for CSV (double-quote escaping for inner quotes)
    static std::string escapeCsv(const std::string& src) {
        std::string dst;
        dst.reserve(src.size());
        for (char c : src) {
            if (c == '"') {
                dst.push_back('"');
                dst.push_back('"');
            } else if (c == '\n') {
                dst.push_back(' ');
            } else if (c == '\r') {
                // skip
            } else {
                dst.push_back(c);
            }
        }
        return dst;
    }

    std::ofstream file_;
};

} // namespace GRIM::Logging

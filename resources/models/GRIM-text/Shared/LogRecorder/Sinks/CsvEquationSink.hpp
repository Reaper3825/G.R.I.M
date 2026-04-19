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
        // Escape message for CSV (replace commas and newlines)
        char safe_msg[520];
        escapeCsv(e.message, safe_msg, sizeof(safe_msg));

        char line[768];
        int written = std::snprintf(line, sizeof(line),
            "%d,%d,%d,%s,%s,%s,%s,%s,\"%s\"",
            e.global_step,
            e.batch_idx,
            static_cast<int>(e.layer_idx),
            logGroupToString(e.group),
            logPhaseToString(e.phase),
            e.tag,
            formatFloat(e.primary),
            formatFloat(e.secondary),
            safe_msg);
        (void)written;

        if (file_.is_open()) {
            file_ << line << '\n';
        }
    }

    /// Format float or "NaN" if not set
    static const char* formatFloat(float v) {
        // Use a thread-local buffer for each float
        // Since we only ever write two floats per line, two buffers suffice
        static thread_local char buf1[32], buf2[32];
        static thread_local int which = 0;
        char* buf = (which++ & 1) ? buf2 : buf1;

        if (std::isnan(v)) {
            return "";
        }
        std::snprintf(buf, 32, "%.6g", v);
        return buf;
    }

    /// Escape string for CSV (double-quote escaping for inner quotes)
    static void escapeCsv(const char* src, char* dst, size_t dst_size) {
        size_t j = 0;
        for (size_t i = 0; src[i] != '\0' && j < dst_size - 2; ++i) {
            char c = src[i];
            if (c == '"') {
                dst[j++] = '"';
                if (j < dst_size - 1) dst[j++] = '"';
            } else if (c == '\n') {
                dst[j++] = ' ';
            } else if (c == '\r') {
                // skip
            } else {
                dst[j++] = c;
            }
        }
        dst[j] = '\0';
    }

    std::ofstream file_;
};

} // namespace GRIM::Logging

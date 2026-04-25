#include "subprocess_status_io.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace GRIMText {
namespace Subprocess {

namespace {

// ---- Local I/O reporting helpers -------------------------------------------
// Collapses every "log a write-side I/O failure" call site into one place so
// the foundational module is not littered with std::cerr boilerplate. Writers
// run in the child (often before any TrainingLogger exists) and in the parent
// before logging is initialized, so std::cerr is the only universally-safe
// channel here. Reader-side contract violations throw std::runtime_error
// instead (those are programmer/protocol errors, not I/O hiccups).

void report_write_error(const std::string& stage, const std::string& detail) {
    std::cerr << "subprocess_status_io: " << stage << ": " << detail << "\n";
}

[[noreturn]] void throw_read_error(const std::string& subprocess_name,
                                   const std::string& path,
                                   const std::string& detail) {
    throw std::runtime_error(
        "subprocess_status_io: subprocess '" + subprocess_name +
        "' " + detail + " at " + path);
}

// Atomic publish: write to <path>.tmp, then rename. Returns false on any
// I/O failure (writer-side: report-and-return, never throw).
bool write_atomic(const std::string& path, const nlohmann::json& doc) {
    fs::path target(path);
    fs::path parent = target.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        fs::create_directories(parent, ec);
        if (ec) {
            report_write_error("create_directories(" + parent.string() + ")",
                               ec.message());
            return false;
        }
    }
    fs::path tmp = target;
    tmp += ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            report_write_error("open temp", tmp.string());
            return false;
        }
        out << doc.dump(2);
        if (!out.good()) {
            report_write_error("write temp", tmp.string());
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, target, ec);
    if (ec) {
        // Some platforms refuse rename-over-existing; remove and retry once.
        fs::remove(target, ec);
        fs::rename(tmp, target, ec);
        if (ec) {
            report_write_error("publish (" + target.string() + ")", ec.message());
            return false;
        }
    }
    return true;
}

} // namespace

bool write_status_success(const std::string& path,
                          const nlohmann::json& payload) {
    nlohmann::json doc = payload.is_object() ? payload : nlohmann::json::object();
    doc["outcome"] = "success";
    return write_atomic(path, doc);
}

bool write_status_error(const std::string& path,
                        const std::string& error_message) {
    nlohmann::json doc;
    doc["outcome"] = "error";
    doc["error_message"] = error_message;
    return write_atomic(path, doc);
}

subprocess_result read_status_file(const std::string& path,
                                   const std::string& subprocess_name) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw_read_error(subprocess_name, path, "did not produce status file");
    }

    nlohmann::json j;
    try {
        in >> j;
    } catch (const std::exception& e) {
        throw_read_error(subprocess_name, path,
                         std::string("wrote malformed status JSON: ") + e.what());
    }

    if (!j.is_object() || !j.contains("outcome") || !j["outcome"].is_string()) {
        throw_read_error(subprocess_name, path,
                         "status file missing required string field 'outcome'");
    }

    subprocess_result result;
    result.subprocess_name = subprocess_name;

    const std::string outcome_str = j["outcome"].get<std::string>();
    if (outcome_str == "success") {
        // Default to ok_proceed; subprocess-specific wrappers may rewrite to
        // ok_one_off based on caller config (NOT this module's job).
        result.outcome = subprocess_outcome::ok_proceed;
        // Hand the full envelope back as the opaque payload, minus the
        // envelope's own "outcome" key. Domain decoding happens in the
        // subprocess-specific wrapper.
        j.erase("outcome");
        result.success_payload = std::move(j);
        return result;
    }

    if (outcome_str == "error") {
        result.outcome = subprocess_outcome::error;
        if (!j.contains("error_message") || !j["error_message"].is_string()) {
            throw_read_error(subprocess_name, path,
                             "reported error but no 'error_message' string field");
        }
        result.error_message = j["error_message"].get<std::string>();
        if (result.error_message.empty()) {
            throw_read_error(subprocess_name, path,
                             "reported error with empty error_message");
        }
        return result;
    }

    throw_read_error(subprocess_name, path,
                     "reported unknown outcome '" + outcome_str +
                     "' (must be 'success' or 'error')");
}

} // namespace Subprocess
} // namespace GRIMText

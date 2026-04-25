#pragma once

// FOUNDATIONAL, subprocess-agnostic I/O for the JSON status-file IPC contract.
//
// This module owns the parent↔child JSON envelope and NOTHING domain-specific.
// It does not know what a vocab is, what a tokenizer is, or what fields a
// particular subprocess emits inside the success payload. Tokenizer-specific
// or any other subprocess-specific decoding MUST live in that subprocess's
// own wrapper (e.g. tokenizer_subprocess.cpp). Adding domain fields here is
// a Rule 20 violation — it forces every unrelated subprocess to depend on
// concepts it has no business with and creates schema drift.
//
// Envelope shapes (the only two the parent ever accepts):
//
//   Success:  { "outcome": "success", ...arbitrary payload object... }
//   Error:    { "outcome": "error", "error_message": "<precise>" }
//
// The success payload is opaque to this module: writers pass in a JSON object,
// readers get one back via subprocess_result::success_payload. The caller
// (subprocess-specific wrapper) is responsible for extracting fields and
// validating their types, and for cross-referencing them against ai_config.json
// when the value is already config-owned (single source of truth).

#include <string>

#include <nlohmann/json.hpp>

#include "subprocess_status.hpp"

namespace GRIMText {
namespace Subprocess {

// ---- Writer side (called by child subprocess executables)

// Write the success envelope atomically (write to <path>.tmp, then rename).
// `payload` is merged into the envelope; "outcome" is set by this function
// and any "outcome" key in `payload` is overwritten. Pass an empty object if
// the subprocess has nothing extra to report.
//
// On any I/O failure, logs to stderr and returns false — the child should NOT
// throw out of write_status_success because it IS the loud-failure channel;
// the caller observes the false return and exits non-zero so the parent's
// "missing status file" path catches it.
bool write_status_success(const std::string& path,
                          const nlohmann::json& payload);

// Write the error envelope atomically. Same I/O semantics as above.
bool write_status_error(const std::string& path,
                        const std::string& error_message);

// ---- Reader side (called by parent / subprocess_manager)

// Parse the status file the child wrote. Throws std::runtime_error with a
// precise message on ANY violation of the envelope contract (file missing,
// malformed JSON, unknown outcome, missing/empty error_message). The
// returned subprocess_result has outcome ∈ { ok_proceed, error }; on
// success, success_payload holds whatever JSON object the child emitted
// (this module does NOT validate domain fields inside it).
subprocess_result read_status_file(const std::string& path,
                                   const std::string& subprocess_name);

} // namespace Subprocess
} // namespace GRIMText

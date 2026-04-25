#pragma once

// Subprocess result contract (FOUNDATIONAL / GENERIC).
//
// This module is intentionally subprocess-AGNOSTIC. It owns:
//   - the three-outcome enum every subprocess wrapper reports,
//   - the generic envelope (outcome / error_message / opaque success payload),
//   - and nothing else. Tokenizer- or vocab-specific fields MUST NOT live
//     here — they belong in the tokenizer wrapper's own result type. Adding
//     domain-specific fields to this struct creates exactly the kind of
//     shared-mutable contract Rule 20 forbids.
//
// A subprocess returns EXACTLY one of three outcomes (Rule 20: no in-between):
//   1. ok_proceed   - Subprocess succeeded; main may continue with provided data.
//   2. ok_one_off   - Subprocess succeeded; main MUST stop because ai_config.json
//                     requested a one-off run (e.g. tokenizer-only build).
//   3. error        - Subprocess failed. error_message is populated with a precise,
//                     accurate description (no fallbacks, no silent recovery).
//
// Each subprocess wrapper produces exactly one subprocess_result. The caller
// switches on `outcome` and reads the relevant payload fields. Reading payload
// fields for the wrong outcome is a programmer error.

#include <string>

#include <nlohmann/json.hpp>

namespace GRIMText {
namespace Subprocess {

enum class subprocess_outcome : int {
    ok_proceed = 0,
    ok_one_off = 1,
    error      = 2,
};

// Generic envelope returned by spawn_and_wait + read_status_file. Domain-
// specific data lives inside `success_payload` (an opaque JSON object) so
// the foundational layer never has to know what fields a particular
// subprocess emits. Tokenizer-specific decoding happens in
// tokenizer_subprocess.cpp.
struct subprocess_result {
    subprocess_outcome outcome = subprocess_outcome::error;

    // Populated on ok_proceed / ok_one_off. Whatever JSON object the child
    // wrote alongside `"outcome":"success"`. Empty object when the child
    // emitted no extra fields.
    nlohmann::json success_payload;

    // Populated on error.
    std::string error_message;

    // Always populated: name of the subprocess that produced this result
    // (used so the caller can attribute errors precisely).
    std::string subprocess_name;
};

} // namespace Subprocess
} // namespace GRIMText

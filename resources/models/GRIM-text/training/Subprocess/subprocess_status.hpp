#pragma once

// Subprocess result contract.
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

#include <cstdint>
#include <string>

namespace GRIMText {
namespace Subprocess {

enum class subprocess_outcome : int {
    ok_proceed = 0,
    ok_one_off = 1,
    error      = 2,
};

// Payload populated by the tokenizer subprocess. Values are only meaningful when
// outcome is ok_proceed or ok_one_off. On error, only error_message is meaningful.
struct subprocess_result {
    subprocess_outcome outcome = subprocess_outcome::error;

    // Populated on ok_proceed / ok_one_off:
    std::string vocab_path;
    std::string training_data_path;
    std::uint32_t vocab_size = 0;

    // Populated on error:
    std::string error_message;

    // Always populated: name of the subprocess that produced this result
    // (used so the caller can attribute errors precisely).
    std::string subprocess_name;
};

} // namespace Subprocess
} // namespace GRIMText

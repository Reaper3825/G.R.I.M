#pragma once

// tokenizer_subprocess
//
// Wraps the standalone `train_tokenizer` executable in a single function call
// suitable for use from the main training entry point. This is the first
// concrete subprocess wired through subprocess_manager.
//
// Behavior:
//   - Loads paths and the `subprocess.tokenizer.only_mode` flag from
//     ai_config.json directly (no hidden defaults; missing required fields
//     throw).
//   - Resolves the train_tokenizer executable as a sibling of the current
//     process binary.
//   - Spawns it with `--status-file <path>` and (if requested) `--force`,
//     waits for completion, and parses the resulting subprocess_result.
//   - If `subprocess.tokenizer.only_mode` is true AND the tokenizer reports
//     success, the returned outcome is rewritten to ok_one_off so the caller
//     stops cleanly instead of proceeding into model training.
//
// Rule 20: there are no fallbacks. Missing executable, missing config field,
// missing status file, or any reported error surfaces precisely.

#include <string>

#include "subprocess_status.hpp"

namespace GRIMText {
namespace Subprocess {

struct tokenizer_subprocess_request {
    // Path to ai_config.json. Required.
    std::string config_path;

    // Optional override of the train_tokenizer executable location. When
    // empty, the manager resolves it as a sibling of the current process.
    std::string executable_path_override;

    // Optional override of the status file path. When empty, the manager
    // writes it next to ai_config.json under
    //   <config_dir>/.subprocess/tokenizer_status.json
    std::string status_file_path_override;

    // When true, propagate `--force` to train_tokenizer (rebuild even if
    // outputs already exist). When false, train_tokenizer skips work whose
    // outputs already exist.
    bool force_rebuild = false;
};

// Run the tokenizer subprocess and return its result. See
// subprocess_status.hpp for the outcome contract.
//
// On infrastructure failure (cannot spawn, malformed status, etc.) this
// throws std::runtime_error. A clean run that reported failure via the status
// file returns a subprocess_result with outcome=error.
subprocess_result run_tokenizer_subprocess(const tokenizer_subprocess_request& req);

} // namespace Subprocess
} // namespace GRIMText

#pragma once

// tokenizer_subprocess
//
// Wraps the standalone `train_tokenizer` executable in a single function call
// suitable for use from the main training entry point. This is the first
// concrete subprocess wired through subprocess_manager.
//
// Behavior:
//   - Consumes a grouped startup snapshot from HyperparameterGroupings.hpp
//     (no hidden defaults; missing required fields throw).
//   - Resolves the train_tokenizer executable as a sibling of the current
//     process binary.
//   - Spawns it with `--status-file <path>`, waits for completion, and
//     decodes the tokenizer-specific success payload (vocab_size) into a
//     tokenizer_subprocess_result.
//   - vocab_path / training_data_path are NOT carried over IPC — they are
//     resolved from TokenizerHP so there is exactly one tokenizer source of truth.
//   - If the grouped snapshot says `only_mode=true` AND the tokenizer reports
//     success, the returned outcome is rewritten to ok_one_off so the caller
//     stops cleanly instead of proceeding into model training.
//
// `training.config.force_rebuild_vocab` is sliced through `TokenizerHP` inside
// train_tokenizer; it is not a subprocess request or CLI flag.
//
// Rule 20: there are no fallbacks. Missing executable, missing config field,
// missing status file, or any reported error surfaces precisely.

#include <cstdint>
#include <string>

#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "subprocess_status.hpp"

namespace GRIMText {
namespace Subprocess {

struct tokenizer_subprocess_request {
    // Grouped startup snapshot authored once by Phase 1.
    GRIM::HyperParameters::TokenizerSubprocessHP hp;

    // Optional override of the train_tokenizer executable location. When
    // empty, the manager resolves it as a sibling of the current process.
    std::string executable_path_override;

    // Optional override of the status file path. When empty, the manager
    // writes it next to the tokenizer vocab artifact under
    //   <vocab_dir>/.subprocess/tokenizer_status.json
    std::string status_file_path_override;

};

// Tokenizer-specific result. Mirrors the foundational subprocess_result
// envelope but adds the tokenizer's domain fields. These domain fields do
// NOT live on the foundational struct — keeping them here means unrelated
// subprocesses are not forced to depend on tokenizer concepts.
struct tokenizer_subprocess_result {
    subprocess_outcome outcome = subprocess_outcome::error;
    std::string subprocess_name;

    // Populated on ok_proceed / ok_one_off. Paths come from TokenizerHP —
    // the IPC envelope does not carry them.
    // vocab_size comes from the child's success payload (it inspected the
    // GRMT header it just wrote).
    std::string vocab_path;
    std::string training_data_path;
    std::uint32_t vocab_size = 0;

    // Populated on error.
    std::string error_message;
};

// Run the tokenizer subprocess and return its decoded result.
//
// On infrastructure failure (cannot spawn, malformed status, missing
// vocab_size in success payload, etc.) this throws std::runtime_error.
// A clean run that reported failure via the status file returns a
// tokenizer_subprocess_result with outcome=error.
tokenizer_subprocess_result run_tokenizer_subprocess(
    const tokenizer_subprocess_request& req);

} // namespace Subprocess
} // namespace GRIMText

#include "tokenizer_subprocess.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "../Phases/Phase1_Startup.hpp"
#include "subprocess_manager.hpp"

namespace fs = std::filesystem;

namespace GRIMText {
namespace Subprocess {

namespace {

// Tokenizer-specific decode of the foundational success payload. Domain
// validation lives HERE — not in subprocess_status_io — so the foundational
// module stays subprocess-agnostic.
std::uint32_t extract_vocab_size(const subprocess_result& env,
                                 const std::string& status_file_path_for_error) {
    if (!env.success_payload.is_object() ||
        !env.success_payload.contains("vocab_size") ||
        !env.success_payload["vocab_size"].is_number_unsigned()) {
        throw std::runtime_error(
            "tokenizer_subprocess: child reported success but status payload "
            "is missing required uint field 'vocab_size' at " +
            status_file_path_for_error);
    }
    return env.success_payload["vocab_size"].get<std::uint32_t>();
}

} // namespace

tokenizer_subprocess_result run_tokenizer_subprocess(
    const Training::TrainingContext& ctx) {
    const auto tokenizer_hp = GRIM::HyperParameters::tokenizerHP(ctx.config);
    if (tokenizer_hp.vocab_path.empty() || tokenizer_hp.data_path.empty()) {
        throw std::runtime_error(
            "tokenizer_subprocess: TokenizerHP missing vocab_path and/or data_path");
    }

    subprocess_request sreq;
    sreq.name = "train_tokenizer";
    sreq.executable_path = resolve_sibling_executable("train_tokenizer");

    fs::path vocab_dir = fs::absolute(tokenizer_hp.vocab_path).parent_path();
    sreq.status_file_path =
        (vocab_dir / ".subprocess" / "tokenizer_status.json").string();

    sreq.arguments.push_back("--status-file");
    sreq.arguments.push_back(sreq.status_file_path);

    const subprocess_result env = spawn_and_wait(sreq);

    tokenizer_subprocess_result result;
    result.outcome = env.outcome;
    result.subprocess_name = env.subprocess_name;
    result.error_message = env.error_message;

    if (env.outcome == subprocess_outcome::error) {
        return result;
    }

    // Decode the tokenizer-specific payload field.
    result.vocab_size = extract_vocab_size(env, sreq.status_file_path);

    // Fill paths from the same tokenizer hyperparameter grouping used by training startup.
    // The child does NOT echo these over IPC; doing so would let the wire schema
    // drift from ai_config.json / TokenizerHP.
    result.vocab_path = tokenizer_hp.vocab_path;
    result.training_data_path = tokenizer_hp.data_path;

    // Rewrite ok_proceed -> ok_one_off when the config requested a one-off.
    if (tokenizer_hp.only_mode && result.outcome == subprocess_outcome::ok_proceed) {
        result.outcome = subprocess_outcome::ok_one_off;
    }

    return result;
}

} // namespace Subprocess
} // namespace GRIMText

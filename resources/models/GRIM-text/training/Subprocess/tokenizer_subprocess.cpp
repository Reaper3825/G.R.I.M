#include "tokenizer_subprocess.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

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
    const tokenizer_subprocess_request& req) {
    if (req.hp.config_path.empty()) {
        throw std::runtime_error(
            "tokenizer_subprocess: tokenizer_subprocess_request.hp.config_path is empty");
    }
    if (!fs::exists(req.hp.config_path)) {
        throw std::runtime_error(
            "tokenizer_subprocess: ai_config.json does not exist at: " + req.hp.config_path);
    }

    const std::string& resolved_config_path = req.hp.config_path;
    const auto& tokenizer_hp = req.hp.tokenizer;
    if (tokenizer_hp.vocab_path.empty() || tokenizer_hp.data_path.empty()) {
        throw std::runtime_error(
            "tokenizer_subprocess: TokenizerHP missing vocab_path and/or data_path for: " +
            resolved_config_path);
    }

    subprocess_request sreq;
    sreq.name = "train_tokenizer";
    sreq.executable_path = req.executable_path_override.empty()
        ? resolve_sibling_executable("train_tokenizer")
        : req.executable_path_override;

    if (req.status_file_path_override.empty()) {
        fs::path config_dir = fs::path(resolved_config_path).parent_path();
        sreq.status_file_path =
            (config_dir / ".subprocess" / "tokenizer_status.json").string();
    } else {
        sreq.status_file_path = req.status_file_path_override;
    }

    sreq.arguments.push_back("--status-file");
    sreq.arguments.push_back(sreq.status_file_path);
    sreq.arguments.push_back("--config");
    sreq.arguments.push_back(resolved_config_path);

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
    if (req.hp.only_mode && result.outcome == subprocess_outcome::ok_proceed) {
        result.outcome = subprocess_outcome::ok_one_off;
    }

    return result;
}

} // namespace Subprocess
} // namespace GRIMText

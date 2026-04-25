#include "tokenizer_subprocess.hpp"

#include <filesystem>
#include <stdexcept>
#include <string>

// HyperParameters_GPU.hpp is the single, mandatory entry point for the
// GRIM::Config::* JSON loaders. It defines the hyperparameter structs and
// then includes control/ai_config_paths.hpp in the only order that header
// permits (Rule 20 / single-owner config layering). Subprocess code MUST
// route ALL ai_config.json reads through this layer; raw json parsing here
// would silently bypass type validation, default propagation, and the
// AiConfigSnapshot contract.
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include "subprocess_manager.hpp"

namespace fs = std::filesystem;

namespace GRIMText {
namespace Subprocess {

subprocess_result run_tokenizer_subprocess(const tokenizer_subprocess_request& req) {
    if (req.config_path.empty()) {
        throw std::runtime_error(
            "tokenizer_subprocess: tokenizer_subprocess_request.config_path is empty");
    }
    if (!fs::exists(req.config_path)) {
        throw std::runtime_error(
            "tokenizer_subprocess: ai_config.json does not exist at: " + req.config_path);
    }

    // Centralized config read. loadSubprocessConfig() throws on a malformed
    // field (Rule 20: fail loud). Returns false ONLY when the file itself is
    // missing/unreadable, which we already handled above — so a false return
    // here is a true infrastructure failure, not a missing-field condition.
    GRIM::Config::SubprocessConfig sub_cfg;
    if (!GRIM::Config::loadSubprocessConfig(sub_cfg, req.config_path)) {
        throw std::runtime_error(
            "tokenizer_subprocess: GRIM::Config::loadSubprocessConfig failed for: " + req.config_path);
    }

    const bool effective_force = req.force_rebuild || sub_cfg.tokenizer_force_rebuild_vocab;

    subprocess_request sreq;
    sreq.name = "train_tokenizer";
    sreq.executable_path = req.executable_path_override.empty()
        ? resolve_sibling_executable("train_tokenizer")
        : req.executable_path_override;

    if (req.status_file_path_override.empty()) {
        fs::path config_dir = fs::absolute(req.config_path).parent_path();
        sreq.status_file_path =
            (config_dir / ".subprocess" / "tokenizer_status.json").string();
    } else {
        sreq.status_file_path = req.status_file_path_override;
    }

    sreq.arguments.push_back("--status-file");
    sreq.arguments.push_back(sreq.status_file_path);
    sreq.arguments.push_back("--config");
    sreq.arguments.push_back(req.config_path);
    if (effective_force) {
        sreq.arguments.push_back("--force");
    }

    subprocess_result result = spawn_and_wait(sreq);

    // Rewrite ok_proceed -> ok_one_off when the config requested a one-off.
    // Errors are passed through unchanged so the caller still sees a precise
    // error_message even when one-off mode is set.
    if (sub_cfg.tokenizer_only_mode && result.outcome == subprocess_outcome::ok_proceed) {
        result.outcome = subprocess_outcome::ok_one_off;
    }

    return result;
}

} // namespace Subprocess
} // namespace GRIMText

#pragma once

// Host-only diagnostic policy and reporting; no model execution or ownership.
#include "../../Shared/Lenses/LensMetadata.hpp"
#include <cmath>
#include <functional>
#include <stdexcept>

namespace GRIMText::Training {
struct DiagnosticLensOptions {
    bool enabled = false;
    bool validate = false;
    int top_k = 10;
    int validation_rounds = 3;
    int validation_tokens = 16;
    int max_replay_rows = 1024;
    double absolute_tolerance = 1e-4;
};

inline DiagnosticLensOptions validateDiagnosticLensOptions(DiagnosticLensOptions options) {
    options.enabled = options.enabled || options.validate;
    if (options.top_k < 1 || options.top_k > 100 ||
        options.validation_rounds < 2 || options.validation_rounds > 20 ||
        options.validation_tokens < 1 || options.validation_tokens > 256 ||
        options.max_replay_rows < 1 || options.max_replay_rows > 1048576 ||
        !std::isfinite(options.absolute_tolerance) || options.absolute_tolerance < 0)
        throw std::runtime_error("Invalid generation_lens_* diagnostic configuration");
    return options;
}

// JSON assembly is compiled as host C++, outside NVCC translation units.
nlohmann::json diagnosticLensReport(
    const GRIM::Lenses::LensSnapshot& snapshot, double tolerance,
    const std::function<std::string(int)>& decode);

nlohmann::json diagnosticLensValidationReport(
    bool token_match, bool text_match, bool identity_pass, int rounds,
    const std::vector<std::size_t>& used_bytes, const std::string& weights);

} // namespace GRIMText::Training

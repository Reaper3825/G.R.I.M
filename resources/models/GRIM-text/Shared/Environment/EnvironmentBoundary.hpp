#pragma once
#include "../Forward/GeneratedSequence.hpp"
#include <optional>
#include <string>

namespace GRIM::Environment {
struct EnvironmentBoundary {
    std::string phase;
    std::size_t begin;
    std::size_t end;
};
// Validate the newly delivered generated span, including atom closure. Runtime
// observations never count as generated delimiters or actionable boundaries.
EnvironmentBoundary inspectEnvironmentBoundary(const GeneratedSequence& sequence,
    std::size_t prompt_tokens, std::optional<std::size_t> previous_close);
}

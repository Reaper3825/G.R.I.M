#pragma once

#include "../Batching/BatchPayload.hpp"

#include <string>
#include <nlohmann/json_fwd.hpp>

namespace GRIM::Config { struct AiConfigSnapshot; }
namespace GRIM::Tokenizer { class UniByte; }

namespace GRIM::Inference {

// Build a compact token-LM prefill from supplied text and optional compatibility
// reasoning_state JSON. Preserves atom side channels and configured span metadata.
// Requires the finalized Phase1 config and its matching tokenizer; no model or
// generation state is consumed or modified.
// Structured input resolves the full configured delimiter vocabulary once for
// this prefill; its immutable map is shared with all sessions using the payload.
// Plain-text input does not opt into structured marker interpretation.
GRIM::Batching::BatchPayload buildInferencePrefill(
    const GRIM::Config::AiConfigSnapshot& config,
    GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt,
    const nlohmann::json* input_state = nullptr);

} // namespace GRIM::Inference

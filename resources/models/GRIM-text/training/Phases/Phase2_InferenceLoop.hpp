#pragma once
//======================================================//
//  Phase2_InferenceLoop.hpp
//  User/session inference orchestration after Phase1 startup
//======================================================//

#include "Phase1_Startup.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Forward/GeneratedSequence.hpp"
#include "../../Shared/Lenses/LensMetadata.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace GRIMText::Training {

struct Phase2TextInferenceResult {
    std::shared_ptr<const GRIM::Lenses::LensSnapshot> prefill_lens_snapshot;
    std::shared_ptr<const GRIM::Lenses::LensCaptureResult> prefill_lens_capture_result;
    std::string text; // Full decoded sequence; atom-insertion models return annotated input.
    std::string continuation_text; // Token-LM output only; never includes prefill state.
    // Exact realized IDs, including the prompt, for diagnostic comparisons.
    std::vector<int> token_ids;
    std::size_t prompt_token_count = 0;
    std::size_t sequence_token_count = 0;
    std::int64_t encode_ms = 0;
    std::int64_t generation_ms = 0;
    std::int64_t decode_ms = 0;
};

/**
 * @brief Execute Phase 2 inference from a text prompt over Phase1-owned state.
 *
 * This text adapter keeps tokenizer
 * access explicit at the call boundary instead of storing a runtime tokenizer
 * on TrainingContext. HTTP/front-end bridge code must send text/options to
 * train_gpu instead of touching TrainingContext, tokenizer artifacts, model
 * config, or Phase1 startup directly.
 *
 * A compiled atom-insertion model uses this same entrypoint as a full-context
 * byte-gap classifier. In that mode the returned text is the original prompt
 * with structurally valid predicted typed delimiters inserted; autoregressive
 * generation fields are not applied.
 */
Phase2TextInferenceResult executePhase2TextInference(
    TrainingContext& ctx,
    GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt,
    const GRIM::HyperParameters::GenerationHP& generation_hp,
    const GRIM::Lenses::LensCaptureRequest* lens_capture = nullptr);

// Build the wire-input adapter into a realized prefill payload. Existing
// reasoning_state JSON is accepted for transport compatibility only.
GRIM::Batching::BatchPayload buildPhase2InferencePrefill(
    TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt, const nlohmann::json& input_state);

// Primary token-LM boundary: caller-authored tokens, GoalTokenSpan tree and atom
// side channels. The generation loop authors ModelForwardRequest per invocation.
Phase2TextInferenceResult executePhase2PayloadInference(
    TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer,
    const GRIM::Batching::BatchPayload& prefill,
    const GRIM::HyperParameters::GenerationHP& generation_hp,
    const GRIM::Lenses::LensCaptureRequest* lens_capture = nullptr);

} // namespace GRIMText::Training

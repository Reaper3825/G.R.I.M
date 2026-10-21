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
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace GRIMText::Training {

struct Phase2TextInferenceResult {
    std::shared_ptr<const GRIM::Lenses::LensSnapshot> prefill_lens_snapshot;
    std::shared_ptr<const GRIM::Lenses::LensCaptureResult> prefill_lens_capture_result;
    // Token-LM continuation excludes prefill state. Atom-insertion models return
    // their annotated input here. Prefill-only inspection has no text output.
    std::string continuation_text;
    // Exact realized IDs, including the prompt, for diagnostic comparisons.
    std::vector<int> token_ids;
    // Retain the complete token-LM trajectory, including scores, positions,
    // provenance, and atom metadata. Absent for paths that do not produce a
    // GeneratedSequence (atom insertion and direct prefill-only inspection).
    std::optional<GRIM::GeneratedSequence> generated_sequence;
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

// Primary token-LM boundary: caller-authored tokens, GoalTokenSpan tree and atom
// side channels. The generation loop authors ModelForwardRequest per invocation.
// Requires Phase1-ready context/tokenizer and a payload validated by batch
// preparation (including any caller-authored metadata). This entry point checks
// only inference-specific mode and compact single-row requirements; device
// upload retains its payload validation and forward geometry checks.
// inspect_only returns after prefill capture, before token sampling.
Phase2TextInferenceResult executePhase2PayloadInference(
    TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer,
    const GRIM::Batching::BatchPayload& prefill,
    const GRIM::HyperParameters::GenerationHP& generation_hp,
    const GRIM::Lenses::LensCaptureRequest* lens_capture = nullptr,
    bool inspect_only = false);

} // namespace GRIMText::Training

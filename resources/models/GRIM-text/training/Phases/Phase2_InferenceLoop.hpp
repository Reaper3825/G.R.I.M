#pragma once
//======================================================//
//  Phase2_InferenceLoop.hpp
//  User/session inference orchestration after Phase1 startup
//======================================================//

#include "Phase1_Startup.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Forward/GeneratedSequence.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/UnigramByte/UniByte.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace GRIM { struct ConceptBlock; }

namespace GRIMText::Training {

struct Phase2TextInferenceResult {
    std::string text;
    std::size_t prompt_token_count = 0;
    std::size_t sequence_token_count = 0;
    std::int64_t encode_ms = 0;
    std::int64_t generation_ms = 0;
    std::int64_t decode_ms = 0;
};

/**
 * @brief Execute Phase 2 inference from a text prompt over Phase1-owned state.
 *
 * This is the only public Phase 2 inference entrypoint. It keeps tokenizer
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
    const GRIM::HyperParameters::GenerationHP& generation_hp);

// Structured state supplied by upstream models. Uses the training renderer;
// an answer present on the input object is excluded from the inference prefix.
Phase2TextInferenceResult executePhase2TextInference(
    TrainingContext& ctx,
    GRIM::Tokenizer::UniByte& tokenizer,
    const GRIM::ConceptBlock& supplied_state,
    const GRIM::HyperParameters::GenerationHP& generation_hp);

} // namespace GRIMText::Training

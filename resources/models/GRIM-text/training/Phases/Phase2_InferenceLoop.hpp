#pragma once
//======================================================//
//  Phase2_InferenceLoop.hpp
//  User/session inference orchestration after Phase1 startup
//======================================================//

#include "Phase1_Startup.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
 * access, BatchPayload construction, payload-level generation, and decoding
 * inside the trainer-owned orchestration process. HTTP/front-end bridge code
 * must send text/options to train_gpu instead of touching TrainingContext,
 * tokenizer artifacts, model config, or Phase1 startup directly.
 */
Phase2TextInferenceResult executePhase2TextInference(
    TrainingContext& ctx,
    const std::string& prompt,
    const GRIM::HyperParameters::GenerationHP& generation_hp);

} // namespace GRIMText::Training
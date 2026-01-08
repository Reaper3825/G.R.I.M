#pragma once

/**
 * @file ForwardOps_Orchestrator.hpp
 * @brief Forward Pass Orchestrator - Main Entry Point
 *
 * Coordinates the 3-phase forward pass:
 * - Phase 3: Input Layer (tokens, embeddings, ScratchBlock)
 * - Phase 2: Encoder (full sequence or incremental)
 * - Phase 1: Output Layer (LM head projection)
 */

#include "ForwardContext.hpp"
#include "ForwardPhase1_OutputLayer.hpp"
#include "ForwardPhase2_Encoder.hpp"
#include "ForwardPhase3_InputLayer.hpp"

namespace GRIM {
class LanguageModel;
}

namespace GRIM {
namespace Forward {

ForwardStatus executeForward(ForwardContext& ctx);

ForwardContext initForwardContext(
    LanguageModel& model,
    ForwardMode mode,
    int batch_size,
    int seq_len,
    ForwardLogitsTarget logits_target,
    const int* host_tokens,
    bool tokens_on_device,
    int new_token,
    int query_pos,
    bool enable_scratch_block,
    bool enable_activation_quantization,
    bool enable_entropy_output);

std::string getForwardErrorReport(const ForwardContext& ctx);
void logForwardSummary(const ForwardContext& ctx);

} // namespace Forward
} // namespace GRIM

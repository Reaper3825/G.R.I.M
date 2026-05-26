#include "PayloadBuildInputs.hpp"

#include "../../Phase1_Startup.hpp"

#include <stdexcept>
#include <string>

namespace GRIMText::Training {

PayloadBuildInputs derivePayloadBuildInputsOrThrow(const TrainingContext& ctx) {
    if (!ctx.model) {
        throw std::runtime_error(
            "FATAL: PayloadBuildInputsReady requires an allocated model — "
            "call ModelAllocated before this step");
    }
    if (!ctx.tokenizer) {
        throw std::runtime_error(
            "FATAL: PayloadBuildInputsReady requires an initialized tokenizer — "
            "call LoadTrainingData before this step");
    }

    const auto fixed_shape = GRIM::HyperParameters::trainingFixedShapeHP(ctx.config);

    PayloadBuildInputs inputs;
    inputs.configured_batch_size     = static_cast<std::size_t>(fixed_shape.batch_size);
    inputs.max_cached_seq            = static_cast<std::size_t>(fixed_shape.max_seq_len);
    inputs.execution_block_num_slots = ctx.config.execution_block_num_slots;
    inputs.execution_block_num_ops   = ctx.config.execution_block_num_ops;
    inputs.execution_block_num_steps = ctx.config.execution_block_num_steps;
    inputs.vocab_size                = ctx.config.vocab_size;
    inputs.train_mtp_k               = ctx.config.mtp_enabled ? ctx.config.mtp_k : 0;
    if (inputs.vocab_size != static_cast<int>(ctx.data_info.actual_vocab_size)) {
        throw std::runtime_error(
            "FATAL: PayloadBuildInputsReady runtime vocab mismatch: actual_vocab_size=" +
            std::to_string(ctx.data_info.actual_vocab_size) +
            " model vocab_size=" + std::to_string(inputs.vocab_size));
    }

    const auto layout = ctx.tokenizer->tokenLayout();
    inputs.token_layout.num_special = layout.num_special;
    inputs.token_layout.num_bytes   = layout.num_bytes;
    inputs.token_layout.num_atoms   = layout.num_atoms;
    inputs.token_layout.num_unigram = layout.num_unigram;
    return inputs;
}

void PayloadBuildInputsReady(TrainingContext& ctx) {
    ctx.payload_build_inputs = derivePayloadBuildInputsOrThrow(ctx);
}

} // namespace GRIMText::Training

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

    const auto& model_cfg = ctx.model->getConfig();

    // Rule 20: TrainingState allocation consumes batch/token capacity only.
    // Sequence capacity remains owned by RunCapacity and the BatchPayload path.
    if (model_cfg.max_cached_batch != static_cast<int>(ctx.run_capacity.batch_rows)) {
        throw std::runtime_error(
            "FATAL: model max_cached_batch does not match RunCapacity (model=" +
            std::to_string(model_cfg.max_cached_batch) +
            " stem=" + std::to_string(ctx.run_capacity.batch_rows) + ")");
    }

    PayloadBuildInputs inputs;
    inputs.max_cached_batch          = static_cast<std::size_t>(ctx.run_capacity.batch_rows);
    inputs.max_cached_seq            = static_cast<std::size_t>(ctx.run_capacity.seq_cap);
    inputs.execution_block_num_slots = model_cfg.execution_block_num_slots;
    inputs.execution_block_num_ops   = model_cfg.execution_block_num_ops;
    inputs.execution_block_num_steps = model_cfg.execution_block_num_steps;
    inputs.actual_vocab_size         = ctx.config.actual_vocab_size;
    inputs.train_mtp_k               = model_cfg.mtp_enabled ? model_cfg.mtp_k : 0;

    const auto layout = ctx.tokenizer.tokenLayout();
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

#include "PayloadBuildInputs.hpp"

#include "../../Phase1_Startup.hpp"

#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"

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
            "call DataInfoReady before this step");
    }

    const auto execution_hp =
        GRIM::HyperParameters::executionBlockConstructionHP(ctx.config.hyperparameters.architecture);
    const auto mtp_hp =
        GRIM::HyperParameters::mtpFeatureHP(ctx.config.hyperparameters.architecture);

    // Rule 20: TrainingState allocation consumes batch/token capacity only.
    // Sequence capacity remains owned by RunCapacity and the BatchPayload path.
    if (ctx.model_allocation.model_max_cached_batch != static_cast<int>(ctx.run_capacity.batch_rows)) {
        throw std::runtime_error(
            "FATAL: model max_cached_batch does not match RunCapacity (model=" +
            std::to_string(ctx.model_allocation.model_max_cached_batch) +
            " stem=" + std::to_string(ctx.run_capacity.batch_rows) + ")");
    }

    PayloadBuildInputs inputs;
    inputs.max_cached_batch          = static_cast<std::size_t>(ctx.run_capacity.batch_rows);
    inputs.max_cached_seq            = static_cast<std::size_t>(ctx.run_capacity.seq_cap);
    inputs.execution_block_num_slots = execution_hp.num_slots;
    inputs.execution_block_num_ops   = execution_hp.num_ops;
    inputs.execution_block_num_steps = execution_hp.num_exec_steps;
    inputs.actual_vocab_size         = ctx.config.actual_vocab_size;
    inputs.train_mtp_k               = mtp_hp.enabled ? mtp_hp.k : 0;

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

#include "ModelAllocationState.hpp"

#include "../../Phase1_Startup.hpp"

#include <stdexcept>
#include <string>

namespace GRIMText::Training {

ModelAllocationState captureAndValidateModelAllocationOrThrow(const TrainingContext& ctx) {
    if (!ctx.model) {
        throw std::runtime_error("FATAL: ModelAllocated validation called before model exists");
    }

    const auto& model_cfg = ctx.model->getConfig();
    const auto& state = ctx.model->getTrainingState();
    const auto& cap = ctx.run_capacity;

    ModelAllocationState allocation;
    allocation.model_max_cached_batch = model_cfg.max_cached_batch;
    allocation.model_max_cached_seq_len = model_cfg.max_cached_seq_len;
    allocation.model_max_tokens_per_batch = model_cfg.max_tokens_per_batch;
    allocation.training_state_max_cached_batch = state.max_cached_batch;
    allocation.training_state_max_cached_seq_len = static_cast<std::uint32_t>(state.max_cached_seq_len);
    allocation.training_state_max_tokens_per_batch = static_cast<int>(state.max_logit_tokens);

    if (allocation.model_max_cached_batch != static_cast<int>(cap.batch_rows)) {
        throw std::runtime_error("FATAL: model max_cached_batch does not match RunCapacity (model=" +
                                 std::to_string(allocation.model_max_cached_batch) +
                                 " stem=" + std::to_string(cap.batch_rows) + ")");
    }
    if (allocation.model_max_cached_seq_len != cap.seq_cap) {
        throw std::runtime_error("FATAL: model max_cached_seq_len does not match RunCapacity (model=" +
                                 std::to_string(allocation.model_max_cached_seq_len) +
                                 " stem=" + std::to_string(cap.seq_cap) + ")");
    }
    if (allocation.model_max_tokens_per_batch != static_cast<int>(cap.max_tokens_per_batch)) {
        throw std::runtime_error("FATAL: model max_tokens_per_batch does not match RunCapacity (model=" +
                                 std::to_string(allocation.model_max_tokens_per_batch) +
                                 " stem=" + std::to_string(cap.max_tokens_per_batch) + ")");
    }

    if (allocation.training_state_max_cached_batch != static_cast<int>(cap.batch_rows)) {
        throw std::runtime_error("FATAL: TrainingState max_cached_batch does not match RunCapacity (state=" +
                                 std::to_string(allocation.training_state_max_cached_batch) +
                                 " stem=" + std::to_string(cap.batch_rows) + ")");
    }
    if (allocation.training_state_max_cached_seq_len != cap.seq_cap) {
        throw std::runtime_error("FATAL: TrainingState max_seq_len_cache does not match RunCapacity (state=" +
                                 std::to_string(allocation.training_state_max_cached_seq_len) +
                                 " stem=" + std::to_string(cap.seq_cap) + ")");
    }
    if (allocation.training_state_max_tokens_per_batch != static_cast<int>(cap.max_tokens_per_batch)) {
        throw std::runtime_error("FATAL: TrainingState max_logit_tokens does not match RunCapacity (state=" +
                                 std::to_string(allocation.training_state_max_tokens_per_batch) +
                                 " stem=" + std::to_string(cap.max_tokens_per_batch) + ")");
    }

    return allocation;
}

} // namespace GRIMText::Training


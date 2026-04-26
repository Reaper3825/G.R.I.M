#pragma once

#include "RunCapacity.hpp"

#include "../../../../Shared/HyperParameters/HyperParameters_GPU.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace GRIMText::Training {

struct TrainingContext;

inline RunCapacity deriveRunCapacityOrThrow(const ::GRIM::HyperParameters::StartupConfig& config) {
    const int batch_size_i = config.hyperparameters.batch_size;
    const int max_seq_len_i = config.max_seq_len;

    if (batch_size_i <= 0) {
        throw std::runtime_error("FATAL: batch_size must be > 0 (got " + std::to_string(batch_size_i) + ")");
    }
    if (max_seq_len_i <= 0) {
        throw std::runtime_error("FATAL: max_seq_len must be > 0 (got " + std::to_string(max_seq_len_i) + ")");
    }

    const uint64_t batch_rows_u64 = static_cast<uint64_t>(static_cast<uint32_t>(batch_size_i));
    const uint64_t seq_cap_u64 = static_cast<uint64_t>(static_cast<uint32_t>(max_seq_len_i));
    const uint64_t tokens_u64 = batch_rows_u64 * seq_cap_u64;

    if (tokens_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error(
            "FATAL: max_tokens_per_batch overflow: batch_size=" + std::to_string(batch_size_i) +
            " max_seq_len=" + std::to_string(max_seq_len_i) +
            " product=" + std::to_string(tokens_u64) +
            " exceeds uint32 max");
    }

    RunCapacity cap;
    cap.batch_rows = static_cast<uint32_t>(batch_size_i);
    cap.seq_cap = static_cast<uint32_t>(max_seq_len_i);
    cap.max_tokens_per_batch = static_cast<uint32_t>(tokens_u64);
    return cap;
}

void HyperparametersReady(TrainingContext& ctx);
void CapacityStemReady(TrainingContext& ctx);

} // namespace GRIMText::Training


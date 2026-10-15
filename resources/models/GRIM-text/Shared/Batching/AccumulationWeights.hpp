#pragma once

#include <cstdint>
#include <stdexcept>

namespace GRIM::Batching {

// Each objective is a microbatch mean. Weight it by its own supervised-row
// count over the optimizer window. The existing clipping boundary subsequently
// divides by configured_accumulation_steps, so cancel that factor here.
inline float objectiveAccumulationScale(
    std::int64_t batch_rows, std::int64_t window_rows,
    int configured_accumulation_steps) {
    if (batch_rows < 0 || window_rows < batch_rows ||
        configured_accumulation_steps <= 0) {
        throw std::runtime_error("invalid objective accumulation counts");
    }
    if (window_rows == 0) return 0.0f;
    return static_cast<float>(
        static_cast<double>(configured_accumulation_steps) *
        static_cast<double>(batch_rows) / static_cast<double>(window_rows));
}

inline int optimizerWindowsPerEpoch(int batches, int accumulation_steps) {
    if (batches <= 0 || accumulation_steps <= 0) {
        throw std::runtime_error("optimizer window counts must be positive");
    }
    return batches / accumulation_steps + (batches % accumulation_steps != 0);
}

} // namespace GRIM::Batching

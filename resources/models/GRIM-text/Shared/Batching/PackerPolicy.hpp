#pragma once

#include <cstdint>

#include "Batching_GPU.hpp" // BatchOrdering

namespace GRIM::Batching {

/**
 * @brief Policy-only scheduler inputs (no capacity ownership).
 *
 * Capacity (max_tokens_per_batch, max_batch_size) must be passed separately.
 */
struct PackerPolicy {
    // === Batch ordering ===
    BatchOrdering batch_ordering = BatchOrdering::PRESERVE;

    // === RNG for shuffling. Zero means preserve input order. ===
    uint64_t rng_seed = 0;
};

} // namespace GRIM::Batching


#pragma once

#include <cstdint>

#include "Batching_GPU.hpp" // PackingStrategy, BatchOrdering

namespace GRIM::Batching {

/**
 * @brief Policy-only scheduler inputs (no capacity ownership).
 *
 * Capacity (max_tokens_per_batch, max_batch_size) must be passed separately.
 */
struct PackerPolicy {
    // === Packing strategy ===
    PackingStrategy strategy = PackingStrategy::SIMILARITY_GROUPED;
    uint32_t bucket_step = 128;
    float similarity_threshold = 0.25f;

    // === Curriculum / ordering hints ===
    bool prefer_short_first = false;
    float curriculum_progress = 1.0f;

    // === Batch ordering (post-packing) ===
    BatchOrdering batch_ordering = BatchOrdering::LENGTH_ASCENDING;
    bool interleave_overflow = true;

    // === RNG for shuffling ===
    uint64_t rng_seed = 0;
};

} // namespace GRIM::Batching


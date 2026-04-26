#pragma once

#include <cstdint>

namespace GRIMText::Training {

/**
 * @brief Single owner of run capacity facts (Rule 20).
 *
 * This is the only place that should "author" the run's batch rows, sequence cap,
 * and the checked padded-token rectangle used for cache sizing. Other layers may
 * mirror these values (e.g. to model config fields) but must not re-derive them.
 */
struct RunCapacity {
    // Effective maximum sequences per batch (batch rows).
    uint32_t batch_rows = 0;

    // Configured sequence length cap for this run.
    uint32_t seq_cap = 0;

    // Checked product: batch_rows * seq_cap (padded rectangle token slots).
    uint32_t max_tokens_per_batch = 0;
};

} // namespace GRIMText::Training


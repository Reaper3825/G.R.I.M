#include "DynaSeq_GPU.hpp"
#include <algorithm>

namespace GRIM::DynaSeq {

BatchPlan planBatches(const Catalog& catalog,
                      uint32_t max_tokens_per_batch,
                      uint32_t max_batch_size,
                      bool prefer_short_first)
{
    BatchPlan plan{};
    if (catalog.entries().empty() || max_tokens_per_batch == 0 || max_batch_size == 0) {
        return plan;
    }

    // Sort sequence ids by length (longest-first default; shortest-first optional).
    std::vector<uint32_t> ids;
    ids.reserve(catalog.entries().size());
    for (const auto& e : catalog.entries()) {
        ids.push_back(e.seq_id);
    }
    std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) {
        const auto& ea = catalog.entries()[a];
        const auto& eb = catalog.entries()[b];
        return prefer_short_first ? (ea.seq_length < eb.seq_length)
                                  : (ea.seq_length > eb.seq_length);
    });

    std::vector<uint32_t> current;
    uint32_t max_len_in_batch = 0;  // Track max sequence length for padding calculation

    auto flush = [&]() {
        if (!current.empty()) {
            // Total tokens = max_seq_length × batch_size (accounts for padding)
            const uint32_t batch_tokens = max_len_in_batch * static_cast<uint32_t>(current.size());
            plan.total_tokens += batch_tokens;
            plan.batches.push_back(current);
            current.clear();
            max_len_in_batch = 0;
        }
    };

    for (uint32_t id : ids) {
        // Bounds check: seq_id must be valid index into entries
        if (id >= catalog.entries().size()) {
            continue;  // Skip invalid seq_id (could also throw per Rule 20)
        }
        
        const auto& meta = catalog.entries()[id];
        
        // Calculate tokens if we add this sequence to current batch:
        // Padded batch tokens = max(current_max_len, new_seq_len) × (batch_size + 1)
        const uint32_t new_max_len = std::max(max_len_in_batch, meta.seq_length);
        const uint32_t new_batch_size = static_cast<uint32_t>(current.size()) + 1;
        const uint32_t tokens_if_added = new_max_len * new_batch_size;
        
        const bool would_overflow_tokens = tokens_if_added > max_tokens_per_batch;
        const bool would_overflow_batch = new_batch_size > max_batch_size;
        
        if ((would_overflow_tokens || would_overflow_batch) && !current.empty()) {
            flush();
        }
        
        current.push_back(id);
        max_len_in_batch = std::max(max_len_in_batch, meta.seq_length);
    }
    flush();

    return plan;
}

} // namespace GRIM::DynaSeq

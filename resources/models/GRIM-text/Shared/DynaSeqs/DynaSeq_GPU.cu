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
    uint32_t current_tokens = 0;

    auto flush = [&]() {
        if (!current.empty()) {
            plan.total_tokens += current_tokens;
            plan.batches.push_back(current);
            current.clear();
            current_tokens = 0;
        }
    };

    for (uint32_t id : ids) {
        const auto& meta = catalog.entries()[id];
        const uint32_t tokens_if_added = current_tokens + meta.seq_length * static_cast<uint32_t>(current.size() + 1);
        const bool would_overflow_tokens = tokens_if_added > max_tokens_per_batch;
        const bool would_overflow_batch = current.size() >= max_batch_size;
        if ((would_overflow_tokens || would_overflow_batch) && !current.empty()) {
            flush();
        }
        current.push_back(id);
        current_tokens = std::max(current_tokens, meta.seq_length * static_cast<uint32_t>(current.size()));
    }
    flush();

    return plan;
}

} // namespace GRIM::DynaSeq

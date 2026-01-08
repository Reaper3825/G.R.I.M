#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>

// Central data structures for dynamic sequence handling. These live on host and are
// consumed by batching utilities; the GPU side only needs the packed batches.
namespace GRIM::DynaSeq {

struct SequenceMetadata {
    uint32_t seq_id;        // unique identifier for this sample
    uint32_t seq_length;    // total tokens incl. special tokens (BOS/EOS/pad)
    uint32_t active_tokens; // non-padding tokens
    uint64_t data_offset;   // offset into dataset/mmap
    uint8_t  difficulty;    // optional curriculum tag
    uint16_t source_id;     // optional shard/source id
};

struct SequenceLengthStats {
    uint32_t count = 0;
    uint32_t min_len = 0;
    uint32_t max_len = 0;
    double mean_len = 0.0;
    uint32_t p50 = 0;
    uint32_t p90 = 0;
    uint32_t p99 = 0;
};

struct BatchPlan {
    std::vector<std::vector<uint32_t>> batches; // seq_ids per batch
    uint64_t total_tokens = 0;
};

// Catalog holds all sequences and exposes lightweight stats for planning.
class Catalog {
public:
    uint32_t add(uint32_t seq_length,
                 uint32_t active_tokens,
                 uint64_t data_offset = 0,
                 uint8_t difficulty = 0,
                 uint16_t source_id = 0)
    {
        SequenceMetadata m{};
        m.seq_id = static_cast<uint32_t>(entries_.size());
        m.seq_length = seq_length;
        m.active_tokens = active_tokens;
        m.data_offset = data_offset;
        m.difficulty = difficulty;
        m.source_id = source_id;
        entries_.push_back(m);
        dirty_stats_ = true;
        return m.seq_id;
    }

    void clear() {
        entries_.clear();
        stats_ = {};
        dirty_stats_ = false;
    }

    const std::vector<SequenceMetadata>& entries() const { return entries_; }

    std::optional<SequenceMetadata> get(uint32_t seq_id) const {
        if (seq_id >= entries_.size()) return std::nullopt;
        return entries_[seq_id];
    }

    SequenceLengthStats stats() {
        if (dirty_stats_) recomputeStats();
        return stats_;
    }

private:
    void recomputeStats() {
        stats_ = {};
        if (entries_.empty()) {
            dirty_stats_ = false;
            return;
        }
        stats_.count = static_cast<uint32_t>(entries_.size());
        std::vector<uint32_t> lens;
        lens.reserve(entries_.size());
        for (const auto& e : entries_) {
            lens.push_back(e.seq_length);
            stats_.min_len = (lens.size() == 1) ? e.seq_length : std::min(stats_.min_len, e.seq_length);
            stats_.max_len = (lens.size() == 1) ? e.seq_length : std::max(stats_.max_len, e.seq_length);
            stats_.mean_len += static_cast<double>(e.seq_length);
        }
        stats_.mean_len /= static_cast<double>(entries_.size());
        std::sort(lens.begin(), lens.end());
        auto pct = [&](double p) -> uint32_t {
            if (lens.empty()) return 0;
            const double idx = p * (lens.size() - 1);
            const size_t lo = static_cast<size_t>(idx);
            const size_t hi = std::min(lens.size() - 1, lo + 1);
            const double alpha = idx - static_cast<double>(lo);
            return static_cast<uint32_t>(static_cast<double>(lens[lo]) * (1.0 - alpha) +
                                         static_cast<double>(lens[hi]) * alpha);
        };
        stats_.p50 = pct(0.50);
        stats_.p90 = pct(0.90);
        stats_.p99 = pct(0.99);
        dirty_stats_ = false;
    }

    std::vector<SequenceMetadata> entries_;
    SequenceLengthStats stats_{};
    bool dirty_stats_ = false;
};

// Greedy packer to group seq_ids into batches under a token budget and batch size cap.
BatchPlan planBatches(const Catalog& catalog,
                      uint32_t max_tokens_per_batch,
                      uint32_t max_batch_size = 32,
                      bool prefer_short_first = false);

} // namespace GRIM::DynaSeq

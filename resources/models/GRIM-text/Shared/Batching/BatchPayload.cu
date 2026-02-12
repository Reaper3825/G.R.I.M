//======================================================//
//  BatchPayload.cu
//  Builder implementation for BatchPayload
//
//  Merges logic from:
//    - Phase2_TrainingLoop.cu::processBatch() sequence extraction (lines 2700-2730)
//    - ComputeLossBatch.cu::prepareLossBatchInputs() padding/masking (lines 52-193)
//    - Token-normalized_clipping.cu::computeBatchTokenStats()
//
//  All metadata computed ONCE here. No downstream recomputation.
//  Rule 20: No fallbacks, crash on any inconsistency.
//======================================================//

#include "BatchPayload.hpp"
#include "Batching_GPU.hpp"
#include "../../training/training_data_loader.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Batching {

BatchPayload buildBatchPayload(
    const BatchAssignment& assignment,
    const std::vector<TrainingSequence*>& views,
    int vocab_size,
    size_t max_cached_batch,
    size_t max_cached_seq_len)
{
    BatchPayload payload;

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 1: Identity — carry forward from assignment
    // ═════════════════════════════════════════════════════════════════════════
    payload.seq_ids = assignment.seq_ids;
    payload.accumulation_group = assignment.accumulation_group;
    payload.batch_size = static_cast<int>(assignment.seq_ids.size());

    if (payload.batch_size <= 0) {
        throw std::runtime_error(
            "buildBatchPayload: batch has 0 sequences — scheduler produced empty batch");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 2: Extract sequences and compute geometry in one pass
    // ═════════════════════════════════════════════════════════════════════════

    // Temporary ragged storage — we extract once, then flatten into padded layout
    struct RawSeq {
        const std::vector<int>* token_ids;
        const std::vector<int>* targets;
        const std::vector<float>* numeric_values;
        const std::vector<uint8_t>* numeric_mask;
        const std::vector<uint16_t>* text_features;
        const std::vector<uint8_t>* text_mask;
        int length;
    };

    std::vector<RawSeq> raw;
    raw.reserve(payload.batch_size);

    payload.seq_lengths.resize(payload.batch_size);
    payload.max_seq_len = 0;
    payload.actual_tokens = 0;

    // Token stats (replaces computeBatchTokenStats)
    payload.token_stats.batch_size = payload.batch_size;
    payload.token_stats.total_tokens = 0;
    payload.token_stats.max_sequence_length = 0;

    for (int b = 0; b < payload.batch_size; ++b) {
        const uint32_t sid = payload.seq_ids[b];

        // Rule 20: crash if view is out of bounds or null
        if (sid >= views.size()) {
            throw std::runtime_error(
                "buildBatchPayload: seq_id=" + std::to_string(sid) +
                " exceeds views.size()=" + std::to_string(views.size()));
        }
        const TrainingSequence* seq = views[sid];
        if (!seq) {
            throw std::runtime_error(
                "buildBatchPayload: views[" + std::to_string(sid) + "] is NULL");
        }

        const int seq_len = static_cast<int>(seq->token_ids.size());
        if (seq_len <= 0) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) + " has 0 tokens");
        }

        // Validate alignment between all per-token arrays
        if (static_cast<int>(seq->targets.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " token_ids.size()=" + std::to_string(seq_len) +
                " != targets.size()=" + std::to_string(seq->targets.size()));
        }
        if (static_cast<int>(seq->token_numeric_values.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " numeric_values.size()=" + std::to_string(seq->token_numeric_values.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }
        if (static_cast<int>(seq->token_numeric_mask.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " numeric_mask.size()=" + std::to_string(seq->token_numeric_mask.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }
        if (static_cast<int>(seq->token_text_mask.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " text_mask.size()=" + std::to_string(seq->token_text_mask.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }
        const int expected_text_feat_len = seq_len * BatchPayload::kTextFeatureDim;
        if (static_cast<int>(seq->token_text_features.size()) != expected_text_feat_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " text_features.size()=" + std::to_string(seq->token_text_features.size()) +
                " != token_ids.size()*kTextFeatureDim=" + std::to_string(expected_text_feat_len));
        }

        // Validate targets for invalid token IDs (root cause of loss=165)
        for (int t = 0; t < seq_len; ++t) {
            const int tid = seq->targets[t];
            if (tid >= 0 && tid >= vocab_size) {
                throw std::runtime_error(
                    "buildBatchPayload: sequence " + std::to_string(sid) +
                    " has invalid target token " + std::to_string(tid) +
                    " at position " + std::to_string(t) +
                    " (vocab_size=" + std::to_string(vocab_size) + ")");
            }
        }

        raw.push_back({
            &seq->token_ids,
            &seq->targets,
            &seq->token_numeric_values,
            &seq->token_numeric_mask,
            &seq->token_text_features,
            &seq->token_text_mask,
            seq_len
        });

        payload.seq_lengths[b] = seq_len;
        payload.max_seq_len = std::max(payload.max_seq_len, seq_len);
        payload.actual_tokens += seq_len;

        // Token stats accumulation
        payload.token_stats.total_tokens += seq_len;
        payload.token_stats.max_sequence_length = std::max(
            payload.token_stats.max_sequence_length, seq_len);
    }

    // Derived geometry
    payload.total_tokens = payload.batch_size * payload.max_seq_len;
    payload.padding_tokens = payload.total_tokens - payload.actual_tokens;
    payload.packing_efficiency = (payload.total_tokens > 0)
        ? static_cast<float>(payload.actual_tokens) / static_cast<float>(payload.total_tokens)
        : 0.0f;

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 3: Cache fit check
    // ═════════════════════════════════════════════════════════════════════════
    payload.fits_in_cache =
        (static_cast<size_t>(payload.batch_size) <= max_cached_batch) &&
        (static_cast<size_t>(payload.max_seq_len) <= max_cached_seq_len);

    if (!payload.fits_in_cache) {
        const char* reason = (static_cast<size_t>(payload.batch_size) > max_cached_batch)
            ? "BATCH_SIZE" : "SEQ_LEN";
        fprintf(stderr,
            "[buildBatchPayload] FATAL: batch does not fit cache (%s): "
            "batch=%d [limit=%zu], max_seq_len=%d [limit=%zu]\n",
            reason,
            payload.batch_size, max_cached_batch,
            payload.max_seq_len, max_cached_seq_len);
        throw std::runtime_error(
            "buildBatchPayload: batch does not fit GPU cache (" +
            std::string(reason) + "): batch=" + std::to_string(payload.batch_size) +
            " limit=" + std::to_string(max_cached_batch) +
            ", seq_len=" + std::to_string(payload.max_seq_len) +
            " limit=" + std::to_string(max_cached_seq_len));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 4: Pad all arrays to flat [batch_size * max_seq_len] layout
    //          and count valid targets — SINGLE PASS
    // ═════════════════════════════════════════════════════════════════════════
    const int S = payload.max_seq_len;
    const size_t flat_size = static_cast<size_t>(payload.total_tokens);
    const size_t text_feat_flat_size = flat_size * BatchPayload::kTextFeatureDim;

    payload.input_ids.assign(flat_size, 0);
    payload.target_ids.assign(flat_size, -1);  // padding targets = masked
    payload.numeric_values.assign(flat_size, 0.0f);
    payload.numeric_mask.assign(flat_size, 0);
    payload.text_features.assign(text_feat_flat_size, 0);
    payload.text_mask.assign(flat_size, 0);
    payload.valid_target_counts.resize(payload.batch_size, 0);

    payload.valid_tokens = 0;

    for (int b = 0; b < payload.batch_size; ++b) {
        const auto& r = raw[b];
        const int seq_len = r.length;
        const size_t row_offset = static_cast<size_t>(b) * S;

        // Copy token IDs
        for (int t = 0; t < seq_len; ++t) {
            payload.input_ids[row_offset + t] = (*r.token_ids)[t];
        }

        // Copy targets and count valid (with final-position masking)
        // ASSUMPTION: Strictly autoregressive (next-token prediction) training.
        // The final position in each window has no valid next-token target,
        // so we mask it with -1 to exclude from loss and gradients.
        int valid_count = 0;
        for (int t = 0; t < seq_len; ++t) {
            int target = (*r.targets)[t];
            // Mask the final position of each sequence (autoregressive boundary)
            if (t == seq_len - 1) {
                target = -1;
            }
            payload.target_ids[row_offset + t] = target;
            if (target >= 0) {
                ++valid_count;
            }
        }
        // Padding positions beyond seq_len already have target=-1 from assign()

        payload.valid_target_counts[b] = valid_count;
        payload.valid_tokens += valid_count;

        // Copy numeric side-channels
        for (int t = 0; t < seq_len; ++t) {
            payload.numeric_values[row_offset + t] = (*r.numeric_values)[t];
            payload.numeric_mask[row_offset + t] = (*r.numeric_mask)[t];
        }

        // Copy text features (kTextFeatureDim values per token)
        for (int t = 0; t < seq_len; ++t) {
            const size_t dst_offset = (row_offset + t) * BatchPayload::kTextFeatureDim;
            const size_t src_offset = static_cast<size_t>(t) * BatchPayload::kTextFeatureDim;
            for (int f = 0; f < BatchPayload::kTextFeatureDim; ++f) {
                payload.text_features[dst_offset + f] = (*r.text_features)[src_offset + f];
            }
            payload.text_mask[row_offset + t] = (*r.text_mask)[t];
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 5: Final validation
    // ═════════════════════════════════════════════════════════════════════════
    if (payload.valid_tokens <= 0) {
        std::ostringstream msg;
        msg << "buildBatchPayload: valid_tokens=0 after padding. "
            << "batch_size=" << payload.batch_size
            << " max_seq_len=" << payload.max_seq_len
            << " total_tokens=" << payload.total_tokens
            << " seq_ids=[";
        for (size_t i = 0; i < payload.seq_ids.size(); ++i) {
            msg << payload.seq_ids[i];
            if (i + 1 < payload.seq_ids.size()) msg << ",";
        }
        msg << "] — all targets are masked (-1), batch has nothing to train on";
        throw std::runtime_error(msg.str());
    }

    // Cross-check geometry invariants (Rule 20: crash if anything is wrong)
    payload.validate("buildBatchPayload");

    return payload;
}

}  // namespace Batching
}  // namespace GRIM

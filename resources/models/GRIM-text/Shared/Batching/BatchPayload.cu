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
#include "../../Shared/UnigramByte/UniByte.hpp"  // TokenLayout

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Batching {

BatchPayload buildBatchPayload(
    const BatchAssignment& assignment,
    const std::vector<TrainingSequence*>& views,
    int vocab_size,
    const GRIM::Tokenizer::TokenLayout& token_layout,
    size_t max_cached_batch,
    size_t max_cached_seq_len)
{
    BatchPayload payload;

    // ═════════════════════════════════════════════════════════════════════════
    // PHASE 1: Identity — carry forward from assignment
    // ═════════════════════════════════════════════════════════════════════════
    payload.seq_ids = assignment.seq_ids;
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
        const std::vector<uint16_t>* text_features;
        const std::vector<uint8_t>* atom_mask;
        const std::vector<uint32_t>* atom_flags;
        std::shared_ptr<const GRIM::Tokenizer::AtomTable> atom_table;
        const std::vector<uint32_t>* atom_entry_ids;
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
        if (static_cast<int>(seq->token_atom_mask.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " atom_mask.size()=" + std::to_string(seq->token_atom_mask.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }
        const int expected_text_feat_len = seq_len * BatchPayload::kTextFeatureDim;
        if (static_cast<int>(seq->token_text_features.size()) != expected_text_feat_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " text_features.size()=" + std::to_string(seq->token_text_features.size()) +
                " != token_ids.size()*kTextFeatureDim=" + std::to_string(expected_text_feat_len));
        }
        if (static_cast<int>(seq->atom_entry_ids.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " atom_entry_ids.size()=" + std::to_string(seq->atom_entry_ids.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
        }
        if (static_cast<int>(seq->token_atom_flags.size()) != seq_len) {
            throw std::runtime_error(
                "buildBatchPayload: sequence " + std::to_string(sid) +
                " token_atom_flags.size()=" + std::to_string(seq->token_atom_flags.size()) +
                " != token_ids.size()=" + std::to_string(seq_len));
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
            &seq->token_text_features,
            &seq->token_atom_mask,
            &seq->token_atom_flags,
            seq->atom_table,
            &seq->atom_entry_ids,
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

    payload.input_ids.assign(flat_size, Tokenizer::PAD_TOKEN_ID);  // PAD=1, NOT UNK=0
    payload.target_ids.assign(flat_size, -1);  // padding targets = masked
    payload.numeric_values.assign(flat_size, 0.0f);
    payload.text_features.assign(text_feat_flat_size, 0);
    payload.atom_mask.assign(flat_size, 0);
    payload.atom_flags.assign(flat_size, 0);
    payload.atom_entry_ids.assign(flat_size, GRIM::Tokenizer::kAtomEntryNone);
    payload.seq_atom_tables.resize(payload.batch_size);
    payload.valid_target_counts.resize(payload.batch_size, 0);

    payload.valid_tokens = 0;

    for (int b = 0; b < payload.batch_size; ++b) {
        const auto& r = raw[b];
        const int seq_len = r.length;
        const size_t row_offset = static_cast<size_t>(b) * S;

        // Bulk copy token IDs (contiguous source → contiguous destination row)
        std::memcpy(&payload.input_ids[row_offset],
                    r.token_ids->data(),
                    seq_len * sizeof(int));

        // Copy targets with final-position masking (autoregressive boundary).
        // All tokens except the last get their target copied; the final position
        // is masked with -1 since there is no valid next-token target.
        if (seq_len > 1) {
            std::memcpy(&payload.target_ids[row_offset],
                        r.targets->data(),
                        (seq_len - 1) * sizeof(int));
        }
        // Mask the final position of the sequence
        payload.target_ids[row_offset + seq_len - 1] = -1;
        // Padding positions beyond seq_len already have target=-1 from assign()

        // Defense-mask non-content tokens and count valid targets — SINGLE PASS
        // isNonContent() = UNK, PAD, BOS (never valid prediction targets).
        // EOS IS a valid target — model must learn to predict end-of-sequence.
        // This is a safety net; DataLoader should already mask these.
        int valid_count = 0;
        for (int t = 0; t < seq_len - 1; ++t) {
            const int target = payload.target_ids[row_offset + t];
            if (target >= 0 && token_layout.isNonContent(target)) {
                // Non-content token leaked through DataLoader — mask it
                payload.target_ids[row_offset + t] = -1;
            } else if (target >= 0) {
                ++valid_count;
            }
        }

        payload.valid_target_counts[b] = valid_count;
        payload.valid_tokens += valid_count;

        // Bulk copy numeric values
        std::memcpy(&payload.numeric_values[row_offset],
                    r.numeric_values->data(),
                    seq_len * sizeof(float));

        // Bulk copy text features (kTextFeatureDim uint16_t values per token)
        const size_t feat_dst_offset = row_offset * BatchPayload::kTextFeatureDim;
        const size_t feat_src_bytes  = static_cast<size_t>(seq_len) * BatchPayload::kTextFeatureDim * sizeof(uint16_t);
        std::memcpy(&payload.text_features[feat_dst_offset],
                    r.text_features->data(),
                    feat_src_bytes);

        // Bulk copy atom mask
        std::memcpy(&payload.atom_mask[row_offset],
                    r.atom_mask->data(),
                    seq_len * sizeof(uint8_t));

        // Bulk copy atom flags (type-specific metadata from AtomTable)
        std::memcpy(&payload.atom_flags[row_offset],
                    r.atom_flags->data(),
                    seq_len * sizeof(uint32_t));

        // Copy atom entry IDs (bulk memcpy — fixed-size uint32_t)
        std::memcpy(&payload.atom_entry_ids[row_offset],
                    r.atom_entry_ids->data(),
                    seq_len * sizeof(uint32_t));
        // Store AtomTable reference for this batch row
        payload.seq_atom_tables[b] = r.atom_table;
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

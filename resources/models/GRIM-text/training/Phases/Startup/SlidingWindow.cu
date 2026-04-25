//======================================================//
//  Startup/SlidingWindow.cu
//
//  Implementation of applySlidingWindows. Pure CPU code, but lives
//  in a .cu so the train_gpu translation unit list stays uniform
//  (mirrors Startup/Logging.cu and Startup/Rng.cu).
//======================================================//

#include "SlidingWindow.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace GRIMText::Training {

void applySlidingWindows(std::vector<TrainingSequence>& sequences,
                         const std::string& split_name,
                         int max_seq_len,
                         int sliding_window_stride,
                         bool add_bos_token,
                         int bos_id,
                         int eos_id,
                         TrainingLogger& logger) {
    std::vector<TrainingSequence> windowed;
    windowed.reserve(sequences.size());

    size_t long_seq_count = 0;
    size_t generated_windows = 0;
    size_t bos_prepended = 0;

    // Final-position autoregressive boundary mask. Required by
    // BatchPayload's Rule 20 invariant.
    auto MaskFinalTarget = [](TrainingSequence& seq) {
        if (!seq.targets.empty()) {
            seq.targets.back() = -1;
        }
    };

    for (const auto& seq : sequences) {
        if (static_cast<int>(seq.token_ids.size()) <= max_seq_len) {
            // Short sequence or exactly max_seq_len — no windowing.
            TrainingSequence copy = seq;
            MaskFinalTarget(copy);
            windowed.push_back(std::move(copy));
            continue;
        }

        // Execution-active rows MUST NOT be fragmented — compiled_bootstrap_bindings
        // and teacher_steps are whole-sequence structures with no windowing semantics.
        if (seq.execution_active) {
            throw std::runtime_error(
                "Execution-active sequence exceeds max_seq_len (" +
                std::to_string(seq.token_ids.size()) + " > " +
                std::to_string(max_seq_len) +
                "). Execution rows cannot be split by sliding window. "
                "Increase max_seq_len or shorten the source data.");
        }

        long_seq_count++;
        const size_t seq_len = seq.token_ids.size();
        size_t start = 0;
        const size_t stride = static_cast<size_t>(sliding_window_stride);
        bool is_first_window = true;
        size_t prev_source_end = 0;  // Track previous window's source end for overlap masking

        while (start < seq_len) {
            // Reserve 1 token for BOS if this is not the first window and BOS is enabled
            const bool prepend_bos = !is_first_window && add_bos_token && bos_id >= 0;
            const size_t effective_max = (is_first_window || !prepend_bos)
                ? static_cast<size_t>(max_seq_len)
                : static_cast<size_t>(max_seq_len - 1);
            size_t end = std::min(seq_len, start + effective_max);

            TrainingSequence window;

            // For non-first windows, prepend BOS token (gated on add_bos_token config)
            if (prepend_bos) {
                window.token_ids.push_back(bos_id);
                window.targets.push_back(-1);  // BOS position masked
                window.token_numeric_values.push_back(0.0f);
                window.token_atom_mask.push_back(0);
                window.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
                window.token_atom_flags.push_back(0);
                for (int i = 0; i < GRIM::Tokenizer::kTextFeatureDim; ++i) {
                    window.token_text_features.push_back(0);
                }
                if (!seq.token_exec_slots.empty())
                    window.token_exec_slots.push_back(static_cast<int32_t>(-1));
                if (!seq.slot_selection_targets.empty())
                    window.slot_selection_targets.push_back(
                        GRIM::Execution::SlotSelectionTarget{GRIM::Execution::SlotSelectionTargetKind::Ignore, -1});
                bos_prepended++;
            }

            // Copy window content
            window.token_ids.insert(window.token_ids.end(),
                seq.token_ids.begin() + start, seq.token_ids.begin() + end);
            window.targets.insert(window.targets.end(),
                seq.targets.begin() + start, seq.targets.begin() + end);
            window.token_numeric_values.insert(window.token_numeric_values.end(),
                seq.token_numeric_values.begin() + start, seq.token_numeric_values.begin() + end);
            window.token_atom_mask.insert(window.token_atom_mask.end(),
                seq.token_atom_mask.begin() + start, seq.token_atom_mask.begin() + end);
            // Atom side channel — share parent sequence's AtomTable
            window.atom_table = seq.atom_table;
            window.atom_entry_ids.insert(window.atom_entry_ids.end(),
                seq.atom_entry_ids.begin() + start, seq.atom_entry_ids.begin() + end);
            window.token_atom_flags.insert(window.token_atom_flags.end(),
                seq.token_atom_flags.begin() + start, seq.token_atom_flags.begin() + end);
            // GRMT v4: slice text features (16 values per token)
            window.token_text_features.insert(window.token_text_features.end(),
                seq.token_text_features.begin() + start * GRIM::Tokenizer::kTextFeatureDim,
                seq.token_text_features.begin() + end * GRIM::Tokenizer::kTextFeatureDim);

            if (!seq.token_exec_slots.empty()) {
                window.token_exec_slots.insert(window.token_exec_slots.end(),
                    seq.token_exec_slots.begin() + static_cast<ptrdiff_t>(start),
                    seq.token_exec_slots.begin() + static_cast<ptrdiff_t>(end));
            }
            if (!seq.slot_selection_targets.empty()) {
                window.slot_selection_targets.insert(window.slot_selection_targets.end(),
                    seq.slot_selection_targets.begin() + static_cast<ptrdiff_t>(start),
                    seq.slot_selection_targets.begin() + static_cast<ptrdiff_t>(end));
            }


            // Mask first position if it's the first window (BOS already there)
            // For non-first windows, BOS was prepended above with target=-1
            if (is_first_window && !window.targets.empty()) {
                window.targets[0] = -1;  // Mask BOS position
            }

            // Issue #143: Mask overlap prefix targets in non-first windows.
            // With stride < max_seq_len, the first (prev_source_end - start)
            // tokens were already trained on in the previous window. Mask them
            // to prevent double-training on the same targets.
            //
            // Issue #147: Subtract 1 from overlap_len. The position at
            // (prev_source_end - 1) was the LAST position in the previous
            // window, which was already  masked there (last-position mask).
            // Its target was NEVER trained. If we mask it here too, we create
            // a one-token training gap at every window boundary. By reducing
            // overlap by 1, this window trains that target instead.
            if (!is_first_window && prev_source_end > start) {
                const size_t raw_overlap = prev_source_end - start;
                const size_t overlap_len = (raw_overlap > 0) ? (raw_overlap - 1) : 0;
                const size_t bos_offset = prepend_bos ? 1 : 0;  // Skip BOS (already masked)
                for (size_t i = bos_offset; i < bos_offset + overlap_len && i < window.targets.size(); ++i) {
                    window.targets[i] = -1;
                }
            }

            // Mask last position for window boundary.
            // BatchPayload requires targets.back() == -1 unconditionally;
            // val no longer gets a special-cased "keep the final target"
            // path because BatchPayload would have masked it anyway.
            if (!window.targets.empty()) {
                window.targets.back() = -1;
            }

            // Issue #146: Inject EOS at end of non-final windows.
            // Without this, ~55% of training windows have NO EOS token,
            // so the model never learns when to stop generating.
            // The last position is already target-masked above, so replacing
            // its token_id with EOS costs nothing — the model sees EOS as
            // input and the second-to-last position learns to predict EOS.
            const bool is_final_window = (end == seq_len);
            if (!is_final_window && eos_id >= 0 && !window.token_ids.empty()) {
                window.token_ids.back() = eos_id;
                window.token_numeric_values.back() = 0.0f;
                window.token_atom_mask.back() = 0;
                if (!window.token_exec_slots.empty())
                    window.token_exec_slots.back() = static_cast<int32_t>(-1);
                if (!window.slot_selection_targets.empty())
                    window.slot_selection_targets.back() =
                        GRIM::Execution::SlotSelectionTarget{GRIM::Execution::SlotSelectionTargetKind::Ignore, -1};
                // Clear text features for the replaced position
                const size_t last_tf_start = (window.token_ids.size() - 1) * GRIM::Tokenizer::kTextFeatureDim;
                for (int i = 0; i < GRIM::Tokenizer::kTextFeatureDim; ++i) {
                    window.token_text_features[last_tf_start + i] = 0;
                }
                // Second-to-last position learns to predict EOS
                if (window.targets.size() >= 2) {
                    window.targets[window.targets.size() - 2] = eos_id;
                }
            }

            // Variable-length window — BatchPayload owns padding.
            windowed.push_back(std::move(window));
            generated_windows++;

            prev_source_end = end;  // Track for overlap masking in next window
            if (end == seq_len) break;
            start += stride;
            is_first_window = false;
        }
    }

    sequences = std::move(windowed);
    if (long_seq_count > 0) {
        logger.log("Sliding window (" + split_name + "): " +
                   std::to_string(long_seq_count) + " long sequences expanded into " +
                   std::to_string(generated_windows) + " windows" +
                   " (BOS prepended to " + std::to_string(bos_prepended) + " mid-sequence windows)");
    }
}

} // namespace GRIMText::Training

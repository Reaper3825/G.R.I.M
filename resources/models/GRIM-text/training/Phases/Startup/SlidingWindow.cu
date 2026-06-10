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

void injectBoundaryTokens(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                          bool add_bos_token,
                          bool add_eos_token,
                          size_t& added_bos_out,
                          size_t& added_eos_out) {
    added_bos_out = 0;
    added_eos_out = 0;

    for (auto& seq : sequences) {
        if (seq.token_ids.empty()) continue;

        // Add BOS if missing at start (controlled by config flag add_bos_token)
        if (add_bos_token && seq.token_ids.front() != GRIM::Tokenizer::BOS_TOKEN_ID) {
            const int old_first_token = seq.token_ids.front();
            seq.token_ids.insert(seq.token_ids.begin(), GRIM::Tokenizer::BOS_TOKEN_ID);
            seq.token_numeric_values.insert(seq.token_numeric_values.begin(), 0.0f);
            seq.token_atom_mask.insert(seq.token_atom_mask.begin(), 0);
            seq.atom_entry_ids.insert(seq.atom_entry_ids.begin(), GRIM::Tokenizer::kAtomEntryNone);
            seq.token_atom_flags.insert(seq.token_atom_flags.begin(), 0);
            // BOS predicts the first real token. This is valid LM supervision;
            // do not mask it away just because the input token is BOS.
            seq.targets.insert(seq.targets.begin(), old_first_token);
            if (!seq.token_exec_slots.empty())
                seq.token_exec_slots.insert(seq.token_exec_slots.begin(), static_cast<int32_t>(-1));
            // BOS insertion shifted all existing token positions right by 1.
            // Remap compiled_bootstrap_bindings token_pos to match.
            for (auto& b : seq.compiled_bootstrap_bindings)
                b.token_pos += 1;
            added_bos_out++;
        }

        // Add EOS if missing at end (controlled by config flag add_eos_token)
        if (add_eos_token && seq.token_ids.back() != GRIM::Tokenizer::EOS_TOKEN_ID) {
            seq.token_ids.push_back(GRIM::Tokenizer::EOS_TOKEN_ID);
            seq.token_numeric_values.push_back(0.0f);
            seq.token_atom_mask.push_back(0);
            seq.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
            seq.token_atom_flags.push_back(0);
            // Fix shift: the PREVIOUS position's target was -1 (no next token existed
            // when DataLoader ran). Now EOS follows it, so set target = EOS.
            if (!seq.targets.empty()) {
                seq.targets.back() = GRIM::Tokenizer::EOS_TOKEN_ID;  // position before EOS → predict EOS
            }
            seq.targets.push_back(-1);  // EOS position itself: nothing follows
            if (!seq.token_exec_slots.empty())
                seq.token_exec_slots.push_back(static_cast<int32_t>(-1));
            added_eos_out++;
        }
    }
}

void applySlidingWindows(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                         const std::string& split_name,
                         int max_seq_len,
                         int sliding_window_stride,
                         int min_seq_valid_tokens,
                         bool add_bos_token,
                         bool add_eos_token,
                         TrainingLogger& logger) {
    // Bracket sequences with BOS/EOS before windowing so window math sees
    // fully-bracketed input. Per-split summary is emitted here so each
    // train/val pass reports its own boundary-injection count.
    size_t added_bos = 0;
    size_t added_eos = 0;
    injectBoundaryTokens(sequences, add_bos_token, add_eos_token, added_bos, added_eos);
    if (added_bos > 0 || added_eos > 0) {
        logger.log("[Data] Boundary tokens (" + split_name + "): added_bos=" +
                   std::to_string(added_bos) + " added_eos=" + std::to_string(added_eos));
    }

    std::vector<GRIM::TokenizerArtifacts::GrmtSequence> windowed;
    windowed.reserve(sequences.size());

    size_t long_seq_count = 0;
    size_t generated_windows = 0;
    size_t bos_prepended = 0;

    // Final-position autoregressive boundary mask. Required by
    // BatchPayload's Rule 20 invariant.
    auto MaskFinalTarget = [](GRIM::TokenizerArtifacts::GrmtSequence& seq) {
        if (!seq.targets.empty()) {
            seq.targets.back() = -1;
        }
    };

    for (const auto& seq : sequences) {
        if (static_cast<int>(seq.token_ids.size()) <= max_seq_len) {
            // Short sequence or exactly max_seq_len — no windowing.
            GRIM::TokenizerArtifacts::GrmtSequence copy = seq;
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
            const bool prepend_bos = !is_first_window && add_bos_token;
            const size_t effective_max = (is_first_window || !prepend_bos)
                ? static_cast<size_t>(max_seq_len)
                : static_cast<size_t>(max_seq_len - 1);
            size_t end = std::min(seq_len, start + effective_max);

            GRIM::TokenizerArtifacts::GrmtSequence window;

            // For non-first windows, prepend BOS token (gated on add_bos_token config)
            if (prepend_bos) {
                window.token_ids.push_back(GRIM::Tokenizer::BOS_TOKEN_ID);
                // BOS predicts the first token of this local window.
                window.targets.push_back(seq.token_ids[start]);
                window.token_numeric_values.push_back(0.0f);
                window.token_atom_mask.push_back(0);
                window.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
                window.token_atom_flags.push_back(0);
                if (!seq.token_exec_slots.empty())
                    window.token_exec_slots.push_back(static_cast<int32_t>(-1));
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

            if (!seq.token_exec_slots.empty()) {
                window.token_exec_slots.insert(window.token_exec_slots.end(),
                    seq.token_exec_slots.begin() + static_cast<ptrdiff_t>(start),
                    seq.token_exec_slots.begin() + static_cast<ptrdiff_t>(end));
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
                // If we prepended a synthetic BOS for this local window, keep its
                // BOS→first-token supervision and only mask true overlapped source
                // positions after it.
                const size_t bos_offset = prepend_bos ? 1 : 0;
                for (size_t i = bos_offset; i < bos_offset + overlap_len && i < window.targets.size(); ++i) {
                    window.targets[i] = -1;
                }
            }

            // Mask last position for window boundary.
            // BatchPayload requires targets.back() == -1 unconditionally;
            // val no longer gets a special-cased "keep the final target"
            // path because BatchPayload would have masked it anyway. Window
            // boundaries stay as pure masked truncations; do not synthesize
            // EOS tokens at non-final window ends.
            if (!window.targets.empty()) {
                window.targets.back() = -1;
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

    // Post-window cleanup. Both filters exist because windowing + BatchPayload
    // masking can leave sequences that downstream code can't consume.
    filterOverlongSequences(sequences, split_name, max_seq_len, logger);
    filterShortSequences(sequences, split_name, min_seq_valid_tokens, logger);
}

void filterOverlongSequences(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                             const std::string& split_name,
                             int max_seq_len,
                             TrainingLogger& logger) {
    // HARD FILTER: Remove any sequences still exceeding max_seq_len after sliding window.
    // This catches cached .grmt files with old sequence lengths.
    const size_t before = sequences.size();
    sequences.erase(
        std::remove_if(sequences.begin(), sequences.end(),
            [max_seq_len](const GRIM::TokenizerArtifacts::GrmtSequence& seq) {
                return static_cast<int>(seq.token_ids.size()) > max_seq_len;
            }),
        sequences.end());
    const size_t removed = before - sequences.size();
    if (removed > 0) {
        logger.log("[FILTER] " + split_name + ": Removed " + std::to_string(removed) +
                   " sequences exceeding max_seq_len=" + std::to_string(max_seq_len));
    }
}

void filterShortSequences(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                          const std::string& split_name,
                          int min_seq_valid_tokens,
                          TrainingLogger& logger) {
    // HARD FILTER: Remove sequences with too few valid tokens after masking.
    // Prevents "valid_tokens=0" errors during loss computation.
    // Count the targets that are ALREADY authored as valid (>= 0); do not hardcode
    // positional assumptions like "index 0 is always BOS" because BOS insertion is
    // config-driven and BOS→first-token is valid supervision.
    if (min_seq_valid_tokens <= 0) return;
    const size_t before = sequences.size();
    sequences.erase(
        std::remove_if(sequences.begin(), sequences.end(),
            [min_seq_valid_tokens](const GRIM::TokenizerArtifacts::GrmtSequence& seq) {
                int valid = 0;
                for (size_t i = 0; i < seq.targets.size(); ++i) {
                    if (seq.targets[i] >= 0) valid++;
                }
                return valid < min_seq_valid_tokens;
            }),
        sequences.end());
    const size_t removed = before - sequences.size();
    if (removed > 0) {
        logger.log("[FILTER] " + split_name + ": Removed " + std::to_string(removed) +
                   " sequences with < " + std::to_string(min_seq_valid_tokens) + " valid tokens");
    }
}

} // namespace GRIMText::Training

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

namespace {

using GrmtSequence = GRIM::TokenizerArtifacts::GrmtSequence;

struct SftWindowConstruction {
    std::vector<GrmtSequence> sequences;
    size_t long_sequence_count = 0;
    size_t generated_window_count = 0;
};

void appendSftTokenRange(GrmtSequence& destination,
                         const GrmtSequence& source,
                         size_t begin,
                         size_t end) {
    destination.token_ids.insert(destination.token_ids.end(),
        source.token_ids.begin() + static_cast<ptrdiff_t>(begin),
        source.token_ids.begin() + static_cast<ptrdiff_t>(end));
    destination.targets.insert(destination.targets.end(),
        source.targets.begin() + static_cast<ptrdiff_t>(begin),
        source.targets.begin() + static_cast<ptrdiff_t>(end));
    destination.token_numeric_values.insert(destination.token_numeric_values.end(),
        source.token_numeric_values.begin() + static_cast<ptrdiff_t>(begin),
        source.token_numeric_values.begin() + static_cast<ptrdiff_t>(end));
    destination.token_atom_mask.insert(destination.token_atom_mask.end(),
        source.token_atom_mask.begin() + static_cast<ptrdiff_t>(begin),
        source.token_atom_mask.begin() + static_cast<ptrdiff_t>(end));
    destination.atom_entry_ids.insert(destination.atom_entry_ids.end(),
        source.atom_entry_ids.begin() + static_cast<ptrdiff_t>(begin),
        source.atom_entry_ids.begin() + static_cast<ptrdiff_t>(end));
    destination.token_atom_flags.insert(destination.token_atom_flags.end(),
        source.token_atom_flags.begin() + static_cast<ptrdiff_t>(begin),
        source.token_atom_flags.begin() + static_cast<ptrdiff_t>(end));
    if (!source.token_exec_slot_indices.empty()) {
        destination.token_exec_slot_indices.insert(destination.token_exec_slot_indices.end(),
            source.token_exec_slot_indices.begin() + static_cast<ptrdiff_t>(begin),
            source.token_exec_slot_indices.begin() + static_cast<ptrdiff_t>(end));
    }
}

SftWindowConstruction constructSftWindows(
    const std::vector<GrmtSequence>& sequences,
    const std::string& split_name,
    int max_seq_len,
    int sliding_window_stride) {
    SftWindowConstruction result;
    result.sequences.reserve(sequences.size());

    const size_t response_overlap = static_cast<size_t>(
        max_seq_len - sliding_window_stride);

    for (const auto& sequence : sequences) {
        const size_t sequence_length = sequence.token_ids.size();
        if (sequence.prompt_length <= 0 || sequence.prompt_end_pos < 0) {
            throw std::runtime_error(
                "Sliding window (" + split_name +
                "): training_stage=sft requires a non-empty prompt span on every sequence");
        }
        if (sequence.prompt_end_pos >= static_cast<int32_t>(sequence_length) ||
            sequence.prompt_end_pos - sequence.prompt_length + 1 < 0) {
            throw std::runtime_error(
                "Sliding window (" + split_name + "): invalid prompt span");
        }

        // Pin the entire prefix through prompt_end_pos. A configured BOS may
        // precede the logical prompt span and remains part of this prefix.
        const size_t prefix_length =
            static_cast<size_t>(sequence.prompt_end_pos) + 1;
        if (prefix_length >= sequence_length) {
            throw std::runtime_error(
                "Sliding window (" + split_name +
                "): training_stage=sft prompt leaves no response tokens");
        }
        if (prefix_length >= static_cast<size_t>(max_seq_len)) {
            throw std::runtime_error(
                "Sliding window (" + split_name + "): SFT prefix length=" +
                std::to_string(prefix_length) +
                " leaves no response capacity within max_seq_len=" +
                std::to_string(max_seq_len));
        }

        const size_t response_length = sequence_length - prefix_length;
        const size_t response_capacity =
            static_cast<size_t>(max_seq_len) - prefix_length;
        const bool fragmented = response_length > response_capacity;
        if (fragmented && response_overlap >= response_capacity) {
            throw std::runtime_error(
                "Sliding window (" + split_name + "): SFT response capacity=" +
                std::to_string(response_capacity) +
                " cannot preserve configured response overlap=" +
                std::to_string(response_overlap));
        }
        if (fragmented &&
            (sequence.execution_active ||
             sequence.execution_gate_target != GRIM::Execution::ExecutionGateTarget::UNSUPERVISED)) {
            throw std::runtime_error(
                "Execution-control-supervised SFT sequence exceeds max_seq_len; "
                "execution-control rows cannot be fragmented");
        }

        const size_t response_hop = fragmented
            ? response_capacity - response_overlap
            : response_capacity;
        size_t response_begin = 0;
        size_t previous_response_end = 0;
        bool first_window = true;
        if (fragmented) {
            ++result.long_sequence_count;
        }

        while (response_begin < response_length) {
            const size_t response_end =
                std::min(response_length, response_begin + response_capacity);
            const size_t owned_response_begin = first_window
                ? 0
                : previous_response_end;

            GrmtSequence window;
            if (!fragmented) {
                window = sequence;
            } else {
                window.atom_table = sequence.atom_table;
                appendSftTokenRange(window, sequence, 0, prefix_length);
                appendSftTokenRange(
                    window,
                    sequence,
                    prefix_length + response_begin,
                    prefix_length + response_end);
                window.prompt_length = sequence.prompt_length;
                window.prompt_end_pos = sequence.prompt_end_pos;
            }

            // Visible prompt/overlap context is not supervision. Re-author the
            // targets so each newly owned response token is predicted once.
            std::fill(window.targets.begin(), window.targets.end(), -1);
            for (size_t response_token = owned_response_begin;
                 response_token < response_end;
                 ++response_token) {
                const size_t source_target_position =
                    prefix_length + response_token - 1;
                const size_t local_token_position =
                    prefix_length + response_token - response_begin;
                window.targets[local_token_position - 1] =
                    sequence.targets[source_target_position];
            }
            window.targets.back() = -1;

            result.sequences.push_back(std::move(window));
            if (fragmented) {
                ++result.generated_window_count;
            }

            previous_response_end = response_end;
            if (response_end == response_length) {
                break;
            }
            response_begin += response_hop;
            first_window = false;
        }
    }

    return result;
}

} // namespace

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
            if (!seq.token_exec_slot_indices.empty())
                seq.token_exec_slot_indices.insert(seq.token_exec_slot_indices.begin(), static_cast<int32_t>(-1));
            // BOS insertion shifted all existing token positions right by 1.
            // Remap compiled_bootstrap_bindings token_pos to match.
            for (auto& b : seq.compiled_bootstrap_bindings)
                b.token_pos += 1;
            if (seq.prompt_end_pos >= 0) {
                seq.prompt_end_pos += 1;
            }
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
            if (!seq.token_exec_slot_indices.empty())
                seq.token_exec_slot_indices.push_back(static_cast<int32_t>(-1));
            added_eos_out++;
        }
    }
}

void applySlidingWindows(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                         const std::string& split_name,
                         GRIM::HyperParameters::TrainingStage training_stage,
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

    if (training_stage == GRIM::HyperParameters::TrainingStage::SFT) {
        auto construction = constructSftWindows(
            sequences, split_name, max_seq_len, sliding_window_stride);
        sequences = std::move(construction.sequences);
        if (construction.long_sequence_count > 0) {
            logger.log("Sliding window (" + split_name + "): " +
                       std::to_string(construction.long_sequence_count) +
                       " long SFT sequences expanded into " +
                       std::to_string(construction.generated_window_count) +
                       " prompt-pinned response windows");
        }
        filterOverlongSequences(sequences, split_name, max_seq_len, logger);
        filterShortSequences(sequences, split_name, min_seq_valid_tokens, logger);
        return;
    }
    if (training_stage != GRIM::HyperParameters::TrainingStage::PT) {
        throw std::runtime_error(
            "Sliding window (" + split_name + "): training_stage=" +
            std::string(GRIM::HyperParameters::trainingStageToJsonString(training_stage)) +
            " has no authored window-construction policy");
    }

    // PT owns the full document stream. Prompt metadata, if present in a
    // reused artifact, must not change PT windowing or downstream pooling.
    for (auto& sequence : sequences) {
        sequence.prompt_length = 0;
        sequence.prompt_end_pos = -1;
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
        // and transition_targets are whole-sequence structures with no windowing semantics.
        if (seq.execution_active ||
            seq.execution_gate_target != GRIM::Execution::ExecutionGateTarget::UNSUPERVISED) {
            throw std::runtime_error(
                "Execution-control-supervised sequence exceeds max_seq_len (" +
                std::to_string(seq.token_ids.size()) + " > " +
                std::to_string(max_seq_len) +
                "). Execution-control rows cannot be split by sliding window. "
                "Increase max_seq_len or shorten the source data.");
        }

        long_seq_count++;
        const size_t seq_len = seq.token_ids.size();
        const bool has_prompt_span = seq.prompt_length > 0;
        size_t prompt_start = 0;
        size_t prompt_end = 0;  // exclusive
        if (has_prompt_span) {
            if (seq.prompt_end_pos < 0 ||
                seq.prompt_end_pos >= static_cast<int32_t>(seq_len)) {
                throw std::runtime_error(
                    "Sliding window (" + split_name + "): invalid prompt_end_pos=" +
                    std::to_string(seq.prompt_end_pos) + " for sequence length=" +
                    std::to_string(seq_len));
            }
            const int32_t derived_start =
                seq.prompt_end_pos - seq.prompt_length + 1;
            if (derived_start < 0) {
                throw std::runtime_error(
                    "Sliding window (" + split_name + "): prompt span extends before sequence start");
            }
            prompt_start = static_cast<size_t>(derived_start);
            prompt_end = static_cast<size_t>(seq.prompt_end_pos) + 1;
        }
        bool prompt_span_assigned = !has_prompt_span;
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
                if (!seq.token_exec_slot_indices.empty())
                    window.token_exec_slot_indices.push_back(static_cast<int32_t>(-1));
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

            if (!seq.token_exec_slot_indices.empty()) {
                window.token_exec_slot_indices.insert(window.token_exec_slot_indices.end(),
                    seq.token_exec_slot_indices.begin() + static_cast<ptrdiff_t>(start),
                    seq.token_exec_slot_indices.begin() + static_cast<ptrdiff_t>(end));
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
            // Logical prompt delimiters are side-channel positions, not token
            // IDs. Preserve the complete prompt on the first window that owns
            // it; partial overlaps must not masquerade as a complete prompt.
            if (!prompt_span_assigned &&
                prompt_start >= start && prompt_end <= end) {
                const size_t local_offset = prepend_bos ? 1 : 0;
                const size_t local_prompt_start =
                    local_offset + (prompt_start - start);
                const size_t local_prompt_end =
                    local_offset + (prompt_end - start);
                window.prompt_length = static_cast<int32_t>(
                    local_prompt_end - local_prompt_start);
                window.prompt_end_pos = static_cast<int32_t>(local_prompt_end - 1);
                prompt_span_assigned = true;
            }

            windowed.push_back(std::move(window));
            generated_windows++;

            prev_source_end = end;  // Track for overlap masking in next window
            if (end == seq_len) break;
            start += stride;
            is_first_window = false;
        }

        if (!prompt_span_assigned) {
            throw std::runtime_error(
                "Sliding window (" + split_name + "): no window contains the complete prompt span; "
                "increase max_seq_len or shorten the prompt");
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

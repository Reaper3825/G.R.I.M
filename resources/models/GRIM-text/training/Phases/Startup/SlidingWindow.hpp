#pragma once
//======================================================//
//  Startup/SlidingWindow.hpp
//
//  Sliding-window expansion for long training sequences.
//  Extracted from Phase1_Startup::loadTrainingData.
//
//  Padding ownership: BatchPayload is the SINGLE owner of padded
//  [batch_size * max_seq_len] layout. This pass produces variable-length
//  TrainingSequences (length in [1, max_seq_len]) and does NOT pre-pad.
//
//  Final-target contract: BatchPayload asserts targets.back() == -1
//  (no autoregressive supervision at the sequence boundary). This pass
//  always masks the final target before handing the sequence off.
//======================================================//

#include "../../training_data_loader.hpp"   // TrainingSequence
#include "../../training_logger.hpp"        // TrainingLogger

#include <string>
#include <vector>

namespace GRIMText::Training {

// Ensure each sequence starts with BOS and ends with EOS (when enabled).
// Mutates each sequence in place across every parallel side-channel
// (token_ids / targets / numeric_values / atom_mask / atom_entry_ids /
// atom_flags / exec_slots / slot_selection_targets) and
// re-maps compiled_bootstrap_bindings token_pos for the BOS shift.
//
// Called by applySlidingWindows before any windowing happens, so all
// downstream window math sees fully-bracketed sequences.
//
// Args:
//   sequences      - in/out: sequences to bracket
//   add_bos_token  - if true, prepend BOS when missing
//   add_eos_token  - if true, append EOS when missing
//   bos_id         - BOS token id (negative => skip BOS)
//   eos_id         - EOS token id (negative => skip EOS)
//   added_bos_out  - out: number of sequences that received a BOS
//   added_eos_out  - out: number of sequences that received an EOS
void injectBoundaryTokens(std::vector<TrainingSequence>& sequences,
                          bool add_bos_token,
                          bool add_eos_token,
                          int bos_id,
                          int eos_id,
                          size_t& added_bos_out,
                          size_t& added_eos_out);

// Drop sequences whose post-window length still exceeds max_seq_len.
// Defends against stale .grmt caches where the on-disk sequence length
// no longer matches the active config. applySlidingWindows calls this
// internally after windowing.
void filterOverlongSequences(std::vector<TrainingSequence>& sequences,
                             const std::string& split_name,
                             int max_seq_len,
                             TrainingLogger& logger);

// Drop sequences with fewer than `min_seq_valid_tokens` non-ignored
// targets, where "valid" mirrors BatchPayload's masking: position 0
// (BOS) and the final position (autoregressive boundary) are excluded.
// applySlidingWindows calls this internally after windowing so callers
// never see a sequence that would trigger "valid_tokens=0" downstream.
// Disabled (no-op) when min_seq_valid_tokens <= 0.
void filterShortSequences(std::vector<TrainingSequence>& sequences,
                          const std::string& split_name,
                          int min_seq_valid_tokens,
                          TrainingLogger& logger);

// Expand any sequences longer than max_seq_len into a series of
// overlapping windows. Sequences <= max_seq_len pass through unchanged
// except for the final-position target mask.
//
// Calls injectBoundaryTokens internally before windowing, then calls
// filterOverlongSequences and filterShortSequences after windowing.
// Callers do not need to bracket or filter sequences themselves.
//
// In-place: `sequences` is replaced with the windowed result.
//
// Args:
//   sequences            - in/out: sequences to window (mutated in place)
//   split_name           - "train" / "val", used only for log lines
//   max_seq_len          - maximum window length
//   sliding_window_stride - hop size between windows; usually < max_seq_len
//   min_seq_valid_tokens - minimum unmasked targets per output sequence
//                          (<= 0 disables the short-sequence filter)
//   add_bos_token        - if true, prepend BOS (boundary + non-first windows)
//   add_eos_token        - if true, append EOS at sequence end
//   bos_id               - BOS token id (negative => disabled)
//   eos_id               - EOS token id (negative => skip non-final EOS injection)
//   logger               - emits per-split summary lines
void applySlidingWindows(std::vector<TrainingSequence>& sequences,
                         const std::string& split_name,
                         int max_seq_len,
                         int sliding_window_stride,
                         int min_seq_valid_tokens,
                         bool add_bos_token,
                         bool add_eos_token,
                         int bos_id,
                         int eos_id,
                         TrainingLogger& logger);

} // namespace GRIMText::Training

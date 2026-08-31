#pragma once
//======================================================//
//  Startup/SlidingWindow.hpp
//
//  Sliding-window expansion for long training sequences.
//  Called after LoadTrainingData reads raw GRMT rows.
//
//  Padding ownership: BatchPayload is the SINGLE owner of padded
//  [batch_size * max_seq_len] layout. This pass produces variable-length
//  GrmtSequence rows (length in [1, max_seq_len]) and does NOT pre-pad.
//
//  Final-target contract: BatchPayload asserts targets.back() == -1
//  (no autoregressive supervision at the sequence boundary). This pass
//  always masks the final target before handing the sequence off.
//======================================================//

#include "../../../Shared/TokenizerArtifacts/GrmtSequence.hpp"  // GRIM::TokenizerArtifacts::GrmtSequence
#include "../../../Shared/HyperParameters/HyperparameterEnums.hpp"
#include "../../training_logger.hpp"        // TrainingLogger

#include <cstddef>
#include <string>
#include <vector>

namespace GRIMText::Training {

// Ensure each sequence starts with BOS and ends with EOS (when enabled).
// Mutates each sequence in place across every parallel side-channel
// (token_ids / targets / numeric_values / atom_mask / atom_entry_ids /
// local_atom_indices / atom_flags / exec_slots) and
// re-maps compiled_bootstrap_bindings token_pos for the BOS shift.
//
// Called by applySlidingWindows before any windowing happens, so all
// downstream window math sees fully-bracketed sequences.
//
// Args:
//   sequences      - in/out: sequences to bracket
//   add_bos_token  - if true, prepend BOS when missing
//   add_eos_token  - if true, append EOS when missing
//   added_bos_out  - out: number of sequences that received a BOS
//   added_eos_out  - out: number of sequences that received an EOS
void injectBoundaryTokens(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                          bool add_bos_token,
                          bool add_eos_token,
                          size_t& added_bos_out,
                          size_t& added_eos_out);

// Drop sequences whose post-window length still exceeds max_seq_len.
// Defends against stale .grmt caches where the on-disk sequence length
// no longer matches the active config. applySlidingWindows calls this
// internally after windowing.
void filterOverlongSequences(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                             const std::string& split_name,
                             int max_seq_len,
                             TrainingLogger& logger);

// Drop sequences with fewer than `min_seq_valid_tokens` non-ignored
// targets. "Valid" means the target value is already authored as a real
// token ID (>= 0) after windowing/boundary logic has run; callers must not
// assume position 0 is always masked because BOS insertion is config-driven
// and BOS→first-token supervision is valid when BOS is present.
// applySlidingWindows calls this internally after windowing so callers
// never see a sequence that would trigger "valid_tokens=0" downstream.
// Disabled (no-op) when min_seq_valid_tokens <= 0.
void filterShortSequences(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                          const std::string& split_name,
                          int min_seq_valid_tokens,
                          TrainingLogger& logger);

// Expand any sequences longer than max_seq_len into a series of
// overlapping windows. Typed atom spans are indivisible: every source and
// output row is checked for matched, same-type boundaries, and a window cut
// is moved to the opening boundary when it would split a span. An atom that
// cannot fit in the available window capacity is rejected. Sequences <=
// max_seq_len pass through unchanged except for the final-position target
// mask. Every finalized output row receives token_atom_aux_target_mask in
// causal prediction coordinates: opening through final value row = 1, typed
// closing-boundary row = 0. BatchPayload uses this channel to suppress LM
// targets inside complete typed spans independently of any reconstruction head.
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
//   training_stage       - selects PT document windows or SFT prompt-pinned windows
//   max_seq_len          - maximum window length
//   sliding_window_stride - hop size between windows; usually < max_seq_len
//   min_seq_valid_tokens - minimum unmasked targets per output sequence
//                          (<= 0 disables the short-sequence filter)
//   add_bos_token        - if true, prepend BOS (boundary + non-first windows)
//   add_eos_token        - if true, append EOS at sequence end
//   logger               - emits per-split summary lines
void applySlidingWindows(std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& sequences,
                         const std::string& split_name,
                         GRIM::HyperParameters::TrainingStage training_stage,
                         int max_seq_len,
                         int sliding_window_stride,
                         int min_seq_valid_tokens,
                         bool add_bos_token,
                         bool add_eos_token,
                         TrainingLogger& logger);

} // namespace GRIMText::Training

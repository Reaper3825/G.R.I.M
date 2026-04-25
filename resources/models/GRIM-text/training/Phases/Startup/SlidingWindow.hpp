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

// Expand any sequences longer than max_seq_len into a series of
// overlapping windows. Sequences <= max_seq_len pass through unchanged
// except for the final-position target mask.
//
// In-place: `sequences` is replaced with the windowed result.
//
// Args:
//   sequences            - in/out: sequences to window (mutated in place)
//   split_name           - "train" / "val", used only for the summary log line
//   max_seq_len          - maximum window length
//   sliding_window_stride - hop size between windows; usually < max_seq_len
//   add_bos_token        - if true, prepend BOS to non-first windows
//   bos_id               - BOS token id (negative => disabled)
//   eos_id               - EOS token id (negative => skip non-final EOS injection)
//   logger               - emits the per-split summary line
void applySlidingWindows(std::vector<TrainingSequence>& sequences,
                         const std::string& split_name,
                         int max_seq_len,
                         int sliding_window_stride,
                         bool add_bos_token,
                         int bos_id,
                         int eos_id,
                         TrainingLogger& logger);

} // namespace GRIMText::Training

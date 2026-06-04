#pragma once
//======================================================//
//  Startup/ClassBalancedWeights.hpp
//
//  Class-balanced loss weight precomputation.
//  Extracted from Phase1_Startup (formerly step "10b").
//
//  Counts per-vocab target frequencies across the training corpus,
//  computes w_v = 1 / freq(v)^β with unseen tokens clamped to the
//  rarest-seen weight, and uploads the resulting [vocab_size] table
//  into TrainingState::class_weights_tensor for use by AutogradLoss.
//
//  No-op for callers: the enabled-flag check lives at the call site
//  in Phase1; this header only exposes the "do the work" entry point.
//======================================================//

#include "../../../Shared/TokenizerArtifacts/GrmtSequence.hpp"  // GRIM::TokenizerArtifacts::GrmtSequence
#include "../../training_logger.hpp"        // TrainingLogger
#include "../../../Shared/TrainingState/TrainingState_GPU.hpp"  // GRIM::TrainingState

#include <cstdint>
#include <vector>

namespace GRIMText::Training {

// Compute class-balanced loss weights from training-target frequencies and
// upload them to TrainingState::class_weights_tensor.
//
// Side effects:
//   - Allocates ts.class_weights_tensor ([1, vocab_size] floats)
//   - cudaMemcpyAsync's the host weight table to that tensor
//   - Sets ts.class_weights_vocab_size = vocab_size
//   - cudaStreamSynchronize on ts.stream_ctrl.getPrimaryStream()
//   - Emits [CLASS_BALANCED] summary + top/bottom-10 weight lines via logger
//
// Throws std::runtime_error if the training data contains zero valid targets.
//
// Args:
//   train_seqs  - training corpus (read-only)
//   vocab_size  - active vocab size; targets outside [0, vocab_size) are skipped
//   beta        - class-balanced exponent (w_v = 1 / freq(v)^β)
//   ts          - training state receiving class_weights_tensor
//   logger      - destination for [CLASS_BALANCED] log lines
void computeAndUploadClassBalancedWeights(
    const std::vector<GRIM::TokenizerArtifacts::GrmtSequence>& train_seqs,
    std::uint32_t vocab_size,
    float beta,
    GRIM::TrainingState& ts,
    TrainingLogger& logger);

} // namespace GRIMText::Training

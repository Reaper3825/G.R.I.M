#pragma once
//======================================================//
//  DiagnosticInference.hpp
//  Isolated inference sampling for training diagnostics
//======================================================//
//
//  PURPOSE
//  =======
//  Houses the training-time inference sample generator
//  in complete isolation from the training loop.
//  This code MUST NEVER modify shared training state
//  (weight tensors, requires_grad flags, optimizer state).
//
//  Generation uses either the KV decode path or the full-context prefill path,
//  depending on whether the active model geometry is sequence-local. Both paths
//  keep inference isolated from model weight tensors and optimizer state.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "../Phases/Phase1_Startup.hpp"
#include "../Phases/Phase2_TrainingLoop.hpp"

namespace GRIMText::Training {

/// Run a diagnostic inference sample if the current optimizer step
/// matches the configured sample interval.  This function is fully
/// self-contained: it rotates through twelve single-operation arithmetic probes
/// with empty persisted state and the curriculum's generic goal, then sends the
/// prefill payload through executePhase2PayloadInference() and logs only its
/// generated continuation. Atom-insertion models use the raw-text classifier path.
/// GRIM_SAMPLE_PROMPT overrides the probe; GRIM_SAMPLE_TOKENS (default 256) and
/// GRIM_SAMPLE_MAX_CHARS (default 2048) control generation and display limits.
/// generation_lens_enabled=true adds final-layer readouts and an identity-head replay.
/// generation_lens_validate=true additionally compares a capture-disabled greedy
/// baseline with repeated captured runs; see Shared/Lenses/README.md for controls.
///
/// SAFETY: This function does NOT modify any model weight tensors,
/// gradient buffers, or optimizer state. Inference paths use no training
/// backward pass and keep generation state separate from optimizer state.
void logDiagnosticSample(TrainingContext& ctx,
                         TrainingLoopState& state,
                         bool inference_diagnostic_enabled,
                         int inference_diagnostic_interval);

}  // namespace GRIMText::Training

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
/// self-contained: it sends a text prompt through executePhase2TextInference()
/// and logs the decoded result.
///
/// SAFETY: This function does NOT modify any model weight tensors,
/// gradient buffers, or optimizer state. Inference paths use no training
/// backward pass and keep generation state separate from optimizer state.
void logDiagnosticSample(TrainingContext& ctx, TrainingLoopState& state);

}  // namespace GRIMText::Training

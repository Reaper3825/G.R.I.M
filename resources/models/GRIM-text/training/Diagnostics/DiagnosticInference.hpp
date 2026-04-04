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
//  The decode forward path (executeDecodeForward_) uses
//  Tensor::detach() views of model weights so that
//  requires_grad=false is set on local non-owning copies,
//  not on the model's actual weight tensors.
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include "../Phases/Phase1_Startup.hpp"
#include "../Phases/Phase2_TrainingLoop.hpp"

namespace GRIMText::Training {

/// Run a diagnostic inference sample if the current optimizer step
/// matches the configured sample interval.  This function is fully
/// self-contained: it tokenizes a prompt, runs model.generate(),
/// decodes the output, and logs the result.
///
/// SAFETY: This function does NOT modify any model weight tensors,
/// gradient buffers, or requires_grad flags.  The underlying decode
/// path uses detach() views for all weight accesses.
void logDiagnosticSample(TrainingContext& ctx, TrainingLoopState& state);

}  // namespace GRIMText::Training

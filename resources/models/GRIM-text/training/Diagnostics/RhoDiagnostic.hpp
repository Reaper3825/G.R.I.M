#pragma once
//======================================================//
//  RhoDiagnostic.hpp
//  Per-Layer Hidden State Correlation Diagnostic
//======================================================//
//
//  PURPOSE
//  =======
//  Computes ρ(l) = avg|cos(h_i^l, h_j^l)| per encoder layer
//  to track where correlation builds through the encoder stack.
//  Extracted from Phase2_TrainingLoop.cu — called once per batch
//  on diagnostic sync intervals.
//
//  EQUATION (Rule 21):
//    ρ(l) = (1/P) Σ_{i<j} |cos(h_i^l, h_j^l)|
//    Δρ(l) = ρ(l) - ρ(l−1)
//
//  Author: Austin Wadkins
//  Date: April 2026
//======================================================//

#include <cstdint>

// Forward declarations — avoid pulling in heavy headers
namespace GRIMText::Training {
    struct TrainingContext;
}

namespace GRIM::Batching {
    struct BatchPayload;
}

namespace GRIM::Diagnostics {

/// Run the RHO_BUILDUP_EQUATION diagnostic once per batch.
///
/// Reads encoder layer outputs from autograd intermediates,
/// computes per-layer ρ and Δρ, writes to telemetry last_obs[5-8],
/// emits an EQ_LOG entry, and logs top-10 batch tokens.
///
/// @param ctx          Full training context (model, tokenizer, logging, telemetry)
/// @param payload      Current batch (for token frequency analysis)
/// @param batch_idx    Batch index within epoch (for logging)
void computeRhoDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx);

} // namespace GRIM::Diagnostics

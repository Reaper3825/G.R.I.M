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

#include "../../Shared/Batching/BatchPayload.hpp"

namespace GRIMText::Training {
    struct TrainingContext;
}

namespace GRIM::Forward {
    struct ModelForwardOutputs;
}

namespace GRIM::Diagnostics {

struct RhoDiagnosticRuntime {
    GRIMText::Training::TrainingContext* training_ctx = nullptr;
    int batch_idx = -1;
};

enum class RhoDiagnosticPhase {
    PostForwardPreBackward,
    PostBackward,
};

struct RhoDiagnosticOptions {
    RhoDiagnosticPhase phase = RhoDiagnosticPhase::PostBackward;
    bool write_telemetry = true;
    bool write_logs = true;
};

/// Run the RHO_BUILDUP_EQUATION diagnostic once per batch.
///
/// Reads encoder layer outputs from the active shared-forward sink,
/// computes per-layer ρ and Δρ, writes to telemetry last_obs[5-8],
/// optionally emits an EQ_LOG/logger entry containing the top-10 batch tokens.
///
/// Phase2 may invoke this helper at multiple points inside the active
/// forward/autograd boundary. The pre-backward emission is intended for
/// forward-vs-backward comparison and is typically log-only; the post-backward
/// emission remains the telemetry-writing path.
///
/// @param ctx          Full training context (model, tokenizer, logging, telemetry)
/// @param payload      Current batch (for token frequency analysis)
/// @param batch_idx    Batch index within epoch (for logging)
/// @param options      Phase/tag selection and independent telemetry/log output controls
void computeRhoDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    const GRIM::Forward::ModelForwardOutputs& forward_outputs,
    int batch_idx,
    const RhoDiagnosticOptions& options = {});

} // namespace GRIM::Diagnostics

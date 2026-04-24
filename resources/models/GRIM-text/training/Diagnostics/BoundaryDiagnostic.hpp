#pragma once
//======================================================//
//  BoundaryDiagnostic.hpp
//  Boundary-crossing check (max_seq_len, training-cache
//  capacity, position-embedding bounds, token-id sanity).
//
//  Lifted verbatim from Phase2_TrainingLoop.cu.
//======================================================//

#include "../../Shared/Batching/BatchPayload.hpp"

namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

void runBoundaryDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx);

} // namespace GRIM::Diagnostics

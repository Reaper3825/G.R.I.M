#pragma once
//======================================================//
//  SpecialTokenDiagnostic.hpp
//  Issue #142 / Rule 21: Special Token Weight & Gradient
//  Verification (UNK/PAD/BOS/EOS row health vs content
//  baseline). Lifted verbatim from Phase2_TrainingLoop.cu.
//======================================================//

#include "../../Shared/Batching/BatchPayload.hpp"

namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

void runSpecialTokenDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    const GRIM::Batching::BatchPayload& payload,
    int batch_idx);

} // namespace GRIM::Diagnostics

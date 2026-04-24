//======================================================//
//  TieVerifyDiagnostic.hpp
//  Runtime tie_embeddings pointer verification.
//  Lifted verbatim from Phase2_TrainingLoop.cu.
//======================================================//

#pragma once

#include <cstddef>

namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

void runTieVerifyDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    std::size_t batch_idx);

} // namespace GRIM::Diagnostics

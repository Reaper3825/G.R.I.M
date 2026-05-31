//======================================================//
//  TieVerifyDiagnostic.hpp
//  Runtime tie_embeddings pointer verification.
//  Lifted verbatim from Phase2_TrainingLoop.cu.
//======================================================//

#pragma once

#include <cstddef>

#include "../Phases/Startup/Model/ParameterRegistry.hpp"

namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

void runTieVerifyDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    std::size_t batch_idx);

} // namespace GRIM::Diagnostics

//======================================================//
//  OptimizerMomentDiagnostic.hpp
//  Adam optimizer m/v moment-state telemetry.
//  Lifted verbatim from Phase2_TrainingLoop.cu —
//  the MomentSample struct, sampleOptimizerMomentStats
//  helper, and the [OptState] log block.
//======================================================//

#pragma once

#include <cstddef>

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Shared/Optimizers/OptimizerState_GPU.hpp"

namespace GRIMText { namespace Training { struct TrainingContext; } }

namespace GRIM::Diagnostics {

constexpr int kMomentSamplePerGroup = 4;

struct MomentSample {
    bool valid = false;
    float m_rms = 0.0f;
    float v_rms = 0.0f;
    std::size_t groups = 0;
    std::size_t samples = 0;
};

MomentSample sampleOptimizerMomentStats(const GRIM::OptimizerState& optimizer_state,
                                        cudaStream_t stream,
                                        bool sync_for_host = false);

void runOptimizerMomentDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    int batch_idx,
    int accumulation_window_micro_batches,
    bool sync_diag);

} // namespace GRIM::Diagnostics

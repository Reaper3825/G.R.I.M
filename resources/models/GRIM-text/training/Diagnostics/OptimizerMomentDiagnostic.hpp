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

MomentSample sampleOptimizerMomentStats(const GRIM::TrainingState& ts, bool sync_for_host = false);

void runOptimizerMomentDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    int batch_idx,
    int micro_step_for_log,
    int accum_steps_for_log,
    bool sync_diag);

} // namespace GRIM::Diagnostics

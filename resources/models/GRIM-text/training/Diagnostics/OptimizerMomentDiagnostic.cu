//======================================================//
//  OptimizerMomentDiagnostic.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  Adam moment-state sampler and the [OptState] log
//  block.
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline block.
//======================================================//

#include "OptimizerMomentDiagnostic.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace GRIM::Diagnostics {

MomentSample sampleOptimizerMomentStats(const GRIM::OptimizerState& optimizer_state,
                                        cudaStream_t stream,
                                        bool sync_for_host) {
    MomentSample sample{};
    if (!sync_for_host) {
        return sample;
    }
    if (stream == nullptr) {
        throw std::runtime_error("[sampleOptimizerMomentStats] stream is NULL - caller MUST provide valid CUDA stream");
    }
    const std::size_t group_count = std::min(optimizer_state.m_states.size(),
                                             optimizer_state.v_states.size());
    if (group_count == 0) {
        return sample;
    }

    std::vector<float> m_host(group_count * kMomentSamplePerGroup, 0.0f);
    std::vector<float> v_host(group_count * kMomentSamplePerGroup, 0.0f);
    std::vector<std::size_t> counts(group_count, 0);

    bool has_copy = false;
    for (std::size_t i = 0; i < group_count; ++i) {
        const float* m_ptr = optimizer_state.m_states[i].data;
        const float* v_ptr = optimizer_state.v_states[i].data;
        const std::size_t size = optimizer_state.m_states[i].numel();
        const std::size_t count = std::min<std::size_t>(kMomentSamplePerGroup, size);
        if (!m_ptr || !v_ptr || count == 0) {
            continue;
        }
        counts[i] = count;
        has_copy = true;
        cudaMemcpyAsync(m_host.data() + i * kMomentSamplePerGroup, m_ptr,
                        count * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(v_host.data() + i * kMomentSamplePerGroup, v_ptr,
                        count * sizeof(float), cudaMemcpyDeviceToHost, stream);
    }

    if (!has_copy) {
        return sample;
    }

    cudaStreamSynchronize(stream);

    double m_sum_sq = 0.0;
    double v_sum_sq = 0.0;
    std::size_t total_samples = 0;
    for (std::size_t i = 0; i < group_count; ++i) {
        const std::size_t count = counts[i];
        for (std::size_t j = 0; j < count; ++j) {
            const float m_val = m_host[i * kMomentSamplePerGroup + j];
            const float v_val = v_host[i * kMomentSamplePerGroup + j];
            m_sum_sq += static_cast<double>(m_val) * m_val;
            v_sum_sq += static_cast<double>(v_val) * v_val;
        }
        total_samples += count;
    }

    if (total_samples == 0) {
        return sample;
    }

    sample.m_rms = static_cast<float>(std::sqrt(m_sum_sq / total_samples));
    sample.v_rms = static_cast<float>(std::sqrt(v_sum_sq / total_samples));
    sample.groups = group_count;
    sample.samples = total_samples;
    sample.valid = true;
    return sample;
}

void runOptimizerMomentDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    int batch_idx,
    int accumulation_window_micro_batches,
    bool sync_diag)
{
    namespace Internal = ::GRIMText::Training::Internal;
    const auto& ts = ctx.model->getTrainingState();
    const auto moment_sample = sampleOptimizerMomentStats(ctx.optimizer.optimizer_state,
                                                         ts.stream_ctrl.getPrimaryStream(),
                                                         sync_diag);
    if (moment_sample.valid) {
        ctx.logging.logger->log("[OptState] batch=" + std::to_string(batch_idx + 1) +
                                " step=" + std::to_string(ctx.optimizer.optimizer_step.step) +
                                " accum_window=" + std::to_string(accumulation_window_micro_batches) +
                                " m_rms=" + Internal::formatScalar(moment_sample.m_rms, 10) +
                                " v_rms=" + Internal::formatScalar(moment_sample.v_rms, 10) +
                                " groups=" + std::to_string(moment_sample.groups) +
                                " samples=" + std::to_string(moment_sample.samples));
    }
}

} // namespace GRIM::Diagnostics

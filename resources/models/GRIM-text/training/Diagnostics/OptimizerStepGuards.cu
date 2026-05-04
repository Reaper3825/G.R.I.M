//======================================================//
//  OptimizerStepGuards.cu
//  Optimizer-owned finite checks around the step boundary.
//======================================================//

#include "OptimizerStepGuards.hpp"

#include <cuda_runtime.h>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace GRIM::Diagnostics {

// ────────────────────────────────────────────────────────────────────────────
// Rule 20: Post-optimizer weight NaN spot check
// ────────────────────────────────────────────────────────────────────────────
void checkPostOptimizerWeightsFinite(
    const GRIM::ParameterGroup* groups,
    std::size_t group_count,
    int optimizer_step,
    float learning_rate,
    int batch_idx)
{
    if (group_count == 0) {
        throw std::runtime_error("ParameterGroup count is zero in post-optimizer finite check");
    }

    if (!groups) {
        throw std::runtime_error("ParameterGroup array is NULL in post-optimizer finite check");
    }

    for (std::size_t g = 0; g < group_count; ++g) {
        const float* w = groups[g].weights();
        const std::size_t n = groups[g].size();

        if (!w || n == 0) continue;

        const std::size_t indices[] = {0, n / 2, n - 1};

        for (std::size_t idx : indices) {
            float h_sample = 0.0f;

            cudaError_t err = cudaMemcpy(
                &h_sample,
                w + idx,
                sizeof(float),
                cudaMemcpyDeviceToHost);

            if (err != cudaSuccess) {
                throw std::runtime_error(
                    "[FATAL] cudaMemcpy failed in post-optimizer finite check for group '" +
                    groups[g].name + "': " + cudaGetErrorString(err));
            }

            if (!std::isfinite(h_sample)) {
                throw std::runtime_error(
                    "[FATAL] Post-optimizer NaN/Inf detected in parameter group '" +
                    groups[g].name + "' at sampled index " + std::to_string(idx) +
                    " batch=" + std::to_string(batch_idx + 1) +
                    " optimizer_step=" + std::to_string(optimizer_step) +
                    " lr=" + std::to_string(learning_rate) +
                    ". Weight corruption was detected after this optimizer step.");
            }
        }
    }
}

} // namespace GRIM::Diagnostics

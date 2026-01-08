#include "GradientCC_Host_GPU.hpp"

#include "GradientCC_GPU.hpp"

#include <cmath>
#include <iostream>

namespace GRIM {

namespace {

bool hasBuffers(const std::vector<GradientBufferView>& buffers)
{
    for (const auto& view : buffers) {
        if (view.data != nullptr && view.length > 0) {
            return true;
        }
    }
    return false;
}

} // namespace

void clampGradientBuffers(const std::vector<GradientBufferView>& buffers,
                          float min_val,
                          float max_val,
                          cudaStream_t stream)
{
    if (!hasBuffers(buffers)) {
        return;
    }

    if (min_val > max_val) {
        std::cerr << "GradientCC Host: invalid clamp range (" << min_val
                  << ", " << max_val << ")" << std::endl;
        return;
    }

    for (const auto& view : buffers) {
        if (!view.data || view.length == 0) {
            continue;
        }

        launchClampGradients(
            view.data,
            static_cast<int>(view.length),
            min_val,
            max_val,
            stream);
    }
}

void clipGradientBuffers(const std::vector<GradientBufferView>& buffers,
                         float max_norm,
                         cudaStream_t stream)
{
    if (!hasBuffers(buffers)) {
        return;
    }

    if (max_norm <= 0.0f || !std::isfinite(max_norm)) {
        std::cerr << "GradientCC Host: invalid clip norm " << max_norm << std::endl;
        return;
    }

    for (const auto& view : buffers) {
        if (!view.data || view.length == 0) {
            continue;
        }

        launchGradientClipping(
            view.data,
            static_cast<int>(view.length),
            max_norm,
            stream);
    }
}

} // namespace GRIM

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace GRIM {

struct GradientBufferView {
    float* data = nullptr;
    size_t length = 0;
};

void clampGradientBuffers(const std::vector<GradientBufferView>& buffers,
                          float min_val,
                          float max_val,
                          cudaStream_t stream);

void clipGradientBuffers(const std::vector<GradientBufferView>& buffers,
                         float max_norm,
                         cudaStream_t stream);

} // namespace GRIM

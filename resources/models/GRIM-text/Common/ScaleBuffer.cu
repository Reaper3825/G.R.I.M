#define USE_CUDA

#include <cmath>
#include <cuda_runtime.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "grim_scale_buffer.hpp"

namespace GRIM {

#ifdef USE_CUDA

__global__ void scaleBufferKernel(float* data, size_t count, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] *= scale;
    }
}

void scaleDeviceBuffer(float* data, size_t count, float scale, cudaStream_t stream) {
    if (!data || count == 0) return;
    if (std::fabs(scale - 1.0f) < 1e-7f) return;
    int block = 256;
    int grid = static_cast<int>((count + block - 1) / block);
    scaleBufferKernel<<<grid, block, 0, stream>>>(data, count, scale);
    CUDA_CHECK(cudaGetLastError());
}

#endif // USE_CUDA

} // namespace GRIM

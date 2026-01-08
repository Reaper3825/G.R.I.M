#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>

namespace GRIM {

void scaleDeviceBuffer(float* data, size_t count, float scale, cudaStream_t stream);

} // namespace GRIM

#endif // USE_CUDA

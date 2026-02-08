#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>

namespace GRIM {

// Forward-declare Tensor (defined in TensorContract_GPU.hpp, also in GRIM namespace)
struct Tensor;

void scaleDeviceBuffer(float* data, size_t count, float scale, cudaStream_t stream);

// Tensor overload — scales tensor.data in-place
void scaleDeviceBuffer(Tensor& tensor, float scale, cudaStream_t stream);

// Tensor gradient overload — scales tensor.grad_data in-place (no-op if no grad)
void scaleGradBuffer(Tensor& tensor, float scale, cudaStream_t stream);

} // namespace GRIM

#endif // USE_CUDA

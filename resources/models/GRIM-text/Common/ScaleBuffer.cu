#define USE_CUDA

#include <cmath>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"
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
    if (!data) {
        throw std::runtime_error("scaleDeviceBuffer: data pointer is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (count == 0) {
        throw std::runtime_error("scaleDeviceBuffer: count is 0 - caller passed empty buffer at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (std::fabs(scale - 1.0f) < 1e-7f) return;  // Optimization: skip identity scaling
    int block = 256;
    int grid = static_cast<int>((count + block - 1) / block);
    scaleBufferKernel<<<grid, block, 0, stream>>>(data, count, scale);
    CUDA_CHECK(cudaGetLastError());
}

void scaleDeviceBuffer(Tensor& tensor, float scale, cudaStream_t stream) {
    if (!tensor.data) {
        throw std::runtime_error("scaleDeviceBuffer(Tensor): tensor.data is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (tensor.numel() == 0) {
        throw std::runtime_error("scaleDeviceBuffer(Tensor): tensor has 0 elements at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    scaleDeviceBuffer(tensor.data, tensor.numel(), scale, stream);
}

void scaleGradBuffer(Tensor& tensor, float scale, cudaStream_t stream) {
    if (!tensor.has_grad()) {
        throw std::runtime_error("scaleGradBuffer: tensor has no gradient buffer - caller must ensure gradient is allocated at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (tensor.numel() == 0) {
        throw std::runtime_error("scaleGradBuffer: tensor has 0 elements at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    scaleDeviceBuffer(tensor.grad_data(), tensor.numel(), scale, stream);
}

#endif // USE_CUDA

} // namespace GRIM

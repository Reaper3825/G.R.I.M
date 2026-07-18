//======================================================//
//  GradientAccumulation.cu
//  Single source of truth for TensorContract gradient accumulation.
//======================================================//

#include "GradientAccumulation.hpp"
#include "TensorContract_GPU.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

constexpr int AUTOGRAD_BLOCK_SIZE = 256;
constexpr int kMaxGridBlocks1DFallback = 65534;
constexpr int kMaxGridDimY = 65535;

inline int getMaxGridBlocks1D() {
    static int cached = -1;
    if (cached < 0) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cached = kMaxGridBlocks1DFallback;
        } else {
            int max_x = 0;
            if (cudaDeviceGetAttribute(&max_x, cudaDevAttrMaxGridDimX, device) != cudaSuccess) {
                cached = kMaxGridBlocks1DFallback;
            } else {
                cached = (max_x > 65534) ? 65534 : max_x;
            }
        }
    }
    return cached;
}

inline dim3 gridForCount(std::size_t count) {
    if (count == 0) {
        return dim3(1, 1, 1);
    }
    const int blocks = static_cast<int>((count + AUTOGRAD_BLOCK_SIZE - 1) / AUTOGRAD_BLOCK_SIZE);
    const int max1d = getMaxGridBlocks1D();
    if (blocks <= max1d) {
        return dim3(blocks, 1, 1);
    }
    const int gy = (blocks + max1d - 1) / max1d;
    if (gy <= kMaxGridDimY) {
        return dim3(max1d, gy, 1);
    }
    const int gx = (blocks + kMaxGridDimY - 1) / kMaxGridDimY;
    return dim3(gx, kMaxGridDimY, 1);
}

__global__ void kernel_accumulate_grad(float* dst, const float* src, std::size_t count, float scale) {
    const std::size_t block_idx = static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
    const std::size_t idx = block_idx * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] += src[idx] * scale;
    }
}

}  // anonymous namespace

namespace GRIM {
namespace autograd {

void accumulate_grad(Tensor& dst,
                     const Tensor& src,
                     float scale,
                     cudaStream_t stream,
                     const char* caller) {
    const char* context = caller ? caller : "autograd::accumulate_grad";
    dst.shape.require(context);
    src.shape.require(context);

    bool matching_shape = dst.shape.layout == src.shape.layout;
    if (matching_shape && dst.shape.is_2d_layout()) {
        matching_shape = dst.shape.as_2d() == src.shape.as_2d();
    }
    if (matching_shape && dst.shape.is_4d()) {
        matching_shape = dst.shape.as_4d() == src.shape.as_4d();
    }
    if (!matching_shape) {
        throw std::runtime_error(std::string(context) + ": tensor shape mismatch");
    }

    accumulate_grad(dst.data, src.data, dst.numel(), scale, stream, context);
}

void accumulate_grad(float* dst,
                     const float* src,
                     std::size_t count,
                     float scale,
                     cudaStream_t stream,
                     const char* caller) {
    const char* context = caller ? caller : "autograd::accumulate_grad";
    if (count == 0) {
        return;
    }
    if (!dst) {
        throw std::runtime_error(std::string(context) + ": dst is NULL");
    }
    if (!src) {
        throw std::runtime_error(std::string(context) + ": src is NULL");
    }
    if (stream == nullptr || stream == 0) {
        throw std::runtime_error(std::string(context) + ": stream is NULL - caller MUST provide valid stream");
    }
    if (!std::isfinite(scale)) {
        throw std::runtime_error(std::string(context) + ": scale is not finite");
    }

    kernel_accumulate_grad<<<gridForCount(count), AUTOGRAD_BLOCK_SIZE, 0, stream>>>(
        dst, src, count, scale);
}

}  // namespace autograd
}  // namespace GRIM

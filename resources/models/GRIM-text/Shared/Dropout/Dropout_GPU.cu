#include "Dropout_GPU.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace GRIM {
namespace {

__device__ inline std::uint64_t xorshift64(std::uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

__device__ inline float to_uniform(std::uint64_t x) {
    // Use top 24 bits → [0,1)
    return static_cast<float>(x >> 40) * (1.0f / static_cast<float>(1u << 24));
}

__global__ void dropoutKernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              int n,
                              float rate,
                              std::uint64_t seed,
                              float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Scrambled per-thread seed (prevents correlation)
    std::uint64_t state =
        seed ^ (0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(idx + 1));
    state = xorshift64(state);

    float r = to_uniform(state);
    float keep = (r >= rate) ? scale : 0.0f;
    output[idx] = input[idx] * keep;
}

} // namespace

void launchDropout(const float* input,
                   float* output,
                   int elements,
                   float rate,
                   std::uint64_t seed,
                   cudaStream_t stream) {
    if (!input || !output || elements <= 0) {
        return;
    }

    if (rate <= 0.0f) {
        if (input != output) {
            cudaMemcpyAsync(output, input,
                            static_cast<size_t>(elements) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
        return;
    }

    if (rate >= 1.0f) {
        cudaMemsetAsync(output, 0,
                        static_cast<size_t>(elements) * sizeof(float),
                        stream);
        return;
    }

    const float scale = 1.0f / (1.0f - rate);
    const int threads = 256;
    const int blocks = (elements + threads - 1) / threads;

    dropoutKernel<<<blocks, threads, 0, stream>>>(
        input, output, elements, rate, seed, scale);
}

} // namespace GRIM

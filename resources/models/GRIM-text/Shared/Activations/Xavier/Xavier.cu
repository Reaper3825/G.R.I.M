#include "Xavier.hpp"
#include "../../HyperParameters/HyperParameters_GPU.hpp"

#include <cmath>
#include <curand_kernel.h>

namespace GRIM {
namespace {

__device__ inline float computeUniformLimit(float fan_in, float fan_out) {
    const float denom = fan_in + fan_out;
    return denom > 0.0f ? sqrtf(6.0f / denom) : 0.0f;
}

__device__ inline float computeNormalStd(float fan_in, float fan_out) {
    const float denom = fan_in + fan_out;
    return denom > 0.0f ? sqrtf(2.0f / denom) : 0.0f;
}

__global__ void XavierUniformKernel(float* data,
                                    std::size_t elements,
                                    float fan_in,
                                    float fan_out,
                                    unsigned long long seed,
                                    unsigned long long subseq) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, subseq, &state);

    const float limit = computeUniformLimit(fan_in, fan_out);
    const float rnd = curand_uniform(&state) * 2.0f - 1.0f; // (-1, 1)
    data[idx] = rnd * limit;
}

__global__ void XavierNormalKernel(float* data,
                                   std::size_t elements,
                                   float fan_in,
                                   float fan_out,
                                   unsigned long long seed,
                                   unsigned long long subseq) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elements) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, subseq, &state);

    const float stddev = computeNormalStd(fan_in, fan_out);
    const float rnd = curand_normal(&state);
    data[idx] = rnd * stddev;
}

inline int computeGridSize(std::size_t elements, int block_size) {
    return static_cast<int>((elements + block_size - 1) / block_size);
}

inline bool validateArgs(const XavierInitArgs& args) {
    return args.data != nullptr && args.elements > 0;
}

} // namespace

void launchXavierUniform(const XavierInitArgs& args) {
    if (!validateArgs(args)) {
        return;
    }
    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid = computeGridSize(args.elements, kBlockSize);
    XavierUniformKernel<<<grid, kBlockSize, 0, args.stream>>>(
        args.data,
        args.elements,
        args.fan_in,
        args.fan_out,
        args.seed,
        args.subsequence);
}

void launchXavierNormal(const XavierInitArgs& args) {
    if (!validateArgs(args)) {
        return;
    }
    constexpr int kBlockSize = HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
    const int grid = computeGridSize(args.elements, kBlockSize);
    XavierNormalKernel<<<grid, kBlockSize, 0, args.stream>>>(
        args.data,
        args.elements,
        args.fan_in,
        args.fan_out,
        args.seed,
        args.subsequence);
}

} // namespace GRIM


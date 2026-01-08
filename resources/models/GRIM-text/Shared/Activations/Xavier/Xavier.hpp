#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace GRIM {

struct XavierConfig {
    std::size_t elements = 0;
    float fan_in = 1.0f;
    float fan_out = 1.0f;
    unsigned long long seed = 1337ULL;
    unsigned long long subsequence = 0ULL;
    cudaStream_t stream = nullptr;
};

struct XavierInitArgs {
    float* data = nullptr;
    std::size_t elements = 0;
    float fan_in = 1.0f;
    float fan_out = 1.0f;
    unsigned long long seed = 1337ULL;
    unsigned long long subsequence = 0ULL;
    cudaStream_t stream = nullptr;
};

void launchXavierUniform(const XavierInitArgs& args);
void launchXavierNormal(const XavierInitArgs& args);

inline void launchXavierInit(const XavierInitArgs& args, bool normal = false) {
    if (normal) {
        launchXavierNormal(args);
    } else {
        launchXavierUniform(args);
    }
}

//======================================================//
//  Convenience Overload (stddev-based initialization)
//  Converts stddev to fan_in/fan_out internally
//======================================================//
inline void launchXavierInit(float* weights, int size, float stddev,
                             unsigned int seed, cudaStream_t stream) {
    // Convert stddev to fan_in/fan_out: stddev = sqrt(2/(fan_in+fan_out))
    // If stddev = sqrt(2/N), then N = 2/stddev^2, so fan_in = fan_out = N/2 = 1/stddev^2
    const float fan = (stddev > 0.0f) ? (1.0f / (stddev * stddev)) : 1.0f;
    XavierInitArgs args{};
    args.data = weights;
    args.elements = static_cast<std::size_t>(size);
    args.fan_in = fan;
    args.fan_out = fan;
    args.seed = seed;
    args.subsequence = 0ULL;
    args.stream = stream;
    launchXavierNormal(args);  // Old code used normal distribution
}

} // namespace GRIM


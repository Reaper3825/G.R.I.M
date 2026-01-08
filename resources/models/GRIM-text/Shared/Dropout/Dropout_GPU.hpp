#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace GRIM {

// In-place or out-of-place inverted dropout on GPU.
// If output == input, operation is in-place.
// rate in [0,1). seed is a host-generated 64-bit value to decorrelate calls.
void launchDropout(const float* input,
                   float* output,
                   int elements,
                   float rate,
                   std::uint64_t seed,
                   cudaStream_t stream);

}  // namespace GRIM

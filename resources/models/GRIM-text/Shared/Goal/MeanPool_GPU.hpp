#pragma once

#include <cuda_runtime.h>

namespace GRIM {

namespace Batching {
struct BatchPayload;
}

namespace Forward {
struct ModelForwardOutputs;
}

void meanPoolHiddenStates(
    const Batching::BatchPayload& payload,
    Forward::ModelForwardOutputs& forward_outputs,
    cudaStream_t stream);

} // namespace GRIM

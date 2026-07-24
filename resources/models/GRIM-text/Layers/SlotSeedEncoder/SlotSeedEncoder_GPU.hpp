//======================================================//
//  SlotSeedEncoder_GPU.hpp
//  Contextual numeric-placeholder -> execution-slot seed
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

#include <cuda_runtime.h>

namespace GRIM {
namespace SlotSeedEncoder {

// Writes every live tensor directly into ModelForwardOutputs. MFO is the sole
// forward-graph owner; this layer creates no secondary output aggregate.
void forward(
    const HyperParameters::SlotSeedEncoderConstructionHP& hp,
    const SlotSeedEncoderParameterTensors& parameter_tensors,
    const Tensor& contextual_hidden_states,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int num_slots,
    cudaStream_t stream,
    Forward::ModelForwardOutputs& forward_outputs);

}  // namespace SlotSeedEncoder
}  // namespace GRIM

#endif  // USE_CUDA

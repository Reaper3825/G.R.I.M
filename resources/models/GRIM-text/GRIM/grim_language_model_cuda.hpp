//======================================================//
//  grim_language_model_cuda.hpp
//======================================================//

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <utility>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

// HyperParameters - single entry point for model configuration snapshot helpers
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"

// Hyperparameter groupings - construction/read views derived from AiConfigSnapshot
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"

// TensorContract - Autograd system (includes ParamGroupType, ParameterGroup, Tensor)
#include "../Shared/TensorContract/TensorContract_GPU.hpp"

// BatchPayload - Single source of truth for per-batch metadata
#include "../Shared/Batching/BatchPayload.hpp"

// BatchDeviceBindings - Explicit device pointers per step (replaces the old
// `mutable d_*` fields that used to live on BatchPayload).
#include "../Shared/Batching/BatchDeviceBindings.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"
#include "../Shared/TrainingState/TrainingState_GPU.hpp"
#include "../Shared/InferenceState/GenerationState_GPU.hpp"
#include "../training/Phases/Startup/Model/ModelGpuState.hpp"
#endif

namespace GRIM {

//======================================================//
//  Forward Declarations
//======================================================//

// Concrete encoder layer type lives in Layers/Encoding/Encoding_GPU.hpp.
// Forward declare it here for pointer-only container access; do not add a
// second alias owner in this header.
class EncodingLayer;

class GPUGrimEncoder;

//======================================================//
//  GPU Classes (Forward Declarations Only)
//======================================================//

#ifdef USE_CUDA
// GPUGrimEncoder - Container for encoder layers, manages layer lifecycle.
// Forward pass logic is in ForwardPhase2_Encoder.cu::runFullEncoder().
// This class only owns layers and provides access to them.
//
// Constructor takes the grouped encoder hyperparameter view plus explicit
// startup/model-assembly resources. Forward pass runtime handles live on the
// forward request.
class GPUGrimEncoder {
public:
    explicit GPUGrimEncoder(const HyperParameters::EncoderLayerConstructionHP& hp);

    EncodingLayer* getLayer(int index);
    const EncodingLayer* getLayer(int index) const;

private:
    struct Impl;
    Impl* pImpl = nullptr;
};

#endif // USE_CUDA

} // namespace GRIM

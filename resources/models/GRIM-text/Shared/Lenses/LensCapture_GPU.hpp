#pragma once
#ifdef USE_CUDA
#include "LensMetadata.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"

namespace GRIM::Lenses {
// Called by the shared forward owner while input/logits/parameters are live.
// Returns only owned host data; synchronizes its copies before returning.
std::shared_ptr<const LensSnapshot> captureFinalLens(
    const LensCaptureRequest& request, LensCaptureMetadata metadata,
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Forward::ModelForwardOutputs& actual,
    cudaStream_t stream, cublasHandle_t handle);
std::uint64_t nextForwardInvocationId();
}
#endif

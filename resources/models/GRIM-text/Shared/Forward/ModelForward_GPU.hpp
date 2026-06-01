//======================================================//
//  ModelForward_GPU.hpp
//  Shared full-model forward primitive
//
//  Phase 2 of the inference/training split:
//  - Shared forward code takes a caller-authored graph-policy request.
//  - Training/autograd owns AutogradContext, loss, backward, optimizer.
//  - This boundary consumes explicit BatchDeviceBindings; it never
//    rediscovers the active step from TrainingState cache fields.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../Batching/BatchPayload.hpp"
#include "../Batching/BatchDeviceBindings.hpp"
#include "ModelForwardOutputs.hpp"
#include "ModelForwardRuntimePayload.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../PBM/PositionalBiasMethod.hpp"

namespace GRIM {

class ScratchBlockLayer;
class GPUGrimEncoder;

} // namespace GRIM

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM {

namespace Forward {

struct MTPHeadForwardView {
    Tensor weight;
    Tensor bias;
};

struct ModelForwardGraphPolicy {
    // Caller-authored graph policy. This is deliberately not a training-vs-inference
    // mode enum: orchestration chooses whether this forward call may connect
    // autograd edges to durable parameters and whether graph outputs must be
    // retained for a later backward owner. Optional forward products such as
    // MTP logits are requested explicitly here rather than inferred from
    // training-vs-inference identity.
    bool connect_parameter_graph = false;
    bool retain_backward_graph = false;
    bool enable_dropout = false;
    bool emit_mtp_logits = false;
};

struct ModelForwardRequest {
    const Config::AiConfigSnapshot* config = nullptr;
    GPUGrimEncoder* gpu_encoder = nullptr;
    ::ParameterRegistry::StartupParameterRegistry* parameter_registry = nullptr;
    const PBM::PBMState* pbm = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;

    ScratchBlockLayer* scratch_block = nullptr;
    bool execution_block_enabled = false;
    std::vector<MTPHeadForwardView> mtp_heads;

    const Batching::BatchPayload* payload = nullptr;
    const Batching::BatchDeviceBindings* bindings = nullptr;
    uint64_t batch_idx = 0;
    ModelForwardGraphPolicy graph{};

    void validate(const char* caller) const;
};

ModelForwardOutputs executeModelForward(const ModelForwardRequest& request,
                                        ModelForwardRuntimePayload& runtime_payload);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA

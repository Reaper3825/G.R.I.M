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

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../Batching/BatchPayload.hpp"
#include "../Batching/BatchDeviceBindings.hpp"
#include "ModelForwardRuntimePayload.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM {

class EmbeddingLayer;
class LMHeadLayer;
class ScratchBlockLayer;
class ReasoningHeadLayer;
class ExecutionBlockLayer;
class GPUGrimEncoder;

namespace Forward {

struct ModelForwardGraphPolicy {
    // Caller-authored graph policy. This is deliberately not a training-vs-inference
    // mode enum: orchestration chooses whether this forward call may connect
    // autograd edges to durable parameters and whether graph outputs must be
    // retained for a later backward owner.
    bool connect_parameter_graph = false;
    bool retain_backward_graph = false;
    bool enable_dropout = false;
};

struct ModelForwardRequest {
    const HyperParameters::LanguageModelConfig* config = nullptr;
    GPUGrimEncoder* gpu_encoder = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;

    EmbeddingLayer* embedding_layer = nullptr;
    LMHeadLayer* lm_head = nullptr;
    ScratchBlockLayer* scratch_block = nullptr;
    ReasoningHeadLayer* reasoning_head = nullptr;
    ExecutionBlockLayer* execution_block = nullptr;

    const Batching::BatchPayload* payload = nullptr;
    const Batching::BatchDeviceBindings* bindings = nullptr;
    uint64_t batch_idx = 0;
    ModelForwardGraphPolicy graph{};

    void validate(const char* caller) const;
};

void executeModelForward(const ModelForwardRequest& request,
                         ModelForwardRuntimePayload& runtime_payload);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA

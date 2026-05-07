//======================================================//
//  ModelForward_GPU.hpp
//  Shared full-model forward primitive
//
//  Phase 2 of the inference/training split:
//  - Shared forward code takes a mode-explicit request.
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

namespace GRIM {

class EmbeddingLayer;
class LMHeadLayer;
class ScratchBlockLayer;
class ReasoningHeadLayer;
class ExecutionBlockLayer;
class GPUGrimEncoder;
struct TrainingState;

namespace HyperParameters {
struct LanguageModelConfig;
}

namespace Forward {

enum class ModelForwardMode {
    TrainingGraph,
    EvalNoGrad,
    InferencePrefill
};

struct ModelForwardResult {
    float* encoder_output = nullptr;
    int total_tokens = 0;
    int vocab_size = 0;
    bool success = false;
    std::string error_message;
};

struct ModelForwardRequest {
    const HyperParameters::LanguageModelConfig* config = nullptr;
    TrainingState* runtime_state = nullptr;
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
    int batch_size = 0;
    int seq_len = 0;
    uint64_t step = 0;
    ModelForwardMode mode = ModelForwardMode::TrainingGraph;

    bool trainingGraph() const { return mode == ModelForwardMode::TrainingGraph; }
    bool preservesLayerIntermediates() const { return mode == ModelForwardMode::InferencePrefill; }
    void validate(const char* caller) const;
};

ModelForwardResult executeModelForward(ModelForwardRequest& request);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA

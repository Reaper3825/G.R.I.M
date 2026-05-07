//======================================================//
//  InferenceForward_GPU.hpp
//  Explicit inference-prefill forward boundary
//
//  Phase 1 of the inference/training split:
//  - Inference does not enter AutogradContext.
//  - Callers provide an explicit BatchDeviceBindings view.
//  - TrainingState cache tensors may still back the view temporarily,
//    but they are not rediscovered by the forward primitive.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../Batching/BatchDeviceBindings.hpp"
#include "../MTP/MTPDiagnostics.hpp"

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

struct InferenceForwardResult {
    float* encoder_output = nullptr;
    int total_tokens = 0;
    int vocab_size = 0;
    bool success = false;
    std::string error_message;
};

struct InferenceForwardRequest {
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

    const Batching::BatchDeviceBindings* bindings = nullptr;
    int batch_size = 0;
    int seq_len = 0;
    uint64_t step = 0;

    void validate(const char* caller) const;
};

InferenceForwardResult executeInferencePrefillForward(InferenceForwardRequest& request);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA

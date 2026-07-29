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

class GPUGrimEncoder;
struct KvCacheState;

} // namespace GRIM

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM {

namespace Forward {

struct ModelForwardGraphPolicy {
    // Caller-authored graph policy. This is deliberately not a training-vs-inference
    // mode enum: orchestration chooses whether this forward call may connect
    // autograd edges to durable parameters.
    bool connect_parameter_graph = false;
    bool enable_dropout = false;
    // Arg/option selector head: when true, the forward encodes candidate atom-entry
    // keys and emits ModelForwardOutputs::selector_logits [total_tokens, num_pool_atoms].
    // Requires the NumberEncoder + selector to be enabled and a non-empty pool.
    bool emit_selector_logits = false;
};

struct ModelForwardRequest {
    const Config::AiConfigSnapshot* config = nullptr;
    GPUGrimEncoder* gpu_encoder = nullptr;
    ::ParameterRegistry::StartupParameterRegistry* parameter_registry = nullptr;
    const PBM::PBMState* pbm = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;

    bool execution_block_enabled = false;

    const Batching::BatchPayload* payload = nullptr;
    const Batching::BatchDeviceBindings* bindings = nullptr;
    uint64_t batch_idx = 0;
    ModelForwardGraphPolicy graph{};

    // Inference-only: when non-null, the encoder attention sublayers run the
    // KV-cache decode/prefill path over this session cache instead of full
    // self-attention. Requires connect_parameter_graph == false, batch_size == 1,
    // and dropout disabled.
    // ExecutionBlock inference may run during InferencePrefill; decode windows do
    // not re-bootstrap or re-execute it. Training callers leave this null.
    KvCacheState* kv_cache = nullptr;

    void validate(const char* caller) const;
};

ModelForwardOutputs executeModelForward(const ModelForwardRequest& request,
                                        ModelForwardRuntimePayload& runtime_payload);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA

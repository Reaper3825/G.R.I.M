//======================================================//
//  AutogradContext.cu
//  Input-context construction for autograd forward/loss/backward
//======================================================//

#include <stdexcept>
#include <string>

#include "AutogradTraining.hpp"

namespace GRIM {
namespace Autograd {

namespace {

void populateCommonContext(
    AutogradContext& ctx,
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    EmbeddingLayer* embedding_layer,
    LMHeadLayer* lm_head,
    ScratchBlockLayer* scratch_block,
    ReasoningHeadLayer* reasoning_head,
    ExecutionBlockLayer* execution_block,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    float grad_scale,
    uint64_t step,
    bool is_training)
{
    ctx.config = config;
    ctx.training_state = training_state;
    ctx.gpu_encoder = gpu_encoder;
    ctx.embedding_layer = embedding_layer;
    ctx.lm_head = lm_head;
    ctx.scratch_block = scratch_block;
    ctx.reasoning_head = reasoning_head;
    ctx.execution_block = execution_block;
    ctx.cublas_handle = cublas_handle;
    ctx.stream = stream;
    ctx.grad_scale = grad_scale;
    ctx.step = step;
    ctx.is_training = is_training;
}

void validateDeviceBindingsForPayload(
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const char* caller)
{
    payload.validate(caller);

    if (bindings.batch_size != payload.batch_size || bindings.max_seq_len != payload.max_seq_len) {
        throw std::runtime_error(
            std::string(caller) + ": BatchDeviceBindings geometry (" +
            std::to_string(bindings.batch_size) + "x" + std::to_string(bindings.max_seq_len) +
            ") does not match payload (" +
            std::to_string(payload.batch_size) + "x" + std::to_string(payload.max_seq_len) +
            ")");
    }
    if (!bindings.d_input_ids || !bindings.d_target_ids || !bindings.d_token_to_slot_map) {
        throw std::runtime_error(
            std::string(caller) + ": BatchDeviceBindings has NULL device pointers - "
            "caller must invoke model.uploadBatchToDevice(payload) before initializing autograd context");
    }
}

} // namespace

// Training overload — derives batch geometry from BatchPayload.
// `bindings` must describe the same batch (geometry-checked) and must already
// have been populated by uploadBatchToDevice(payload) at the H2D sync slice.
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    EmbeddingLayer* embedding_layer,
    LMHeadLayer* lm_head,
    ScratchBlockLayer* scratch_block,
    ReasoningHeadLayer* reasoning_head,
    ExecutionBlockLayer* execution_block,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    float grad_scale,
    uint64_t step,
    bool is_training
) {
    validateDeviceBindingsForPayload(payload, bindings, "initAutogradContext(payload)");

    AutogradContext ctx{};
    populateCommonContext(
        ctx, config, training_state, gpu_encoder, embedding_layer, lm_head,
        scratch_block, reasoning_head, execution_block, cublas_handle, stream,
        grad_scale, step, is_training);

    ctx.payload = &payload;
    ctx.device_bindings = &bindings;
    ctx.batch_size = payload.batch_size;
    ctx.seq_len = payload.max_seq_len;

    ctx.validate("initAutogradContext(payload)");
    return ctx;
}

// Inference overload — builds a geometry-only BatchPayload so ctx.payload is never null.
AutogradContext initAutogradContext(
    const LanguageModelConfig* config,
    TrainingState* training_state,
    GPUGrimEncoder* gpu_encoder,
    EmbeddingLayer* embedding_layer,
    LMHeadLayer* lm_head,
    ScratchBlockLayer* scratch_block,
    ReasoningHeadLayer* reasoning_head,
    ExecutionBlockLayer* execution_block,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    int batch_size,
    int seq_len,
    float grad_scale,
    uint64_t step,
    bool is_training
) {
    AutogradContext ctx{};
    populateCommonContext(
        ctx, config, training_state, gpu_encoder, embedding_layer, lm_head,
        scratch_block, reasoning_head, execution_block, cublas_handle, stream,
        grad_scale, step, is_training);

    ctx.batch_size = batch_size;
    ctx.seq_len = seq_len;

    // Build geometry-only inference payload (vectors stay empty).
    ctx.inference_payload_.batch_size   = batch_size;
    ctx.inference_payload_.max_seq_len  = seq_len;
    ctx.inference_payload_.total_tokens = batch_size * seq_len;
    ctx.inference_payload_.actual_tokens = batch_size * seq_len;
    // NOTE: payload pointer is set by executeAutogradForward() after return-by-value,
    // because NRVO/move invalidates self-pointers.

    ctx.validate("initAutogradContext(inference)");
    return ctx;
}

}  // namespace Autograd
}  // namespace GRIM

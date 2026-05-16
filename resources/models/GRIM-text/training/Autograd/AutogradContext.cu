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
    if (!payload.mtp_shifted_targets.empty()) {
        if (!bindings.d_mtp_shifted_targets) {
            throw std::runtime_error(
                std::string(caller) + ": BatchDeviceBindings.d_mtp_shifted_targets is NULL for MTP payload");
        }
    }
}

} // namespace

// Training overload — borrows batch geometry from BatchPayload.
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
    const HyperParameters::LossConfigHP& loss_config,
    uint64_t step,
    bool is_training
) {
    validateDeviceBindingsForPayload(payload, bindings, "initAutogradContext(payload)");

    AutogradContext ctx(loss_config);
    populateCommonContext(
        ctx, config, training_state, gpu_encoder, embedding_layer, lm_head,
        scratch_block, reasoning_head, execution_block, cublas_handle, stream,
        step, is_training);

    ctx.payload = &payload;
    ctx.device_bindings = &bindings;

    ctx.validate("initAutogradContext(payload)");
    return ctx;
}

}  // namespace Autograd
}  // namespace GRIM

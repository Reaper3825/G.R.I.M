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
    const Config::AiConfigSnapshot* config,
    TrainingState* training_state,
    Forward::ModelForwardOutputs& forward_outputs,
    AutogradLossState& loss_state,
    GPUGrimEncoder* gpu_encoder,
    bool execution_block_enabled,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    uint64_t batch_idx)
{
    ctx.config = config;
    ctx.training_state = training_state;
    ctx.forward_outputs = &forward_outputs;
    ctx.loss_state = &loss_state;
    ctx.gpu_encoder = gpu_encoder;
    ctx.execution_block_enabled = execution_block_enabled;
    ctx.parameter_registry = &parameter_registry;
    ctx.cublas_handle = cublas_handle;
    ctx.stream = stream;
    ctx.batch_idx = batch_idx;
}

void validateDeviceBindingsForPayload(
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const char* caller)
{
    payload.validate(caller);

    if (!bindings.d_input_ids || !bindings.d_target_ids || !bindings.d_token_to_slot_map) {
        throw std::runtime_error(
            std::string(caller) + ": BatchDeviceBindings has NULL device pointers - "
            "caller must invoke Batching::uploadBatchToDevice(config, training_state, payload) before initializing autograd context");
    }
    if (!payload.mtp_shifted_targets.empty()) {
        if (!bindings.d_mtp_shifted_targets) {
            throw std::runtime_error(
                std::string(caller) + ": BatchDeviceBindings.d_mtp_shifted_targets is NULL for MTP payload");
        }
    }
}

} // namespace

// Training overload — payload carries the realized host batch datum, while the
// fixed-shape training geometry itself is enforced against config at the upload
// boundary. `bindings` must already have been populated by
// Batching::uploadBatchToDevice(config, training_state, payload) at the H2D sync slice.
AutogradContext initAutogradContext(
    const Config::AiConfigSnapshot* config,
    TrainingState* training_state,
    Forward::ModelForwardOutputs& forward_outputs,
    AutogradLossState& loss_state,
    GPUGrimEncoder* gpu_encoder,
    bool execution_block_enabled,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    uint64_t batch_idx
) {
    validateDeviceBindingsForPayload(payload, bindings, "initAutogradContext(payload)");

    AutogradContext ctx;
    populateCommonContext(
        ctx, config, training_state, forward_outputs, loss_state,
        gpu_encoder,
        execution_block_enabled, parameter_registry, cublas_handle, stream,
        batch_idx);

    ctx.payload = &payload;
    ctx.device_bindings = &bindings;

    ctx.validate("initAutogradContext(payload)");
    return ctx;
}

}  // namespace Autograd
}  // namespace GRIM

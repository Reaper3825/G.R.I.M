//======================================================//
//  AutogradTraining.cu
//  Autograd loss and backward orchestration
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "AutogradTraining.hpp"

#include "../Diagnostics/GradientConnectivityDiagnostic.hpp"
#include "../../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/Loss/ComputeLoss/AtomInsertionLoss.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/VerboseLogging.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#define AG_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[AutogradTraining] INFO: " << msg << std::endl; \
    } \
} while(0)

namespace GRIM {
namespace Autograd {
namespace {

void validateLossBoundaryInputs(
    const AutogradContext& ctx,
    const Batching::BatchPayload& payload,
    const char* caller
) {
    if (!ctx.config) throw std::runtime_error(std::string(caller) + ": ctx.config is NULL");
    if (!ctx.training_state) throw std::runtime_error(std::string(caller) + ": ctx.training_state is NULL");
    if (!ctx.forward_outputs) throw std::runtime_error(std::string(caller) + ": ctx.forward_outputs is NULL");
    if (!ctx.loss_state) throw std::runtime_error(std::string(caller) + ": ctx.loss_state is NULL");
    if (!ctx.gpu_encoder) throw std::runtime_error(std::string(caller) + ": ctx.gpu_encoder is NULL");
    if (!ctx.parameter_registry) throw std::runtime_error(std::string(caller) + ": ctx.parameter_registry is NULL");
    if (!ctx.device_bindings) throw std::runtime_error(std::string(caller) + ": ctx.device_bindings is NULL");

    payload.validate(caller);
    const bool supervision_uploaded = payload.EnableAtomIdentification
        ? ctx.device_bindings->d_atom_insertion_gap_targets &&
            ctx.device_bindings->d_atom_insertion_valid_gap_mask
        : ctx.device_bindings->d_target_ids != nullptr;
    if (!ctx.device_bindings->d_input_ids ||
        !supervision_uploaded ||
        !ctx.device_bindings->d_token_to_slot_index_map) {
        throw std::runtime_error(
            std::string(caller) + ": BatchDeviceBindings has NULL device pointers");
    }
}

}  // namespace

LossResult computeAutogradLoss(
    AutogradContext& ctx,
    const Batching::BatchPayload& payload,
    const HyperParameters::LossConfigHP& loss_config
) {
    LossResult result{};
    result.success = false;
    validateLossBoundaryInputs(ctx, payload, "computeAutogradLoss");

    auto& forward_outputs = *ctx.forward_outputs;
    auto& loss_state = *ctx.loss_state;
    if (!forward_outputs.logits_tensor.data) {
        throw std::runtime_error(
            "computeAutogradLoss: Logits tensor not initialized - caller must run shared forward before loss");
    }

    const bool atom_insertion_enabled = payload.EnableAtomIdentification;
    const int valid_rows = atom_insertion_enabled
        ? payload.atom_insertion_valid_gap_count
        : payload.lm_valid_tokens;
    AG_INFO("Computing loss: tokens=" << payload.total_tokens
            << " vocab=" << payload.vocab_size
            << " valid_rows=" << valid_rows
            << " atom_insertion=" << atom_insertion_enabled);

    if (atom_insertion_enabled) {
        AtomInsertion::AtomInsertionLossStats atom_stats{};
        loss_state.loss_tensor = AtomInsertion::atomInsertionLoss(
            forward_outputs.logits_tensor,
            forward_outputs,
            payload,
            *ctx.device_bindings,
            true,
            AtomInsertion::AtomInsertionLossConfig{},
            &atom_stats,
            ctx.stream);
        if (atom_stats.valid_gap_count != valid_rows) {
            throw std::runtime_error(
                "computeAutogradLoss: atom loss valid-gap count mismatch");
        }
    } else {
        loss_state.loss_tensor = autograd::unified_loss(
            forward_outputs.logits_tensor,
            payload,
            *ctx.device_bindings,
            loss_config,
            ctx.d_class_weights,
            ctx.stream);
    }

    float total_loss = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(
        &total_loss, loss_state.loss_tensor.data, sizeof(float),
        cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    const float primary_loss = total_loss;
    if (!std::isfinite(primary_loss)) {
        throw std::runtime_error(
            std::string("computeAutogradLoss: ") +
            (atom_insertion_enabled ? "atom BCE" : "pure text CE") +
            " is non-finite (" + std::to_string(primary_loss) + ")");
    }

    // Phase2's historical primary-loss channel is named text_loss. Atom mode
    // uses the same scalar channel so telemetry/validation remain task-neutral.
    result.text_loss = primary_loss;
    result.valid_tokens = valid_rows;
    result.loss_value = total_loss;
    result.selector_loss = 0.0f;
    result.weight_text = 1.0f;
    if (!std::isfinite(result.loss_value)) {
        throw std::runtime_error(
            "computeAutogradLoss: loss is non-finite (primary=" +
            std::to_string(primary_loss) + ")");
    }

    AG_INFO("Loss computed: total=" << total_loss
            << " primary=" << primary_loss
            << " selector=" << result.selector_loss
            << " valid_rows=" << valid_rows);
    result.success = true;
    return result;
}

BackwardResult executeAutogradBackward(
    AutogradContext& ctx,
    bool accumulate
) {
    BackwardResult result{};
    result.success = false;
    result.grad_rms = 0.0f;
    ctx.validate("executeAutogradBackward");

    auto& loss_state = *ctx.loss_state;
    if (!loss_state.loss_tensor.data) {
        throw std::runtime_error(
            "executeAutogradBackward: Loss tensor not initialized - call computeAutogradLoss() first");
    }
    if (!loss_state.loss_tensor.grad_fn) {
        throw std::runtime_error(
            "executeAutogradBackward: Loss tensor has no grad_fn - autograd chain broken");
    }

    AG_INFO("Executing backward pass (accumulate=" << accumulate << ")");
    if (!accumulate) {
        GRIM::zeroParameterGradients(
            ctx.parameter_registry->requireParameterGroups("executeAutogradBackward"),
            ctx.stream);
    }

    Diagnostics::GradientVerificationSession gradient_verification(ctx, accumulate);

    AG_INFO("Calling loss_tensor.backward(nullptr, 1.0f, ctx.payload, ctx.device_bindings)...");
    loss_state.loss_tensor.backward(
        nullptr, 1.0f, ctx.payload, ctx.device_bindings);
    AG_INFO("loss_tensor.backward() returned successfully");

    Diagnostics::logPostBackwardGradientSamples(ctx, accumulate);

    AG_INFO("Verifying gradients are connected to optimizer...");
    if (!gradient_verification.verify(ctx)) {
        throw std::runtime_error(
            "executeAutogradBackward: Gradient connectivity verification failed after backward mutated parameter gradients");
    }
    AG_INFO("Gradient connectivity verified");
    AG_INFO("Backward complete");

    result.success = true;
    return result;
}

bool verifyGradientsAreConnected(AutogradContext& ctx) {
    return Diagnostics::verifyGradientsAreConnected(ctx);
}

}  // namespace Autograd
}  // namespace GRIM

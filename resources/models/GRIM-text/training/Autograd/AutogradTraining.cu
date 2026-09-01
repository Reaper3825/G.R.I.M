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
#include "../../Shared/LocalAtomRetrieval/LocalAtomRetrievalLoss.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/VerboseLogging.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

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

    Tensor primary_loss_tensor;
    if (atom_insertion_enabled) {
        AtomInsertion::AtomInsertionLossStats atom_stats{};
        primary_loss_tensor = AtomInsertion::atomInsertionLoss(
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
        primary_loss_tensor = autograd::unified_loss(
            forward_outputs.logits_tensor,
            payload,
            *ctx.device_bindings,
            loss_config,
            ctx.d_class_weights,
            ctx.stream);
    }

    const int retrieval_query_count = payload.localAtomQueryCount();
    const auto model_hp = HyperParameters::modelHP(*ctx.config);
    const bool retrieval_active =
        model_hp.local_atom_retrieval_enabled &&
        !atom_insertion_enabled &&
        retrieval_query_count > 0;

    Tensor retrieval_loss_tensor;
    Tensor weighted_retrieval_loss_tensor;
    if (retrieval_active) {
        if (!forward_outputs.local_atom_retrieval_logits.data) {
            throw std::runtime_error(
                "computeAutogradLoss: local atom retrieval is enabled for a batch "
                "with queries, but the shared forward did not emit retrieval logits");
        }
        retrieval_loss_tensor = LocalAtomRetrieval::LocalAtomRetrievalLoss(
            forward_outputs,
            payload,
            *ctx.device_bindings,
            ctx.stream);
        weighted_retrieval_loss_tensor = autograd::mul_scalar(
            retrieval_loss_tensor,
            loss_config.local_atom_retrieval_weight,
            ctx.stream);
        loss_state.loss_tensor = autograd::add(
            primary_loss_tensor,
            weighted_retrieval_loss_tensor,
            ctx.stream);
    } else {
        loss_state.loss_tensor = std::move(primary_loss_tensor);
    }

    float primary_loss = 0.0f;
    float retrieval_loss_raw = 0.0f;
    float weighted_retrieval_loss = 0.0f;
    float total_loss = 0.0f;
    if (retrieval_active) {
        CUDA_CHECK(cudaMemcpyAsync(
            &primary_loss, primary_loss_tensor.data, sizeof(float),
            cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            &retrieval_loss_raw, retrieval_loss_tensor.data, sizeof(float),
            cudaMemcpyDeviceToHost, ctx.stream));
        CUDA_CHECK(cudaMemcpyAsync(
            &weighted_retrieval_loss, weighted_retrieval_loss_tensor.data,
            sizeof(float), cudaMemcpyDeviceToHost, ctx.stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(
            &primary_loss, loss_state.loss_tensor.data, sizeof(float),
            cudaMemcpyDeviceToHost, ctx.stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(
        &total_loss, loss_state.loss_tensor.data, sizeof(float),
        cudaMemcpyDeviceToHost, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));

    if (!std::isfinite(primary_loss)) {
        throw std::runtime_error(
            std::string("computeAutogradLoss: ") +
            (atom_insertion_enabled ? "atom BCE" : "pure text CE") +
            " is non-finite (" + std::to_string(primary_loss) + ")");
    }
    if (!std::isfinite(retrieval_loss_raw) ||
        !std::isfinite(weighted_retrieval_loss)) {
        throw std::runtime_error(
            "computeAutogradLoss: local atom retrieval loss is non-finite "
            "(raw=" + std::to_string(retrieval_loss_raw) +
            ", weighted=" + std::to_string(weighted_retrieval_loss) + ")");
    }

    // Phase2's historical primary-loss channel is named text_loss. Atom mode
    // uses the same scalar channel so telemetry/validation remain task-neutral.
    result.text_loss = primary_loss;
    result.valid_tokens = valid_rows;
    result.loss_value = total_loss;
    result.selector_loss = 0.0f;
    result.local_atom_retrieval_loss_raw = retrieval_loss_raw;
    result.local_atom_retrieval_loss = weighted_retrieval_loss;
    result.weight_text = 1.0f;
    result.weight_local_atom_retrieval =
        retrieval_active ? loss_config.local_atom_retrieval_weight : 0.0f;
    result.local_atom_retrieval_queries =
        retrieval_active ? retrieval_query_count : 0;
    result.local_atom_reference_targets =
        retrieval_active ? payload.local_atom_reference_target_count : 0;
    if (!std::isfinite(result.loss_value)) {
        throw std::runtime_error(
            "computeAutogradLoss: loss is non-finite (primary=" +
            std::to_string(primary_loss) + ", retrieval=" +
            std::to_string(weighted_retrieval_loss) + ")");
    }

    AG_INFO("Loss computed: total=" << total_loss
            << " primary=" << primary_loss
            << " local_atom_retrieval_raw=" << retrieval_loss_raw
            << " local_atom_retrieval_weighted=" << weighted_retrieval_loss
            << " local_atom_retrieval_queries=" << result.local_atom_retrieval_queries
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

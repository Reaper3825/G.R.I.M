//======================================================//
//  AutogradTraining.cu
//  Implementation of autograd-based training flow
//======================================================//

#include "AutogradTraining.hpp"

// MUST include full definition of GPUGrimEncoder for method calls
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/grim_layer_gpu.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/Encoding/AblationFlags.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/Loss/ComputeLoss/ArgSelectorLoss.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <algorithm>  // std::clamp, std::min, std::max (+ std::min_element/max_element in diagnostics)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "../../Shared/VerboseLogging.hpp"

// Issue #142: DELETED kernelMaskNonContentLogits / launchMaskNonContentLogits.
// Setting special token logits to -inf is NON-STANDARD and was a workaround for
// mode collapse (tok1=PAD at 98% argmax). The standard approach is loss masking
// via target=-1 which is already implemented in:
//   - AutogradLoss.cu: forward skips target==-1, backward zeros grad for target==-1
//   - BatchPayload.cu: defense-masks non-content tokens with target=-1
//   - DataLoader.cu: masks non-content tokens during data loading
// The -inf masking was poisoning every diagnostic that sampled logits through
// the old durable cached-logits side channel
// (logit_min=-inf → logit_range=inf → logit_mean=-inf → logit_std=NaN).
// Inference-time sampling/masking policy remains Phase2-owned in
// training/Phases/Phase2_InferenceLoop.cu over the shared-forward boundary.

// Logging macros - guarded by VerboseLogging flags for production
#define AG_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[AutogradTraining] INFO: " << msg << std::endl; \
    } \
} while(0)
#define AG_ERROR(msg) do { std::cerr << "[AutogradTraining] ERROR: " << msg << std::endl; } while(0)
#define AG_WARN(msg) do { std::cerr << "[AutogradTraining] WARN: " << msg << std::endl; } while(0)

namespace GRIM {
namespace Autograd {

// Finding 1 (Rule 26): countValidTokensKernel/countValidTokens DELETED — zero callers
// Finding 2 (Rule 26): sumSquaredKernel/computeSumSquared DELETED — only caller was
//   computeGradientNorm() which is redundant with Phase2's computeGradNorm()

// NOTE: linkEncoderWeightsToTrainingState was removed.
// Encoder trainable tensors are registry-owned; optimizer and diagnostics access
// them via StartupParameterRegistry::requireEncodingLayerParameters(...).
// See Startup/Model/ParameterGroupRegistration.{hpp,cu}.

// Context initialization lives in AutogradContext.cu so this file can focus on
// the autograd math path: forward, loss, backward, and the training-step bridge.

//======================================================================
// Autograd Forward Pass
// PRODUCTION-READY: Runs entire model with autograd graph intact
//======================================================================

namespace {
struct GradientSignalProbe {
    bool allocated = false;
    bool finite = true;
    bool nonzero = false;
    float rms = 0.0f;
    size_t checked = 0;
};

struct GradientDeltaProbe {
    bool comparable = false;
    bool finite = true;
    bool changed = false;
    float delta_rms = 0.0f;
    size_t checked = 0;
    std::string error_message;
};

struct GradientSignalExpectation {
    std::string label;
    bool had_grad_storage = false;
    const float* grad_ptr = nullptr;
    size_t count = 0;
    std::vector<float> before;
};

struct GradientVerificationActivity {
    bool text_loss_active = false;
};

struct GradientSignalBaselines {
    bool require_current_microbatch_delta = false;
    std::vector<GradientSignalExpectation> expected;

    const GradientSignalExpectation* find(const std::string& label) const {
        for (const auto& entry : expected) {
            if (entry.label == label) {
                return &entry;
            }
        }
        return nullptr;
    }
};

GradientSignalProbe probeGradientSignal(Tensor& tensor, cudaStream_t stream) {
    GradientSignalProbe probe{};
    if (!tensor.data || !tensor.has_grad() || !tensor.grad_data()) {
        return probe;
    }
    probe.allocated = true;
    const size_t count = static_cast<size_t>(tensor.numel());
    probe.checked = count;
    if (probe.checked == 0) {
        probe.finite = false;
        return probe;
    }

    std::vector<float> h_grad(probe.checked);
    CUDA_CHECK(cudaMemcpyAsync(h_grad.data(), tensor.grad_data(),
                               probe.checked * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double sum_sq = 0.0;
    for (float v : h_grad) {
        if (!std::isfinite(v)) {
            probe.finite = false;
        }
        if (v != 0.0f) {
            probe.nonzero = true;
        }
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    probe.rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(probe.checked)));
    if (!std::isfinite(probe.rms)) {
        probe.finite = false;
    }
    return probe;
}

GradientSignalExpectation captureGradientExpectation(
    Tensor& tensor,
    const std::string& label,
    cudaStream_t stream
) {
    GradientSignalExpectation expectation{};
    expectation.label = label;
    if (!tensor.data || !tensor.has_grad() || !tensor.grad_data()) {
        return expectation;
    }

    expectation.had_grad_storage = true;
    expectation.grad_ptr = tensor.grad_data();
    expectation.count = static_cast<size_t>(tensor.numel());
    if (expectation.count == 0) {
        return expectation;
    }

    expectation.before.resize(expectation.count);
    CUDA_CHECK(cudaMemcpyAsync(expectation.before.data(), tensor.grad_data(),
                               expectation.count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return expectation;
}

GradientDeltaProbe probeGradientDelta(
    Tensor& tensor,
    const GradientSignalExpectation& expectation,
    cudaStream_t stream
) {
    GradientDeltaProbe probe{};
    if (!tensor.data || !tensor.has_grad() || !tensor.grad_data()) {
        probe.error_message = ".grad_data is NULL after backward";
        return probe;
    }

    const size_t count = static_cast<size_t>(tensor.numel());
    probe.checked = count;
    if (count == 0) {
        probe.error_message = ".grad has zero elements";
        return probe;
    }

    std::vector<float> after(count);
    CUDA_CHECK(cudaMemcpyAsync(after.data(), tensor.grad_data(),
                               count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!expectation.had_grad_storage) {
        probe.comparable = true;
        double sum_sq = 0.0;
        for (float v : after) {
            if (!std::isfinite(v)) {
                probe.finite = false;
            }
            if (v != 0.0f) {
                probe.changed = true;
            }
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
        }
        probe.delta_rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count)));
        if (!std::isfinite(probe.delta_rms)) {
            probe.finite = false;
        }
        return probe;
    }

    if (expectation.grad_ptr != tensor.grad_data()) {
        probe.error_message = ".grad_data pointer changed during backward";
        return probe;
    }
    if (expectation.count != count || expectation.before.size() != count) {
        probe.error_message = ".grad shape/size changed during backward";
        return probe;
    }

    probe.comparable = true;
    double sum_sq = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float before = expectation.before[i];
        const float current = after[i];
        if (!std::isfinite(before) || !std::isfinite(current)) {
            probe.finite = false;
        }
        const float delta = current - before;
        if (delta != 0.0f) {
            probe.changed = true;
        }
        sum_sq += static_cast<double>(delta) * static_cast<double>(delta);
    }
    probe.delta_rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(count)));
    if (!std::isfinite(probe.delta_rms)) {
        probe.finite = false;
    }
    return probe;
}

GradientVerificationActivity detectGradientVerificationActivity(AutogradContext& ctx) {
    GradientVerificationActivity activity{};
    activity.text_loss_active = ctx.payload && ctx.payload->lm_valid_tokens > 0;
    return activity;
}

GradientSignalBaselines captureGradientVerificationBaselines(
    AutogradContext& ctx,
    bool require_current_microbatch_delta
) {
    GradientSignalBaselines baselines{};
    baselines.require_current_microbatch_delta = require_current_microbatch_delta;
    if (!require_current_microbatch_delta) {
        return baselines;
    }

    const GradientVerificationActivity activity = detectGradientVerificationActivity(ctx);
    const auto model_hp = GRIM::HyperParameters::modelHP(*ctx.config);
    auto captureExpected = [&](Tensor& tensor, const std::string& label) {
        if (tensor.data) {
            baselines.expected.push_back(captureGradientExpectation(tensor, label, ctx.stream));
        }
    };

    if (activity.text_loss_active) {
        auto& lm_head_parameters = ctx.parameter_registry->requireLmHeadParameters("captureGradientSignalBaselines");
        captureExpected(lm_head_parameters.weights, "lm_head weights");
        if (model_hp.encoder_num_layers > 0) {
            // Ablation-aware (AblationFlags.hpp): snapshot the baseline for the
            // LIVE sublayer so the accumulation-delta check has a matching entry.
            // Labels and branch conditions MUST match verifyGradientsAreConnectedImpl exactly.
            if (GRIM::Ablation::kAttnDeliversParamGradient) {
                auto& enc0 = ctx.parameter_registry->requireEncodingLayerParameters(0, "captureGradientSignalBaselines");
                captureExpected(enc0.W_qkv, "layer 0 attnWqkv");
                if (model_hp.encoder_attention_residual_gate_enabled) {
                    auto& gate0 = ctx.parameter_registry->requireAttentionResidualGateParameters(
                        0, "captureGradientSignalBaselines");
                    captureExpected(
                        gate0.W_gate, "layer 0 attentionResidualGateW");
                }
            } else if (!GRIM::Ablation::kZeroFfnResidual) {
                auto& ffn0 = ctx.parameter_registry->requireFeedForwardParameters(0, "captureGradientSignalBaselines");
                captureExpected(ffn0.W2, "layer 0 ffnW2 (attn ablated)");
            }
        }
    }

    AG_INFO("Captured " << baselines.expected.size()
            << " pre-backward gradient baselines for accumulation-slot verification");
    return baselines;
}

bool verifyGradientsAreConnectedImpl(
    AutogradContext& ctx,
    const GradientSignalBaselines* baselines
);

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

    if (!ctx.device_bindings->d_input_ids || !ctx.device_bindings->d_target_ids || !ctx.device_bindings->d_token_to_slot_index_map) {
        throw std::runtime_error(std::string(caller) + ": BatchDeviceBindings has NULL device pointers");
    }
}
}  // namespace

//======================================================================
// Autograd Loss Computation
//======================================================================

LossResult computeAutogradLoss(
    AutogradContext& ctx,
    const Batching::BatchPayload& payload,
    const HyperParameters::LossConfigHP& loss_config
) {
    LossResult result{};
    result.success = false;
    
    // RULE 20: Fail loud
    validateLossBoundaryInputs(ctx, payload, "computeAutogradLoss");
    const auto* cfg = ctx.config;
    
    // RULE 20: Fail loud - validate logits tensor was populated by forward pass
    auto& forward_outputs = *ctx.forward_outputs;
    auto& loss_state = *ctx.loss_state;
    if (!forward_outputs.logits_tensor.data) {
        throw std::runtime_error("computeAutogradLoss: Logits tensor not initialized - caller must run shared forward before loss");
    }
    
    // BatchPayload owns target semantics; BatchDeviceBindings is the explicit
    // per-step device view of the uploaded target mirror.
    if (!ctx.device_bindings || !ctx.device_bindings->d_target_ids) {
        throw std::runtime_error(
            "computeAutogradLoss: BatchDeviceBindings target pointer is NULL - "
            "caller must upload the Phase1-authored BatchPayload before loss");
    }
    
    const int total_tokens = payload.total_tokens;
    const int vocab_size = payload.vocab_size;
    const int lm_valid_tokens = payload.lm_valid_tokens;
    
    AG_INFO("Computing loss: tokens=" << total_tokens << " vocab=" << vocab_size
            << " lm_valid=" << lm_valid_tokens);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 1. TEXT CROSS-ENTROPY LOSS (autograd::unified_loss)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Compute text CE directly into the canonical Category 1 owner consumed
    // later by executeAutogradBackward(). Phase2 receives the host-side
    // LossResult from computeAutogradLoss(); it does not own a second loss Tensor.
    loss_state.loss_tensor = autograd::unified_loss(
        forward_outputs.logits_tensor,
        payload,
        *ctx.device_bindings,
        loss_config,
        ctx.d_class_weights,
        ctx.stream
    );

    float text_ce_loss = 0.0f;
    cudaMemcpyAsync(&text_ce_loss, loss_state.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    if (!std::isfinite(text_ce_loss)) {
        throw std::runtime_error("computeAutogradLoss: pure text CE is non-finite (" +
                                 std::to_string(text_ce_loss) + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. Arg/option selector auxiliary loss: L_total += α_sel * CE(select_logits, target_entry)
    //
    // Trains the selector to pick the correct candidate atom-entry ("option") at
    // each position. Execution-INDEPENDENT: targets are the next atom's entry
    // (data-derived). No-op unless selector_enabled and the forward emitted
    // selector_logits with a non-empty candidate pool + supervised targets.
    // ═══════════════════════════════════════════════════════════════════════════
    // Scaled selector contribution actually added to loss_tensor (0 when the
    // selector is disabled or fires no supervised candidates this step). Reported
    // as its own channel.
    float selector_loss_scaled = 0.0f;
    const bool selector_enabled =
        GRIM::HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "selector_enabled");
    if (selector_enabled) {
        const int sel_candidates = ctx.device_bindings->num_pool_atoms;
        const int sel_valid = payload.arg_select_valid_count;
        const float selector_alpha =
            GRIM::HyperParameters::snapshotTrainingConfigField<float>(*cfg, "selector_alpha");
        const bool sel_can_fire =
            forward_outputs.selector_logits.data && sel_candidates > 0 &&
            ctx.device_bindings->d_arg_select_targets && sel_valid > 0;

        if (sel_can_fire && selector_alpha > 0.0f) {
            Tensor selector_loss_tensor = autograd::argSelectorLoss(
                forward_outputs.selector_logits,
                ctx.device_bindings->d_arg_select_targets,
                total_tokens,
                sel_candidates,
                sel_valid,
                ctx.stream);

            float selector_loss = 0.0f;
            cudaMemcpyAsync(&selector_loss, selector_loss_tensor.data, sizeof(float),
                            cudaMemcpyDeviceToHost, ctx.stream);
            cudaStreamSynchronize(ctx.stream);
            if (!std::isfinite(selector_loss)) {
                throw std::runtime_error("computeAutogradLoss: selector loss is non-finite (" +
                                         std::to_string(selector_loss) + ")");
            }
            // Always-on supervision telemetry: confirms the selector is actually
            // trained this step (loss flows into W_q AND the NumberEncoder keys).
            std::cerr << "[SELECTOR] CE=" << selector_loss
                      << " valid=" << sel_valid
                      << " candidates=" << sel_candidates
                      << " alpha=" << selector_alpha << std::endl;

            selector_loss_scaled = selector_alpha * selector_loss;
            Tensor scaled_selector = autograd::scale_scalar(selector_loss_tensor, selector_alpha, ctx.stream);
            loss_state.loss_tensor = autograd::add(loss_state.loss_tensor, scaled_selector, ctx.stream);
        } else {
            // Always-on: the selector is enabled but contributed NOTHING this
            // step. Most commonly the batch has no supervised next-atom target
            // (valid=0) — i.e. the curriculum lacks multi-candidate option rows.
            std::cerr << "[SELECTOR] skipped (no supervised candidates this step):"
                      << " logits=" << (forward_outputs.selector_logits.data ? 1 : 0)
                      << " candidates=" << sel_candidates
                      << " valid=" << sel_valid
                      << " alpha=" << selector_alpha << std::endl;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    float text_plus_aux_loss = 0.0f;
    cudaMemcpyAsync(&text_plus_aux_loss, loss_state.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    if (!std::isfinite(text_plus_aux_loss)) {
        throw std::runtime_error("computeAutogradLoss: text/selector loss is non-finite (" +
                                 std::to_string(text_plus_aux_loss) + ")");
    }

    result.text_loss = text_ce_loss;
    result.valid_tokens = lm_valid_tokens;

    // ═══════════════════════════════════════════════════════════════════════════
    // POST-SELECTOR OBJECTIVE BOUNDARY
    // ═══════════════════════════════════════════════════════════════════════════
    // ═══════════════════════════════════════════════════════════════════════════
    // GROUND-TRUTH LOSS: Read the ACTUAL tensor that backward will differentiate.
    // This is the single source of truth — no manual reconstruction from stale
    // host-side scalars. text_loss was snapshot before exec additions.
    // ═══════════════════════════════════════════════════════════════════════════
    float actual_loss = 0.0f;
    cudaMemcpyAsync(&actual_loss, loss_state.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    result.loss_value = actual_loss;
    result.selector_loss = selector_loss_scaled;
    result.weight_text = 1.0f;
    
    if (!std::isfinite(result.loss_value)) {
        throw std::runtime_error("computeAutogradLoss: combined loss is non-finite (actual_tensor=" 
            + std::to_string(actual_loss) + " text_ce=" + std::to_string(text_ce_loss)
            + " selector=" + std::to_string(selector_loss_scaled) + ")");
    }
    
    AG_INFO("Loss computed: total=" << actual_loss << " text_ce=" << text_ce_loss
            << " selector=" << result.selector_loss
            << " lm_valid=" << lm_valid_tokens);
    
    result.success = true;
    return result;
}

//======================================================================
// Autograd Backward Pass
//======================================================================

BackwardResult executeAutogradBackward(
    AutogradContext& ctx,
    bool accumulate
) {
    BackwardResult result{};
    result.success = false;
    result.grad_rms = 0.0f;
    
    ctx.validate("executeAutogradBackward");
    
    auto& loss_state = *ctx.loss_state;
    const auto model_hp = GRIM::HyperParameters::modelHP(*ctx.config);
    if (!loss_state.loss_tensor.data) {
        throw std::runtime_error("executeAutogradBackward: Loss tensor not initialized - call computeAutogradLoss() first");
    }
    
    if (!loss_state.loss_tensor.grad_fn) {
        throw std::runtime_error("executeAutogradBackward: Loss tensor has no grad_fn - autograd chain broken");
    }
    if (!ctx.parameter_registry) {
        throw std::runtime_error("executeAutogradBackward: parameter_registry is NULL - registered parameter gradient lifecycle requires StartupParameterRegistry");
    }
    
        AG_INFO("Executing backward pass (accumulate=" << accumulate << ")");

    // Parameter gradients are lifecycle-managed through the registered
    // TensorContract ParameterGroup inventory. AutogradTraining must not walk
    // layer internals or special-case optional heads here.
    if (!accumulate) {
        GRIM::zeroParameterGradients(ctx.parameter_registry->requireParameterGroups("executeAutogradBackward"), ctx.stream);
    }

    // On accumulation slots, existing parameter grad buffers intentionally hold
    // earlier microbatch contributions. A post-backward nonzero check alone can
    // therefore pass even if this microbatch delivered no signal. Snapshot only
    // the tensors that verification expects to receive signal, then require a
    // nonzero pre/post delta for those tensors after backward.
    GradientSignalBaselines gradient_signal_baselines =
        captureGradientVerificationBaselines(ctx, accumulate);

    AG_INFO("Calling loss_tensor.backward(nullptr, 1.0f, ctx.payload, ctx.device_bindings)...");
    loss_state.loss_tensor.backward(nullptr, 1.0f, ctx.payload, ctx.device_bindings);
    AG_INFO("loss_tensor.backward() returned successfully");

    // ════════════════════════════════════════════════════════════════════
    // DIAGNOSTIC: Sample gradient values immediately after backward to
    // determine if backward itself produces zeros or if something later
    // corrupts the buffers.  Issue: "zero gradients every other batch"
    // ════════════════════════════════════════════════════════════════════
    {
        cudaStreamSynchronize(ctx.stream);
        auto& embedding_parameters = ctx.parameter_registry->requireEmbeddingParameters("executeAutogradBackward");
        auto& lm_head_parameters = ctx.parameter_registry->requireLmHeadParameters("executeAutogradBackward");

        float emb_sample = 0.0f;
        float* emb_grads = embedding_parameters.token_weights.grad_data();
        if (emb_grads) {
            cudaMemcpy(&emb_sample, emb_grads, sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Sample LM head weight gradient (first element)
        float lm_sample = 0.0f;
        float* lm_grads = lm_head_parameters.weights.grad_data();
        if (lm_grads) {
            cudaMemcpy(&lm_sample, lm_grads, sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Sample first encoder layer attnWqkv gradient
        float enc_sample = 0.0f;
        if (model_hp.encoder_num_layers > 0) {
            auto& enc0 = ctx.parameter_registry->requireEncodingLayerParameters(0, "executeAutogradBackward");
            float* wqkv_grads = enc0.W_qkv.grad_data();
            if (wqkv_grads) {
                cudaMemcpy(&enc_sample, wqkv_grads, sizeof(float), cudaMemcpyDeviceToHost);
            }
        }

        // Sample RMSNorm gamma gradient (layer 0)
        float rms_sample = 0.0f;
        if (model_hp.encoder_num_layers > 0) {
            auto& enc0 = ctx.parameter_registry->requireEncodingLayerParameters(0, "executeAutogradBackward");
            float* rms_grads = enc0.rms1_gamma.grad_data();
            if (rms_grads) {
                cudaMemcpy(&rms_sample, rms_grads, sizeof(float), cudaMemcpyDeviceToHost);
            }
        }

        fprintf(stderr,
            "[GRAD_DIAG] POST-BACKWARD accumulate=%d "
            "emb_grad[0]=%.10e lm_grad[0]=%.10e enc_wqkv_grad[0]=%.10e enc0_rms1_gamma_grad[0]=%.10e "
            "lm_ptr=%p\n",
            static_cast<int>(accumulate),
            emb_sample, lm_sample, enc_sample, rms_sample,
            static_cast<void*>(lm_grads));
    }

    // Verify gradients are properly connected before optimizer runs
    AG_INFO("Verifying gradients are connected to optimizer...");
    if (!verifyGradientsAreConnectedImpl(ctx, &gradient_signal_baselines)) {
        throw std::runtime_error(
            "executeAutogradBackward: Gradient connectivity verification failed after backward mutated parameter gradients");
    }
    AG_INFO("Gradient connectivity verified");
    
    // Accumulation-window normalization is intentionally not part of backward.
    // Phase2 scales registered parameter gradients exactly once after the full
    // accumulation window completes, before clipping and optimizer update.
    
    AG_INFO("Backward complete");
    
    result.success = true;
    return result;
}

//======================================================================
// Helper Functions
//======================================================================

namespace {
bool verifyGradientsAreConnectedImpl(
    AutogradContext& ctx,
    const GradientSignalBaselines* baselines
) {
    bool ok = true;
    const GradientVerificationActivity activity = detectGradientVerificationActivity(ctx);
    const auto model_hp = GRIM::HyperParameters::modelHP(*ctx.config);

    auto requireRegisteredGroup = [&](Tensor& tensor, const std::string& label) -> ParameterGroup& {
        return ctx.parameter_registry->requireParameterGroupForTensor(
            tensor, label, "verifyGradientsAreConnectedImpl");
    };

    auto recordNumericalSignal = [&](ParameterGroup& group, const std::string& label, bool received) -> bool {
        auto& state = group.gradient_verification;
        if (received) {
            state.record_nonzero_signal(ctx.global_step);
            return false;
        }

        const uint64_t zero_streak = state.record_zero_signal();
        if (state.ever_received_nonzero_signal) {
            // A parameter that already proved numerical connectivity may later
            // enter a legitimate saturated/dead-zone interval. Keep checking
            // exactly, but rate-limit the persistent warning to powers of two.
            if (zero_streak == 1 || (zero_streak & (zero_streak - 1)) == 0) {
                AG_WARN(label << ".grad received structural delivery but no representable numerical signal"
                        << " (consecutive_zero_signal_checks=" << zero_streak
                        << ", last_nonzero_step=" << state.last_nonzero_step << ")");
            }
            return false;
        }

        if (zero_streak == 1) {
            AG_WARN(label << ".grad delivered no representable numerical signal; "
                    << "tolerating the first active check because this parameter has not yet "
                    << "established a nonzero history");
            return false;
        }

        AG_WARN(label << ".grad delivered no representable numerical signal for "
                << zero_streak << " consecutive active checks and has never produced "
                << "a nonzero gradient in this training process");
        return true;
    };

    auto requireAllocatedFinite = [&](Tensor& t, const std::string& label) {
        if (!t.data) return;
        if (!t.has_grad()) {
            AG_WARN(label << ".grad is NULL - allocated parameter did not expose optimizer gradient storage");
            ok = false;
            return;
        }
        GradientSignalProbe probe = probeGradientSignal(t, ctx.stream);
        if (!probe.allocated) {
            AG_WARN(label << ".grad_data is NULL despite has_grad=true");
            ok = false;
            return;
        }
        if (!probe.finite) {
            AG_WARN(label << ".grad contains non-finite values in full tensor (checked="
                << probe.checked << ", rms=" << probe.rms << ")");
            ok = false;
        } else {
            AG_INFO(label << ".grad allocated; checked=" << probe.checked
                    << " rms=" << probe.rms
                    << " received_nonzero=" << (probe.nonzero ? "yes" : "no"));
        }
    };

    auto requireReceivedGradient = [&](Tensor& t, const std::string& label) {
        requireAllocatedFinite(t, label);
        if (!t.data || !t.has_grad() || !t.grad_data()) return;

        auto& group = requireRegisteredGroup(t, label);
        auto& verification = group.gradient_verification;
        const uint64_t delivery_count = t.gradient_delivery_count();
        if (!verification.observe_active_check(delivery_count)) {
            AG_WARN(label << ".grad received no leaf-gradient delivery during this active backward"
                    << " (active_check=" << verification.active_check_count
                    << ", delivery_count=" << delivery_count << ")");
            ok = false;
            return;
        }

        if (baselines && baselines->require_current_microbatch_delta) {
            const GradientSignalExpectation* expectation = baselines->find(label);
            if (!expectation) {
                AG_WARN(label << ".grad current-microbatch baseline is missing during accumulation verification");
                ok = false;
                return;
            }
            GradientDeltaProbe delta_probe = probeGradientDelta(t, *expectation, ctx.stream);
            if (!delta_probe.comparable) {
                AG_WARN(label << ".grad could not be compared against its pre-backward accumulation baseline"
                        << delta_probe.error_message);
                ok = false;
                return;
            }
            if (!delta_probe.finite) {
                AG_WARN(label << ".grad current-microbatch delta contains non-finite values (checked="
                        << delta_probe.checked << ", delta_rms=" << delta_probe.delta_rms << ")");
                ok = false;
                return;
            }
            const bool numerical_signal_missing =
                !delta_probe.changed || delta_probe.delta_rms == 0.0f;
            if (!numerical_signal_missing) {
                (void)recordNumericalSignal(group, label, true);
            }
            if (numerical_signal_missing && recordNumericalSignal(group, label, false)) {
                AG_WARN(label << ".grad did not change during this accumulation microbatch "
                        << "(checked=" << delta_probe.checked << ") — current backward delivered "
                        << "structurally but has never established nonzero numerical signal");
                ok = false;
                return;
            }
            if (numerical_signal_missing) {
                AG_INFO(label << ".grad received current-microbatch structural delivery; checked="
                        << delta_probe.checked << " delta_rms=0");
            } else {
                AG_INFO(label << ".grad received current-microbatch numerical signal; checked="
                        << delta_probe.checked << " delta_rms=" << delta_probe.delta_rms);
            }
            return;
        }

        GradientSignalProbe probe = probeGradientSignal(t, ctx.stream);
        const bool numerical_signal_missing = !probe.nonzero || probe.rms == 0.0f;
        if (!numerical_signal_missing) {
            (void)recordNumericalSignal(group, label, true);
        }
        if (numerical_signal_missing && recordNumericalSignal(group, label, false)) {
            AG_WARN(label << ".grad is allocated but full-tensor RMS is zero after this backward "
                    << "(checked=" << probe.checked << ") — leaf delivery occurred but this "
                    << "parameter has never established nonzero numerical signal");
            ok = false;
        }
    };
    
    // The autograd system stores gradients in Tensor.grad_ fields (shared_ptr<Tensor>)
    // The optimizer accesses them directly via Tensor.grad_data() pointers.
    // 
    // This function verifies that gradients are properly allocated and accessible.
    // It does NOT copy — gradients are already wired up during initialization.
    // This is a diagnostic check to catch pointer setup bugs before optimizer runs.
    
    auto& embedding_parameters = ctx.parameter_registry->requireEmbeddingParameters("verifyGradientsAreConnectedImpl");

    // ═══════════════════════════════════════════════════════════════════════════
    // Embedding gradients (may be tied with LM head) — owned by StartupParameterRegistry
    // ═══════════════════════════════════════════════════════════════════════════
    if (embedding_parameters.token_weights.data) {
        requireAllocatedFinite(embedding_parameters.token_weights, "embedding token_weights");
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LM head gradients (Pattern B: owned by persistent LMHeadLayer)
    // May be tied to embedding (same underlying grad Tensor)
    // ═══════════════════════════════════════════════════════════════════════════
    auto& lm_head_parameters = ctx.parameter_registry->requireLmHeadParameters("verifyGradientsAreConnectedImpl");
    if (lm_head_parameters.weights.data) {
        if (!lm_head_parameters.weights.has_grad()) {
            AG_WARN("lm_head weights.grad is NULL - gradients NOT flowing to optimizer!");
            ok = false;
        } else {
            // Check if tied to embedding (same underlying grad Tensor)
            if (lm_head_parameters.weights.grad_data() == embedding_parameters.token_weights.grad_data()) {
                AG_INFO("LM head gradients TIED to embedding: " << lm_head_parameters.weights.numel() << " elements");
            } else {
                AG_INFO("LM head gradients SEPARATE: " << lm_head_parameters.weights.numel() << " elements at " << lm_head_parameters.weights.grad_data());
            }
            if (activity.text_loss_active) {
                requireReceivedGradient(lm_head_parameters.weights, "lm_head weights");
            } else {
                requireAllocatedFinite(lm_head_parameters.weights, "lm_head weights");
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Logits gradients are TensorContract-owned.
    // ═══════════════════════════════════════════════════════════════════════════
    // LMHead returns non-leaf logits, so LogSoftmaxGradFn allocates its own
    // non-leaf input_grad buffer and passes that view directly to the upstream
    // logits grad_fn. TrainingState must not mirror or copy logits gradients.
    
    // NOTE: Encoder gradients live in registry-owned encoder parameter tensors,
    // not TrainingState mirrors or EncodingLayer-owned state.
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Final RMSNorm gamma (Pattern B: owned by LMHeadLayer)
    // When freeze_learned_rms_gammas=true, final_rms_gamma requires_grad=false and
    // we skip the check (no grad is correct, not a bug).
    // ═══════════════════════════════════════════════════════════════════════════
    if (lm_head_parameters.final_rms_gamma.data
        && lm_head_parameters.final_rms_gamma.requires_grad
        && !lm_head_parameters.final_rms_gamma.has_grad()) {
        AG_WARN("final_rms_gamma.grad is NULL - gradients NOT flowing!");
        ok = false;
    } else if (lm_head_parameters.final_rms_gamma.data && lm_head_parameters.final_rms_gamma.requires_grad) {
        requireAllocatedFinite(lm_head_parameters.final_rms_gamma,
                               "final_rms_gamma");
    }

    if (lm_head_parameters.bias.data && !lm_head_parameters.bias.has_grad()) {
        AG_WARN("lm_head_bias.grad is NULL - gradients NOT flowing!");
        ok = false;
    } else if (lm_head_parameters.bias.data) {
        requireAllocatedFinite(lm_head_parameters.bias, "lm_head_bias");
    }

    if (ctx.gpu_encoder) {
        if (!ctx.parameter_registry) {
            throw std::runtime_error("verifyGradientsAreConnectedImpl: gpu_encoder exists but ctx.parameter_registry is NULL");
        }
        const int num_layers = model_hp.encoder_num_layers;
        if (static_cast<int>(ctx.parameter_registry->feedForwardParameterTensors().size()) != num_layers) {
            throw std::runtime_error("verifyGradientsAreConnectedImpl: feedForwardParameterTensors size mismatch. size=" +
                                     std::to_string(ctx.parameter_registry->feedForwardParameterTensors().size()) +
                                     " num_layers=" + std::to_string(num_layers));
        }
        for (int layer = 0; layer < num_layers; ++layer) {
            auto& enc = ctx.parameter_registry->requireEncodingLayerParameters(layer, "verifyGradientsAreConnectedImpl");
            auto check = [&](Tensor& t, const char* name) {
                if (t.data) requireAllocatedFinite(t, "layer " + std::to_string(layer) + " " + std::string(name));
            };
            if (model_hp.encoder_freeze_learned_rms_gammas) {
                if (enc.rms1_gamma.has_grad()) {
                    AG_WARN("layer " << layer << " rms1Gamma has a grad buffer while freeze_learned_rms_gammas=true");
                    ok = false;
                }
                if (enc.rms2_gamma.has_grad()) {
                    AG_WARN("layer " << layer << " rms2Gamma has a grad buffer while freeze_learned_rms_gammas=true");
                    ok = false;
                }
            } else {
                check(enc.rms1_gamma, "rms1Gamma");
                check(enc.rms2_gamma, "rms2Gamma");
            }
            // Issue #148: Sandwich norm gammas REMOVED
            check(enc.W_qkv, "attnWqkv");
            check(enc.b_qkv, "attnBqkv");
            check(enc.W_o, "attnWo");
            check(enc.b_o, "attnBo");
            if (model_hp.encoder_attention_residual_gate_enabled) {
                auto& gate = ctx.parameter_registry->requireAttentionResidualGateParameters(
                    layer, "verifyGradientsAreConnectedImpl");
                check(gate.W_gate, "attentionResidualGateW");
                check(gate.b_gate, "attentionResidualGateB");
            }
            auto& ffn_parameters = ctx.parameter_registry->requireFeedForwardParameters(layer, "verifyGradientsAreConnectedImpl");
            check(ffn_parameters.W_gate, "ffnWGate");
            check(ffn_parameters.W1, "ffnW1");
            check(ffn_parameters.W2, "ffnW2");
            check(ffn_parameters.b2, "ffnB2");
            check(enc.layer_scale1, "layerScale1");
            check(enc.layer_scale2, "layerScale2");
        }
        if (activity.text_loss_active && num_layers > 0) {
            // Ablation-aware encoder signal check (AblationFlags.hpp):
            // a sublayer zeroed by kZeroAttnResidual / kZeroFfnResidual or by a
            // fine-grained attention probe (kZeroAttnV / kZeroAttnOProj)
            // legitimately receives ZERO gradient, so requiring it to receive
            // signal would be a false positive. Verify the live sublayer instead
            // so we still catch genuine backward-wiring breakage; if BOTH paths
            // are ablated there is no encoder gradient path to validate.
            auto& enc0 = ctx.parameter_registry->requireEncodingLayerParameters(0, "verifyGradientsAreConnectedImpl");
            if (GRIM::Ablation::kAttnDeliversParamGradient) {
                requireReceivedGradient(enc0.W_qkv, "layer 0 attnWqkv");
                if (model_hp.encoder_attention_residual_gate_enabled) {
                    auto& gate0 = ctx.parameter_registry->requireAttentionResidualGateParameters(
                        0, "verifyGradientsAreConnectedImpl");
                    requireReceivedGradient(
                        gate0.W_gate, "layer 0 attentionResidualGateW");
                }
            } else if (!GRIM::Ablation::kZeroFfnResidual) {
                auto& ffn0 = ctx.parameter_registry->requireFeedForwardParameters(0, "verifyGradientsAreConnectedImpl");
                requireReceivedGradient(ffn0.W2, "layer 0 ffnW2 (attn ablated)");
            }
        }

        // ════════════════════════════════════════════════════════════════════
        // ABLATION GRADIENT-LEAK PROBE (AblationFlags.hpp)
        //
        // kZeroAttnV and kZeroAttnResidual both force attn_out == 0 in the
        // forward pass, so the residual stream is provably identical between
        // them. Pure autograd math then says the attention parameters (W_qkv,
        // W_o) MUST receive EXACTLY ZERO gradient in both configs.
        //
        // The two configs differ ONLY in the backward pass: kZeroAttnResidual
        // kills the gradient at the residual add (FlashAttention backward sees
        // dO == 0), while kZeroAttnV leaves the FlashAttention forward AND
        // backward fully live (real Q/K/softmax + extreme ALiBi bias, nonzero
        // dO). If the attention backward LEAKS a nonzero dQ/dK under that ALiBi
        // regime, W_qkv receives signal it should not, trains the trunk, and
        // becomes a collapse driver that exists ONLY in kZeroAttnV.
        //
        // This probe reports the per-layer W_qkv / W_o gradient RMS so the leak
        // is a logged NUMBER, not an argument: rms == 0 / nonzero=no => clean
        // (configs provably identical); rms > 0 / nonzero=yes => BACKWARD LEAK.
        // ════════════════════════════════════════════════════════════════════
        if constexpr (GRIM::Ablation::kZeroAttnV || GRIM::Ablation::kZeroAttnResidual) {
            const char* ablation_tag =
                GRIM::Ablation::kZeroAttnV ? "kZeroAttnV" : "kZeroAttnResidual";
            for (int layer = 0; layer < num_layers; ++layer) {
                auto& enc_probe = ctx.parameter_registry->requireEncodingLayerParameters(
                    layer, "verifyGradientsAreConnectedImpl");
                auto probe_attn_param = [&](Tensor& t, const char* param_name) {
                    if (!t.data || !t.has_grad()) {
                        AG_INFO("[ABLATION-GRADLEAK][" << ablation_tag << "] layer=" << layer
                                << " " << param_name << ".grad NOT ALLOCATED (expected for a frozen sublayer)");
                        return;
                    }
                    GradientSignalProbe probe = probeGradientSignal(t, ctx.stream);
                    const bool leaked = probe.nonzero && probe.rms != 0.0f;
                    AG_INFO("[ABLATION-GRADLEAK][" << ablation_tag << "] layer=" << layer
                            << " " << param_name << ".grad rms=" << probe.rms
                            << " nonzero=" << (probe.nonzero ? "yes" : "no")
                            << " finite=" << (probe.finite ? "yes" : "no")
                            << " checked=" << probe.checked
                            << " -> " << (leaked
                                ? "LEAK (attention backward delivered signal that math says must be ZERO -> collapse-driver candidate)"
                                : "clean-zero (no backward leak)"));
                };
                probe_attn_param(enc_probe.W_qkv, "attnWqkv");
                probe_attn_param(enc_probe.W_o, "attnWo");
            }
        }
    }

    return ok;
}
}  // namespace

bool verifyGradientsAreConnected(AutogradContext& ctx) {
    return verifyGradientsAreConnectedImpl(ctx, nullptr);
}

// Finding 2 (Rule 26): computeGradientNorm() DELETED — redundant with
// Phase2's registered-parameter gradient diagnostics.
// The old function duplicated a full L2 norm scan + cudaStreamSynchronize per batch
// whose result was only logged and never consumed by Phase2.

}  // namespace Autograd
}  // namespace GRIM

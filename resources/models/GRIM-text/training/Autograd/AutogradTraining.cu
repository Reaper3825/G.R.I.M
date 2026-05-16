//======================================================//
//  AutogradTraining.cu
//  Implementation of autograd-based training flow
//======================================================//

#include "AutogradTraining.hpp"
#include "AutogradSelectorSupervisionLoss.hpp"

// MUST include full definition of GPUGrimEncoder for method calls
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/grim_layer_gpu.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/Forward/ModelForward_GPU.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/MTP/MTP_GPU.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include "../../Shared/Execution/ExecutionPayloadValidation.hpp"
#include "../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../../Shared/Execution/DecodeTimeNumPolicy.hpp"

#include <iostream>
#include <cmath>
#include <vector>
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
// The -inf masking was poisoning every diagnostic that read cached_logits_tensor
// (logit_min=-inf → logit_range=inf → logit_mean=-inf → logit_std=NaN).
// Inference-time masking in grim_language_model_gpu.cu is SEPARATE and stays.

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
// Encoder owns its weights internally; optimizer accesses gradients via
// Tensor& accessors (enc->attnWqkv().grad_data() etc.).
// See Startup/Model/ParameterGroupRegistration.{hpp,cu}.

// Context initialization lives in AutogradContext.cu so this file can focus on
// the autograd math path: forward, loss, backward, and the training-step bridge.

//======================================================================
// Autograd Forward Pass
// PRODUCTION-READY: Runs entire model with autograd graph intact
//======================================================================

// Rule 20 explicit tape sealing: skip equation-tape D2H/fprintf on non-initial
// accumulation slots. Scope MUST cover the full autograd step (forward + loss +
// backward) — sealing only forward leaves loss/backward to log identical output
// per slot, defeating the optimization and producing duplicate logs.
namespace {
struct TapeSkipScope {
    GRIM::Logging::BatchLogTape* tape;
    bool prev;
    explicit TapeSkipScope(bool skip)
        : tape(GRIM::Logging::getGlobalTape()),
          prev(tape ? tape->skipThisPass() : false) {
        if (tape) tape->setSkipThisPass(skip);
    }
    ~TapeSkipScope() { if (tape) tape->setSkipThisPass(prev); }
    TapeSkipScope(const TapeSkipScope&) = delete;
    TapeSkipScope& operator=(const TapeSkipScope&) = delete;
};

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
    bool selector_loss_active = false;
    bool exec_op_loss_active = false;
    bool exec_arg_loss_active = false;
    bool exec_write_selection_ce_active = false;
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
    activity.selector_loss_active = ctx.training_state
        && !ctx.training_state->autograd_intermediates.selector_fwd_results.empty();
    if (ctx.training_state) {
        const auto& intermediates = ctx.training_state->autograd_intermediates;
        // These flags are set only by computeAutogradLoss() when an execution
        // loss term passes execution_active / teacher_step_mask filtering and is
        // actually added into the normalized execution auxiliary objective.
        // Do not infer activity from exec_outputs_per_row: that includes padded
        // or inactive diagnostics that may never reach loss_tensor.
        activity.exec_op_loss_active = intermediates.exec_op_ce_added
            || intermediates.exec_transition_added;
        activity.exec_arg_loss_active = intermediates.exec_arg_ce_added
            || intermediates.exec_transition_added;
        activity.exec_write_selection_ce_active = intermediates.exec_write_ce_added;
    }
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
    auto captureExpected = [&](Tensor& tensor, const std::string& label) {
        if (tensor.data) {
            baselines.expected.push_back(captureGradientExpectation(tensor, label, ctx.stream));
        }
    };

    if (activity.text_loss_active) {
        captureExpected(ctx.lm_head->weights(), "lm_head weights");
        if (ctx.gpu_encoder && ctx.gpu_encoder->getNumLayers() > 0) {
            auto* enc0 = ctx.gpu_encoder->getLayer(0);
            if (enc0) {
                captureExpected(enc0->attnWqkv(), "layer 0 attnWqkv");
            }
        }
    }

    if (ctx.execution_block && ctx.config->execution_block_enabled) {
        auto& eb = *ctx.execution_block;
        if (activity.exec_op_loss_active) {
            captureExpected(eb.W_op_select(), "exec block W_op_select");
        }
        if (activity.exec_arg_loss_active) {
            captureExpected(eb.w_arg1_select(), "exec block w_arg1_select");
            captureExpected(eb.w_arg2_select(), "exec block w_arg2_select");
        }
        if (activity.exec_write_selection_ce_active) {
            captureExpected(eb.W_write_query(), "exec block W_write_query");
        }
    }

    if (ctx.model && ctx.config->selector_enabled && activity.selector_loss_active) {
        auto* selector = ctx.model->getDecodeTimeSlotSelectorLayer();
        if (selector) {
            captureExpected(selector->W_q_select(), "selector W_q_select");
            captureExpected(selector->W_k_select(), "selector W_k_select");
            captureExpected(selector->null_logit_bias(), "selector null_logit_bias");
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
}  // namespace

ForwardResult executeAutogradForward(AutogradContext& ctx) {
    ctx.validate("executeAutogradForward");

    Forward::ModelForwardRequest request{};
    request.config = ctx.config;
    request.runtime_state = ctx.training_state;
    request.gpu_encoder = ctx.gpu_encoder;
    request.cublas_handle = ctx.cublas_handle;
    request.stream = ctx.stream;
    request.embedding_layer = ctx.embedding_layer;
    request.lm_head = ctx.lm_head;
    request.scratch_block = ctx.scratch_block;
    request.reasoning_head = ctx.reasoning_head;
    request.execution_block = ctx.execution_block;
    request.payload = ctx.payload;
    request.bindings = ctx.device_bindings;
    request.step = ctx.step;
    request.mode = ctx.is_training
        ? Forward::ModelForwardMode::TrainingGraph
        : Forward::ModelForwardMode::EvalNoGrad;

    Forward::ModelForwardResult shared_result = Forward::executeModelForward(request);

    ForwardResult result{};
    result.encoder_output = shared_result.encoder_output;
    result.total_tokens = shared_result.total_tokens;
    result.vocab_size = shared_result.vocab_size;
    result.success = shared_result.success;
    result.error_message = std::move(shared_result.error_message);
    return result;
}
//======================================================================
// Autograd Loss Computation
//======================================================================

LossResult computeAutogradLoss(
    AutogradContext& ctx
) {
    LossResult result{};
    result.success = false;
    
    // RULE 20: Fail loud
    ctx.validate("computeAutogradLoss");
    if (!ctx.payload) {
        throw std::runtime_error("computeAutogradLoss: ctx.payload is NULL — training path MUST set payload via initAutogradContext(const BatchPayload&, ...)");
    }
    const auto& payload = *ctx.payload;
    payload.validate("computeAutogradLoss");
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    
    // RULE 20: Fail loud - validate logits tensor was populated by forward pass
    auto& intermediates = ts->autograd_intermediates;
    if (!intermediates.logits_tensor.data) {
        throw std::runtime_error("computeAutogradLoss: Logits tensor not initialized - call executeAutogradForward() first");
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
    
    // Compute text CE - returns scalar Tensor with NLLLossGradFn → LogSoftmaxGradFn chain
    Tensor loss_tensor = autograd::unified_loss(
        intermediates.logits_tensor,
        payload,
        *ctx.device_bindings,
        ctx.loss_config,
        ctx.d_class_weights,
        ctx.stream
    );
    
    // Move loss tensor to intermediates (TrainingState owns it during backward)
    intermediates.loss_tensor = std::move(loss_tensor);

    float text_ce_loss = 0.0f;
    cudaMemcpyAsync(&text_ce_loss, intermediates.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    if (!std::isfinite(text_ce_loss)) {
        throw std::runtime_error("computeAutogradLoss: pure text CE is non-finite (" +
                                 std::to_string(text_ce_loss) + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. MTP (multi-token prediction) auxiliary losses: L_total += α/K * Σ_k L_k
    // Delegated to MTP_GPU module (see Shared/MTP/MTP_GPU.cu)
    // ═══════════════════════════════════════════════════════════════════════════
    GRIM::MTP::computeMTPAuxiliaryLosses(ctx, intermediates, result.mtp_diagnostics);

    float mtp_loss = 0.0f;
    if (result.mtp_diagnostics.valid) {
        for (float head_contribution : result.mtp_diagnostics.head_loss) {
            mtp_loss += head_contribution;
        }
        result.mtp_diagnostics.L_total = text_ce_loss + mtp_loss;
    }

    float text_plus_mtp_loss = 0.0f;
    cudaMemcpyAsync(&text_plus_mtp_loss, intermediates.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    if (!std::isfinite(text_plus_mtp_loss)) {
        std::string msg = "computeAutogradLoss: text+MTP loss is non-finite (" + std::to_string(text_plus_mtp_loss) + ")";
        if (result.mtp_diagnostics.valid) {
            msg += " L0_main=" + std::to_string(result.mtp_diagnostics.L0_main);
            for (size_t i = 0; i < result.mtp_diagnostics.head_loss.size(); ++i)
                msg += " head_loss[" + std::to_string(i) + "]=" + std::to_string(result.mtp_diagnostics.head_loss[i]);
        }
        throw std::runtime_error(msg);
    }

    result.text_loss = text_ce_loss;
    result.mtp_loss = mtp_loss;
    result.valid_tokens = lm_valid_tokens;

    // ═══════════════════════════════════════════════════════════════════════════
    // EXECUTION BLOCK LOSS — autograd-connected CE + transition loss
    //
    // Two gradient-connected paths:
    //   1. transition_loss — L1 on soft-computed value vs teacher expected_value
    //   2. selection CE — logits-space CE on op/arg1/arg2/write selections
    //      (computed inside executeStep via autograd::cross_entropy_logits)
    //
    // The execution auxiliary objective is averaged over active scalar loss
    // terms before it is added to loss_tensor. This keeps effective execution
    // weight invariant to teacher-step count / batch composition.
    //
    // Monitoring-only (host scalars, no gradients):
    //   - exec_entropy_monitor — distribution collapse detection
    //   - exec_structured_ce — scalar readback of device CE for logging parity
    // ═══════════════════════════════════════════════════════════════════════════
    float exec_structured_ce = 0.0f;
    float exec_entropy_monitor = 0.0f;

    if (cfg->execution_block_enabled && ctx.execution_block
        && !intermediates.exec_outputs_per_row.empty()) {

        if (static_cast<int>(intermediates.exec_outputs_per_row.size()) != payload.batch_size) {
            throw std::runtime_error(
                "computeAutogradLoss: exec_outputs_per_row size="
                + std::to_string(intermediates.exec_outputs_per_row.size())
                + " does not match payload.batch_size=" + std::to_string(payload.batch_size)
                + " — execution forward must produce exactly one row output per payload row");
        }
        if (!ctx.payload->execution_active.empty()
            && static_cast<int>(ctx.payload->execution_active.size()) != payload.batch_size) {
            throw std::runtime_error(
                "computeAutogradLoss: execution_active size="
                + std::to_string(ctx.payload->execution_active.size())
                + " does not match payload.batch_size=" + std::to_string(payload.batch_size)
                + " — Phase1 payload row masks must align with execution outputs");
        }

        intermediates.exec_op_ce_added = false;
        intermediates.exec_arg_ce_added = false;
        intermediates.exec_write_ce_added = false;
        intermediates.exec_transition_added = false;

        const bool have_step_mask = (ctx.payload && !ctx.payload->teacher_step_mask.empty());
        const float ce_weight = cfg->structured_ce_weight;

        // Accumulate autograd losses locally, then normalize once before adding
        // to the main loss. Adding raw per-step sums makes rows with more
        // teacher steps exert larger pressure than rows with fewer steps.
        Tensor exec_loss_sum;
        bool have_exec_loss_sum = false;
        int exec_active_step_count = 0;
        int exec_loss_term_count = 0;
        int ce_tensor_count = 0;
        float ce_scalar_sum = 0.0f;

        enum class ExecLossFlag {
            Op,
            Arg,
            Write,
            Transition
        };

        auto addExecLossTerm = [&](Tensor&& contribution, const char* name, int b, int k, ExecLossFlag flag) {
            if (!contribution.data || !contribution.grad_fn) {
                throw std::runtime_error(
                    std::string("AutogradTraining: execution loss tensor '") + name
                    + "' has no data/grad_fn at row=" + std::to_string(b)
                    + " step=" + std::to_string(k)
                    + " — execution auxiliary loss must remain autograd-connected");
            }
            if (contribution.numel() != 1) {
                throw std::runtime_error(
                    std::string("AutogradTraining: execution loss tensor '") + name
                    + "' is not scalar at row=" + std::to_string(b)
                    + " step=" + std::to_string(k)
                    + " numel=" + std::to_string(contribution.numel()));
            }
            if (!have_exec_loss_sum) {
                exec_loss_sum = std::move(contribution);
                have_exec_loss_sum = true;
            } else {
                exec_loss_sum = autograd::add(exec_loss_sum, contribution, ctx.stream);
            }
            switch (flag) {
                case ExecLossFlag::Op:
                    intermediates.exec_op_ce_added = true;
                    break;
                case ExecLossFlag::Arg:
                    intermediates.exec_arg_ce_added = true;
                    break;
                case ExecLossFlag::Write:
                    intermediates.exec_write_ce_added = true;
                    break;
                case ExecLossFlag::Transition:
                    intermediates.exec_transition_added = true;
                    break;
            }
            exec_loss_term_count++;
        };

        for (int b = 0; b < payload.batch_size; ++b) {
            if (!ctx.payload->execution_active.empty()
                && !ctx.payload->execution_active[b])
                continue;

            const auto& row_steps = intermediates.exec_outputs_per_row[b].steps;
            for (int k = 0; k < static_cast<int>(row_steps.size()); ++k) {
                // Skip padded steps — no gradient contribution
                if (have_step_mask
                    && b < static_cast<int>(ctx.payload->teacher_step_mask.size())
                    && k < static_cast<int>(ctx.payload->teacher_step_mask[b].size())
                    && ctx.payload->teacher_step_mask[b][k] == 0)
                    continue;

                exec_active_step_count++;

                const auto& sout = row_steps[k];

                // transition_loss → autograd graph (L1 value supervision)
                if (sout.transition_loss.data && sout.transition_loss.grad_fn) {
                    auto scaled = autograd::scale_scalar(
                        sout.transition_loss,
                        cfg->execution_block_causal_w1_transition,
                        ctx.stream);
                    addExecLossTerm(std::move(scaled), "transition_loss", b, k, ExecLossFlag::Transition);
                }

                // Fix #6: div_invalid_penalty → autograd graph
                // Penalizes p_op[3] when division was clamped (|v2| < eps).
                // Gradient flows: penalty → p_op → softmax → op_logits → W_op_select.
                if (sout.div_invalid_penalty.data && sout.div_invalid_penalty.grad_fn) {
                    auto scaled = autograd::scale_scalar(
                        sout.div_invalid_penalty,
                        1.0f,
                        ctx.stream);
                    addExecLossTerm(std::move(scaled), "div_invalid_penalty", b, k, ExecLossFlag::Op);
                }

                // Fix #8: div_magnitude_penalty → autograd graph
                // Penalizes large |v_out| after clamped division.
                // Gradient flows: penalty → v_out → FourOpMixGradFn → v1, v2.
                if (sout.div_magnitude_penalty.data && sout.div_magnitude_penalty.grad_fn) {
                    auto scaled = autograd::scale_scalar(
                        sout.div_magnitude_penalty,
                        1.0f,
                        ctx.stream);
                    addExecLossTerm(std::move(scaled), "div_magnitude_penalty", b, k, ExecLossFlag::Transition);
                }

                // Fix #7: arg_reinforce_loss → autograd graph
                // REINFORCE: λ * detached(|v_out-target|) * (-log p_arg[k])
                // Gradient flows ONLY to arg logits. No soft weighting of values.
                if (sout.arg_reinforce_loss.data && sout.arg_reinforce_loss.grad_fn) {
                    auto scaled = autograd::scale_scalar(
                        sout.arg_reinforce_loss,
                        1.0f,
                        ctx.stream);
                    addExecLossTerm(std::move(scaled), "arg_reinforce_loss", b, k, ExecLossFlag::Arg);
                }

                // selection CE → autograd graph (direct per-decision supervision)
                if (cfg->structured_ce_enabled && sout.has_selection_ce) {
                    auto accumulate_ce = [&](const Tensor& ce_tensor, const char* name, ExecLossFlag flag) {
                        if (!ce_tensor.data || !ce_tensor.grad_fn)
                            throw std::runtime_error(
                                std::string("AutogradTraining: selection CE tensor '")
                                + name + "' has no data/grad_fn at row=" + std::to_string(b)
                                + " step=" + std::to_string(k)
                                + " — CrossEntropyLogitsGradFn was not attached");
                        auto scaled = autograd::scale_scalar(ce_tensor, ce_weight, ctx.stream);
                        addExecLossTerm(std::move(scaled), name, b, k, flag);
                    };

                    accumulate_ce(sout.selection_ce_op,    "selection_ce_op",    ExecLossFlag::Op);
                    accumulate_ce(sout.selection_ce_arg1,  "selection_ce_arg1",  ExecLossFlag::Arg);
                    accumulate_ce(sout.selection_ce_arg2,  "selection_ce_arg2",  ExecLossFlag::Arg);
                    accumulate_ce(sout.selection_ce_write, "selection_ce_write", ExecLossFlag::Write);

                    // Scalar readback for logging parity (one sync per step is acceptable
                    // since we're already doing cudaMemcpy for expected_value upload)
                    float h_ce[4];
                    cudaMemcpyAsync(&h_ce[0], sout.selection_ce_op.data,    sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                    cudaMemcpyAsync(&h_ce[1], sout.selection_ce_arg1.data,  sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                    cudaMemcpyAsync(&h_ce[2], sout.selection_ce_arg2.data,  sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                    cudaMemcpyAsync(&h_ce[3], sout.selection_ce_write.data, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                    cudaStreamSynchronize(ctx.stream);
                    ce_scalar_sum += h_ce[0] + h_ce[1] + h_ce[2] + h_ce[3];
                    ce_tensor_count++;
                }
            }
        }

        if (ce_tensor_count > 0) {
            exec_structured_ce = ce_scalar_sum / static_cast<float>(ce_tensor_count);
        }

        if (have_exec_loss_sum) {
            if (exec_loss_term_count <= 0) {
                throw std::runtime_error(
                    "AutogradTraining: have_exec_loss_sum=true but exec_loss_term_count<=0 — execution loss normalization invariant broken");
            }
            const float exec_loss_norm = 1.0f / static_cast<float>(exec_loss_term_count);
            Tensor normalized_exec_loss = autograd::scale_scalar(
                exec_loss_sum,
                exec_loss_norm,
                ctx.stream);
            intermediates.loss_tensor = autograd::add(
                intermediates.loss_tensor, normalized_exec_loss, ctx.stream);
            AG_INFO("Execution auxiliary loss normalized over " << exec_loss_term_count
                    << " scalar loss terms across " << exec_active_step_count
                    << " active execution steps");
        }

        // Optional entropy monitor (non-differentiable; not added to loss_tensor)
        if (cfg->entropy_aux_weight > 0.0f) {
            int monitored_entropy_rows = 0;
            for (int b = 0; b < payload.batch_size; ++b) {
                if (!ctx.payload->execution_active.empty()
                    && !ctx.payload->execution_active[b])
                    continue;

                const auto& all_steps = intermediates.exec_outputs_per_row[b].steps;
                std::vector<const ExecutionBlockStepOutput*> real_steps;
                if (have_step_mask
                    && b < static_cast<int>(ctx.payload->teacher_step_mask.size())
                    && !ctx.payload->teacher_step_mask[b].empty()) {
                    for (int k = 0; k < static_cast<int>(all_steps.size()); ++k) {
                        if (k < static_cast<int>(ctx.payload->teacher_step_mask[b].size())
                            && ctx.payload->teacher_step_mask[b][k] == 0)
                            continue;
                        real_steps.push_back(&all_steps[k]);
                    }
                } else {
                    for (const auto& s : all_steps)
                        real_steps.push_back(&s);
                }
                if (real_steps.empty()) {
                    continue;
                }
                Tensor ent = ctx.execution_block->computeEntropyLoss(
                    real_steps,
                    cfg->entropy_aux_weight,
                    ctx.stream);
                if (!ent.data) {
                    throw std::runtime_error(
                        "computeAutogradLoss: execution entropy monitor returned NULL tensor data for row="
                        + std::to_string(b) + " with real_steps=" + std::to_string(real_steps.size()));
                }
                float h_ent = 0.0f;
                cudaStreamSynchronize(ctx.stream);
                cudaMemcpy(&h_ent, ent.data, sizeof(float), cudaMemcpyDeviceToHost);
                exec_entropy_monitor += h_ent;
                monitored_entropy_rows++;
            }
            if (monitored_entropy_rows > 0)
                exec_entropy_monitor /= static_cast<float>(monitored_entropy_rows);
        }
    }
    result.entropy_monitor = exec_entropy_monitor;

    // ═══════════════════════════════════════════════════════════════════════════
    // SELECTOR SUPERVISION LOSS — autograd CE through TensorContract
    // Selector supervision is FINAL-STATE selector-only supervision: each row may
    // provide at most one non-Ignore target, and it MUST be attached to the last
    // decode position because the available ExecutionMemory is the row's final
    // post-execution state. Per-token non-Ignore targets would require a
    // timestep-aligned ExecutionMemory snapshot and are rejected loudly.
    // The hidden input is copied into an owned detached Tensor. Gradients train
    // DecodeTimeSlotSelector parameters only; they do not slice back into the
    // encoder hidden tensor because TensorContract has no parent slice/view op.
    // Weighted by cfg->selector_supervision_weight (0 = disabled).
    // ═══════════════════════════════════════════════════════════════════════════
    float selector_supervision_loss = addSelectorSupervisionLoss(ctx, intermediates);
    result.selector_loss = selector_supervision_loss;

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUND-TRUTH LOSS: Read the ACTUAL tensor that backward will differentiate.
    // This is the single source of truth — no manual reconstruction from stale
    // host-side scalars. text_loss was snapshot before exec/selector additions.
    // ═══════════════════════════════════════════════════════════════════════════
    float actual_loss = 0.0f;
    cudaMemcpyAsync(&actual_loss, intermediates.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    result.loss_value = actual_loss;
    result.numeric_loss = actual_loss - text_ce_loss - mtp_loss - selector_supervision_loss;
    result.aux_loss = actual_loss - text_ce_loss;  // MTP + execution/numeric + selector
    result.weight_text = 1.0f;
    
    if (!std::isfinite(result.loss_value)) {
        throw std::runtime_error("computeAutogradLoss: combined loss is non-finite (actual_tensor=" 
            + std::to_string(actual_loss) + " text_ce=" + std::to_string(text_ce_loss)
            + " mtp=" + std::to_string(mtp_loss)
            + " exec_ce=" + std::to_string(exec_structured_ce)
            + " selector=" + std::to_string(selector_supervision_loss) + ")");
    }
    
    AG_INFO("Loss computed: total=" << actual_loss << " text_ce=" << text_ce_loss
            << " mtp=" << mtp_loss
            << " numeric_exec=" << result.numeric_loss
            << " exec_ce=" << exec_structured_ce
            << " exec_entropy_monitor=" << exec_entropy_monitor
            << " selector=" << selector_supervision_loss
            << " lm_valid=" << lm_valid_tokens);
    
    result.success = true;
    return result;
}

//======================================================================
// Autograd Backward Pass
//======================================================================

BackwardResult executeAutogradBackward(
    AutogradContext& ctx,
    bool accumulate,
    float grad_scale
) {
    BackwardResult result{};
    result.success = false;
    result.grad_rms = 0.0f;
    
    ctx.validate("executeAutogradBackward");
    if (!std::isfinite(grad_scale) || grad_scale <= 0.0f) {
        throw std::runtime_error("executeAutogradBackward: grad_scale must be finite and > 0, got " +
                                 std::to_string(grad_scale));
    }
    
    auto* ts = ctx.training_state;
    auto& intermediates = ts->autograd_intermediates;
    if (!intermediates.loss_tensor.data) {
        throw std::runtime_error("executeAutogradBackward: Loss tensor not initialized - call computeAutogradLoss() first");
    }
    
    if (!intermediates.loss_tensor.grad_fn) {
        throw std::runtime_error("executeAutogradBackward: Loss tensor has no grad_fn - autograd chain broken");
    }
    if (!ctx.model) {
        throw std::runtime_error("executeAutogradBackward: model is NULL - registered parameter gradient lifecycle requires LanguageModel");
    }
    
    AG_INFO("Executing backward pass (accumulate=" << accumulate << ", scale=" << grad_scale << ")");

    // Parameter gradients are lifecycle-managed through the registered
    // TensorContract ParameterGroup inventory. AutogradTraining must not walk
    // layer internals or special-case optional heads here.
    if (!accumulate) {
        GRIM::zeroParameterGradients(ctx.model->parameterGroups(), ctx.stream);
    }

    // On accumulation slots, existing parameter grad buffers intentionally hold
    // earlier microbatch contributions. A post-backward nonzero check alone can
    // therefore pass even if this microbatch delivered no signal. Snapshot only
    // the tensors that verification expects to receive signal, then require a
    // nonzero pre/post delta for those tensors after backward.
    GradientSignalBaselines gradient_signal_baselines =
        captureGradientVerificationBaselines(ctx, accumulate);
    
    // Call backward on the text loss (single loss path)
    // Starting with grad_scale (usually 1/accumulation_steps)
    AG_INFO("Calling loss_tensor.backward(nullptr, " << grad_scale << ")...");
    intermediates.loss_tensor.backward(nullptr, grad_scale);
    AG_INFO("loss_tensor.backward() returned successfully");

    // ════════════════════════════════════════════════════════════════════
    // DIAGNOSTIC: Sample gradient values immediately after backward to
    // determine if backward itself produces zeros or if something later
    // corrupts the buffers.  Issue: "zero gradients every other batch"
    // ════════════════════════════════════════════════════════════════════
    {
        cudaStreamSynchronize(ctx.stream);

        // Sample LM head weight gradient (first element)
        float lm_sample = 0.0f;
        float* lm_grads = ctx.lm_head->weights().grad_data();
        if (lm_grads) {
            cudaMemcpy(&lm_sample, lm_grads, sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Sample first encoder layer attnWqkv gradient
        float enc_sample = 0.0f;
        if (ctx.gpu_encoder && ctx.gpu_encoder->getNumLayers() > 0) {
            auto* enc0 = ctx.gpu_encoder->getLayer(0);
            if (enc0) {
                float* wqkv_grads = enc0->attnWqkv().grad_data();
                if (wqkv_grads) {
                    cudaMemcpy(&enc_sample, wqkv_grads, sizeof(float), cudaMemcpyDeviceToHost);
                }
            }
        }

        // Sample RMSNorm gamma gradient (layer 0)
        float rms_sample = 0.0f;
        if (ctx.gpu_encoder && ctx.gpu_encoder->getNumLayers() > 0) {
            auto* enc0 = ctx.gpu_encoder->getLayer(0);
            if (enc0) {
                float* rms_grads = enc0->rms1Gamma().grad_data();
                if (rms_grads) {
                    cudaMemcpy(&rms_sample, rms_grads, sizeof(float), cudaMemcpyDeviceToHost);
                }
            }
        }

        fprintf(stderr,
            "[GRAD_DIAG] POST-BACKWARD accumulate=%d grad_scale=%.4f "
            "lm_grad[0]=%.10e enc_wqkv_grad[0]=%.10e rms_gamma_grad[0]=%.10e "
            "lm_ptr=%p\n",
            static_cast<int>(accumulate), grad_scale,
            lm_sample, enc_sample, rms_sample,
            static_cast<void*>(lm_grads));
    }

    // ScratchBlock backward is now automatic via ScratchBlockGradFn in the autograd chain.
    // loss_tensor.backward() → ... → ScratchBlockGradFn::apply() handles parameter gradients.
    
    // Verify gradients are properly connected before optimizer runs
    AG_INFO("Verifying gradients are connected to optimizer...");
    if (!verifyGradientsAreConnectedImpl(ctx, &gradient_signal_baselines)) {
        throw std::runtime_error(
            "executeAutogradBackward: Gradient connectivity verification failed after backward mutated parameter gradients");
    }
    AG_INFO("Gradient connectivity verified");
    
    // ISSUE #149: Manual parameter gradient scaling REMOVED.
    // We now scale the root gradient (the loss) at the start of backward()
    // which propagates the scale through the entire computation graph.
    // This is mathematically equivalent, more efficient (fewer kernels),
    // and safer against omission bugs when adding new layers.
    
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
    const bool reasoning_loss_active = false;  // ReasoningHead forward is diagnostic-only until a real reasoning loss path is assembled.

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
            if (!delta_probe.changed || delta_probe.delta_rms == 0.0f) {
                AG_WARN(label << ".grad did not change during this accumulation microbatch "
                        << "(checked=" << delta_probe.checked << ") — current backward path did not deliver signal");
                ok = false;
                return;
            }
            AG_INFO(label << ".grad received current-microbatch signal; checked="
                    << delta_probe.checked << " delta_rms=" << delta_probe.delta_rms);
            return;
        }

        GradientSignalProbe probe = probeGradientSignal(t, ctx.stream);
        if (!probe.nonzero || probe.rms == 0.0f) {
            AG_WARN(label << ".grad is allocated but full-tensor RMS is zero after this backward "
                    << "(checked=" << probe.checked << ") — gradient path did not deliver signal");
            ok = false;
        }
    };
    
    // The autograd system stores gradients in Tensor.grad_ fields (shared_ptr<Tensor>)
    // The optimizer accesses them directly via Tensor.grad_data() pointers.
    // 
    // This function verifies that gradients are properly allocated and accessible.
    // It does NOT copy — gradients are already wired up during initialization.
    // This is a diagnostic check to catch pointer setup bugs before optimizer runs.
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Embedding gradients (may be tied with LM head) — Pattern B: owned by EmbeddingLayer
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() accessor
    if (ctx.embedding_layer->tokenWeights().data) {
        requireAllocatedFinite(ctx.embedding_layer->tokenWeights(), "embedding token_weights");
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LM head gradients (Pattern B: owned by persistent LMHeadLayer)
    // May be tied to embedding (same underlying grad Tensor)
    // ═══════════════════════════════════════════════════════════════════════════
    if (ctx.lm_head->weights().data) {
        if (!ctx.lm_head->weights().has_grad()) {
            AG_WARN("lm_head weights.grad is NULL - gradients NOT flowing to optimizer!");
            ok = false;
        } else {
            // Check if tied to embedding (same underlying grad Tensor)
            if (ctx.lm_head->weights().grad_data() == ctx.embedding_layer->tokenWeights().grad_data()) {
                AG_INFO("LM head gradients TIED to embedding: " << ctx.lm_head->weights().numel() << " elements");
            } else {
                AG_INFO("LM head gradients SEPARATE: " << ctx.lm_head->weights().numel() << " elements at " << ctx.lm_head->weights().grad_data());
            }
            if (activity.text_loss_active) {
                requireReceivedGradient(ctx.lm_head->weights(), "lm_head weights");
            } else {
                requireAllocatedFinite(ctx.lm_head->weights(), "lm_head weights");
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Logits gradients are TensorContract-owned.
    // ═══════════════════════════════════════════════════════════════════════════
    // LMHead returns non-leaf logits, so LogSoftmaxGradFn allocates its own
    // non-leaf input_grad buffer and passes that view directly to the upstream
    // logits grad_fn. TrainingState must not mirror or copy logits gradients.
    
    // NOTE: Encoder gradients are in encoder's internal Tensors, not TrainingState.
    // The optimizer accesses them via Tensor& accessors (enc->attnWqkv() etc.).
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Final RMSNorm gamma (Pattern B: owned by LMHeadLayer)
    // When lm_head_freeze_final_rms_gamma=true, requires_grad=false and we skip the check
    // (no grad is correct, not a bug).
    // ═══════════════════════════════════════════════════════════════════════════
    if (ctx.lm_head->finalRmsGamma().data
        && ctx.lm_head->finalRmsGamma().requires_grad
        && !ctx.lm_head->finalRmsGamma().has_grad()) {
        AG_WARN("final_rms_gamma.grad is NULL - gradients NOT flowing!");
        ok = false;
    } else if (ctx.lm_head->finalRmsGamma().data && ctx.lm_head->finalRmsGamma().requires_grad) {
        requireAllocatedFinite(ctx.lm_head->finalRmsGammaMutable_UnfrozenOnly("verifyGradientsAreConnected"),
                               "final_rms_gamma");
    }

    if (ctx.lm_head->bias().data && !ctx.lm_head->bias().has_grad()) {
        AG_WARN("lm_head_bias.grad is NULL - gradients NOT flowing!");
        ok = false;
    } else if (ctx.lm_head->bias().data) {
        requireAllocatedFinite(ctx.lm_head->bias(), "lm_head_bias");
    }

    if (ctx.gpu_encoder) {
        const int num_layers = ctx.gpu_encoder->getNumLayers();
        for (int layer = 0; layer < num_layers; ++layer) {
            auto* enc = ctx.gpu_encoder->getLayer(layer);
            if (!enc) {
                AG_WARN("encoder layer " << layer << " is NULL during gradient verification");
                ok = false;
                continue;
            }
            auto check = [&](Tensor& t, const char* name) {
                if (t.data) requireAllocatedFinite(t, "layer " + std::to_string(layer) + " " + std::string(name));
            };
            check(enc->rms1Gamma(), "rms1Gamma");
            check(enc->rms2Gamma(), "rms2Gamma");
            // Issue #148: Sandwich norm gammas REMOVED
            check(enc->attnWqkv(), "attnWqkv");
            check(enc->attnBqkv(), "attnBqkv");
            check(enc->attnWo(), "attnWo");
            check(enc->attnBo(), "attnBo");
            check(enc->ffnWGate(), "ffnWGate");
            check(enc->ffnW1(), "ffnW1");
            check(enc->ffnW2(), "ffnW2");
            check(enc->ffnB2(), "ffnB2");
            check(enc->layerScale1(), "layerScale1");
            check(enc->layerScale2(), "layerScale2");
        }
        if (activity.text_loss_active && num_layers > 0) {
            auto* enc0 = ctx.gpu_encoder->getLayer(0);
            if (enc0) {
                requireReceivedGradient(enc0->attnWqkv(), "layer 0 attnWqkv");
            }
        }
    }

    if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
        auto checkScratch = [&](Tensor& t, const char* name) {
            if (t.data) requireAllocatedFinite(t, "scratch block " + std::string(name));
        };
        checkScratch(ctx.scratch_block->atomTypeEmbeddings(), "atomTypeEmbeddings");
        checkScratch(ctx.scratch_block->atomProjection(), "atomProjection");
    }

    if (ctx.model && !GRIM::MTP::verifyMTPGradients(*ctx.model)) {
        ok = false;
    }

    // ExecutionBlock parameters
    if (ctx.execution_block && ctx.config->execution_block_enabled) {
        auto checkEB = [&](Tensor& t, const char* name) {
            if (t.data) requireAllocatedFinite(t, "exec block " + std::string(name));
        };
        auto& eb = *ctx.execution_block;
        checkEB(eb.w_decode_1(), "w_decode_1");
        checkEB(eb.b_decode_1(), "b_decode_1");
        checkEB(eb.w_decode_2(), "w_decode_2");
        checkEB(eb.w_arg1_select(), "w_arg1_select");
        checkEB(eb.w_arg2_select(), "w_arg2_select");
        checkEB(eb.W_op_select(), "W_op_select");
        checkEB(eb.W_key_proj(), "W_key_proj");
        checkEB(eb.W_write_query(), "W_write_query");
        checkEB(eb.W_write_key(), "W_write_key");
        checkEB(eb.alpha(), "alpha");
        checkEB(eb.beta(), "beta");
        checkEB(eb.step_embeddings(), "step_embeddings");
        checkEB(eb.type_num_embed(), "type_num_embed");
        checkEB(eb.W_value_to_emb(), "W_value_to_emb");
        checkEB(eb.b_value_to_emb(), "b_value_to_emb");
        checkEB(eb.w_inject_gate(), "w_inject_gate");
        checkEB(eb.W_Q_read(), "W_Q_read");
        checkEB(eb.W_K_read(), "W_K_read");
        checkEB(eb.W_V_read(), "W_V_read");
        checkEB(eb.W_O_read(), "W_O_read");
        checkEB(eb.W_gate_read(), "W_gate_read");
        checkEB(eb.tau(), "tau");
        checkEB(eb.E_slot(), "E_slot");
        checkEB(eb.E_op(), "E_op");
        checkEB(eb.W_scal(), "W_scal");
        checkEB(eb.b_scal(), "b_scal");
        checkEB(eb.W_trace(), "W_trace");
        checkEB(eb.b_trace(), "b_trace");
        checkEB(eb.W_reason_gate(), "W_reason_gate");
        checkEB(eb.W_trace_gate(), "W_trace_gate");
        if (activity.exec_op_loss_active) {
            requireReceivedGradient(eb.W_op_select(), "exec block W_op_select");
        }
        if (activity.exec_arg_loss_active) {
            requireReceivedGradient(eb.w_arg1_select(), "exec block w_arg1_select");
            requireReceivedGradient(eb.w_arg2_select(), "exec block w_arg2_select");
        }
        if (activity.exec_write_selection_ce_active) {
            requireReceivedGradient(eb.W_write_query(), "exec block W_write_query");
        }
    }

    // DecodeTimeSlotSelector parameters
    if (ctx.model && ctx.config->selector_enabled) {
        auto* selector = ctx.model->getDecodeTimeSlotSelectorLayer();
        if (selector) {
            auto checkSel = [&](Tensor& t, const char* name) {
                if (t.data) requireAllocatedFinite(t, "selector " + std::string(name));
            };
            checkSel(selector->W_q_select(), "W_q_select");
            checkSel(selector->W_k_select(), "W_k_select");
            checkSel(selector->null_key_select(), "null_key_select");
            checkSel(selector->null_logit_bias(), "null_logit_bias");
            if (activity.selector_loss_active) {
                requireReceivedGradient(selector->W_q_select(), "selector W_q_select");
                requireReceivedGradient(selector->W_k_select(), "selector W_k_select");
                requireReceivedGradient(selector->null_logit_bias(), "selector null_logit_bias");
            }
        }
    }
    
    // ReasoningHead parameters: executeAutogradForward currently invokes the
    // head for structured diagnostics only. No reasoning loss is assembled into
    // intermediates.loss_tensor, so verifying these params for received gradient
    // would falsely claim training connectivity. Re-enable only alongside a real
    // reasoning loss path.
    if (ctx.reasoning_head && reasoning_loss_active) {
        auto checkRH = [&](Tensor& t, const char* name) {
            if (t.data) requireAllocatedFinite(t, "reasoning head " + std::string(name));
        };
        checkRH(ctx.reasoning_head->W_op(), "W_op");
        checkRH(ctx.reasoning_head->b_op(), "b_op");
        checkRH(ctx.reasoning_head->w_arg1(), "w_arg1");
        checkRH(ctx.reasoning_head->w_arg2(), "w_arg2");
    } else if (ctx.reasoning_head) {
        AG_INFO("ReasoningHead gradient verification skipped: no reasoning loss path is connected to loss_tensor");
    }
    
    return ok;
}
}  // namespace

bool verifyGradientsAreConnected(AutogradContext& ctx) {
    return verifyGradientsAreConnectedImpl(ctx, nullptr);
}

// Finding 2 (Rule 26): computeGradientNorm() DELETED — redundant with
// Phase2's ctx.model->computeGradNorm(true) which produces per-component breakdown.
// The old function duplicated a full L2 norm scan + cudaStreamSynchronize per batch
// whose result was only logged and never consumed by Phase2.

//======================================================================
// Main Entry Point
//======================================================================

LossResult autogradTrainingStep(
    LanguageModel& model,
    TrainingState& training_state,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const HyperParameters::LossConfigHP& loss_config,
    bool accumulate,
    float grad_scale,
    uint64_t step
) {
    payload.validate("autogradTrainingStep");

    const auto& cfg = model.getConfig();

    // Execution payload validation (WS4: single shared validator)
    GRIM::Execution::validateExecutionPayload(
        payload, "autogradTrainingStep",
        cfg.execution_block_num_slots, cfg.execution_block_num_ops, cfg.execution_block_num_steps);

    // When execution_block is disabled, teacher_steps are ignored — batch trains with plain cross-entropy.
    if (!payload.teacher_steps.empty() && !cfg.execution_block_enabled) {
        AG_WARN("batch has teacher_steps (arithmetic) but execution_block_enabled=false; "
                "training with plain cross-entropy over text tokens (teacher supervision skipped)");
    }

    // WS8: Structural layer availability — crash loud if config says enabled but layers are missing.
    if (cfg.execution_block_enabled) {
        if (!model.getExecutionBlockLayer()) {
            throw std::runtime_error(
                "autogradTrainingStep: execution_block_enabled but ExecutionBlock layer is null");
        }
        ScratchBlockLayer* sb_check = model.getScratchBlockLayer();
        if (!sb_check || !sb_check->isEnabled()) {
            throw std::runtime_error(
                "autogradTrainingStep: execution_block_enabled requires ScratchBlock enabled");
        }
    }

    const int total_tokens = payload.total_tokens;
    
    // Get encoder for autograd forward
    GPUGrimEncoder& gpu_encoder = model.getGpuEncoder();
    EmbeddingLayer* embedding_layer = model.getEmbeddingLayer();
    LMHeadLayer* lm_head = model.getLmHeadLayer();
    ScratchBlockLayer* scratch_block = model.getScratchBlockLayer();
    ReasoningHeadLayer* reasoning_head = model.getReasoningHeadLayer();
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();

    // ═══════════════════════════════════════════════════════════════════════════
    // GPU COPIES: handled upstream by LanguageModel::uploadBatchToDevice(payload).
    // initAutogradContext is the single sync-boundary validator for the returned
    // BatchDeviceBindings; this step never authors payload geometry or H2D copies.
    // ═══════════════════════════════════════════════════════════════════════════

    // Logit buffer capacity is still enforced here so a mis-sized payload trips
    // immediately, before any forward kernel is launched.
    const auto& logits_shape = training_state.cached_logits_tensor.shape.require("autogradTrainingStep cached_logits_tensor");
    if (!logits_shape.is_2d_layout()) {
        throw std::runtime_error("autogradTrainingStep: cached_logits_tensor must be a 2D LOGITS buffer");
    }
    const auto logits_dims = logits_shape.as_2d();
    const size_t logit_limit = static_cast<size_t>(logits_dims.rows);
    if (static_cast<size_t>(total_tokens) > logit_limit) {
        throw std::runtime_error(
            "autogradTrainingStep: total_tokens=" + std::to_string(total_tokens) +
            " exceeds logit buffer capacity=" + std::to_string(logit_limit));
    }
    if (logits_dims.cols != payload.vocab_size) {
        throw std::runtime_error(
            "autogradTrainingStep: cached_logits_tensor cols=" + std::to_string(logits_dims.cols) +
            " != payload/cfg vocab_size=" + std::to_string(payload.vocab_size) +
            " (rows=" + std::to_string(logits_dims.rows) +
            ", total_tokens=" + std::to_string(total_tokens) + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AUTOGRAD CONTEXT
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Write authoritative training step to TrainingState BEFORE building context.
    // This is the ONLY mutation site for autograd_step.
    training_state.autograd_step = step;

    // Rule 20 explicit tape sealing: skip equation-tape D2H/fprintf on non-initial
    // accumulation slots across the ENTIRE step (forward + loss + backward). Sealing
    // only forward (the previous behavior) was a Rule 20 violation: loss and
    // backward kept logging duplicated tape entries on every slot.
    TapeSkipScope tape_skip_scope(accumulate);

    ExecutionBlockLayer* execution_block = model.getExecutionBlockLayer();

    AutogradContext ctx = initAutogradContext(
        &cfg,
        &training_state,
        &gpu_encoder,
        embedding_layer,
        lm_head,
        scratch_block,
        reasoning_head,
        execution_block,
        training_state.cublas_handle.get(),
        stream,
        payload,
        bindings,
        loss_config,
        step,
        true
    );
    if (loss_config.class_balanced_enabled) {
        if (!training_state.class_weights_tensor.data) {
            throw std::runtime_error("autogradTrainingStep: class_balanced_enabled=true but class_weights_tensor is NULL");
        }
        if (training_state.class_weights_vocab_size != payload.vocab_size) {
            throw std::runtime_error("autogradTrainingStep: class_weights_vocab_size=" +
                std::to_string(training_state.class_weights_vocab_size) + " != payload.vocab_size=" +
                std::to_string(payload.vocab_size));
        }
        ctx.d_class_weights = training_state.class_weights_tensor.data;
    } else {
        ctx.d_class_weights = nullptr;
    }
    ctx.skip_equation_logging = accumulate;  // Skip D2H + fprintf on non-initial accumulation slots
    ctx.model = &model;  // For MTP head access in computeAutogradLoss

    // ═══════════════════════════════════════════════════════════════════════════
    // FORWARD → LOSS → BACKWARD
    // ═══════════════════════════════════════════════════════════════════════════

    // Allocate read-gate accumulator once (Category 3 workspace on TrainingState)
    if (!training_state.read_gate_accum_tensor.data && cfg.execution_block_enabled) {
        training_state.read_gate_accum_tensor = Tensor::zeros({2}, stream, "read_gate_accum");
    }
    // Zero the accumulator before forward (sum=0, count=0)
    if (training_state.read_gate_accum_tensor.data) {
        CUDA_CHECK(cudaMemsetAsync(training_state.read_gate_accum_tensor.data, 0, 2 * sizeof(float), stream));
    }
    
    ForwardResult fwd_result = executeAutogradForward(ctx);
    if (!fwd_result.success) {
        throw std::runtime_error("autogradTrainingStep: Forward failed - " + fwd_result.error_message);
    }

    // Read back the cross-attention read gate accumulator (sum/count on device)
    // Snapshot Category 3 workspace into Category 2 telemetry scalar BEFORE the
    // autograd boundary (Rule 20).
    if (training_state.read_gate_accum_tensor.data) {
        float h_accum[2] = {0.0f, 0.0f};
        CUDA_CHECK(cudaMemcpyAsync(h_accum, training_state.read_gate_accum_tensor.data,
                                   2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        training_state.h_read_gate_mean = (h_accum[1] > 0.0f)
            ? (h_accum[0] / h_accum[1])
            : 0.0f;
    }
    
    LossResult loss_result = computeAutogradLoss(ctx);
    if (!loss_result.success) {
        loss_result.error_message = "autogradTrainingStep: Loss failed - " + loss_result.error_message;
        // Rule 20 single-owner clear: caller's AutogradStepScope handles intermediates.
        return loss_result;
    }
    
    // Rule 20: Non-finite loss means forward produced garbage.
    // Skip backward entirely — don't propagate NaN/Inf gradients.
    if (!std::isfinite(loss_result.loss_value)) {
        loss_result.success = false;
        loss_result.error_message = "Non-finite loss: " + std::to_string(loss_result.loss_value);
        // Rule 20 single-owner clear: caller's AutogradStepScope handles intermediates.
        return loss_result;
    }
    
    BackwardResult bwd_result = executeAutogradBackward(ctx, accumulate, grad_scale);
    if (!bwd_result.success) {
        loss_result.success = false;
        loss_result.error_message = "autogradTrainingStep: Backward failed - " + bwd_result.error_message;
        // Rule 20 single-owner clear: caller's AutogradStepScope handles intermediates.
        return loss_result;
    }
    
    // Post-backward cleanup (matches LanguageModel::backward() behavior)
    training_state.sequence_weight_count = 0;
    
    // Rule 20 ownership taxonomy: AutogradIntermediates::clear() is owned by
    // the caller's AutogradStepScope RAII guard. Do NOT clear here. Post-step
    // diagnostics (Phase2_TrainingLoop, GuessCache) read
    // from TrainingState::cached_logits_tensor (Cat 3 step-output snapshot),
    // never from intermediates.logits_tensor (Cat 1, transient).
    
    AG_INFO("Training step complete: loss=" << loss_result.loss_value);
    
    return loss_result;
}
}  // namespace Autograd
}  // namespace GRIM

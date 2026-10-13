//======================================================//
// TimeConditionedEulerOrchestration.hpp
// Unwired payload contracts and NON-EXECUTABLE orchestration sketch.
// No entry point, implementation, config reader, or caller is installed here.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <array>
#include <cstdint>
#include <vector>

#include "ModelForward_GPU.hpp"
#include "../../Layers/FeedForward/Feed_Forward_GPU.hpp"
#include "../../Layers/FlashAttention/EncoderSelfAttention_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

namespace GRIM::Forward {

// Authored numerical policy, separate from batch geometry and device bindings.
// No defaults: a future caller must supply the entire integration contract.
// Diagram instance: {12, 0.0f, 12.0f, beta}; dt = 1, tau_k = k / 12.
struct TimeConditionedEulerPlan {
    int num_steps;
    float time_begin;
    float time_end;
    float attention_to_ffn_beta;
};

// Proposed shared conditioner: [tau] -> linear -> SiLU -> linear.
// Output layout: [attention_shift(D), ffn_shift(D), attention_gain(1), ffn_gain(1)].
// Gains are direct learned outputs; no hidden sigmoid or fixed depth multiplier.
// These are borrowed graph-policy-selected views, not new parameter owners.
struct TimeConditionerParameterPayload {
    const Tensor& input_weight;   // [1, conditioner_width], DIRECT_WEIGHT
    const Tensor& input_bias;     // [1, conditioner_width]
    const Tensor& output_weight;  // [conditioner_width, 2 * d_model + 2]
    const Tensor& output_bias;    // [1, 2 * d_model + 2]
};

struct SharedTimeConditionedParameterPayload {
    const EncodingLayerParameterTensors& encoding;
    const FeedForwardParameterTensors& ffn;
    const TimeConditionerParameterPayload& conditioner;

    // Optional borrowed projections of the SAME shared block. A future adapter
    // must bind these into every execution slot without registering copies.
    // Null is valid only when LoRA is disabled by the authored model contract.
    const std::array<LoRAProjectionView, 5>* lora_projections;
};

// Execution dependencies are supplied explicitly. No TrainingState lookup,
// global "current batch", registry indexing, or device upload occurs here.
struct TimeConditionedEulerExecutionPayload {
    const Batching::BatchPayload& batch;
    const Batching::BatchDeviceBindings& bindings;
    const PBM::PBMState& pbm;
    const HyperParameters::EncoderLayerConstructionHP& encoder_hp;
    FeedForwardLayer& ffn_compute;
    ModelForwardGraphPolicy graph;
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    std::uint64_t batch_idx;

    // Optional session owner, already allocated with num_steps cache slots.
    // Weight sharing NEVER maps all execution steps to cache slot zero.
    KvCacheState* kv_cache;
};

struct TimeConditionedEulerRequest {
    const TimeConditionedEulerPlan& plan;
    const TimeConditionedEulerExecutionPayload& execution;
    const SharedTimeConditionedParameterPayload& parameters;
    const Tensor& initial_hidden;  // [batch.total_tokens, d_model], already embedded
};

// One immutable step datum. Time is integration time, not token position;
// RoPE/ALiBi continue to consume the existing PBM and batch bindings.
struct TimeConditionedEulerStepPayload {
    int step_index;                // activation/cache/dropout slot, NOT parameter index
    float time;
    float normalized_time;
    float delta_time;
    const Tensor& hidden;
    const KvCacheLayerView* cache; // null for full-sequence attention
};

struct TimeConditionedVectorFieldRequest {
    const TimeConditionedEulerExecutionPayload& execution;
    const SharedTimeConditionedParameterPayload& parameters;
    const TimeConditionedEulerStepPayload& step;
    float attention_to_ffn_beta;
};

// Extra retained tensors alongside the existing ModelForwardOutputs step slot.
// Keep these alive through loss/backward: some GradFns borrow activation data.
// In particular, do not reuse one conditioner buffer across all Euler steps.
struct TimeConditionedEulerStepOutputs {
    Tensor conditioner_time;       // [1, 1], constant tau, requires_grad=false
    Tensor conditioner_linear;
    Tensor conditioner_hidden;
    Tensor conditioner_projected;
    Tensor conditioner_output;     // [1, 2 * d_model + 2]
    Tensor attention_shift;        // [1, d_model]
    Tensor ffn_shift;              // [1, d_model]
    Tensor attention_gain;         // [1, 1], stays on device/in the graph
    Tensor ffn_gain;               // [1, 1], stays on device/in the graph
    Tensor attention_gain_channels; // [1, d_model], replicated scalar for layer_scale
    Tensor ffn_gain_channels;      // [1, d_model], replicated scalar for layer_scale
    Tensor attention_input;        // RMSNorm(H) + attention_shift
    Tensor ffn_input;              // RMSNorm(U) + ffn_shift
    Tensor beta_attention;         // beta * D_A
    Tensor gained_attention;       // g_A(t) * D_A
    Tensor gained_ffn;             // g_F(t) * D_F
    Tensor vector_field;           // f_theta(H, t), [tokens, d_model]
    Tensor euler_increment;        // dt * f_theta(H, t)
};

// Caller-owned per-call sink. Keep BOTH the canonical sink and this extension
// through the graph window. Canonical per-layer vectors are execution slots.
// final_hidden lives in canonical.output_per_layer.back(); no independent copy
// or second parameter owner is implied by this payload.
struct TimeConditionedEulerOutputPayload {
    ModelForwardOutputs& canonical;
    std::vector<TimeConditionedEulerStepOutputs> steps;
    Tensor scalar_to_channels;     // constant ones [1, d_model], no grad
};

/*
NON-EXECUTABLE PSEUDOCODE — proposed helper names below are not callable APIs.

void orchestrateTimeConditionedEuler(
    const TimeConditionedEulerRequest& request,
    TimeConditionedEulerOutputPayload& outputs)
{
    const auto& plan = request.plan;
    const auto& exec = request.execution;

    validateEulerRequest(request, outputs);
    // Fail loudly unless:
    // - num_steps > 0; times, duration, dt, beta are finite; time_end > time_begin.
    // - initial_hidden is [batch.total_tokens, encoder_hp.d_model].
    // - batch/bindings/PBM geometry and all shared parameter shapes agree.
    // - stream and cuBLAS handle are valid; the input/parameters obey graph policy.
    // - parameter views are connected when requested, detached otherwise. The
    //   caller owns selected views through backward; no temporary leaf copies.
    // - canonical step vectors and outputs.steps are empty on entry. Embedding
    //   and metadata already on canonical are allowed and remain untouched.
    // - optional gates/LoRA are explicitly bound according to encoder_hp.
    // - cached execution has batch_size == 1, no parameter graph, no dropout,
    //   no input graph, causal attention, matching cache geometry, num_steps
    //   cache slots, and enough remaining capacity for this forward's tokens.
    // Do not call ModelForwardRequest::validate(): its existing per-layer
    // parameter-count contract does not describe this proposed architecture.

    const float duration = plan.time_end - plan.time_begin;
    const float dt = duration / plan.num_steps;
    autograd::set_autograd_cublas_handle(exec.cublas_handle);
    outputs.canonical.reserveLayerOutputs(plan.num_steps);
    outputs.steps.resize(plan.num_steps); // stable owners before constructing tape
    outputs.scalar_to_channels = constantOnes(1, exec.encoder_hp.d_model, exec.stream);

    for (int k = 0; k < plan.num_steps; ++k) {
        outputs.canonical.pushLayerOutputs();
        const Tensor& H = k == 0 ? request.initial_hidden
                                : outputs.canonical.output_per_layer[k - 1];
        KvCacheLayerView cache_view{};
        if (exec.kv_cache) cache_view = exec.kv_cache->layerView(k);

        const TimeConditionedEulerStepPayload step{
            k, plan.time_begin + k * dt,
            float(k) / float(plan.num_steps), dt, H,
            exec.kv_cache ? &cache_view : nullptr
        };
        const TimeConditionedVectorFieldRequest field{
            exec, request.parameters, step, plan.attention_to_ffn_beta
        };
        auto& retained = outputs.steps[k];

        bindSharedLoRAViews(field, outputs.canonical);
        // Bind the same borrowed projections to slot k; no per-step weights.
        // Dropout keys use batch_idx AND k, never shared parameter index zero.

        evaluateTimeConditioner(field, retained);
        // tau [1,1] -> matmul(input_weight) -> broadcast_add(input_bias)
        // -> SiLU -> matmul(output_weight) -> broadcast_add(output_bias).
        // Save EVERY intermediate in retained, then slice_columns into the
        // two D-wide shifts and two scalar gains. All conditioner weights are
        // shared across steps. Never extract learned gains as host floats.
        retained.attention_gain_channels = autograd::matmul(
            retained.attention_gain, outputs.scalar_to_channels, exec.stream);
        retained.ffn_gain_channels = autograd::matmul(
            retained.ffn_gain, outputs.scalar_to_channels, exec.stream);

        auto& norm_A = outputs.canonical.rms1_out_per_layer[k];
        norm_A = autograd::rms_norm(H, field.parameters.encoding.rms1_gamma,
                                   exec.encoder_hp.rms_epsilon, exec.stream);
        retained.attention_input = autograd::broadcast_add(
            norm_A, retained.attention_shift, exec.stream);

        evaluateAttentionDelta(field, retained.attention_input, outputs.canonical);
        // Adapt to EXISTING Attention::EncoderSelfAttentionForwardRequest:
        // {batch, bindings, attention_hp, stream, cublas_handle, batch_idx, k}.
        // Use the existing full/cached attention facade according to step.cache.
        // Retain a stable attention_hp for the facade call. Preserve configured
        // token/head gates, dropout, and optional LayerScale explicitly.
        // Gated attention -> canonical.attention_residual_branch_per_layer[k];
        // optional LayerScale -> canonical.scaled_proj_per_layer[k].
        // NO fixed 1/sqrt(2*num_layers) forward multiplier in this mode.

        const Tensor& D_A = exec.encoder_hp.use_layer_scale
            ? outputs.canonical.scaled_proj_per_layer[k]
            : outputs.canonical.attention_residual_branch_per_layer[k];
        retained.beta_attention = autograd::mul_scalar(
            D_A, field.attention_to_ffn_beta, exec.stream);
        auto& U = outputs.canonical.residual1_per_layer[k];
        U = autograd::add(H, retained.beta_attention, exec.stream);

        auto& norm_F = outputs.canonical.rms2_out_per_layer[k];
        norm_F = autograd::rms_norm(U, field.parameters.encoding.rms2_gamma,
                                   exec.encoder_hp.rms_epsilon, exec.stream);
        retained.ffn_input = autograd::broadcast_add(
            norm_F, retained.ffn_shift, exec.stream);
        evaluateFFNDelta(field, retained.ffn_input, outputs.canonical);
        // Delegate SwiGLU to exec.ffn_compute.forward using shared FFN views
        // and execution slot k. Preserve configured branch modifiers explicitly.
        // Raw FFN -> canonical.ffn_out_per_layer[k]; optional LayerScale ->
        // canonical.scaled_ffn_per_layer[k]. NO fixed depth multiplier.
        const Tensor& D_F = exec.encoder_hp.use_layer_scale
            ? outputs.canonical.scaled_ffn_per_layer[k]
            : outputs.canonical.ffn_out_per_layer[k];

        retained.gained_attention = autograd::layer_scale(
            D_A, retained.attention_gain_channels, exec.stream);
        retained.gained_ffn = autograd::layer_scale(
            D_F, retained.ffn_gain_channels, exec.stream);
        retained.vector_field = autograd::add(
            retained.gained_attention, retained.gained_ffn, exec.stream);
        retained.euler_increment = autograd::mul_scalar(
            retained.vector_field, step.delta_time, exec.stream);
        outputs.canonical.output_per_layer[k] = autograd::add(
            H, retained.euler_increment, exec.stream);
        // Final update starts from H, NOT U; beta is only the FFN conditioning
        // path. Do not add the attention contribution a second time via U.
    }

    // All steps used the SAME pre-forward token offset, but separate caches.
    // Advance once after successful encoder execution, never inside the loop.
    // A future enclosing full-model caller must not also advance this cache.
    if (exec.kv_cache) exec.kv_cache->advance(exec.batch.total_tokens, exec.stream);

    // Final H is canonical.output_per_layer.back(). Caller decides LM-head
    // readout and capture. No embedding, loss, backward, optimizer, config
    // mutation, initialization-gain change, or parameter registration here.
    // Execute on the caller stream; outputs survive its asynchronous work.
}
*/

} // namespace GRIM::Forward

#endif // USE_CUDA

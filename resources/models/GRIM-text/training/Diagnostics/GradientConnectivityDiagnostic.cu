//======================================================//
//  GradientConnectivityDiagnostic.cu
//  Backward gradient connectivity diagnostics
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "GradientConnectivityDiagnostic.hpp"

#include "../Autograd/AutogradTraining.hpp"
#include "../../Layers/Encoding/AblationFlags.hpp"
#include "../../Shared/GPUBuffer/GPUBuffer.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/VerboseLogging.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define GD_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[GradientConnectivityDiagnostic] INFO: " << msg << std::endl; \
    } \
} while(0)
#define GD_WARN(msg) do { \
    std::cerr << "[GradientConnectivityDiagnostic] WARN: " << msg << std::endl; \
} while(0)
#define GD_CUDA_CHECK(call) do { \
    const cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + ":" + \
                                 std::to_string(__LINE__) + " - " + \
                                 cudaGetErrorString(status)); \
    } \
} while(0)

namespace GRIM {
namespace Autograd {
namespace Diagnostics {
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

GradientSignalProbe probeGradientSignal(Tensor& tensor, cudaStream_t stream) {
    GradientSignalProbe probe{};
    if (!tensor.data || !tensor.has_grad() || !tensor.grad_data()) {
        return probe;
    }
    probe.allocated = true;
    probe.checked = static_cast<size_t>(tensor.numel());
    if (probe.checked == 0) {
        probe.finite = false;
        return probe;
    }

    std::vector<float> host_grad(probe.checked);
    GD_CUDA_CHECK(cudaMemcpyAsync(host_grad.data(), tensor.grad_data(),
                                  probe.checked * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
    GD_CUDA_CHECK(cudaStreamSynchronize(stream));

    double sum_sq = 0.0;
    for (float value : host_grad) {
        if (!std::isfinite(value)) probe.finite = false;
        if (value != 0.0f) probe.nonzero = true;
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    probe.rms = static_cast<float>(
        std::sqrt(sum_sq / static_cast<double>(probe.checked)));
    if (!std::isfinite(probe.rms)) probe.finite = false;
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
    if (expectation.count == 0) return expectation;

    expectation.before.resize(expectation.count);
    GD_CUDA_CHECK(cudaMemcpyAsync(expectation.before.data(), tensor.grad_data(),
                                  expectation.count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
    GD_CUDA_CHECK(cudaStreamSynchronize(stream));
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
    GD_CUDA_CHECK(cudaMemcpyAsync(after.data(), tensor.grad_data(),
                                  count * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
    GD_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (expectation.had_grad_storage) {
        if (expectation.grad_ptr != tensor.grad_data()) {
            probe.error_message = ".grad_data pointer changed during backward";
            return probe;
        }
        if (expectation.count != count || expectation.before.size() != count) {
            probe.error_message = ".grad shape/size changed during backward";
            return probe;
        }
    }

    probe.comparable = true;
    double sum_sq = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float before = expectation.had_grad_storage ? expectation.before[i] : 0.0f;
        const float current = after[i];
        if (!std::isfinite(before) || !std::isfinite(current)) probe.finite = false;
        const float delta = current - before;
        if (delta != 0.0f) probe.changed = true;
        sum_sq += static_cast<double>(delta) * static_cast<double>(delta);
    }
    probe.delta_rms = static_cast<float>(
        std::sqrt(sum_sq / static_cast<double>(count)));
    if (!std::isfinite(probe.delta_rms)) probe.finite = false;
    return probe;
}

GradientVerificationActivity detectActivity(const AutogradContext& ctx) {
    GradientVerificationActivity activity{};
    activity.text_loss_active = ctx.payload &&
        (ctx.payload->lm_valid_tokens > 0 ||
         ctx.payload->atom_insertion_valid_gap_count > 0);
    return activity;
}

}  // namespace

struct GradientVerificationSession::Impl {
    bool require_current_microbatch_delta = false;
    std::vector<GradientSignalExpectation> expected;

    const GradientSignalExpectation* find(const std::string& label) const {
        for (const auto& entry : expected) {
            if (entry.label == label) return &entry;
        }
        return nullptr;
    }
};

GradientVerificationSession::GradientVerificationSession(
    AutogradContext& ctx,
    bool require_current_microbatch_delta
) : impl_(std::make_unique<Impl>()) {
    impl_->require_current_microbatch_delta = require_current_microbatch_delta;
    if (!require_current_microbatch_delta) return;

    const GradientVerificationActivity activity = detectActivity(ctx);
    const auto model_hp = HyperParameters::modelHP(*ctx.config);
    auto capture = [&](Tensor& tensor, const std::string& label) {
        if (tensor.data) {
            impl_->expected.push_back(
                captureGradientExpectation(tensor, label, ctx.stream));
        }
    };

    if (activity.text_loss_active) {
        auto& lm_head = ctx.parameter_registry->requireLmHeadParameters(
            "GradientVerificationSession");
        capture(lm_head.weights, "lm_head weights");
        if (model_hp.atom_insertion_enabled) {
            auto& atom_boundary =
                ctx.parameter_registry->requireAtomInsertionBoundaryParameters(
                    "GradientVerificationSession");
            capture(
                atom_boundary.left_projection_weight,
                "atom insertion left projection weight");
            capture(
                atom_boundary.right_projection_weight,
                "atom insertion right projection weight");
            capture(
                atom_boundary.projection_bias,
                "atom insertion projection bias");
        }
        if (model_hp.encoder_num_layers > 0) {
            if (Ablation::kAttnDeliversParamGradient) {
                auto& encoder = ctx.parameter_registry->requireEncodingLayerParameters(
                    0, "GradientVerificationSession");
                capture(encoder.W_qkv, "layer 0 attnWqkv");
                if (model_hp.encoder_attention_residual_gate_enabled) {
                    auto& gate = ctx.parameter_registry->requireAttentionResidualGateParameters(
                        0, "GradientVerificationSession");
                    capture(gate.W_gate, "layer 0 attentionResidualGateW");
                }
            } else if (!Ablation::kZeroFfnResidual) {
                auto& ffn = ctx.parameter_registry->requireFeedForwardParameters(
                    0, "GradientVerificationSession");
                capture(ffn.W2, "layer 0 ffnW2 (attn ablated)");
            }
        }
    }

    GD_INFO("Captured " << impl_->expected.size()
            << " pre-backward gradient baselines for accumulation-slot verification");
}

GradientVerificationSession::~GradientVerificationSession() = default;
GradientVerificationSession::GradientVerificationSession(
    GradientVerificationSession&&) noexcept = default;
GradientVerificationSession& GradientVerificationSession::operator=(
    GradientVerificationSession&&) noexcept = default;

bool GradientVerificationSession::verify(AutogradContext& ctx) const {
    bool ok = true;
    const GradientVerificationActivity activity = detectActivity(ctx);
    const auto model_hp = HyperParameters::modelHP(*ctx.config);

    auto requireRegisteredGroup = [&](Tensor& tensor, const std::string& label) -> ParameterGroup& {
        return ctx.parameter_registry->requireParameterGroupForTensor(
            tensor, label, "GradientVerificationSession::verify");
    };

    auto recordNumericalSignal = [&](ParameterGroup& group, const std::string& label, bool received) {
        auto& state = group.gradient_verification;
        if (received) {
            state.record_nonzero_signal(ctx.global_step);
            return false;
        }
        const uint64_t zero_streak = state.record_zero_signal();
        if (state.ever_received_nonzero_signal) {
            if (zero_streak == 1 || (zero_streak & (zero_streak - 1)) == 0) {
                GD_WARN(label << ".grad received structural delivery but no representable numerical signal"
                        << " (consecutive_zero_signal_checks=" << zero_streak
                        << ", last_nonzero_step=" << state.last_nonzero_step << ")");
            }
            return false;
        }
        if (zero_streak == 1) {
            GD_WARN(label << ".grad delivered no representable numerical signal; tolerating "
                    << "the first active check because this parameter has not yet established "
                    << "a nonzero history");
            return false;
        }
        GD_WARN(label << ".grad delivered no representable numerical signal for "
                << zero_streak << " consecutive active checks and has never produced "
                << "a nonzero gradient in this training process");
        return true;
    };

    auto requireAllocatedFinite = [&](Tensor& tensor, const std::string& label) {
        if (!tensor.data) return;
        if (!tensor.has_grad()) {
            GD_WARN(label << ".grad is NULL - allocated parameter did not expose optimizer gradient storage");
            ok = false;
            return;
        }
        const GradientSignalProbe probe = probeGradientSignal(tensor, ctx.stream);
        if (!probe.allocated) {
            GD_WARN(label << ".grad_data is NULL despite has_grad=true");
            ok = false;
        } else if (!probe.finite) {
            GD_WARN(label << ".grad contains non-finite values in full tensor (checked="
                    << probe.checked << ", rms=" << probe.rms << ")");
            ok = false;
        } else {
            GD_INFO(label << ".grad allocated; checked=" << probe.checked
                    << " rms=" << probe.rms
                    << " received_nonzero=" << (probe.nonzero ? "yes" : "no"));
        }
    };

    auto requireReceivedGradient = [&](Tensor& tensor, const std::string& label) {
        requireAllocatedFinite(tensor, label);
        if (!tensor.data || !tensor.has_grad() || !tensor.grad_data()) return;

        auto& group = requireRegisteredGroup(tensor, label);
        auto& verification = group.gradient_verification;
        const uint64_t delivery_count = tensor.gradient_delivery_count();
        if (!verification.observe_active_check(delivery_count)) {
            GD_WARN(label << ".grad received no leaf-gradient delivery during this active backward"
                    << " (active_check=" << verification.active_check_count
                    << ", delivery_count=" << delivery_count << ")");
            ok = false;
            return;
        }

        if (impl_->require_current_microbatch_delta) {
            const GradientSignalExpectation* expectation = impl_->find(label);
            if (!expectation) {
                GD_WARN(label << ".grad current-microbatch baseline is missing during accumulation verification");
                ok = false;
                return;
            }
            const GradientDeltaProbe probe = probeGradientDelta(tensor, *expectation, ctx.stream);
            if (!probe.comparable) {
                GD_WARN(label << ".grad could not be compared against its pre-backward accumulation baseline"
                        << probe.error_message);
                ok = false;
                return;
            }
            if (!probe.finite) {
                GD_WARN(label << ".grad current-microbatch delta contains non-finite values (checked="
                        << probe.checked << ", delta_rms=" << probe.delta_rms << ")");
                ok = false;
                return;
            }
            const bool missing = !probe.changed || probe.delta_rms == 0.0f;
            if (!missing) (void)recordNumericalSignal(group, label, true);
            if (missing && recordNumericalSignal(group, label, false)) {
                GD_WARN(label << ".grad did not change during this accumulation microbatch "
                        << "(checked=" << probe.checked << ") - current backward delivered "
                        << "structurally but has never established nonzero numerical signal");
                ok = false;
                return;
            }
            GD_INFO(label << ".grad received current-microbatch "
                    << (missing ? "structural delivery" : "numerical signal")
                    << "; checked=" << probe.checked << " delta_rms=" << probe.delta_rms);
            return;
        }

        const GradientSignalProbe probe = probeGradientSignal(tensor, ctx.stream);
        const bool missing = !probe.nonzero || probe.rms == 0.0f;
        if (!missing) (void)recordNumericalSignal(group, label, true);
        if (missing && recordNumericalSignal(group, label, false)) {
            GD_WARN(label << ".grad is allocated but full-tensor RMS is zero after this backward "
                    << "(checked=" << probe.checked << ") - leaf delivery occurred but this "
                    << "parameter has never established nonzero numerical signal");
            ok = false;
        }
    };

    auto& embedding = ctx.parameter_registry->requireEmbeddingParameters(
        "GradientVerificationSession::verify");
    requireAllocatedFinite(embedding.token_weights, "embedding token_weights");

    auto& lm_head = ctx.parameter_registry->requireLmHeadParameters(
        "GradientVerificationSession::verify");
    if (lm_head.weights.data) {
        if (!lm_head.weights.has_grad()) {
            GD_WARN("lm_head weights.grad is NULL - gradients NOT flowing to optimizer!");
            ok = false;
        } else {
            GD_INFO("LM head gradients "
                    << (lm_head.weights.grad_data() == embedding.token_weights.grad_data()
                        ? "TIED to embedding: " : "SEPARATE: ")
                    << lm_head.weights.numel() << " elements");
            if (activity.text_loss_active) {
                requireReceivedGradient(lm_head.weights, "lm_head weights");
            } else {
                requireAllocatedFinite(lm_head.weights, "lm_head weights");
            }
        }
    }

    if (lm_head.final_rms_gamma.data && lm_head.final_rms_gamma.requires_grad) {
        requireAllocatedFinite(lm_head.final_rms_gamma, "final_rms_gamma");
    }
    if (lm_head.bias.data) requireAllocatedFinite(lm_head.bias, "lm_head_bias");

    if (model_hp.atom_insertion_enabled) {
        auto& atom_boundary =
            ctx.parameter_registry->requireAtomInsertionBoundaryParameters(
                "GradientVerificationSession::verify");
        auto check_atom_parameter = [&](Tensor& tensor, const std::string& label) {
            if (activity.text_loss_active) {
                requireReceivedGradient(tensor, label);
            } else {
                requireAllocatedFinite(tensor, label);
            }
        };
        check_atom_parameter(
            atom_boundary.left_projection_weight,
            "atom insertion left projection weight");
        check_atom_parameter(
            atom_boundary.right_projection_weight,
            "atom insertion right projection weight");
        check_atom_parameter(
            atom_boundary.projection_bias,
            "atom insertion projection bias");
    }

    const int num_layers = model_hp.encoder_num_layers;
    if (static_cast<int>(ctx.parameter_registry->feedForwardParameterTensors().size()) != num_layers) {
        throw std::runtime_error(
            "GradientVerificationSession::verify: feedForwardParameterTensors size mismatch. size=" +
            std::to_string(ctx.parameter_registry->feedForwardParameterTensors().size()) +
            " num_layers=" + std::to_string(num_layers));
    }

    for (int layer = 0; layer < num_layers; ++layer) {
        auto& encoder = ctx.parameter_registry->requireEncodingLayerParameters(
            layer, "GradientVerificationSession::verify");
        auto check = [&](Tensor& tensor, const char* name) {
            requireAllocatedFinite(
                tensor, "layer " + std::to_string(layer) + " " + std::string(name));
        };
        if (model_hp.encoder_freeze_learned_rms_gammas) {
            if (encoder.rms1_gamma.has_grad() || encoder.rms2_gamma.has_grad()) {
                GD_WARN("layer " << layer << " frozen RMS gamma has a grad buffer");
                ok = false;
            }
        } else {
            check(encoder.rms1_gamma, "rms1Gamma");
            check(encoder.rms2_gamma, "rms2Gamma");
        }
        check(encoder.W_qkv, "attnWqkv");
        check(encoder.b_qkv, "attnBqkv");
        check(encoder.W_o, "attnWo");
        check(encoder.b_o, "attnBo");
        if (model_hp.encoder_attention_residual_gate_enabled) {
            auto& gate = ctx.parameter_registry->requireAttentionResidualGateParameters(
                layer, "GradientVerificationSession::verify");
            check(gate.W_gate, "attentionResidualGateW");
            check(gate.b_gate, "attentionResidualGateB");
        }
        auto& ffn = ctx.parameter_registry->requireFeedForwardParameters(
            layer, "GradientVerificationSession::verify");
        check(ffn.W_gate, "ffnWGate");
        check(ffn.W1, "ffnW1");
        check(ffn.W2, "ffnW2");
        check(ffn.b2, "ffnB2");
        check(encoder.layer_scale1, "layerScale1");
        check(encoder.layer_scale2, "layerScale2");
    }

    if (activity.text_loss_active && num_layers > 0) {
        auto& encoder = ctx.parameter_registry->requireEncodingLayerParameters(
            0, "GradientVerificationSession::verify");
        if (Ablation::kAttnDeliversParamGradient) {
            requireReceivedGradient(encoder.W_qkv, "layer 0 attnWqkv");
            if (model_hp.encoder_attention_residual_gate_enabled) {
                auto& gate = ctx.parameter_registry->requireAttentionResidualGateParameters(
                    0, "GradientVerificationSession::verify");
                requireReceivedGradient(
                    gate.W_gate, "layer 0 attentionResidualGateW");
            }
        } else if (!Ablation::kZeroFfnResidual) {
            auto& ffn = ctx.parameter_registry->requireFeedForwardParameters(
                0, "GradientVerificationSession::verify");
            requireReceivedGradient(ffn.W2, "layer 0 ffnW2 (attn ablated)");
        }
    }

    if constexpr (Ablation::kZeroAttnV || Ablation::kZeroAttnResidual) {
        const char* tag = Ablation::kZeroAttnV ? "kZeroAttnV" : "kZeroAttnResidual";
        for (int layer = 0; layer < num_layers; ++layer) {
            auto& encoder = ctx.parameter_registry->requireEncodingLayerParameters(
                layer, "GradientVerificationSession::verify");
            auto probeAttentionParameter = [&](Tensor& tensor, const char* name) {
                if (!tensor.data || !tensor.has_grad()) {
                    GD_INFO("[ABLATION-GRADLEAK][" << tag << "] layer=" << layer
                            << " " << name << ".grad NOT ALLOCATED (expected for a frozen sublayer)");
                    return;
                }
                const GradientSignalProbe probe = probeGradientSignal(tensor, ctx.stream);
                const bool leaked = probe.nonzero && probe.rms != 0.0f;
                GD_INFO("[ABLATION-GRADLEAK][" << tag << "] layer=" << layer
                        << " " << name << ".grad rms=" << probe.rms
                        << " nonzero=" << (probe.nonzero ? "yes" : "no")
                        << " finite=" << (probe.finite ? "yes" : "no")
                        << " checked=" << probe.checked << " -> "
                        << (leaked ? "LEAK" : "clean-zero"));
            };
            probeAttentionParameter(encoder.W_qkv, "attnWqkv");
            probeAttentionParameter(encoder.W_o, "attnWo");
        }
    }

    return ok;
}

void logPostBackwardGradientSamples(AutogradContext& ctx, bool accumulate) {
    if constexpr (!VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) return;

    GD_CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
    auto& embedding = ctx.parameter_registry->requireEmbeddingParameters(
        "logPostBackwardGradientSamples");
    auto& lm_head = ctx.parameter_registry->requireLmHeadParameters(
        "logPostBackwardGradientSamples");
    const auto model_hp = HyperParameters::modelHP(*ctx.config);

    auto readFirst = [](Tensor& tensor) {
        float sample = 0.0f;
        if (tensor.grad_data()) {
            GD_CUDA_CHECK(cudaMemcpy(
                &sample, tensor.grad_data(), sizeof(float), cudaMemcpyDeviceToHost));
        }
        return sample;
    };

    float encoder_sample = 0.0f;
    float rms_sample = 0.0f;
    if (model_hp.encoder_num_layers > 0) {
        auto& encoder = ctx.parameter_registry->requireEncodingLayerParameters(
            0, "logPostBackwardGradientSamples");
        encoder_sample = readFirst(encoder.W_qkv);
        rms_sample = readFirst(encoder.rms1_gamma);
    }

    std::fprintf(stderr,
        "[GRAD_DIAG] POST-BACKWARD accumulate=%d "
        "emb_grad[0]=%.10e lm_grad[0]=%.10e enc_wqkv_grad[0]=%.10e "
        "enc0_rms1_gamma_grad[0]=%.10e lm_ptr=%p\n",
        static_cast<int>(accumulate),
        readFirst(embedding.token_weights), readFirst(lm_head.weights),
        encoder_sample, rms_sample, static_cast<void*>(lm_head.weights.grad_data()));
}

bool verifyGradientsAreConnected(AutogradContext& ctx) {
    GradientVerificationSession session(ctx, false);
    return session.verify(ctx);
}

}  // namespace Diagnostics
}  // namespace Autograd
}  // namespace GRIM

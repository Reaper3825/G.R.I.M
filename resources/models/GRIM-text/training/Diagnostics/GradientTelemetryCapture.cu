#include "GradientTelemetryCapture.hpp"
#include "../Phases/Phase2_TrainingLoop.hpp"
#include "../../Shared/Telemetry/TelemetryLattice_GPU.hpp"
#include "../../Shared/Gradients/GradientCC_GPU.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM::Diagnostics {
namespace {
using Telemetry::MetricStream;
constexpr int kDynamicBase = static_cast<int>(MetricStream::GRADIENT_CAPTURE_DYNAMIC_BASE);
constexpr int kLayerMetrics = 34;
constexpr int kGroupMetrics = 4;
const char* const kStatsNames[] = {"mean", "std", "rms", "min", "max"};
const char* const kGroupNames[] = {"grad_rms_pre", "grad_sum_sq_pre", "grad_energy_pct", "grad_parameter_count"};
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(
        std::string("GradientTelemetryCapture: ") + cudaGetErrorString(error));
}
void publish(float* obs, int id, double value) {
    const float converted = static_cast<float>(value);
    if (!std::isfinite(converted)) throw std::runtime_error("GradientTelemetryCapture: invalid observation");
    obs[id] = converted;
}
struct Stats {
    // POD storage is required for the uninitialized CUDA shared-memory array.
    unsigned long long count, below_one, low, high;
    double mean, m2, sum_sq, slope_sum;
    double min, max;
    int nonfinite;
};
__device__ Stats merge(Stats a, const Stats& b) {
    a.nonfinite |= b.nonfinite;
    if (!b.count) return a;
    if (!a.count) { Stats result = b; result.nonfinite |= a.nonfinite; return result; }
    const double delta = b.mean - a.mean;
    const unsigned long long n = a.count + b.count;
    a.mean += delta * static_cast<double>(b.count) / n;
    a.m2 += b.m2 + delta * delta * static_cast<double>(a.count) * b.count / n;
    a.min = fmin(a.min, b.min); a.max = fmax(a.max, b.max);
    a.count = n; a.sum_sq += b.sum_sq; a.slope_sum += b.slope_sum;
    a.below_one += b.below_one; a.low += b.low; a.high += b.high;
    return a;
}
// One block per tensor. Welford reductions preserve small gate variance near 1.
__global__ void reduceGateStats(const float* data, size_t count, Stats* output,
                                const int* lengths, int max_seq_len) {
    Stats local{};
    for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
        if (lengths && static_cast<int>(i % max_seq_len) >= lengths[i / max_seq_len]) continue;
        const double value = data[i];
        if (!isfinite(value)) { local.nonfinite = 1; continue; }
        Stats one{};
        one.count = 1; one.mean = one.min = one.max = value;
        one.sum_sq = value * value;
        one.slope_sum = value * (1.0 - 0.5 * value);
        one.below_one = value < 1.0; one.low = value <= 0.1; one.high = value >= 1.9;
        local = merge(local, one);
    }
    __shared__ Stats partial[256];
    partial[threadIdx.x] = local;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (threadIdx.x < offset) partial[threadIdx.x] = merge(partial[threadIdx.x], partial[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *output = partial[0];
}

} // namespace

struct GradientTelemetryCaptureState {
    int layers = 0, group_base = kDynamicBase;
    Stats* device_stats = nullptr;
    Stats* host_stats = nullptr;
    std::vector<bool> window_valid;
    bool forward_pending = false, readback_pending = false;
    int forward_step = 0, forward_tokens = 0;
    ~GradientTelemetryCaptureState() {
        if (device_stats) cudaFree(device_stats);
        if (host_stats) cudaFreeHost(host_stats);
    }
};

void registerGradientMetricStreams(GRIMText::Training::TrainingContext& ctx,
                                   Telemetry::LatticeConfig& config) {
    const auto hp = HyperParameters::encoderLayerConstructionHP(ctx.config);
    const auto& groups = ctx.parameter_registry.requireParameterGroups("registerGradientMetricStreams");
    auto state = std::make_shared<GradientTelemetryCaptureState>();
    state->layers = hp.attention_residual_gate_enabled ? hp.num_layers : 0;
    state->group_base = kDynamicBase + state->layers * kLayerMetrics;
    const size_t required = state->group_base + groups.size() * kGroupMetrics;
    if (required > HyperParameters::TELEMETRY_MAX_STREAMS)
        throw std::runtime_error("GradientTelemetryCapture: registered metrics exceed telemetry capacity");
    // Stream count is a storage dimension derived from actual registrations.
    // Preserve the authored capacity floor and all old stable/retired IDs.
    config.num_streams = std::max(config.num_streams, static_cast<int>(required));
    if (config.num_streams > HyperParameters::TELEMETRY_MAX_STREAMS)
        throw std::runtime_error("GradientTelemetryCapture: configured telemetry capacity is too large");
    config.metric_stream_names.resize(config.num_streams);
    for (int layer = 0; layer < state->layers; ++layer) {
        int id = kDynamicBase + layer * kLayerMetrics;
        const auto name = [&](const std::string& suffix) {
            config.metric_stream_names[id++] = "layer" + std::to_string(layer) + "_attention_residual_gate_" + suffix;
        };
        for (const char* suffix : kStatsNames) name(std::string("multiplier_") + suffix);
        for (const char* suffix : {"below_identity_fraction", "low_saturation_fraction", "high_saturation_fraction", "sigmoid_slope_mean"})
            name(std::string("multiplier_") + suffix);
        for (const char* suffix : kStatsNames) name(std::string("logit_") + suffix);
        for (const char* type : {"weight", "bias", "weight_grad_postclip", "bias_grad_postclip"})
            for (const char* suffix : kStatsNames) name(std::string(type) + "_" + suffix);
    }
    for (size_t group = 0; group < groups.size(); ++group)
        for (int metric = 0; metric < kGroupMetrics; ++metric)
            config.metric_stream_names[state->group_base + group * kGroupMetrics + metric] = groups[group].name + "_" + kGroupNames[metric];
    for (int id = 94; id < config.num_streams; ++id)
        ctx.telemetry.last_obs[id] = std::numeric_limits<float>::quiet_NaN();
    if (state->layers) {
        const size_t count = 6 * state->layers;
        check(cudaMallocAsync(reinterpret_cast<void**>(&state->device_stats), count * sizeof(Stats), config.stream));
        check(cudaMemsetAsync(state->device_stats, 0, count * sizeof(Stats), config.stream));
        check(cudaMallocHost(reinterpret_cast<void**>(&state->host_stats), count * sizeof(Stats)));
        state->window_valid.resize(4 * state->layers);
    }
    ctx.telemetry.gradient_capture = std::move(state);
}

namespace {
void reduce(const float* data, size_t count, Stats* dest, cudaStream_t stream,
            const int* lengths = nullptr, int max_seq_len = 0) {
    if (!data || !count || !stream) throw std::runtime_error("GradientTelemetryCapture: missing input");
    reduceGateStats<<<1, 256, 0, stream>>>(data, count, dest, lengths, max_seq_len);
    check(cudaGetLastError());
}
void publishStats(float* obs, int id, const Stats& stats, bool multiplier = false) {
    if (stats.nonfinite || !stats.count) throw std::runtime_error("GradientTelemetryCapture: invalid summary");
    publish(obs, id++, stats.mean);
    publish(obs, id++, std::sqrt(std::max(0.0, stats.m2 / stats.count)));
    publish(obs, id++, std::sqrt(stats.sum_sq / stats.count));
    publish(obs, id++, stats.min); publish(obs, id++, stats.max);
    if (multiplier) {
        publish(obs, id++, double(stats.below_one) / stats.count);
        publish(obs, id++, double(stats.low) / stats.count);
        publish(obs, id++, double(stats.high) / stats.count);
        publish(obs, id++, stats.slope_sum / stats.count);
    }
}
}

void captureGateForwardTelemetry(GRIMText::Training::TrainingContext& ctx,
    const Forward::ModelForwardOutputs& outputs, const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings, int batch_idx, cudaStream_t stream) {
    if (!ctx.telemetry.enabled || !ctx.telemetry.gradient_capture) return;
    auto& state = *ctx.telemetry.gradient_capture;
    const int interval = HyperParameters::trainingRuntimeControlHP(ctx.config).log_interval;
    if (!state.layers || interval <= 0 || (batch_idx + 1) % interval != 0) return;
    if (!bindings.d_sequence_lengths || payload.max_seq_len <= 0 || payload.actual_tokens <= 0 ||
        outputs.attention_residual_gate_multiplier_per_layer.size() != static_cast<size_t>(state.layers) ||
        outputs.attention_residual_gate_logits_per_layer.size() != static_cast<size_t>(state.layers))
        throw std::runtime_error("GradientTelemetryCapture: invalid forward geometry");
    for (int layer = 0; layer < state.layers; ++layer) {
        const auto& multiplier = outputs.attention_residual_gate_multiplier_per_layer[layer];
        const auto& logits = outputs.attention_residual_gate_logits_per_layer[layer];
        if (multiplier.numel() != static_cast<size_t>(payload.total_tokens) || logits.numel() != multiplier.numel())
            throw std::runtime_error("GradientTelemetryCapture: invalid gate shape");
        reduce(multiplier.data, multiplier.numel(), state.device_stats + 2 * layer,
               stream, bindings.d_sequence_lengths, payload.max_seq_len);
        reduce(logits.data, logits.numel(), state.device_stats + 2 * layer + 1,
               stream, bindings.d_sequence_lengths, payload.max_seq_len);
    }
    state.forward_pending = true;
    state.forward_step = ctx.global_step;
    state.forward_tokens = payload.actual_tokens;
}

void captureGradientWindowTelemetry(GRIMText::Training::TrainingContext& ctx,
    const GradClip::ClipResult& clip, const GradClip::ClipScratch& scratch,
    int optimizer_step, cudaStream_t stream) {
    if (!ctx.telemetry.enabled || !ctx.telemetry.gradient_capture) return;
    auto& state = *ctx.telemetry.gradient_capture;
    float* obs = ctx.telemetry.last_obs;
    for (int id = 94; id < ctx.telemetry.lattice->config().num_streams; ++id)
        obs[id] = std::numeric_limits<float>::quiet_NaN();
    const auto& groups = ctx.parameter_registry.requireParameterGroups("captureGradientWindowTelemetry");
    if (!scratch.h_partial_sums || clip.measured_group_count != groups.size())
        throw std::runtime_error("GradientTelemetryCapture: missing gradient measurements");
    publish(obs, int(MetricStream::GRADIENT_GLOBAL_SUM_SQ_PRE), clip.global_sum_sq_pre);
    publish(obs, int(MetricStream::GRADIENT_GLOBAL_RMS_POST), clip.global_rms_post);
    publish(obs, int(MetricStream::GRADIENT_CLIPPED), clip.clipped ? 1.0 : 0.0);
    publish(obs, int(MetricStream::GRADIENT_CLIP_SCALE), clip.global_rms_pre > 0.0f
        ? double(clip.global_rms_post) / clip.global_rms_pre : 1.0);
    publish(obs, int(MetricStream::GRADIENT_CAPTURE_OPTIMIZER_ITERATION), optimizer_step + 1);
    double gate_sum_sq = 0.0;
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& group = groups[i];
        if (!group.grads() || !group.size()) continue;
        const double energy = scratch.h_partial_sums[i];
        const int id = state.group_base + i * kGroupMetrics;
        publish(obs, id, std::sqrt(energy / group.size())); publish(obs, id + 1, energy);
        publish(obs, id + 2, clip.global_sum_sq_pre > 0.0 ? 100.0 * energy / clip.global_sum_sq_pre : 0.0);
        publish(obs, id + 3, group.size());
        if (group.name.find("attention_residual_gate_") != std::string::npos) gate_sum_sq += energy;
    }
    publish(obs, int(MetricStream::ATTENTION_RESIDUAL_GATE_ENERGY_PCT),
        clip.global_sum_sq_pre > 0.0 ? 100.0 * gate_sum_sq / clip.global_sum_sq_pre : 0.0);
    for (int layer = 0; layer < state.layers; ++layer) {
        const auto& gate = ctx.parameter_registry.requireAttentionResidualGateParameters(layer, "captureGradientWindowTelemetry");
        const Tensor* tensors[] = {&gate.W_gate, &gate.b_gate, &gate.W_gate, &gate.b_gate};
        for (int part = 0; part < 4; ++part) {
            const Tensor& tensor = *tensors[part];
            const float* data = part < 2 ? tensor.data : (tensor.grad_ ? tensor.grad_->data : nullptr);
            state.window_valid[4 * layer + part] = data != nullptr;
            if (data) reduce(data, tensor.numel(), state.device_stats + 2 * state.layers + 4 * layer + part, stream);
        }
    }
    if (state.layers) {
        // Enqueued before the optimizer boundary's EXISTING stream sync. No
        // capture-specific host wait and no token arrays copied to the CPU.
        check(cudaMemcpyAsync(state.host_stats, state.device_stats,
            6 * state.layers * sizeof(Stats), cudaMemcpyDeviceToHost, stream));
        state.readback_pending = true;
    }
}

void collectGradientTelemetry(GRIMText::Training::TrainingContext& ctx) {
    if (!ctx.telemetry.gradient_capture) return;
    auto& state = *ctx.telemetry.gradient_capture;
    if (!state.readback_pending) return;
    float* obs = ctx.telemetry.last_obs;
    for (int layer = 0; layer < state.layers; ++layer) {
        const int id = kDynamicBase + layer * kLayerMetrics;
        if (state.forward_pending) {
            const Stats& multiplier = state.host_stats[2 * layer];
            const Stats& logits = state.host_stats[2 * layer + 1];
            if (multiplier.count != static_cast<size_t>(state.forward_tokens) || logits.count != multiplier.count)
                throw std::runtime_error("GradientTelemetryCapture: forward token count mismatch");
            publishStats(obs, id, multiplier, true); publishStats(obs, id + 9, logits);
        }
        for (int part = 0; part < 4; ++part) if (state.window_valid[4 * layer + part])
            publishStats(obs, id + 14 + part * 5, state.host_stats[2 * state.layers + 4 * layer + part]);
    }
    if (state.forward_pending) {
        publish(obs, int(MetricStream::GATE_FORWARD_SAMPLE_GLOBAL_STEP), state.forward_step);
        publish(obs, int(MetricStream::GATE_FORWARD_SAMPLE_TOKEN_COUNT), state.forward_tokens);
    }
    state.forward_pending = false; state.readback_pending = false;
}
} // namespace GRIM::Diagnostics

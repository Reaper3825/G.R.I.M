#include "LensCapture_GPU.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace GRIM::Lenses {
namespace {
std::atomic<std::uint64_t> invocation_counter{0};
std::atomic<std::uint64_t> capture_counter{0};

void check(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("Lens capture: ") + cudaGetErrorString(error));
}

// Complete each transfer before any destination vector can be destroyed on an
// exception. Captures are deliberately synchronous and opt-in.
std::vector<float> copyRow(const Tensor& tensor, std::size_t row, int width,
                           cudaStream_t stream) {
    if (!tensor.data || !tensor.shape.is_2d_layout() ||
        tensor.shape.as_2d().cols != width || row >= static_cast<std::size_t>(tensor.shape.as_2d().rows))
        throw std::runtime_error("Lens capture: tensor row geometry mismatch");
    std::vector<float> host(static_cast<std::size_t>(width));
    try {
        check(cudaMemcpyAsync(host.data(), tensor.data + row * width,
                              host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
    return host;
}
}

std::uint64_t nextForwardInvocationId() {
    return invocation_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}


LensCaptureSession::LensCaptureSession(
    const LensCaptureRequest& request, const Batching::BatchPayload& payload,
    const NamedConceptSpanDefinitions& definitions, int layer_count,
    int cache_prefix_length, bool effective_causal, bool dropout_enabled,
    std::uint64_t invocation_id,
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameters,
    cudaStream_t stream, cublasHandle_t handle)
    : request_(request), payload_(payload), hp_(hp), stream_(stream), handle_(handle),
      result_(std::make_shared<LensCaptureResult>()) {
    auto base = makeCaptureMetadata(request, payload, definitions, layer_count,
        hp.d_model, cache_prefix_length, effective_causal, dropout_enabled, invocation_id);
    if (!stream || !handle || hp.atom_insertion_enabled || hp.vocab_size != payload.vocab_size)
        throw std::runtime_error("Lens capture: invalid execution or head geometry");
    if (hp.mlp_enabled && hp.mlp_d_ff <= 0)
        throw std::runtime_error("Lens capture: invalid adapter width");
    if (!parameters.weights.data || !parameters.weights.shape.is_2d_layout() ||
        parameters.weights.shape.as_2d().rows != hp.vocab_size ||
        parameters.weights.shape.as_2d().cols != hp.d_model)
        throw std::runtime_error("Lens capture: invalid vocabulary projection geometry");
    const int count = request.all_positions ? payload.seq_lengths.at(request.batch_row) : 1;
    positions_.reserve(count);
    for (int i = 0; i < count; ++i) {
        auto selected = request;
        selected.all_positions = false;
        selected.token_position = request.all_positions ? i : base.token_position;
        positions_.push_back(makeCaptureMetadata(selected, payload, definitions, layer_count,
            hp.d_model, cache_prefix_length, effective_causal, dropout_enabled, invocation_id));
    }
    result_->temporary_memory_budget_bytes = request.temporary_memory_budget_bytes;
    result_->chunk_rows = planCaptureChunkRows(request, count, hp.d_model, hp.vocab_size,
        hp.mlp_enabled ? hp.mlp_d_ff : 0, parameters.final_rms_gamma.data != nullptr,
        parameters.bias.data != nullptr, kEnableLmHeadTokenTypeGateExperiment,
        result_->planned_temporary_bytes);
    result_->snapshots.reserve(static_cast<std::size_t>(count) * (request.all_layers ? layer_count : 1));
    next_layer_ = request.all_layers ? 0 : layer_count - 1;
    parameters_.owns_weights = false;
    parameters_.weights = parameters.weights.detach(stream);
    parameters_.bias = parameters.bias.detach(stream);
    parameters_.final_rms_gamma = parameters.final_rms_gamma.detach(stream);
    parameters_.mlp_W_gate = parameters.mlp_W_gate.detach(stream);
    parameters_.mlp_W_up = parameters.mlp_W_up.detach(stream);
    parameters_.mlp_W_down = parameters.mlp_W_down.detach(stream);
    if (kEnableLmHeadTokenTypeGateExperiment) {
        try {
            effective_weights_ = autograd::type_gate_rows_by_token_type(parameters_.weights, stream);
            check(cudaStreamSynchronize(stream));
        } catch (...) {
            (void)cudaStreamSynchronize(stream);
            throw;
        }
    }
}

void LensCaptureSession::captureLayer(int layer, const Tensor& hidden,
                                      const Tensor* actual_logits) {
    const int final_layer = positions_.front().layer_count - 1;
    if (!request_.all_layers && layer != final_layer) return;
    // Final output remains live through the ordinary head. Read it once there,
    // so validation compares the complete logits without retaining V per token.
    if (layer == final_layer && !actual_logits) return;
    if (layer != next_layer_ || layer > final_layer)
        throw std::runtime_error("Lens capture: missing, repeated, or out-of-order layer");
    if (!hidden.data || !hidden.shape.is_2d_layout() ||
        hidden.shape.as_2d().rows != payload_.total_tokens ||
        hidden.shape.as_2d().cols != hp_.d_model)
        throw std::runtime_error("Lens capture: encoder rectangle mismatch");
    const bool replay = !actual_logits || request_.replay_identity_control ||
                        request_.all_layers || request_.all_positions;
    for (std::size_t begin = 0; begin < positions_.size(); begin += result_->chunk_rows) {
        const int rows = static_cast<int>(std::min<std::size_t>(
            result_->chunk_rows, positions_.size() - begin));
        const auto flat = static_cast<std::size_t>(request_.batch_row) * payload_.max_seq_len +
                          positions_[begin].token_position;
        const LMHeadReadoutSelection selection{request_.batch_row,
            positions_[begin].token_position, positions_[begin].token_position + rows};
        Forward::ModelForwardOutputs scratch;
        // Scratch follows the existing ModelForwardOutputs/Tensor RAII lifecycle.
        if (replay)
            forwardLmHeadReadoutChunk(hp_, parameters_, hidden, payload_, selection,
                effective_weights_.data ? &effective_weights_ : nullptr,
                stream_, handle_, scratch);
        for (int i = 0; i < rows; ++i) {
            LensSnapshot snapshot;
            snapshot.metadata = positions_[begin + i];
            auto& m = snapshot.metadata;
            m.layer_index = layer;
            if (layer != final_layer)
                m.target_boundary = "encoder_block_output/post_block";
            m.capture_sequence = capture_counter.fetch_add(1, std::memory_order_relaxed) + 1;
            m.capture_timestamp_unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            m.capture_monotonic_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            if (request_.retain_readout_input)
                snapshot.readout_input = copyRow(hidden, flat + i, hp_.d_model, stream_);
            std::vector<float> actual;
            if (actual_logits) {
                actual = copyRow(*actual_logits, flat + i, hp_.vocab_size, stream_);
                snapshot.readouts.push_back(summarizeLogits(actual, request_.top_k, ReadoutKind::ActualFinal));
            }
            if (replay) {
                const auto logits = copyRow(scratch.logits_tensor, i, hp_.vocab_size, stream_);
                auto direct = summarizeLogits(logits, request_.top_k, ReadoutKind::Direct);
                snapshot.readouts.push_back(direct);
                if (actual_logits) {
                    double error = 0;
                    for (std::size_t v = 0; v < logits.size(); ++v)
                        error = std::max(error, std::abs(static_cast<double>(actual[v]) - logits[v]));
                    snapshot.identity_max_abs_logit_error = error;
                    if (request_.replay_identity_control) {
                        direct.kind = ReadoutKind::JacobianIdentity;
                        snapshot.readouts.push_back(std::move(direct));
                    }
                }
            }
            result_->snapshots.push_back(std::move(snapshot));
        }

    }
    ++next_layer_;
}

void LensCaptureSession::publish(Forward::ModelForwardOutputs& outputs) {
    const int layers = positions_.front().layer_count;
    if (next_layer_ != layers || result_->snapshots.size() !=
        positions_.size() * static_cast<std::size_t>(request_.all_layers ? layers : 1))
        throw std::runtime_error("Lens capture: incomplete layer/position coverage");
    outputs.lens_capture_result = result_;
    // Existing single-snapshot consumers see the last real position/final layer.
    // Aliasing shares the canonical collection's ownership; no duplicate result.
    outputs.lens_snapshot = std::shared_ptr<const LensSnapshot>(
        outputs.lens_capture_result, &result_->snapshots.back());
    result_.reset();
}

} // namespace GRIM::Lenses

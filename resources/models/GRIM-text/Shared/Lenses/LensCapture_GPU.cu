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
// exception. Captures are deliberately synchronous and opt-in in v1.
std::vector<float> copyRow(const Tensor& tensor, std::size_t row, int width,
                           cudaStream_t stream) {
    if (!tensor.data || !tensor.shape.is_2d_layout() ||
        tensor.shape.as_2d().cols != width || row >= static_cast<std::size_t>(tensor.shape.as_2d().rows))
        throw std::runtime_error("Lens capture: tensor row geometry mismatch");
    std::vector<float> host(static_cast<std::size_t>(width));
    check(cudaMemcpyAsync(host.data(), tensor.data + row * width,
                          host.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    return host;
}
}

std::uint64_t nextForwardInvocationId() {
    return invocation_counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

std::shared_ptr<const LensSnapshot> captureFinalLens(
    const LensCaptureRequest& request, LensCaptureMetadata metadata,
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Forward::ModelForwardOutputs& actual,
    cudaStream_t stream, cublasHandle_t handle) {
    auto snapshot = std::make_shared<LensSnapshot>();
    metadata.capture_sequence = capture_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    metadata.capture_timestamp_unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    metadata.capture_monotonic_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    snapshot->metadata = std::move(metadata);
    const auto row = static_cast<std::size_t>(snapshot->metadata.batch_row) * payload.max_seq_len
        + snapshot->metadata.token_position;
    snapshot->readout_input = copyRow(actual.encoder_output_tensor, row, hp.d_model, stream);
    const auto actual_logits = copyRow(actual.logits_tensor, row, hp.vocab_size, stream);
    snapshot->readouts.push_back(summarizeLogits(actual_logits, request.top_k, ReadoutKind::ActualFinal));

    if (request.replay_identity_control) {
        // Borrow detached parameters; never attach diagnostic gradients to the
        // training graph or overwrite the owner's live forward outputs.
        LMHeadParameterTensors detached;
        detached.owns_weights = false;
        detached.weights = parameters.weights.detach(stream);
        if (parameters.bias.data) detached.bias = parameters.bias.detach(stream);
        if (parameters.final_rms_gamma.data) detached.final_rms_gamma = parameters.final_rms_gamma.detach(stream);
        if (parameters.mlp_W_gate.data) detached.mlp_W_gate = parameters.mlp_W_gate.detach(stream);
        if (parameters.mlp_W_up.data) detached.mlp_W_up = parameters.mlp_W_up.detach(stream);
        if (parameters.mlp_W_down.data) detached.mlp_W_down = parameters.mlp_W_down.detach(stream);
        auto input = actual.encoder_output_tensor.detach(stream);
        Forward::ModelForwardOutputs replay;
        try {
            // Preserve the entire input rectangle and its payload: causal
            // centering and PC1 are not generally single-row operations.
            forwardLmHead(hp, detached, input, payload, stream, handle, replay);
            const auto replay_logits = copyRow(replay.logits_tensor, row, hp.vocab_size, stream);
            auto direct = summarizeLogits(replay_logits, request.top_k, ReadoutKind::Direct);
            snapshot->readouts.push_back(direct);
            direct.kind = ReadoutKind::JacobianIdentity;
            snapshot->readouts.push_back(std::move(direct));
            double maximum_error = 0;
            for (std::size_t i = 0; i < actual_logits.size(); ++i)
                maximum_error = std::max(maximum_error,
                    std::abs(static_cast<double>(actual_logits[i]) - replay_logits[i]));
            snapshot->identity_max_abs_logit_error = maximum_error;
        } catch (...) {
            // Finish queued readout work before destroying detached views and
            // replay scratch. Preserve the original diagnostic exception.
            (void)cudaStreamSynchronize(stream);
            throw;
        }
    }
    return snapshot;
}
} // namespace GRIM::Lenses

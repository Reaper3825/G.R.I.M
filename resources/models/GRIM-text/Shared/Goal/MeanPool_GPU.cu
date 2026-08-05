#include "MeanPool_GPU.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../Forward/ModelForwardOutputs.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {

namespace {

constexpr int kBlockSize = 256;

__global__ void meanPoolHiddenStatesKernel(
    const float* __restrict__ hidden_states,
    float* __restrict__ mean_pool,
    int first_response_row,
    int response_token_count,
    int d_model)
{
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= d_model) {
        return;
    }

    double sum = 0.0;
    for (int response_token_index = 0;
         response_token_index < response_token_count;
         ++response_token_index) {
        sum += static_cast<double>(
            hidden_states[
                static_cast<std::size_t>(
                    first_response_row + response_token_index) * d_model +
                feature]);
    }
    mean_pool[feature] = static_cast<float>(
        sum / static_cast<double>(response_token_count));
}

} // namespace

void meanPoolHiddenStates(
    const Batching::BatchPayload& payload,
    Forward::ModelForwardOutputs& forward_outputs,
    cudaStream_t stream)
{
    if (!stream) {
        throw std::runtime_error(
            "meanPoolHiddenStates: stream is NULL");
    }
    // Inference prefill/decode payloads do not yet carry a complete
    // prompt-plus-response span. The operation remains callable there but has
    // no target state to materialize.
    if (payload.prompt_lengths.empty() &&
        payload.prompt_end_positions.empty()) {
        return;
    }
    if (payload.batch_size <= 0 || payload.max_seq_len <= 0) {
        throw std::runtime_error(
            "meanPoolHiddenStates: invalid batch geometry");
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size ||
        static_cast<int>(payload.prompt_lengths.size()) != payload.batch_size ||
        static_cast<int>(payload.prompt_end_positions.size()) != payload.batch_size) {
        throw std::runtime_error(
            "meanPoolHiddenStates: payload row metadata size mismatch");
    }

    const Tensor* hidden_states = nullptr;
    if (forward_outputs.final_normalized_hidden_states.data) {
        hidden_states = &forward_outputs.final_normalized_hidden_states;
    } else if (forward_outputs.encoder_output_tensor.data) {
        hidden_states = &forward_outputs.encoder_output_tensor;
    }
    if (!hidden_states) {
        throw std::runtime_error(
            "meanPoolHiddenStates: ModelForwardOutputs has no final hidden states");
    }
    hidden_states->require("meanPoolHiddenStates hidden_states");
    if (!hidden_states->shape.is_2d_layout()) {
        throw std::runtime_error(
            "meanPoolHiddenStates: hidden_states must be 2D "
            "[batch_size * max_seq_len, d_model]");
    }

    const auto shape = hidden_states->shape.as_2d();
    if (shape.rows <= 0 || shape.cols <= 0) {
        throw std::runtime_error(
            "meanPoolHiddenStates: hidden-state dimensions must be greater than zero");
    }
    const int expected_hidden_rows = payload.batch_size * payload.max_seq_len;
    if (shape.rows != expected_hidden_rows) {
        throw std::runtime_error(
            "meanPoolHiddenStates: hidden-state row count " +
            std::to_string(shape.rows) + " != batch_size * max_seq_len " +
            std::to_string(expected_hidden_rows));
    }
    const int d_model = shape.cols;

    struct ResponseSpan {
        int first_hidden_row = 0;
        int token_count = 0;
    };
    std::vector<ResponseSpan> response_spans;
    response_spans.reserve(static_cast<std::size_t>(payload.batch_size));

    for (int batch_row = 0; batch_row < payload.batch_size; ++batch_row) {
        const std::size_t row = static_cast<std::size_t>(batch_row);
        const int prompt_length = payload.prompt_lengths[row];
        const int prompt_end = payload.prompt_end_positions[row];
        const int sequence_length = payload.seq_lengths[row];

        if (prompt_length == 0 && prompt_end == -1) {
            continue;
        }
        const int prompt_start = prompt_end - prompt_length + 1;
        if (prompt_length <= 0 || prompt_start < 0 || prompt_end < 0 ||
            prompt_end >= sequence_length || sequence_length > payload.max_seq_len) {
            throw std::runtime_error(
                "meanPoolHiddenStates: invalid prompt/sequence span at batch row " +
                std::to_string(batch_row));
        }

        const int response_start = prompt_end + 1;
        const int response_token_count = sequence_length - response_start;
        if (response_token_count == 0) {
            continue;
        }
        response_spans.push_back(ResponseSpan{
            batch_row * payload.max_seq_len + response_start,
            response_token_count});
    }

    if (response_spans.empty()) {
        return;
    }

    if (!forward_outputs.goal.target_state) {
        forward_outputs.goal.target_state.emplace();
    }
    Tensor& mean_pool = forward_outputs.goal.target_state->norm_mean_pool;
    mean_pool = Tensor::empty(
        TensorContract::TensorShape::make_BSM(
            static_cast<int>(response_spans.size()),
            d_model),
        false,
        stream,
        "target_state_mean_pool");

    const int blocks = (d_model + kBlockSize - 1) / kBlockSize;
    for (std::size_t output_row = 0;
         output_row < response_spans.size();
         ++output_row) {
        const ResponseSpan& response_span = response_spans[output_row];
        meanPoolHiddenStatesKernel<<<blocks, kBlockSize, 0, stream>>>(
            hidden_states->data,
            mean_pool.data + output_row * static_cast<std::size_t>(d_model),
            response_span.first_hidden_row,
            response_span.token_count,
            d_model);
    }

    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("meanPoolHiddenStates: kernel launch failed: ") +
            cudaGetErrorString(launch_error));
    }
}

} // namespace GRIM

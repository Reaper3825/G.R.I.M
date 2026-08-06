#include "MeanPool_GPU.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {

namespace {

constexpr int kBlockSize = 256;

__global__ void meanPoolHiddenStatesKernel(
    const float* __restrict__ hidden_states,
    float* __restrict__ mean_pool,
    int flat_token_offset,
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
                    flat_token_offset + response_token_index) * d_model +
                feature]);
    }
    mean_pool[feature] = static_cast<float>(
        sum / static_cast<double>(response_token_count));
}

} // namespace

Tensor meanPoolHiddenStates(
    const Tensor& hidden_states,
    int tokens_per_sequence,
    const std::vector<MeanPoolSequenceSpan>& spans,
    cudaStream_t stream)
{
    if (!stream) {
        throw std::runtime_error(
            "meanPoolHiddenStates: stream is NULL");
    }
    hidden_states.require("meanPoolHiddenStates hidden_states");
    if (!hidden_states.shape.is_2d_layout()) {
        throw std::runtime_error(
            "meanPoolHiddenStates: hidden_states must be 2D [rows, d_model]");
    }

    const auto shape = hidden_states.shape.as_2d();
    if (shape.rows <= 0 || shape.cols <= 0) {
        throw std::runtime_error(
            "meanPoolHiddenStates: hidden-state dimensions must be greater than zero");
    }
    if (spans.empty()) {
        throw std::runtime_error(
            "meanPoolHiddenStates: spans must not be empty");
    }
    if (tokens_per_sequence <= 0 || shape.rows % tokens_per_sequence != 0) {
        throw std::runtime_error(
            "meanPoolHiddenStates: invalid tokens_per_sequence for hidden-state layout");
    }
    const int sequence_count = shape.rows / tokens_per_sequence;
    const int d_model = shape.cols;

    for (std::size_t span_index = 0; span_index < spans.size(); ++span_index) {
        const MeanPoolSequenceSpan& span = spans[span_index];
        if (span.sequence_index < 0 || span.sequence_index >= sequence_count ||
            span.token_begin < 0 || span.token_end <= span.token_begin ||
            span.token_end > tokens_per_sequence) {
            throw std::runtime_error(
                "meanPoolHiddenStates: invalid span at index " +
                std::to_string(span_index));
        }
    }

    Tensor mean_pool = Tensor::empty(
        TensorContract::TensorShape::make_BSM(
            static_cast<int>(spans.size()),
            d_model),
        false,
        stream,
        "mean_pool");

    const int blocks = (d_model + kBlockSize - 1) / kBlockSize;
    for (std::size_t pooled_sequence_index = 0;
         pooled_sequence_index < spans.size();
         ++pooled_sequence_index) {
        const MeanPoolSequenceSpan& span = spans[pooled_sequence_index];
        const int flat_token_offset =
            span.sequence_index * tokens_per_sequence + span.token_begin;
        const int token_count = span.token_end - span.token_begin;
        meanPoolHiddenStatesKernel<<<blocks, kBlockSize, 0, stream>>>(
            hidden_states.data,
            mean_pool.data +
                pooled_sequence_index * static_cast<std::size_t>(d_model),
            flat_token_offset,
            token_count,
            d_model);
    }

    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("meanPoolHiddenStates: kernel launch failed: ") +
            cudaGetErrorString(launch_error));
    }

    return mean_pool;
}

} // namespace GRIM

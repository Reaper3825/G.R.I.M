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
                static_cast<std::size_t>(response_token_index) * d_model +
                feature]);
    }
    mean_pool[feature] = static_cast<float>(
        sum / static_cast<double>(response_token_count));
}

} // namespace

Tensor meanPoolHiddenStates(
    const Tensor& hidden_states,
    cudaStream_t stream)
{
    if (!stream) {
        throw std::runtime_error(
            "meanPoolHiddenStates: stream is NULL");
    }
    hidden_states.require("meanPoolHiddenStates hidden_states");
    if (!hidden_states.shape.is_2d_layout()) {
        throw std::runtime_error(
            "meanPoolHiddenStates: hidden_states must be 2D "
            "[response_token_count, d_model]");
    }

    const auto shape = hidden_states.shape.as_2d();
    if (shape.rows <= 0 || shape.cols <= 0) {
        throw std::runtime_error(
            "meanPoolHiddenStates: response_token_count and d_model must be "
            "greater than zero");
    }
    const int response_token_count = shape.rows;
    const int d_model = shape.cols;

    Tensor mean_pool = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, d_model),
        false,
        stream,
        "target_state_mean_pool");

    const int blocks = (d_model + kBlockSize - 1) / kBlockSize;
    meanPoolHiddenStatesKernel<<<blocks, kBlockSize, 0, stream>>>(
        hidden_states.data,
        mean_pool.data,
        response_token_count,
        d_model);

    const cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("meanPoolHiddenStates: kernel launch failed: ") +
            cudaGetErrorString(launch_error));
    }

    return mean_pool;
}

} // namespace GRIM

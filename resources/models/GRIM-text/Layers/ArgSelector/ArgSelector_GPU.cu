//======================================================//
//  ArgSelector_GPU.cu
//  Arg/option selector scoring head implementation.
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ArgSelector_GPU.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace GRIM {
namespace ArgSelector {

namespace {

constexpr int kMaskBlock = 256;
constexpr float kMaskNegInf = -1e30f;

__global__ void kernel_row_window_mask(
    float* __restrict__ mask,                // [total_tokens * num_pool_atoms]
    const int* __restrict__ row_atom_offset, // [batch_size + 1]
    long total_elems,
    int max_seq_len,
    int batch_size,
    int num_pool_atoms)
{
    const long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    const int t = static_cast<int>(idx / num_pool_atoms);
    const int e = static_cast<int>(idx % num_pool_atoms);
    const int row = t / max_seq_len;
    int lo = 0;
    int hi = 0;
    if (row >= 0 && row < batch_size) {
        lo = row_atom_offset[row];
        hi = row_atom_offset[row + 1];
    }
    mask[idx] = (e >= lo && e < hi) ? 0.0f : kMaskNegInf;
}

}  // namespace

void launchRowWindowMask(
    float* d_mask,
    const int* d_row_atom_offset,
    int total_tokens,
    int max_seq_len,
    int batch_size,
    int num_pool_atoms,
    cudaStream_t stream)
{
    if (!d_mask || !d_row_atom_offset) {
        throw std::runtime_error("launchRowWindowMask: NULL mask or row_atom_offset");
    }
    const long total_elems = static_cast<long>(total_tokens) * num_pool_atoms;
    if (total_elems <= 0) {
        return;
    }
    const int blocks = static_cast<int>((total_elems + kMaskBlock - 1) / kMaskBlock);
    kernel_row_window_mask<<<blocks, kMaskBlock, 0, stream>>>(
        d_mask, d_row_atom_offset, total_elems, max_seq_len, batch_size, num_pool_atoms);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("launchRowWindowMask: kernel launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

Tensor argSelectorForward(
    const Tensor& encoder_output,
    const Tensor& W_q,
    const Tensor& keys,
    const Batching::BatchPayload& payload,
    const int* d_row_atom_offset,
    int num_pool_atoms,
    float softmax_scale,
    cudaStream_t stream)
{
    if (!stream) {
        throw std::runtime_error("argSelectorForward: stream is NULL");
    }
    if (!d_row_atom_offset) {
        throw std::runtime_error("argSelectorForward: d_row_atom_offset is NULL");
    }
    if (num_pool_atoms <= 0) {
        throw std::runtime_error("argSelectorForward: num_pool_atoms must be > 0");
    }
    if (!std::isfinite(softmax_scale) || softmax_scale <= 0.0f) {
        throw std::runtime_error("argSelectorForward: softmax_scale must be finite and > 0");
    }
    encoder_output.shape.require("argSelectorForward encoder_output");
    W_q.shape.require("argSelectorForward W_q");
    keys.shape.require("argSelectorForward keys");
    if (!encoder_output.shape.is_2d_layout() || !W_q.shape.is_2d_layout() || !keys.shape.is_2d_layout()) {
        throw std::runtime_error("argSelectorForward: encoder_output / W_q / keys must all be 2D");
    }
    const auto enc_shape = encoder_output.shape.as_2d();
    const auto wq_shape = W_q.shape.as_2d();
    const auto key_shape = keys.shape.as_2d();
    const int total_tokens = payload.total_tokens;
    if (enc_shape.rows != total_tokens) {
        throw std::runtime_error("argSelectorForward: encoder_output rows=" +
                                 std::to_string(enc_shape.rows) + " != payload.total_tokens=" +
                                 std::to_string(total_tokens));
    }
    const int d_model = enc_shape.cols;
    if (wq_shape.rows != d_model || wq_shape.cols != d_model) {
        throw std::runtime_error("argSelectorForward: W_q must be [d_model,d_model]=[" +
                                 std::to_string(d_model) + "," + std::to_string(d_model) + "], got [" +
                                 std::to_string(wq_shape.rows) + "," + std::to_string(wq_shape.cols) + "]");
    }
    if (key_shape.rows != num_pool_atoms || key_shape.cols != d_model) {
        throw std::runtime_error("argSelectorForward: keys must be [num_pool_atoms,d_model]=[" +
                                 std::to_string(num_pool_atoms) + "," + std::to_string(d_model) + "], got [" +
                                 std::to_string(key_shape.rows) + "," + std::to_string(key_shape.cols) + "]");
    }

    // Q = h · W_qᵀ  (matmul transpose_b convention matches the encoder QKV path).
    Tensor query = autograd::matmul(encoder_output, W_q, stream, /*transpose_b=*/true);
    // scores = Q · Keysᵀ  → [total_tokens, num_pool_atoms]
    Tensor scores = autograd::matmul(query, keys, stream, /*transpose_b=*/true);
    if (softmax_scale != 1.0f) {
        scores = autograd::mul_scalar(scores, softmax_scale, stream);
    }

    // Additive row-window mask: candidates outside token t's row get -inf so each
    // token selects only among its own row's entries. Constant (non-grad) tensor;
    // masked positions contribute ~0 probability and ~0 gradient.
    Tensor mask = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(total_tokens, num_pool_atoms),
        false, stream, "selector_window_mask");
    launchRowWindowMask(
        mask.data, d_row_atom_offset, total_tokens, payload.max_seq_len,
        payload.batch_size, num_pool_atoms, stream);

    Tensor selection_logits = autograd::add(scores, mask, stream);
    return selection_logits;
}

}  // namespace ArgSelector
}  // namespace GRIM

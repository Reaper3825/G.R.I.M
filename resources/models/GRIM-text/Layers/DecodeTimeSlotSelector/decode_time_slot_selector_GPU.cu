//======================================================//
//  DecodeTimeSlotSelectorLayer — implementation
//
//  Sole owner of trainable decode-time selector tensors.
//  Computes score vector over { NULL } ∪ L via pointer-
//  selector baseline:
//    q = W_q * h_t
//    key[s] = W_k * slot_features[s]
//    score[0] = q · null_key + null_logit_bias
//    score[1+i] = q · key[L_i]
//======================================================//

#include "decode_time_slot_selector_GPU.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

namespace GRIM {

// ─── CUDA helpers ────────────────────────────────────

#define SELECTOR_CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string("DecodeTimeSlotSelector CUDA error at ") + \
            __FILE__ + ":" + std::to_string(__LINE__) + " — " + cudaGetErrorString(_e)); \
    } \
} while (0)

static constexpr int kBlock = 256;

// ─── Kernels ─────────────────────────────────────────

// MatVec: out[j] = sum_i(A[i * cols + j] * x[i])  where A is [rows, cols], x is [rows], out is [cols]
// One thread per output column.
__global__ void kernelMatVecTranspose(const float* __restrict__ A,
                                       const float* __restrict__ x,
                                       float* __restrict__ out,
                                       int rows, int cols) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float acc = 0.0f;
    for (int i = 0; i < rows; ++i) {
        acc += A[i * cols + j] * x[i];
    }
    out[j] = acc;
}

// Dot products: scores[s] = sum_j(q[j] * keys[s * d + j])  for s in [0, num_keys)
__global__ void kernelDotScores(const float* __restrict__ q,
                                 const float* __restrict__ keys,
                                 float* __restrict__ scores,
                                 int num_keys, int d) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_keys) return;
    float acc = 0.0f;
    const float* key_s = keys + s * d;
    for (int j = 0; j < d; ++j) {
        acc += q[j] * key_s[j];
    }
    scores[s] = acc;
}

// Add scalar bias to scores[0]
__global__ void kernelAddBias(float* __restrict__ scores, const float* __restrict__ bias) {
    scores[0] += bias[0];
}

// ─── Constructor ─────────────────────────────────────

DecodeTimeSlotSelectorLayer::DecodeTimeSlotSelectorLayer(
    const DecodeTimeSlotSelectorConfig& config,
    uint64_t seed,
    cudaStream_t init_stream)
    : config_(config)
{
    if (config_.d_model <= 0) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer: d_model must be positive, got " +
                                 std::to_string(config_.d_model));
    }
    if (config_.d_selector <= 0) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer: d_selector must be positive, got " +
                                 std::to_string(config_.d_selector));
    }
    if (config_.d_slot_features <= 0) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer: d_slot_features must be positive, got " +
                                 std::to_string(config_.d_slot_features));
    }
    if (!init_stream) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer: init_stream is NULL");
    }

    const int dm = config_.d_model;
    const int ds = config_.d_selector;
    const int df = config_.d_slot_features;

    auto make_param = [&](int rows, int cols, uint64_t s, const char* name) -> Tensor {
        auto t = Tensor::zeros(TensorContract::TensorShape::make_BSM(rows, cols),
                               true, init_stream, name);
        t.requires_grad_();
        t.ensure_grad();
        Tensor::xavier_uniform_(t, s, init_stream);
        return t;
    };

    // W_q_select: [d_model, d_selector]
    W_q_select_ = make_param(dm, ds, seed, "selector.W_q_select");

    // W_k_select: [d_slot_features, d_selector]
    W_k_select_ = make_param(df, ds, seed + 1, "selector.W_k_select");

    // null_key_select: [1, d_selector] — learnable NULL key
    null_key_select_ = make_param(1, ds, seed + 2, "selector.null_key_select");

    // null_logit_bias: [1, 1] scalar — init to 0
    null_logit_bias_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, 1),
                                     true, init_stream, "selector.null_logit_bias");
    null_logit_bias_.requires_grad_();
    null_logit_bias_.ensure_grad();

    // Scratch buffers
    SELECTOR_CUDA_CHECK(cudaMalloc(&d_query_buf_, static_cast<size_t>(ds) * sizeof(float)));
    SELECTOR_CUDA_CHECK(cudaMemsetAsync(d_query_buf_, 0, static_cast<size_t>(ds) * sizeof(float), init_stream));

    const int max_keys = kMaxSlots + 1; // NULL + up to kMaxSlots live slots
    SELECTOR_CUDA_CHECK(cudaMalloc(&d_keys_buf_, static_cast<size_t>(max_keys) * ds * sizeof(float)));
    SELECTOR_CUDA_CHECK(cudaMemsetAsync(d_keys_buf_, 0, static_cast<size_t>(max_keys) * ds * sizeof(float), init_stream));

    SELECTOR_CUDA_CHECK(cudaMalloc(&d_score_buffer_, static_cast<size_t>(max_keys) * sizeof(float)));
    SELECTOR_CUDA_CHECK(cudaMemsetAsync(d_score_buffer_, 0, static_cast<size_t>(max_keys) * sizeof(float), init_stream));
}

// ─── Destructor ──────────────────────────────────────

DecodeTimeSlotSelectorLayer::~DecodeTimeSlotSelectorLayer() {
    if (d_query_buf_)    { cudaFree(d_query_buf_);    d_query_buf_ = nullptr; }
    if (d_keys_buf_)     { cudaFree(d_keys_buf_);     d_keys_buf_ = nullptr; }
    if (d_score_buffer_) { cudaFree(d_score_buffer_);  d_score_buffer_ = nullptr; }
}

// ─── Move semantics ──────────────────────────────────

DecodeTimeSlotSelectorLayer::DecodeTimeSlotSelectorLayer(DecodeTimeSlotSelectorLayer&& other) noexcept
    : config_(other.config_),
      W_q_select_(std::move(other.W_q_select_)),
      W_k_select_(std::move(other.W_k_select_)),
      null_key_select_(std::move(other.null_key_select_)),
      null_logit_bias_(std::move(other.null_logit_bias_)),
      d_query_buf_(other.d_query_buf_),
      d_keys_buf_(other.d_keys_buf_),
      d_score_buffer_(other.d_score_buffer_)
{
    other.d_query_buf_ = nullptr;
    other.d_keys_buf_ = nullptr;
    other.d_score_buffer_ = nullptr;
}

DecodeTimeSlotSelectorLayer& DecodeTimeSlotSelectorLayer::operator=(DecodeTimeSlotSelectorLayer&& other) noexcept {
    if (this != &other) {
        if (d_query_buf_)    cudaFree(d_query_buf_);
        if (d_keys_buf_)     cudaFree(d_keys_buf_);
        if (d_score_buffer_) cudaFree(d_score_buffer_);

        config_ = other.config_;
        W_q_select_ = std::move(other.W_q_select_);
        W_k_select_ = std::move(other.W_k_select_);
        null_key_select_ = std::move(other.null_key_select_);
        null_logit_bias_ = std::move(other.null_logit_bias_);
        d_query_buf_ = other.d_query_buf_;
        d_keys_buf_ = other.d_keys_buf_;
        d_score_buffer_ = other.d_score_buffer_;

        other.d_query_buf_ = nullptr;
        other.d_keys_buf_ = nullptr;
        other.d_score_buffer_ = nullptr;
    }
    return *this;
}

// ─── Forward ─────────────────────────────────────────

SelectorScoreResult DecodeTimeSlotSelectorLayer::forward(
    const float* h_t,
    const float* slot_features,
    int num_live_slots,
    cudaStream_t stream)
{
    if (!h_t) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: h_t is NULL");
    }
    if (!stream) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: stream is NULL");
    }
    if (num_live_slots < 0 || num_live_slots > kMaxSlots) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: num_live_slots=" +
                                 std::to_string(num_live_slots) + " out of range [0, " +
                                 std::to_string(kMaxSlots) + "]");
    }
    if (num_live_slots > 0 && !slot_features) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: slot_features is NULL with num_live_slots > 0");
    }

    const int ds = config_.d_selector;
    const int dm = config_.d_model;
    const int df = config_.d_slot_features;
    const int total_keys = 1 + num_live_slots; // NULL + live slots

    // Step 1: q = W_q^T * h_t  (h_t [1, dm], W_q [dm, ds] → q [1, ds])
    // Transposed matmul: q[j] = sum_i W_q[i,j] * h_t[i]
    {
        const int grid = (ds + kBlock - 1) / kBlock;
        kernelMatVecTranspose<<<grid, kBlock, 0, stream>>>(
            W_q_select_.data, h_t, d_query_buf_, dm, ds);
    }

    // Step 2: Build keys array in d_keys_buf_
    // Key[0] = null_key_select (already stored, just copy)
    SELECTOR_CUDA_CHECK(cudaMemcpyAsync(
        d_keys_buf_, null_key_select_.data,
        static_cast<size_t>(ds) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    // Key[1..num_live] = W_k^T * slot_features[s]
    if (num_live_slots > 0) {
        for (int s = 0; s < num_live_slots; ++s) {
            const float* feat_s = slot_features + s * df;
            float* key_s = d_keys_buf_ + (1 + s) * ds;
            const int grid = (ds + kBlock - 1) / kBlock;
            kernelMatVecTranspose<<<grid, kBlock, 0, stream>>>(
                W_k_select_.data, feat_s, key_s, df, ds);
        }
    }

    // Step 3: Compute dot-product scores
    {
        const int grid = (total_keys + kBlock - 1) / kBlock;
        kernelDotScores<<<grid, kBlock, 0, stream>>>(
            d_query_buf_, d_keys_buf_, d_score_buffer_, total_keys, ds);
    }

    // Step 4: Add null_logit_bias to score[0]
    kernelAddBias<<<1, 1, 0, stream>>>(d_score_buffer_, null_logit_bias_.data);

    SelectorScoreResult result;
    result.d_scores = d_score_buffer_;
    result.num_live_slots = num_live_slots;
    return result;
}

} // namespace GRIM

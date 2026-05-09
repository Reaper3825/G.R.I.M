//======================================================//
//  DecodeTimeSlotSelectorLayer — implementation
//
//  Pattern B (self-managed autograd):
//  Forward uses autograd::matmul, autograd::add, autograd::concat.
//  Backward is automatic via the grad_fn chain — NO manual kernels.
//
//  Score vector over { NULL } ∪ L:
//    q = h_t @ W_q                             [1, ds]
//    null_score = q @ null_key^T                [1, 1]
//    null_score_biased = null_score + bias       [1, 1]
//    slot_keys = feats @ W_k                    [L, ds]
//    slot_scores = q @ slot_keys^T              [1, L]
//    scores = concat(null_score_biased, slot_scores) [1, 1+L]
//======================================================//

#include "decode_time_slot_selector_GPU.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>
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

// ─── Constructor ─────────────────────────────────────

DecodeTimeSlotSelectorLayer::DecodeTimeSlotSelectorLayer(
    const HyperParameters::DecodeTimeSelectorConstructionHP& config,
    uint64_t seed,
    cudaStream_t init_stream,
    cublasHandle_t cublas_handle)
    : config_(config), cublas_handle_(cublas_handle)
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
    if (!cublas_handle_) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer: cublas_handle is NULL");
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

    autograd::set_autograd_cublas_handle(cublas_handle_);

    std::fprintf(stderr, "[DecodeTimeSlotSelectorLayer] Autograd self-allocated: "
                 "W_q=[%d,%d], W_k=[%d,%d], null_key=[1,%d], bias=[1,1]\n",
                 dm, ds, df, ds, ds);
}

// ─── Destructor ──────────────────────────────────────

DecodeTimeSlotSelectorLayer::~DecodeTimeSlotSelectorLayer() {
    // Tensors clean up via RAII — no manual buffers
}

// ─── Move semantics ──────────────────────────────────

DecodeTimeSlotSelectorLayer::DecodeTimeSlotSelectorLayer(DecodeTimeSlotSelectorLayer&& other) noexcept
    : config_(other.config_),
    cublas_handle_(other.cublas_handle_),
      W_q_select_(std::move(other.W_q_select_)),
      W_k_select_(std::move(other.W_k_select_)),
      null_key_select_(std::move(other.null_key_select_)),
      null_logit_bias_(std::move(other.null_logit_bias_))
{
    other.cublas_handle_ = nullptr;
}

DecodeTimeSlotSelectorLayer& DecodeTimeSlotSelectorLayer::operator=(DecodeTimeSlotSelectorLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        cublas_handle_ = other.cublas_handle_;
        W_q_select_ = std::move(other.W_q_select_);
        W_k_select_ = std::move(other.W_k_select_);
        null_key_select_ = std::move(other.null_key_select_);
        null_logit_bias_ = std::move(other.null_logit_bias_);
        other.cublas_handle_ = nullptr;
    }
    return *this;
}

// ─── Forward (Pattern B: autograd primitives only) ───

SelectorForwardResult DecodeTimeSlotSelectorLayer::forward(
    const Tensor& h_t,
    const Tensor& slot_features,
    int num_live_slots,
    cudaStream_t stream)
{
    if (!stream) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: stream is NULL");
    }
    if (!h_t.data) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: h_t.data is NULL");
    }
    if (num_live_slots < 0 || num_live_slots > kMaxSlots) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: num_live_slots=" +
                                 std::to_string(num_live_slots) + " out of range [0, " +
                                 std::to_string(kMaxSlots) + "]");
    }
    if (num_live_slots > 0 && !slot_features.data) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: slot_features.data is NULL with num_live_slots > 0");
    }
    if (num_live_slots > 0) {
        // Validate slot_features shape matches [num_live_slots, d_slot_features].
        // A shape mismatch means a stale or mis-sized buffer would silently produce
        // wrong score vectors via the W_k matmul.
        if (slot_features.shape.layout == TensorContract::Layout::UNKNOWN) {
            throw std::runtime_error(
                "DecodeTimeSlotSelectorLayer::forward: slot_features has UNKNOWN layout "
                "with num_live_slots=" + std::to_string(num_live_slots));
        }
        const auto& sf = slot_features.shape.as_2d();
        if (sf.rows != num_live_slots) {
            throw std::runtime_error(
                "DecodeTimeSlotSelectorLayer::forward: slot_features.rows=" +
                std::to_string(sf.rows) + " != num_live_slots=" +
                std::to_string(num_live_slots));
        }
        if (sf.cols != config_.d_slot_features) {
            throw std::runtime_error(
                "DecodeTimeSlotSelectorLayer::forward: slot_features.cols=" +
                std::to_string(sf.cols) + " != config.d_slot_features=" +
                std::to_string(config_.d_slot_features));
        }
    }

    // Set cuBLAS handle for autograd matmul
    if (!cublas_handle_) {
        throw std::runtime_error("DecodeTimeSlotSelectorLayer::forward: cublas_handle is NULL");
    }
    autograd::set_autograd_cublas_handle(cublas_handle_);

    SelectorForwardResult result;
    result.num_live_slots = num_live_slots;

    // Step 1: q = h_t @ W_q  →  [1, d_selector]
    result.q = autograd::matmul(h_t, W_q_select_, stream, h_t.data, nullptr);

    // Step 2: null_score = q @ null_key^T  →  [1, 1]
    Tensor null_score = autograd::matmul(result.q, null_key_select_, stream,
                                          result.q.data, nullptr, /*transpose_b=*/true);

    // Step 3: null_score_biased = null_score + null_logit_bias  →  [1, 1]
    Tensor null_score_biased = autograd::add(null_score, null_logit_bias_, stream);

    if (num_live_slots > 0) {
        // Step 4: slot_keys = slot_features @ W_k  →  [L, d_selector]
        result.slot_keys = autograd::matmul(slot_features, W_k_select_, stream,
                                             slot_features.data, nullptr);

        // Step 5: slot_scores = q @ slot_keys^T  →  [1, L]
        Tensor slot_scores = autograd::matmul(result.q, result.slot_keys, stream,
                                               result.q.data, result.slot_keys.data,
                                               /*transpose_b=*/true);

        // Step 6: scores = concat(null_score_biased, slot_scores)  →  [1, 1+L]
        result.scores = autograd::concat(null_score_biased, slot_scores, stream);
    } else {
        // No live slots: scores = null_score_biased  →  [1, 1]
        result.scores = std::move(null_score_biased);
    }

    return result;
}

} // namespace GRIM

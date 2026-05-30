//======================================================//
//  DecodeTimeSlotSelector — implementation
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

void validateDecodeTimeSlotSelectorConfig(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp)
{
    if (hp.d_model <= 0) {
        throw std::runtime_error("DecodeTimeSlotSelector: d_model must be positive, got " +
                                 std::to_string(hp.d_model));
    }
    if (hp.d_selector <= 0) {
        throw std::runtime_error("DecodeTimeSlotSelector: d_selector must be positive, got " +
                                 std::to_string(hp.d_selector));
    }
    if (hp.d_slot_features <= 0) {
        throw std::runtime_error("DecodeTimeSlotSelector: d_slot_features must be positive, got " +
                                 std::to_string(hp.d_slot_features));
    }
    if (hp.num_slots <= 0) {
        throw std::runtime_error("DecodeTimeSlotSelector: num_slots must be positive, got " +
                                 std::to_string(hp.num_slots));
    }
    if (hp.scratch_slots < 0 || hp.scratch_slots >= hp.num_slots) {
        throw std::runtime_error("DecodeTimeSlotSelector: scratch_slots=" +
                                 std::to_string(hp.scratch_slots) +
                                 " out of range [0, " + std::to_string(hp.num_slots) + ")");
    }
}

DecodeTimeSlotSelector createDecodeTimeSlotSelector(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    uint64_t seed,
    cudaStream_t init_stream)
{
    validateDecodeTimeSlotSelectorConfig(hp);
    if (!init_stream) {
        throw std::runtime_error("createDecodeTimeSlotSelector: init_stream is NULL");
    }

    const int dm = hp.d_model;
    const int ds = hp.d_selector;
    const int df = hp.d_slot_features;

    auto make_param = [&](int rows, int cols, uint64_t s, const char* name) -> Tensor {
        auto t = Tensor::zeros(TensorContract::TensorShape::make_BSM(rows, cols),
                               true, init_stream, name);
        t.requires_grad_();
        t.ensure_grad();
        Tensor::xavier_uniform_(t, s, init_stream);
        return t;
    };

    DecodeTimeSlotSelector selector{};
    selector.W_q_select = make_param(dm, ds, seed, "selector.W_q_select");
    selector.W_k_select = make_param(df, ds, seed + 1, "selector.W_k_select");
    selector.null_key_select = make_param(1, ds, seed + 2, "selector.null_key_select");
    selector.null_logit_bias = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, 1),
                                             true, init_stream, "selector.null_logit_bias");
    selector.null_logit_bias.requires_grad_();
    selector.null_logit_bias.ensure_grad();

    std::fprintf(stderr, "[DecodeTimeSlotSelector] Autograd self-allocated: "
                 "W_q=[%d,%d], W_k=[%d,%d], null_key=[1,%d], bias=[1,1]\n",
                 dm, ds, df, ds, ds);
    return selector;
}

// ─── Forward (Pattern B: autograd primitives only) ───

GRIM::Forward::SelectorForwardResult forwardDecodeTimeSlotSelector(
    const DecodeTimeSlotSelector& selector,
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    const Tensor& h_t,
    const Tensor& slot_features,
    int num_live_slots,
    cudaStream_t stream,
    cublasHandle_t cublas_handle)
{
    validateDecodeTimeSlotSelectorConfig(hp);
    if (!stream) {
        throw std::runtime_error("forwardDecodeTimeSlotSelector: stream is NULL");
    }
    if (!h_t.data) {
        throw std::runtime_error("forwardDecodeTimeSlotSelector: h_t.data is NULL");
    }
    const int max_live_slots = hp.num_slots - hp.scratch_slots;
    if (num_live_slots < 0 || num_live_slots > max_live_slots) {
        throw std::runtime_error("forwardDecodeTimeSlotSelector: num_live_slots=" +
                                 std::to_string(num_live_slots) + " out of range [0, " +
                                 std::to_string(max_live_slots) + "]");
    }
    if (num_live_slots > 0 && !slot_features.data) {
        throw std::runtime_error("forwardDecodeTimeSlotSelector: slot_features.data is NULL with num_live_slots > 0");
    }
    if (num_live_slots > 0) {
        // Validate slot_features shape matches [num_live_slots, d_slot_features].
        // A shape mismatch means a stale or mis-sized buffer would silently produce
        // wrong score vectors via the W_k matmul.
        if (slot_features.shape.layout == TensorContract::Layout::UNKNOWN) {
            throw std::runtime_error(
                "forwardDecodeTimeSlotSelector: slot_features has UNKNOWN layout "
                "with num_live_slots=" + std::to_string(num_live_slots));
        }
        const auto& sf = slot_features.shape.as_2d();
        if (sf.rows != num_live_slots) {
            throw std::runtime_error(
                "forwardDecodeTimeSlotSelector: slot_features.rows=" +
                std::to_string(sf.rows) + " != num_live_slots=" +
                std::to_string(num_live_slots));
        }
        if (sf.cols != hp.d_slot_features) {
            throw std::runtime_error(
                "forwardDecodeTimeSlotSelector: slot_features.cols=" +
                std::to_string(sf.cols) + " != hp.d_slot_features=" +
                std::to_string(hp.d_slot_features));
        }
    }

    // Set cuBLAS handle for autograd matmul
    if (!cublas_handle) {
        throw std::runtime_error("forwardDecodeTimeSlotSelector: cublas_handle is NULL");
    }
    autograd::set_autograd_cublas_handle(cublas_handle);

    GRIM::Forward::SelectorForwardResult result;
    result.num_live_slots = num_live_slots;

    // Step 1: q = h_t @ W_q  →  [1, d_selector]
    result.q = autograd::matmul(h_t, selector.W_q_select, stream, h_t.data, nullptr);

    // Step 2: null_score = q @ null_key^T  →  [1, 1]
    Tensor null_score = autograd::matmul(result.q, selector.null_key_select, stream,
                                          result.q.data, nullptr, /*transpose_b=*/true);

    // Step 3: null_score_biased = null_score + null_logit_bias  →  [1, 1]
    Tensor null_score_biased = autograd::add(null_score, selector.null_logit_bias, stream);

    if (num_live_slots > 0) {
        // Step 4: slot_keys = slot_features @ W_k  →  [L, d_selector]
        result.slot_keys = autograd::matmul(slot_features, selector.W_k_select, stream,
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

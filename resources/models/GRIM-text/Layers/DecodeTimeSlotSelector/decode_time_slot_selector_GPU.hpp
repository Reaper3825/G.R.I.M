#pragma once
//======================================================//
//  DecodeTimeSlotSelectorLayer — sole owner of trainable
//  decode-time selector tensors and learned slot-state
//  encoding.
//
//  Returns an ordered score vector over { NULL } ∪ L
//  where L is the live slot candidate set.
//
//  Required baseline tensors:
//    W_q_select       — query projection (d_model → d_selector)
//    W_k_select       — key projection (d_slot_features → d_selector)
//    null_key_select  — learnable NULL key vector (d_selector)
//    null_logit_bias  — learnable scalar NULL bias
//
//  No other module may own trainable decode-time selector
//  tensors. Policy code consumes the score vector as-is.
//
//  Pattern B (self-managed autograd): forward uses
//  autograd::matmul, autograd::add, autograd::concat.
//  Backward is automatic via the grad_fn chain.
//======================================================//

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cublas_v2.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
struct cublasContext;
using cublasHandle_t = cublasContext*;
#endif

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

// Forward result: autograd-tracked score tensor + keep-alive intermediates
struct SelectorForwardResult {
    Tensor scores;           // [1, 1+num_live_slots] — logits over { NULL } ∪ L
    int num_live_slots = 0;

    // Keep-alive: intermediate Tensors whose .data is cached by upstream MatMulGradFn nodes.
    // These MUST stay alive until backward() completes.
    Tensor q;                // [1, d_selector]
    Tensor slot_keys;        // [num_live, d_selector] (empty if no slots)
};

class DecodeTimeSlotSelectorLayer {
public:
    DecodeTimeSlotSelectorLayer(const HyperParameters::DecodeTimeSelectorConstructionHP& config,
                                uint64_t seed,
                                cudaStream_t init_stream,
                                cublasHandle_t cublas_handle);
    ~DecodeTimeSlotSelectorLayer();

    DecodeTimeSlotSelectorLayer(DecodeTimeSlotSelectorLayer&& other) noexcept;
    DecodeTimeSlotSelectorLayer& operator=(DecodeTimeSlotSelectorLayer&& other) noexcept;

    DecodeTimeSlotSelectorLayer(const DecodeTimeSlotSelectorLayer&) = delete;
    DecodeTimeSlotSelectorLayer& operator=(const DecodeTimeSlotSelectorLayer&) = delete;

    // Forward: compute scores over { NULL } ∪ L using autograd primitives.
    //
    // h_t:             [1, d_model] hidden state (non-owning view OK)
    // slot_features:   [num_live, d_slot_features] fixed slot features (non-owning view OK)
    // num_live_slots:  number of live candidates in L
    // stream:          CUDA stream for async execution
    //
    // Returns SelectorForwardResult with autograd-tracked scores tensor.
    // Caller MUST keep the result alive until backward() completes.
    SelectorForwardResult forward(const Tensor& h_t,
                                  const Tensor& slot_features,
                                  int num_live_slots,
                                  cudaStream_t stream);

    const HyperParameters::DecodeTimeSelectorConstructionHP& config() const { return config_; }

    // ── Tensor accessors (const + non-const) ──
    Tensor& W_q_select()             { return W_q_select_; }
    const Tensor& W_q_select() const { return W_q_select_; }

    Tensor& W_k_select()             { return W_k_select_; }
    const Tensor& W_k_select() const { return W_k_select_; }

    Tensor& null_key_select()             { return null_key_select_; }
    const Tensor& null_key_select() const { return null_key_select_; }

    Tensor& null_logit_bias()             { return null_logit_bias_; }
    const Tensor& null_logit_bias() const { return null_logit_bias_; }

private:
    HyperParameters::DecodeTimeSelectorConstructionHP config_;
    cublasHandle_t cublas_handle_ = nullptr;

    // Required baseline trainable tensors
    Tensor W_q_select_;       // [d_model, d_selector]
    Tensor W_k_select_;       // [d_slot_features, d_selector]
    Tensor null_key_select_;  // [1, d_selector]
    Tensor null_logit_bias_;  // [1, 1] scalar

    static constexpr int kMaxSlots = 16; // Upper bound for validation
};

} // namespace GRIM

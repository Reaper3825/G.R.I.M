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
//======================================================//

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct DecodeTimeSlotSelectorConfig {
    int d_model = 0;         // Hidden state dimension (query source)
    int d_selector = 0;      // Selector key/query projection dimension
    int d_slot_features = 0; // Fixed slot feature vector size from policy
};

// Score result: ordered over { NULL } ∪ L
// Index 0 = NULL, indices 1..|L| = candidates in L-order
struct SelectorScoreResult {
    float* d_scores = nullptr;  // Device pointer, length = 1 + num_live_slots
    int num_live_slots = 0;
};

class DecodeTimeSlotSelectorLayer {
public:
    DecodeTimeSlotSelectorLayer(const DecodeTimeSlotSelectorConfig& config,
                                uint64_t seed,
                                cudaStream_t init_stream);
    ~DecodeTimeSlotSelectorLayer();

    DecodeTimeSlotSelectorLayer(DecodeTimeSlotSelectorLayer&& other) noexcept;
    DecodeTimeSlotSelectorLayer& operator=(DecodeTimeSlotSelectorLayer&& other) noexcept;

    DecodeTimeSlotSelectorLayer(const DecodeTimeSlotSelectorLayer&) = delete;
    DecodeTimeSlotSelectorLayer& operator=(const DecodeTimeSlotSelectorLayer&) = delete;

    // Forward: compute scores over { NULL } ∪ L
    //
    // h_t:             device pointer to decode hidden state [1, d_model]
    // slot_features:   device pointer to fixed slot features [num_live, d_slot_features]
    // num_live_slots:  number of live candidates in L
    // stream:          CUDA stream for async execution
    //
    // Returns scores written to internal buffer d_score_buffer_.
    // Index 0 = NULL score, indices 1..num_live = candidate scores.
    SelectorScoreResult forward(const float* h_t,
                                const float* slot_features,
                                int num_live_slots,
                                cudaStream_t stream);

    const DecodeTimeSlotSelectorConfig& config() const { return config_; }

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
    DecodeTimeSlotSelectorConfig config_;

    // Required baseline trainable tensors
    Tensor W_q_select_;       // [d_model, d_selector]
    Tensor W_k_select_;       // [d_slot_features, d_selector]
    Tensor null_key_select_;  // [1, d_selector]
    Tensor null_logit_bias_;  // [1, 1] scalar

    // Internal scratch buffers
    float* d_query_buf_ = nullptr;   // [1, d_selector]
    float* d_keys_buf_ = nullptr;    // [max_slots + 1, d_selector]
    float* d_score_buffer_ = nullptr; // [max_slots + 1]

    static constexpr int kMaxSlots = 16; // Upper bound for buffer allocation
};

} // namespace GRIM

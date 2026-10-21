#pragma once

#include <algorithm>

namespace GRIM::Inference {

enum class InferenceReturnReason {
    Continue,
    ExplicitFinish,
    TokenLimit,
    ContextLimit,
    EOS,
    SpanBoundary,
};

// Decide when to leave token generation and return to the caller's text decode.
// These functions have no GPU, tokenizer, sampling, or trajectory ownership.
inline int remainingInferenceTokenCapacity(int new_tokens, int total_tokens,
                                           int max_new_tokens, int max_sequence_tokens) {
    return std::min(max_new_tokens - new_tokens, max_sequence_tokens - total_tokens);
}

inline InferenceReturnReason evaluateInferenceReturn(
    bool explicitly_finished, int new_tokens, int total_tokens, int last_token_id,
    int max_new_tokens, int min_new_tokens, int max_sequence_tokens, int eos_token_id,
    bool span_closed = false) {
    // Preserve existing precedence: explicit finish, capacity, then EOS.
    if (explicitly_finished) return InferenceReturnReason::ExplicitFinish;
    if (new_tokens >= max_new_tokens) return InferenceReturnReason::TokenLimit;
    if (total_tokens >= max_sequence_tokens) return InferenceReturnReason::ContextLimit;
    if (eos_token_id >= 0 && new_tokens > 0 &&
        new_tokens >= min_new_tokens && last_token_id == eos_token_id)
        return InferenceReturnReason::EOS;
    if (span_closed) return InferenceReturnReason::SpanBoundary;
    return InferenceReturnReason::Continue;
}

inline bool isTerminalInferenceReturn(InferenceReturnReason reason) {
    return reason != InferenceReturnReason::Continue &&
           reason != InferenceReturnReason::SpanBoundary;
}

} // namespace GRIM::Inference

// Training-only corpus expectation step for Unigram vocabulary fitting.
// Owns segment traversal and posterior aggregation; forward-backward math
// lives in UnigramForwardBackward. Does not mutate vocabulary scores.

#pragma once

#include "../Unigram.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM {
namespace Tokenizer {

struct UnigramExpectationStepRequest {
    const std::vector<UnigramPiece>& pieces;
    const std::vector<std::string>& normalized_texts;
    // One sorted, non-overlapping exclusion list per normalized text.
    const std::vector<std::vector<AtomSpan>>& normalized_exclusion_spans;
    bool enable_byte_fallback;
};

struct UnigramExpectationStepResult {
    // Keys are vocabulary token IDs, not piece indices.
    std::unordered_map<int, double> learned_token_counts;
    double expected_learned_piece_tokens = 0.0;
    double expected_fixed_penalty_byte_fallback_tokens = 0.0;
    double log_likelihood = 0.0;
};

// Evaluates the current vocabulary over all eligible corpus segments.
// The caller retains corpus/vocabulary ownership. Trie and posterior buffers
// are local to this call; fallback mass stays separate from learned counts.
UnigramExpectationStepResult runUnigramExpectationStep(
    const UnigramExpectationStepRequest& request);

} // namespace Tokenizer
} // namespace GRIM

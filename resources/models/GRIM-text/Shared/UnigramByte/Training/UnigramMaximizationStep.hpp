// Training-only score update for the Unigram EM M-step.

#pragma once

#include "../Unigram.hpp"

#include <unordered_map>
#include <vector>

namespace GRIM {
namespace Tokenizer {

struct UnigramMaximizationStepRequest {
    // Only ordinary piece scores are mutated; order and text stay unchanged.
    std::vector<UnigramPiece>& pieces;
    // Expected counts keyed by vocabulary token ID, not piece index.
    const std::unordered_map<int, double>& learned_token_counts;
    double expected_learned_piece_tokens;
    double smoothing;
};

// Updates log-probability scores from smoothed expected counts. User-defined
// pieces are untouched and fixed-penalty byte fallback is excluded from the
// normalizer. Returns the number of ordinary pieces with zero expected usage
// before smoothing (the trainer's "unused" telemetry).
int runUnigramMaximizationStep(const UnigramMaximizationStepRequest& request);

} // namespace Tokenizer
} // namespace GRIM

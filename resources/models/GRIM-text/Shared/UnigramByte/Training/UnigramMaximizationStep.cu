// Learned-piece probability normalization for the Unigram EM M-step.

#include "UnigramMaximizationStep.hpp"

#include <algorithm>
#include <cmath>

namespace GRIM {
namespace Tokenizer {

int runUnigramMaximizationStep(const UnigramMaximizationStepRequest& request) {
    // Byte fallback remains outside the normalized learned-piece distribution.
    // Its UNKNOWN_SCORE path is a fixed unnormalized per-byte coverage penalty, so the
    // M-step normalizer is learned-piece posterior mass + learned-piece smoothing only.
    const size_t ordinary_piece_count = static_cast<size_t>(std::count_if(
        request.pieces.begin(), request.pieces.end(),
        [](const UnigramPiece& piece) { return !piece.is_user_defined; }));
    double smoothed_total = request.expected_learned_piece_tokens +
        request.smoothing * static_cast<double>(ordinary_piece_count);
    int zero_count = 0;
    for (size_t i = 0; i < request.pieces.size(); ++i) {
        auto& piece = request.pieces[i];
        if (piece.is_user_defined) continue;
        int tid = tokenIdForIndex(static_cast<int>(i));
        double expected_count = 0.0;
        auto count_it = request.learned_token_counts.find(tid);
        if (count_it != request.learned_token_counts.end()) {
            expected_count = count_it->second;
        }
        double count = expected_count + request.smoothing;
        piece.score = static_cast<float>(std::log(count / smoothed_total));
        if (expected_count == 0.0) {
            zero_count++;
        }
    }
    return zero_count;
}

} // namespace Tokenizer
} // namespace GRIM

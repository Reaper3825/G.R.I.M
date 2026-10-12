// Training-only vocabulary selection by estimated likelihood loss.

#pragma once

#include "UnigramTrainingSession.hpp"

#include <vector>

namespace GRIM {
namespace Tokenizer {

struct UnigramLikelihoodPruningResult {
    std::vector<UnigramPiece> survivors;
    int removed = 0;
    int protected_count = 0;
    int irreplaceable_no_alt = 0;
};

// Ranks pieces by expected usage times the log-score advantage over their
// best alternative learned-piece segmentation. Protects user-defined,
// single-codepoint, and irreplaceable pieces. Byte fallback is not used as
// an alternative segmentation.
//
// Expected counts are indexed by piece position. Survivors are moved out of
// pieces in original order; the caller owns vocabulary rewriting and EM refits.
UnigramLikelihoodPruningResult runUnigramLikelihoodPruning(
    UnigramTrainingSession& session, const char* caller);

} // namespace Tokenizer
} // namespace GRIM

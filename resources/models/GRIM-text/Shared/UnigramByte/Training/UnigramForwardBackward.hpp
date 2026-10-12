//======================================================//
//  UnigramForwardBackward.hpp
//  True Unigram LM forward-backward estimator for training
//
//  This is training-only math. Production segmentation stays in
//  UnigramViterbi.hpp/.cu so encode() keeps its single-best path
//  behavior while trainFromCorpus() can use soft EM counts.
//======================================================//

#pragma once

#include "../Unigram.hpp"
#include "UnigramTrainingSession.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace GRIM {
namespace Tokenizer {

struct UnigramForwardBackwardStats {
    explicit UnigramForwardBackwardStats(size_t piece_count);

    std::vector<double> piece_expected_counts;
    double expected_learned_piece_tokens = 0.0;
    // Byte fallback is a fixed unnormalized per-byte coverage penalty path. This counter
    // is telemetry for fallback posterior usage; it is not normalized into the
    // learned-piece distribution during the M-step.
    double expected_fixed_penalty_byte_fallback_tokens = 0.0;
    double log_likelihood = 0.0;
};

class UnigramForwardBackwardLattice final {
public:
    UnigramForwardBackwardLattice(const std::vector<UnigramPiece>& pieces,
                                  bool enable_byte_fallback,
                                  const char* caller);

    void accumulateSegment(const std::string& normalized_segment,
                           UnigramForwardBackwardStats& stats,
                           const char* caller) const;

    size_t pieceCount() const noexcept { return piece_count_; }

    // Export score-independent matching edges once per vocabulary shape.
    void appendSegmentMatches(std::string_view segment, size_t byte_offset,
                              UnigramTrainingStorage& storage) const;

private:
    struct TrieNode final {
        std::array<int, 256> children;
        int piece_index = -1;
        double score = 0.0;

        TrieNode();
    };

    std::vector<TrieNode> trie_;
    size_t piece_count_ = 0;
    bool enable_byte_fallback_ = false;

    void insertPiece(const UnigramPiece& piece,
                     int piece_index,
                     const char* caller);
};

} // namespace Tokenizer
} // namespace GRIM

namespace GRIM::Tokenizer {
// Session path uses cached edges and reusable scratch, retaining the same
// log-domain forward/backward and raw-byte fixed-penalty fallback math.
void accumulateUnigramTrainingSegment(UnigramTrainingSession& session,
    const UnigramTrainingSegment& segment, const char* caller);
}

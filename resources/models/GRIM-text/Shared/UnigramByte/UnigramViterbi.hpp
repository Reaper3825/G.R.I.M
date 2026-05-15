//======================================================//
//  UnigramViterbi.hpp
//  RAII Viterbi segmentation session for UnigramLM
//
//  UnigramLM owns learned vocab and trie state. This file owns
//  per-segmentation dynamic-programming state and path materialization.
//======================================================//

#pragma once

#include "Unigram.hpp"

#include <string>
#include <utility>
#include <vector>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  Viterbi Node
//======================================================//
struct UnigramViterbiNode {
    float score = 0.0f;     // Best score to reach this position
    int prev_pos = -1;      // Previous position in best path
    int token_id = -1;      // Token ID of piece ending here
    int piece_length = 0;   // Length of piece in bytes
};

//======================================================//
//  UnigramViterbiSession
//======================================================//
class UnigramViterbiSession final {
public:
    UnigramViterbiSession(const UnigramLM& model,
                          const std::string& normalized_text,
                          const char* caller);
    ~UnigramViterbiSession() = default;

    UnigramViterbiSession(const UnigramViterbiSession&) = delete;
    UnigramViterbiSession& operator=(const UnigramViterbiSession&) = delete;

    UnigramViterbiSession(UnigramViterbiSession&&) noexcept = default;
    UnigramViterbiSession& operator=(UnigramViterbiSession&&) noexcept = default;

    const std::vector<int>& tokens() const noexcept { return tokens_; }
    std::vector<int> takeTokens() { return std::move(tokens_); }
    float pathScore() const noexcept { return path_score_; }

private:
    static std::vector<UnigramViterbiNode> runForward(const UnigramLM& model,
                                                      const std::string& normalized_text,
                                                      const char* caller);
    static std::vector<int> runBacktrack(const std::vector<UnigramViterbiNode>& nodes,
                                         int end_pos,
                                         const char* caller);

    std::vector<UnigramViterbiNode> nodes_;
    std::vector<int> tokens_;
    float path_score_ = 0.0f;
};

} // namespace Tokenizer
} // namespace GRIM
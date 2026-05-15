//======================================================//
//  UnigramViterbi.hpp
//  CUDA-backed RAII Viterbi segmentation session for UnigramLM
//
//  UnigramLM owns learned vocab and trie state. This file owns
//  per-segmentation launch validation and path materialization.
//======================================================//

#pragma once

#include "Unigram.hpp"

#include <string>
#include <utility>
#include <vector>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  CUDA Kernel Status Codes
//======================================================//
inline constexpr int kUnigramViterbiCudaOk = 0;
inline constexpr int kUnigramViterbiCudaNullText = 1;
inline constexpr int kUnigramViterbiCudaNullTrieChildren = 2;
inline constexpr int kUnigramViterbiCudaNullTrieTokenIds = 3;
inline constexpr int kUnigramViterbiCudaNullTrieScores = 4;
inline constexpr int kUnigramViterbiCudaNullViterbiScores = 5;
inline constexpr int kUnigramViterbiCudaNullViterbiPrev = 6;
inline constexpr int kUnigramViterbiCudaNullViterbiTokens = 7;
inline constexpr int kUnigramViterbiCudaEmptyTrie = 8;
inline constexpr int kUnigramViterbiCudaTrieChildOutOfRange = 9;
inline constexpr int kUnigramViterbiCudaNullOutputTokens = 10;
inline constexpr int kUnigramViterbiCudaNullOutputCount = 11;
inline constexpr int kUnigramViterbiCudaInvalidMaxTokens = 12;
inline constexpr int kUnigramViterbiCudaBacktrackLengthTooLarge = 13;
inline constexpr int kUnigramViterbiCudaInvalidBackpointer = 14;
inline constexpr int kUnigramViterbiCudaOutputBufferTooSmall = 15;
inline constexpr int kUnigramViterbiCudaByteFallbackSpanInvalid = 16;
inline constexpr int kUnigramViterbiCudaBacktrackSafetyLimit = 17;
inline constexpr int kUnigramViterbiCudaNullViterbiPrevIsFallback = 18;
inline constexpr int kUnigramViterbiCudaNullOutputIsFallback = 19;

inline const char* unigramViterbiCudaErrorName(int code) {
    switch (code) {
        case kUnigramViterbiCudaOk: return "OK";
        case kUnigramViterbiCudaNullText: return "NullText";
        case kUnigramViterbiCudaNullTrieChildren: return "NullTrieChildren";
        case kUnigramViterbiCudaNullTrieTokenIds: return "NullTrieTokenIds";
        case kUnigramViterbiCudaNullTrieScores: return "NullTrieScores";
        case kUnigramViterbiCudaNullViterbiScores: return "NullViterbiScores";
        case kUnigramViterbiCudaNullViterbiPrev: return "NullViterbiPrev";
        case kUnigramViterbiCudaNullViterbiTokens: return "NullViterbiTokens";
        case kUnigramViterbiCudaEmptyTrie: return "EmptyTrie";
        case kUnigramViterbiCudaTrieChildOutOfRange: return "TrieChildOutOfRange";
        case kUnigramViterbiCudaNullOutputTokens: return "NullOutputTokens";
        case kUnigramViterbiCudaNullOutputCount: return "NullOutputCount";
        case kUnigramViterbiCudaInvalidMaxTokens: return "InvalidMaxTokens";
        case kUnigramViterbiCudaBacktrackLengthTooLarge: return "BacktrackLengthTooLarge";
        case kUnigramViterbiCudaInvalidBackpointer: return "InvalidBackpointer";
        case kUnigramViterbiCudaOutputBufferTooSmall: return "OutputBufferTooSmall";
        case kUnigramViterbiCudaByteFallbackSpanInvalid: return "ByteFallbackSpanInvalid";
        case kUnigramViterbiCudaBacktrackSafetyLimit: return "BacktrackSafetyLimit";
        case kUnigramViterbiCudaNullViterbiPrevIsFallback: return "NullViterbiPrevIsFallback";
        case kUnigramViterbiCudaNullOutputIsFallback: return "NullOutputIsFallback";
        default: return "UnknownUnigramViterbiCudaError";
    }
}

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
    const std::vector<bool>& fallbackFlags() const noexcept { return token_is_fallback_; }
    std::vector<int> takeTokens() { return std::move(tokens_); }
    float pathScore() const noexcept { return path_score_; }

private:
    struct CudaResult {
        std::vector<int> tokens;
        std::vector<bool> token_is_fallback;
        float path_score = 0.0f;
    };

    static CudaResult runCuda(const UnigramLM& model,
                              const std::string& normalized_text,
                              const char* caller);

    std::vector<int> tokens_;
    std::vector<bool> token_is_fallback_;
    float path_score_ = 0.0f;
};

} // namespace Tokenizer
} // namespace GRIM
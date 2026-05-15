//======================================================//
//  VocabWriteOp.hpp
//  Single learned-vocab mutation primitive for UnigramLM
//
//  Token IDs are position-derived:
//    token_id = UNIGRAM_VOCAB_OFFSET + index_in_pieces
//  Therefore the vector write and text->index write must happen as one
//  validated operation. Call this primitive instead of writing pieces_ or
//  piece_to_id_ directly.
//======================================================//

#pragma once

#include "Unigram.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace GRIM {
namespace Tokenizer {

enum class UnigramVocabWriteMode {
    AppendOnly,
    AppendOrUpdate
};

struct UnigramVocabWriteTarget {
    std::vector<UnigramPiece>& pieces;
    std::unordered_map<std::string, int>& piece_to_id;
};

struct UnigramVocabWriteRequest {
    UnigramVocabWriteTarget target;
    UnigramPiece piece;
    int expected_token_id;
    UnigramVocabWriteMode mode;
    const char* caller;
};

struct UnigramVocabWriteResult {
    int index;
    int token_id;
    bool inserted;
};

inline void requireUnigramVocabWriteCaller(const char* caller) {
    if (!caller || caller[0] == '\0') {
        throw std::runtime_error("UnigramVocabWriteOp requires a non-empty caller label");
    }
}

inline int requireUnigramVocabWriteIndex(int token_id, const char* caller) {
    requireUnigramVocabWriteCaller(caller);
    if (token_id < UNIGRAM_VOCAB_OFFSET) {
        throw std::runtime_error(std::string(caller) +
                                 ": expected learned unigram token_id >= UNIGRAM_VOCAB_OFFSET=" +
                                 std::to_string(UNIGRAM_VOCAB_OFFSET) +
                                 ", got " + std::to_string(token_id));
    }
    return token_id - UNIGRAM_VOCAB_OFFSET;
}

inline int tokenIdForUnigramVocabWriteIndex(int index, const char* caller) {
    requireUnigramVocabWriteCaller(caller);
    if (index < 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned unigram index must be >= 0, got " +
                                 std::to_string(index));
    }
    return UNIGRAM_VOCAB_OFFSET + index;
}

inline int learnedPieceLimitFromTokenizerHP(
    const ::GRIM::HyperParameters::TokenizerHP& tokenizer_hp,
    const char* caller) {
    requireUnigramVocabWriteCaller(caller);
    ::GRIM::HyperParameters::requirePositiveGroupingValue(
        tokenizer_hp.target_vocab_size,
        "target_vocab_size",
        caller);
    return tokenizer_hp.target_vocab_size;
}

inline void validateUnigramVocabWritePiece(
    const UnigramPiece& piece,
    const char* caller) {
    requireUnigramVocabWriteCaller(caller);
    if (piece.text.empty()) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned unigram piece text is empty");
    }
    if (piece.text.size() > static_cast<std::size_t>(MAX_PIECE_LENGTH)) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned unigram piece text has " +
                                 std::to_string(piece.text.size()) +
                                 " bytes, max=" + std::to_string(MAX_PIECE_LENGTH));
    }
    if (!std::isfinite(piece.score)) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned unigram piece score is not finite for text='" +
                                 piece.text.substr(0, 64) + "'");
    }
}

inline void requireUnigramVocabWriteSizeFitsInt(
    std::size_t size,
    const char* caller) {
    requireUnigramVocabWriteCaller(caller);
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(caller) +
                                 ": learned unigram vocab size exceeds int range: " +
                                 std::to_string(size));
    }
}

inline void requireUnigramVocabWriteExistingIndex(
    const UnigramVocabWriteTarget& target,
    int index,
    const std::string& text,
    const char* caller) {
    requireUnigramVocabWriteCaller(caller);
    if (index < 0 || index >= static_cast<int>(target.pieces.size())) {
        throw std::runtime_error(std::string(caller) +
                                 ": piece_to_id index out of range for text='" +
                                 text.substr(0, 64) + "': index=" +
                                 std::to_string(index) + " pieces=" +
                                 std::to_string(target.pieces.size()));
    }
    if (target.pieces[static_cast<std::size_t>(index)].text != text) {
        throw std::runtime_error(std::string(caller) +
                                 ": piece_to_id is inconsistent for text='" +
                                 text.substr(0, 64) + "' at index=" +
                                 std::to_string(index));
    }
}

inline UnigramVocabWriteResult applyUnigramVocabWriteOp(
    UnigramVocabWriteRequest request) {
    const char* caller = request.caller;
    requireUnigramVocabWriteCaller(caller);
    validateUnigramVocabWritePiece(request.piece, caller);

    const int expected_index = requireUnigramVocabWriteIndex(
        request.expected_token_id,
        caller);

    auto existing = request.target.piece_to_id.find(request.piece.text);
    if (existing != request.target.piece_to_id.end()) {
        const int existing_index = existing->second;
        requireUnigramVocabWriteExistingIndex(
            request.target,
            existing_index,
            request.piece.text,
            caller);
        if (expected_index != existing_index) {
            throw std::runtime_error(std::string(caller) +
                                     ": expected token_id=" +
                                     std::to_string(request.expected_token_id) +
                                     " maps to index=" + std::to_string(expected_index) +
                                     " but existing piece index=" +
                                     std::to_string(existing_index));
        }
        if (request.mode == UnigramVocabWriteMode::AppendOnly) {
            throw std::runtime_error(std::string(caller) +
                                     ": duplicate learned unigram piece in append-only write: '" +
                                     request.piece.text.substr(0, 64) + "'");
        }

        UnigramPiece& target_piece =
            request.target.pieces[static_cast<std::size_t>(existing_index)];
        target_piece.score = request.piece.score;
        if (!target_piece.is_user_defined) {
            target_piece.is_user_defined = request.piece.is_user_defined;
        }
        return UnigramVocabWriteResult{
            existing_index,
            tokenIdForUnigramVocabWriteIndex(existing_index, caller),
            false};
    }

    requireUnigramVocabWriteSizeFitsInt(request.target.pieces.size(), caller);
    const int append_index = static_cast<int>(request.target.pieces.size());
    if (expected_index != append_index) {
        throw std::runtime_error(std::string(caller) +
                                 ": expected append token_id=" +
                                 std::to_string(request.expected_token_id) +
                                 " maps to index=" + std::to_string(expected_index) +
                                 " but next append index=" +
                                 std::to_string(append_index));
    }

    request.target.pieces.push_back(std::move(request.piece));
    bool inserted = false;
    try {
        inserted = request.target.piece_to_id.emplace(
            request.target.pieces.back().text,
            append_index).second;
    } catch (...) {
        request.target.pieces.pop_back();
        throw;
    }

    if (!inserted) {
        const std::string duplicate = request.target.pieces.back().text;
        request.target.pieces.pop_back();
        throw std::runtime_error(std::string(caller) +
                                 ": duplicate learned unigram piece appeared during append: '" +
                                 duplicate.substr(0, 64) + "'");
    }

    return UnigramVocabWriteResult{
        append_index,
        tokenIdForUnigramVocabWriteIndex(append_index, caller),
        true};
}

inline void clearUnigramVocab(
    UnigramVocabWriteTarget target,
    const char* caller) {
    requireUnigramVocabWriteCaller(caller);
    target.piece_to_id.clear();
    target.pieces.clear();
}

inline void rewriteUnigramVocab(
    UnigramVocabWriteTarget target,
    std::vector<UnigramPiece> new_pieces,
    const char* caller) {
    requireUnigramVocabWriteCaller(caller);
    requireUnigramVocabWriteSizeFitsInt(new_pieces.size(), caller);

    std::vector<UnigramPiece> rewritten_pieces;
    std::unordered_map<std::string, int> rewritten_piece_to_id;
    rewritten_pieces.reserve(new_pieces.size());
    rewritten_piece_to_id.reserve(new_pieces.size());
    UnigramVocabWriteTarget rewritten_target{
        rewritten_pieces,
        rewritten_piece_to_id};

    for (UnigramPiece& piece : new_pieces) {
        const int next_index = static_cast<int>(rewritten_pieces.size());
        applyUnigramVocabWriteOp(UnigramVocabWriteRequest{
            rewritten_target,
            std::move(piece),
            tokenIdForUnigramVocabWriteIndex(next_index, caller),
            UnigramVocabWriteMode::AppendOnly,
            caller});
    }

    target.pieces = std::move(rewritten_pieces);
    target.piece_to_id = std::move(rewritten_piece_to_id);
}

} // namespace Tokenizer
} // namespace GRIM
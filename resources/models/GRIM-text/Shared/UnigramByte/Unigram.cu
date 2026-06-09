//======================================================//
//  Unigram.cu
//  CUDA implementation of Unigram Language Model tokenizer
//
//  Inference wrappers, vocab I/O, and trie building.
//  Training pipeline is in UnigramTrainer.cu.
//  Text utilities are in TextUtils.cu.
//======================================================//

#include "Unigram.hpp"
#include "TextUtils.hpp"
#include "UnigramViterbi.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

namespace GRIM {

namespace Tokenizer {

//======================================================//
//  UnigramLM Implementation
//======================================================//

//--------------------------------------------------//
// Vocabulary Management
//--------------------------------------------------//

const UnigramPiece* UnigramLM::getPiece(int token_id) const {
    // Layout special tokens are not in pieces_ — use TokenLayout helpers for those IDs.
    int idx = token_id - UNIGRAM_VOCAB_OFFSET;
    if (idx < 0 || idx >= static_cast<int>(pieces_.size())) {
        return nullptr;
    }
    return &pieces_[idx];
}

int UnigramLM::getPieceId(const std::string& text) const {
    auto it = piece_to_id_.find(text);
    if (it == piece_to_id_.end()) {
        return UNK_TOKEN_ID;
    }
    return UNIGRAM_VOCAB_OFFSET + it->second;
}

bool UnigramLM::hasPiece(const std::string& text) const {
    return piece_to_id_.count(text) > 0;
}

//--------------------------------------------------//
// Trie Building
//--------------------------------------------------//

void UnigramLM::buildTrie() {
    trie_.clear();
    trie_.push_back(TrieNode());  // Root node
    
    for (size_t i = 0; i < pieces_.size(); ++i) {
        const auto& piece = pieces_[i];
        if (piece.text.empty()) {
            throw std::runtime_error("UnigramLM::buildTrie: piece index=" + std::to_string(i) +
                                     " has empty text; an empty piece would assign a token to the trie root");
        }
        if (piece.text.size() > static_cast<size_t>(MAX_PIECE_LENGTH)) {
            throw std::runtime_error("UnigramLM::buildTrie: piece index=" + std::to_string(i) +
                                     " length=" + std::to_string(piece.text.size()) +
                                     " exceeds MAX_PIECE_LENGTH=" + std::to_string(MAX_PIECE_LENGTH) +
                                     "; the Viterbi kernel caps walks at MAX_PIECE_LENGTH so this piece could never be matched (silent dead vocab)");
        }
        int node = 0;
        
        for (unsigned char c : piece.text) {
            if (trie_[node].children[c] < 0) {
                trie_[node].children[c] = static_cast<int>(trie_.size());
                trie_.push_back(TrieNode());
            }
            node = trie_[node].children[c];
        }
        
        if (trie_[node].token_id >= 0) {
            throw std::runtime_error("UnigramLM::buildTrie: duplicate piece text '" + piece.text +
                                     "' at index=" + std::to_string(i) +
                                     "; would silently overwrite token_id=" + std::to_string(trie_[node].token_id) +
                                     " and desynchronize Viterbi output from piece_to_id_");
        }
        // Token ID is ALWAYS position-derived. No stored field.
        trie_[node].token_id = tokenIdForIndex(static_cast<int>(i));
        trie_[node].score = piece.score;
    }

    ++trie_generation_;
}

std::vector<int> UnigramLM::encode(const std::string& text, bool prepend_space) const {
    if (text.empty()) return {};

    // SentencePiece-style normalization: spaces → ▁
    // prepend_space=true adds leading ▁ (start of text / first segment)
    // prepend_space=false skips prepend (mid-text segment after atom)
    std::string normalized = normalizeSpaces(text, prepend_space);
    UnigramViterbiSession session(*this, normalized, "UnigramLM::encode");
    return session.takeTokens();
}

} // namespace Tokenizer
} // namespace GRIM

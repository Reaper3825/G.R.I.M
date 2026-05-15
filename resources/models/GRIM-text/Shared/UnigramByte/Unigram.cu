//======================================================//
//  Unigram.cu
//  CUDA implementation of Unigram Language Model tokenizer
//
//  Inference wrappers, vocab I/O, trie building, and GPU decode kernel.
//  Training pipeline is in UnigramTrainer.cu.
//  Text utilities are in TextUtils.cu.
//======================================================//

#include "Unigram.hpp"
#include "TextUtils.hpp"
#include "UnigramViterbi.hpp"
#include "VocabWriteOp.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <utility>

namespace GRIM {

// Public static method wrappers — delegate to TextUtils
std::string Tokenizer::UnigramLM::normalizeForTokenization(const std::string& text) {
    return Tokenizer::normalizeSpaces(text);
}
std::string Tokenizer::UnigramLM::denormalizeFromTokenization(const std::string& text) {
    return Tokenizer::denormalizeSpaces(text);
}

namespace Tokenizer {

//======================================================//
//  CUDA Kernels
//======================================================//

// Kernel: Decode tokens to text
__global__ void kernelUnigramDecode(
    const int* __restrict__ token_ids,
    size_t count,
    const char* __restrict__ piece_data,
    const int* __restrict__ piece_offsets,
    const int* __restrict__ piece_lengths,
    int vocab_offset,
    int vocab_size,
    char* __restrict__ output,
    size_t* __restrict__ output_length,
    size_t max_output
) {
    // Single thread for now (could parallelize with scan)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    size_t out_pos = 0;
    
    for (size_t i = 0; i < count && out_pos < max_output; ++i) {
        int tid = token_ids[i];
        
        // Check if it's a unigram token
        if (tid >= vocab_offset && tid < vocab_offset + vocab_size) {
            int piece_idx = tid - vocab_offset;
            int offset = piece_offsets[piece_idx];
            int len = piece_lengths[piece_idx];
            
            for (int j = 0; j < len && out_pos < max_output; ++j) {
                output[out_pos++] = piece_data[offset + j];
            }
        }
        // Byte tokens handled by caller
    }
    
    *output_length = out_pos;
}

//======================================================//
//  UnigramLM Implementation
//======================================================//

//--------------------------------------------------//
// Vocabulary Management
//--------------------------------------------------//

void UnigramLM::writePiece(const std::string& text, float score, bool is_user_defined) {
    UnigramPiece piece;
    piece.text = text;
    piece.score = score;
    // token_id is NOT stored — it's ALWAYS (UNIGRAM_VOCAB_OFFSET + index).
    piece.is_user_defined = is_user_defined;

    const auto existing = piece_to_id_.find(text);
    int expected_token_id = tokenIdForIndex(static_cast<int>(pieces_.size()));
    if (existing != piece_to_id_.end()) {
        expected_token_id = tokenIdForIndex(existing->second);
    }

    applyUnigramVocabWriteOp(UnigramVocabWriteRequest{
        UnigramVocabWriteTarget{pieces_, piece_to_id_},
        std::move(piece),
        expected_token_id,
        UnigramVocabWriteMode::AppendOrUpdate,
        "UnigramLM::writePiece"});
}

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
        int node = 0;
        
        for (unsigned char c : piece.text) {
            if (trie_[node].children[c] < 0) {
                trie_[node].children[c] = static_cast<int>(trie_.size());
                trie_.push_back(TrieNode());
            }
            node = trie_[node].children[c];
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

std::string UnigramLM::decode(const std::vector<int>& token_ids) const {
    std::string result;
    
    for (int tid : token_ids) {
        if (tid >= BYTE_TOKEN_OFFSET && tid < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) {
            // Byte token (subtract BYTE_TOKEN_OFFSET to get raw byte)
            result.push_back(static_cast<char>(tid - BYTE_TOKEN_OFFSET));
        } else if (tid >= UNIGRAM_VOCAB_OFFSET) {
            // Unigram token
            const UnigramPiece* p = getPiece(tid);
            if (!p) {
                throw std::runtime_error("UnigramLM::decode: unigram token_id=" + std::to_string(tid) +
                                         " has no backing UnigramPiece");
            }
            result += p->text;
        } else {
            throw std::runtime_error("UnigramLM::decode: token_id=" + std::to_string(tid) +
                                     " is outside byte/unigram primitive ranges; use UniByte::decode for layout-aware decoding");
        }
    }
    
    return denormalizeSpaces(result);
}

} // namespace Tokenizer
} // namespace GRIM

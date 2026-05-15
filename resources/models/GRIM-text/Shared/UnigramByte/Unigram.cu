//======================================================//
//  Unigram.cu
//  CUDA implementation of Unigram Language Model tokenizer
//
//  Inference, vocab I/O, trie building, Viterbi, GPU encoding.
//  Training pipeline is in UnigramTrainer.cu.
//  Text utilities are in TextUtils.cu.
//======================================================//

#include "Unigram.hpp"
#include "TextUtils.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

namespace GRIM {

// Punctuation isolation guard for Viterbi.
// Returns true if the character at the given position is a punctuation character
// that should be tokenized in isolation (never merged with adjacent letters/digits).
// Used in both CPU Viterbi and GPU kernel to enforce punctuation boundary splitting.
// Note: space (0x20) is NOT punctuation — it's a normal character that can appear
// in multi-word vocab tokens like "of the".
// MUST be __host__ __device__ for CUDA kernel use.
__host__ __device__ static inline bool isPunctBoundary(unsigned char c) {
    return (c >= '!' && c <= '/') ||  // !"#$%&'()*+,-./
           (c >= ':' && c <= '@') ||  // :;<=>?@
           (c >= '[' && c <= '`') ||  // [\]^_`
           (c >= '{' && c <= '~');    // {|}~
}

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

__global__ void kernelViterbiForward(
    const char* __restrict__ text,
    size_t length,
    const int* __restrict__ trie_children,    // [num_nodes * 256]
    const int* __restrict__ trie_token_ids,   // [num_nodes]
    const float* __restrict__ trie_scores,    // [num_nodes]
    int num_trie_nodes,
    float* __restrict__ viterbi_scores,       // [length + 1]
    int* __restrict__ viterbi_prev,           // [length + 1]
    int* __restrict__ viterbi_tokens,         // [length + 1]
    bool* __restrict__ needs_fallback,        // [length]
    int unk_id,
    float unk_score,
    bool enable_byte_fallback
) {
    // Single thread processes positions SEQUENTIALLY to maintain Viterbi invariants
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Initialize position 0
    viterbi_scores[0] = 0.0f;
    viterbi_prev[0] = -1;
    viterbi_tokens[0] = -1;
    
    // Process each position sequentially (required for correctness)
    for (size_t pos = 1; pos <= length; ++pos) {
        unsigned char cur_byte = static_cast<unsigned char>(text[pos - 1]);
        bool cur_is_punct = isPunctBoundary(cur_byte);
        
        // PUNCTUATION ISOLATION GUARD:
        // If this position's character is punctuation, force it to be a single
        // byte token. Skip trie matching entirely — punctuation is never merged
        // with adjacent letters/digits.
        if (cur_is_punct) {
            // Emit as byte token: BYTE_TOKEN_OFFSET + byte_value
            viterbi_scores[pos] = viterbi_scores[pos - 1] + unk_score;
            viterbi_prev[pos] = static_cast<int>(pos - 1);
            viterbi_tokens[pos] = static_cast<int>(cur_byte) + 4;  // BYTE_TOKEN_OFFSET = 4
            continue;
        }
        
        float best_score = -1e30f;
        int best_prev = -1;
        int best_token = unk_id;
        bool found_match = false;
        
        // Try all possible pieces ending at position `pos`
        // Walk backwards from pos, traversing trie
        int node = 0;  // Start at trie root
        
        for (size_t start = pos; start > 0 && (pos - start) < MAX_PIECE_LENGTH; --start) {
            size_t idx = start - 1;
            unsigned char c = static_cast<unsigned char>(text[idx]);
            
            // PUNCTUATION ISOLATION GUARD:
            // Stop backward walk if we hit a punctuation character —
            // pieces must not span across a punctuation boundary.
            if (isPunctBoundary(c)) break;
            
            // Navigate trie
            int child = trie_children[node * 256 + c];
            if (child < 0) break;  // No path in trie
            
            node = child;
            
            // Check if this node is end of a token
            int token_id = trie_token_ids[node];
            if (token_id >= 0) {
                float piece_score = trie_scores[node];
                // Safe: viterbi_scores[start - 1] was computed in previous iteration
                float candidate_score = viterbi_scores[start - 1] + piece_score;
                
                if (candidate_score > best_score) {
                    best_score = candidate_score;
                    best_prev = static_cast<int>(start - 1);
                    best_token = token_id;
                    found_match = true;
                }
            }
        }
        
        // If no match found, mark for byte fallback
        if (!found_match) {
            best_prev = static_cast<int>(pos - 1);
            best_token = unk_id;
            best_score = viterbi_scores[pos - 1] + unk_score;

            if (enable_byte_fallback && needs_fallback) {
                needs_fallback[pos - 1] = true;
            }
        }
        
        viterbi_scores[pos] = best_score;
        viterbi_prev[pos] = best_prev;
        viterbi_tokens[pos] = best_token;
    }
}

// Kernel: Backtrack Viterbi path
__global__ void kernelViterbiBacktrack(
    size_t length,
    const int* __restrict__ viterbi_prev,
    const int* __restrict__ viterbi_tokens,
    int* __restrict__ output_tokens,
    int* __restrict__ output_count,
    int max_tokens
) {
    // Single thread does backtracking
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Count tokens first
    int count = 0;
    int pos = static_cast<int>(length);
    while (pos > 0) {
        count++;
        pos = viterbi_prev[pos];
    }
    
    // Clamp to max_tokens to prevent buffer overflow
    if (count > max_tokens) {
        count = max_tokens;
    }
    
    // Write tokens in reverse
    *output_count = count;
    pos = static_cast<int>(length);
    int write_idx = count - 1;
    while (pos > 0 && write_idx >= 0) {
        output_tokens[write_idx] = viterbi_tokens[pos];
        pos = viterbi_prev[pos];
        write_idx--;
    }
}

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

// Kernel: Trie lookup for batch encoding
__global__ void kernelTrieLookup(
    const char* __restrict__ text,
    size_t length,
    size_t start_pos,
    const int* __restrict__ trie_children,
    const int* __restrict__ trie_token_ids,
    const float* __restrict__ trie_scores,
    int* __restrict__ match_token,
    int* __restrict__ match_length,
    float* __restrict__ match_score
) {
    // Single thread traverses trie from start_pos
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    int node = 0;
    int best_token = -1;
    int best_length = 0;
    float best_score = -1e30f;
    
    for (size_t i = start_pos; i < length && (i - start_pos) < MAX_PIECE_LENGTH; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int child = trie_children[node * 256 + c];
        
        if (child < 0) break;
        node = child;
        
        int token_id = trie_token_ids[node];
        if (token_id >= 0) {
            float score = trie_scores[node];
            if (score > best_score) {
                best_token = token_id;
                best_length = static_cast<int>(i - start_pos + 1);
                best_score = score;
            }
        }
    }
    
    *match_token = best_token;
    *match_length = best_length;
    *match_score = best_score;
}

//======================================================//
//  UnigramLM Implementation
//======================================================//

//--------------------------------------------------//
// Vocabulary Management
//--------------------------------------------------//

void UnigramLM::addPiece(const std::string& text, float score, bool is_user_defined) {
    if (piece_to_id_.count(text)) {
        // Update existing piece's score. Token ID is immutable (= UNIGRAM_VOCAB_OFFSET + index).
        int idx = piece_to_id_[text];
        pieces_[idx].score = score;
        if (!pieces_[idx].is_user_defined) {
            pieces_[idx].is_user_defined = is_user_defined;
        }
        return;
    }
    
    UnigramPiece piece;
    piece.text = text;
    piece.score = score;
    // token_id is NOT stored — it's ALWAYS (UNIGRAM_VOCAB_OFFSET + index).
    piece.is_user_defined = is_user_defined;
    
    piece_to_id_[text] = static_cast<int>(pieces_.size());
    pieces_.push_back(piece);
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
}

//--------------------------------------------------//
// CPU Viterbi Encoding
//--------------------------------------------------//

std::vector<ViterbiNode> UnigramLM::viterbi(const std::string& text) const {
    size_t n = text.size();
    std::vector<ViterbiNode> nodes(n + 1);
    
    // Initialize
    nodes[0].score = 0.0f;
    nodes[0].prev_pos = -1;
    nodes[0].token_id = -1;
    nodes[0].piece_length = 0;
    
    for (size_t i = 1; i <= n; ++i) {
        nodes[i].score = -1e30f;
        nodes[i].prev_pos = -1;
        nodes[i].token_id = UNK_TOKEN_ID;
        nodes[i].piece_length = 1;
    }
    
    // Forward pass
    for (size_t pos = 0; pos < n; ++pos) {
        if (nodes[pos].score < -1e20f) continue;  // Unreachable
        
        unsigned char cur_byte = static_cast<unsigned char>(text[pos]);
        bool cur_is_punct = isPunctBoundary(cur_byte);
        
        // PUNCTUATION ISOLATION GUARD:
        // If current position is a punctuation character, force it to be emitted
        // as a single byte token and skip trie matching entirely.
        // This prevents tokens like "however," or "al." from ever being selected.
        if (cur_is_punct) {
            float byte_score = nodes[pos].score + UNKNOWN_SCORE;
            if (byte_score > nodes[pos + 1].score) {
                nodes[pos + 1].score = byte_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = static_cast<int>(cur_byte) + BYTE_TOKEN_OFFSET;
                nodes[pos + 1].piece_length = 1;
            }
            continue;  // Skip trie search — punctuation is always isolated
        }
        
        // Try all pieces starting at pos
        {
            if (trie_.empty()) {
                throw std::runtime_error("viterbi(): trie_ is empty — buildTrie() was never called. "
                                         "Caller MUST build trie before encoding at " + 
                                         std::string(__FILE__) + ":" + std::to_string(__LINE__));
            }
            int node = 0;
            for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= n; ++len) {
                unsigned char c = static_cast<unsigned char>(text[pos + len - 1]);
                
                // PUNCTUATION ISOLATION GUARD:
                // Stop extending the piece if we hit a punctuation character.
                // This prevents the trie from matching tokens that contain
                // punctuation mixed with letters (e.g., "al.", "et al.,").
                if (isPunctBoundary(c)) break;
                
                if (trie_[node].children[c] < 0) break;
                node = trie_[node].children[c];
                
                if (trie_[node].token_id >= 0) {
                    float score = nodes[pos].score + trie_[node].score;
                    
                    if (score > nodes[pos + len].score) {
                        nodes[pos + len].score = score;
                        nodes[pos + len].prev_pos = static_cast<int>(pos);
                        nodes[pos + len].token_id = trie_[node].token_id;
                        nodes[pos + len].piece_length = static_cast<int>(len);
                    }
                }
            }
        }
        
        if (enable_byte_fallback_) {
            // Byte fallback: allow single byte advance
            float byte_score = nodes[pos].score + UNKNOWN_SCORE;
            if (byte_score > nodes[pos + 1].score) {
                unsigned char byte_val = static_cast<unsigned char>(text[pos]);
                nodes[pos + 1].score = byte_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = static_cast<int>(byte_val) + BYTE_TOKEN_OFFSET;  // Byte token ID (offset by specials)
                nodes[pos + 1].piece_length = 1;
            }
        } else {
            // Byte fallback disabled: advance with <unk> instead of byte tokens.
            std::cout << "[UnigramLM] Warning: byte fallback disabled, using <unk> token for unknown bytes" << std::endl;
            float unk_score = nodes[pos].score + UNKNOWN_SCORE;
            if (unk_score > nodes[pos + 1].score) {
                nodes[pos + 1].score = unk_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = UNK_TOKEN_ID;
                nodes[pos + 1].piece_length = 1;
            }
        }
    }
    
    return nodes;
}

std::vector<int> UnigramLM::backtrack(const std::vector<ViterbiNode>& nodes, int end_pos) const {
    std::vector<int> tokens;
    int pos = end_pos;
    
    while (pos > 0) {
        tokens.push_back(nodes[pos].token_id);
        pos = nodes[pos].prev_pos;
    }
    
    std::reverse(tokens.begin(), tokens.end());
    return tokens;
}

std::vector<int> UnigramLM::encode(const std::string& text, bool prepend_space) const {
    if (text.empty()) return {};

    // SentencePiece-style normalization: spaces → ▁
    // prepend_space=true adds leading ▁ (start of text / first segment)
    // prepend_space=false skips prepend (mid-text segment after atom)
    std::string normalized = normalizeSpaces(text, prepend_space);
    auto nodes = viterbi(normalized);
    return backtrack(nodes, static_cast<int>(normalized.size()));
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

//======================================================//
//  UnigramViterbi.cu
//  RAII Viterbi segmentation for UnigramLM
//
//  Owns per-run dynamic-programming state. Durable device
//  buffers stay in UnigramGpuMemory; learned vocab/trie state
//  stays in UnigramLM.
//======================================================//

#include "UnigramViterbi.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {

namespace {

static void requireCallerLabel(const char* caller) {
    if (caller == nullptr) {
        throw std::runtime_error("UnigramViterbiSession requires a non-null caller label at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (caller[0] == '\0') {
        throw std::runtime_error("UnigramViterbiSession requires a non-empty caller label at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
}

__device__ static bool initializeCudaErrorCode(int* error_code) {
    if (error_code == nullptr) {
        asm("trap;");
        return false;
    }
    *error_code = kUnigramViterbiCudaOk;
    return true;
}

__device__ static void setCudaErrorCode(int* error_code, int code) {
    *error_code = code;
}

} // namespace

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
    bool* __restrict__ selected_fallback,     // [length], cleared here; selected path is marked by backtrack
    int unk_id,
    bool enable_byte_fallback,
    int* __restrict__ error_code
) {
    // Single thread processes positions SEQUENTIALLY to maintain Viterbi invariants.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!initializeCudaErrorCode(error_code)) return;
    if (text == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullText); return; }
    if (trie_children == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieChildren); return; }
    if (trie_token_ids == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieTokenIds); return; }
    if (trie_scores == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieScores); return; }
    if (viterbi_scores == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiScores); return; }
    if (viterbi_prev == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiPrev); return; }
    if (viterbi_tokens == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiTokens); return; }
    if (num_trie_nodes <= 0) { setCudaErrorCode(error_code, kUnigramViterbiCudaEmptyTrie); return; }
    
    // Initialize all DP states before forward relaxation.
    for (size_t i = 0; i <= length; ++i) {
        viterbi_scores[i] = -1e30f;
        viterbi_prev[i] = -1;
        viterbi_tokens[i] = unk_id;
    }
    viterbi_scores[0] = 0.0f;
    viterbi_tokens[0] = -1;

    if (selected_fallback != nullptr) {
        for (size_t i = 0; i < length; ++i) {
            selected_fallback[i] = false;
        }
    }
    
    // Mirror the CPU Viterbi pass: from each reachable start position, walk
    // the forward trie over text[pos], text[pos + 1], ... and relax end states.
    for (size_t pos = 0; pos < length; ++pos) {
        if (viterbi_scores[pos] < -1e20f) continue;
        
        unsigned char cur_byte = static_cast<unsigned char>(text[pos]);

        int node = 0;
        for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= length; ++len) {
            unsigned char c = static_cast<unsigned char>(text[pos + len - 1]);
            
            int child = trie_children[node * 256 + c];
            if (child < 0) break;
            if (child >= num_trie_nodes) { setCudaErrorCode(error_code, kUnigramViterbiCudaTrieChildOutOfRange); return; }
            node = child;
            
            int token_id = trie_token_ids[node];
            if (token_id >= 0) {
                float score = viterbi_scores[pos] + trie_scores[node];
                
                if (score > viterbi_scores[pos + len]) {
                    viterbi_scores[pos + len] = score;
                    viterbi_prev[pos + len] = static_cast<int>(pos);
                    viterbi_tokens[pos + len] = token_id;
                }
            }
        }
        
        float fallback_score = viterbi_scores[pos] + UNKNOWN_SCORE;
        if (fallback_score > viterbi_scores[pos + 1]) {
            viterbi_scores[pos + 1] = fallback_score;
            viterbi_prev[pos + 1] = static_cast<int>(pos);

            if (enable_byte_fallback) {
                viterbi_tokens[pos + 1] = static_cast<int>(cur_byte) + BYTE_TOKEN_OFFSET;
            } else {
                viterbi_tokens[pos + 1] = unk_id;
            }
        }
    }
}

__global__ void kernelViterbiBacktrack(
    size_t length,
    const int* __restrict__ viterbi_prev,
    const int* __restrict__ viterbi_tokens,
    int* __restrict__ output_tokens,
    int* __restrict__ output_count,
    int max_tokens,
    bool* __restrict__ selected_fallback,
    int* __restrict__ error_code
) {
    // Single thread does backtracking.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!initializeCudaErrorCode(error_code)) return;
    if (viterbi_prev == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiPrev); return; }
    if (viterbi_tokens == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullViterbiTokens); return; }
    if (output_tokens == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullOutputTokens); return; }
    if (output_count == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullOutputCount); return; }
    if (max_tokens <= 0) { setCudaErrorCode(error_code, kUnigramViterbiCudaInvalidMaxTokens); return; }
    if (length > static_cast<size_t>(INT_MAX)) { setCudaErrorCode(error_code, kUnigramViterbiCudaBacktrackLengthTooLarge); return; }

    *output_count = 0;
    
    if (selected_fallback != nullptr) {
        for (size_t i = 0; i < length; ++i) {
            selected_fallback[i] = false;
        }
    }
    
    int count = 0;
    int pos = static_cast<int>(length);
    int safety_counter = 0;
    while (pos > 0) {
        ++safety_counter;
        if (safety_counter > static_cast<int>(length)) { setCudaErrorCode(error_code, kUnigramViterbiCudaBacktrackSafetyLimit); return; }
        int prev_pos = viterbi_prev[pos];
        if (prev_pos < 0 || prev_pos >= pos) { setCudaErrorCode(error_code, kUnigramViterbiCudaInvalidBackpointer); return; }
        count++;
        if (count > max_tokens) { setCudaErrorCode(error_code, kUnigramViterbiCudaOutputBufferTooSmall); return; }
        pos = prev_pos;
    }
    
    *output_count = count;
    pos = static_cast<int>(length);
    int write_idx = count - 1;
    safety_counter = 0;
    while (pos > 0) {
        ++safety_counter;
        if (safety_counter > static_cast<int>(length)) { setCudaErrorCode(error_code, kUnigramViterbiCudaBacktrackSafetyLimit); return; }
        if (write_idx < 0) { setCudaErrorCode(error_code, kUnigramViterbiCudaOutputBufferTooSmall); return; }
        int prev_pos = viterbi_prev[pos];
        if (prev_pos < 0 || prev_pos >= pos) { setCudaErrorCode(error_code, kUnigramViterbiCudaInvalidBackpointer); return; }
        int token_id = viterbi_tokens[pos];

        if (selected_fallback != nullptr &&
            token_id >= BYTE_TOKEN_OFFSET && token_id < BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE) {
            if (prev_pos != pos - 1) { setCudaErrorCode(error_code, kUnigramViterbiCudaByteFallbackSpanInvalid); return; }
            selected_fallback[prev_pos] = true;
        }

        output_tokens[write_idx] = token_id;
        pos = prev_pos;
        write_idx--;
    }
}

__global__ void kernelTrieLookup(
    const char* __restrict__ text,
    size_t length,
    size_t start_pos,
    const int* __restrict__ trie_children,
    const int* __restrict__ trie_token_ids,
    const float* __restrict__ trie_scores,
    int num_trie_nodes,
    int* __restrict__ match_token,
    int* __restrict__ match_length,
    float* __restrict__ match_score,
    int* __restrict__ error_code
) {
    // Single thread traverses trie from start_pos.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (!initializeCudaErrorCode(error_code)) return;
    if (text == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullText); return; }
    if (trie_children == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieChildren); return; }
    if (trie_token_ids == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieTokenIds); return; }
    if (trie_scores == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullTrieScores); return; }
    if (match_token == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullMatchToken); return; }
    if (match_length == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullMatchLength); return; }
    if (match_score == nullptr) { setCudaErrorCode(error_code, kUnigramViterbiCudaNullMatchScore); return; }
    if (num_trie_nodes <= 0) { setCudaErrorCode(error_code, kUnigramViterbiCudaEmptyTrie); return; }
    
    int node = 0;
    int best_token = -1;
    int best_length = 0;
    float best_score = -1e30f;
    
    for (size_t i = start_pos; i < length && (i - start_pos) < MAX_PIECE_LENGTH; ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        int child = trie_children[node * 256 + c];
        
        if (child < 0) break;
        if (child >= num_trie_nodes) { setCudaErrorCode(error_code, kUnigramViterbiCudaTrieChildOutOfRange); return; }
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
//  UnigramViterbiSession Implementation
//======================================================//

UnigramViterbiSession::UnigramViterbiSession(const UnigramLM& model,
                                             const std::string& normalized_text,
                                             const char* caller) {
    requireCallerLabel(caller);
    nodes_ = runForward(model, normalized_text, caller);
    if (nodes_.empty()) {
        throw std::runtime_error(std::string(caller) +
                                 ": Viterbi forward returned no nodes at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    path_score_ = nodes_.back().score;
    tokens_ = runBacktrack(nodes_, static_cast<int>(normalized_text.size()), caller);
}

std::vector<UnigramViterbiNode> UnigramViterbiSession::runForward(
    const UnigramLM& model,
    const std::string& normalized_text,
    const char* caller) {
    requireCallerLabel(caller);
    if (model.trie_.empty()) {
        throw std::runtime_error(std::string(caller) +
                                 ": trie_ is empty; caller MUST call UnigramLM::buildTrie() before Viterbi segmentation at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    const size_t n = normalized_text.size();
    std::vector<UnigramViterbiNode> nodes(n + 1);
    
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
    
    for (size_t pos = 0; pos < n; ++pos) {
        if (nodes[pos].score < -1e20f) continue;
        
        unsigned char cur_byte = static_cast<unsigned char>(normalized_text[pos]);

        int node = 0;
        for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= n; ++len) {
            unsigned char c = static_cast<unsigned char>(normalized_text[pos + len - 1]);
            
            if (model.trie_[node].children[c] < 0) break;
            node = model.trie_[node].children[c];
            
            if (model.trie_[node].token_id >= 0) {
                float score = nodes[pos].score + model.trie_[node].score;
                
                if (score > nodes[pos + len].score) {
                    nodes[pos + len].score = score;
                    nodes[pos + len].prev_pos = static_cast<int>(pos);
                    nodes[pos + len].token_id = model.trie_[node].token_id;
                    nodes[pos + len].piece_length = static_cast<int>(len);
                }
            }
        }
        
        float fallback_score = nodes[pos].score + UNKNOWN_SCORE;
        if (fallback_score > nodes[pos + 1].score) {
            nodes[pos + 1].score = fallback_score;
            nodes[pos + 1].prev_pos = static_cast<int>(pos);

            if (model.enable_byte_fallback_) {
                unsigned char byte_val = static_cast<unsigned char>(normalized_text[pos]);
                nodes[pos + 1].token_id = static_cast<int>(byte_val) + BYTE_TOKEN_OFFSET;
                nodes[pos + 1].piece_length = 1;
            } else {
                nodes[pos + 1].token_id = UNK_TOKEN_ID;
                nodes[pos + 1].piece_length = 1;
            }
        }
    }
    
    return nodes;
}

std::vector<int> UnigramViterbiSession::runBacktrack(
    const std::vector<UnigramViterbiNode>& nodes,
    int end_pos,
    const char* caller) {
    requireCallerLabel(caller);
    if (nodes.empty()) {
        throw std::runtime_error(std::string(caller) +
                                 ": cannot backtrack an empty Viterbi node buffer at " +
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (end_pos < 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": Viterbi end_pos is negative: " + std::to_string(end_pos) +
                                 " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }
    if (end_pos >= static_cast<int>(nodes.size())) {
        throw std::runtime_error(std::string(caller) +
                                 ": Viterbi end_pos exceeds node buffer: end_pos=" +
                                 std::to_string(end_pos) + ", nodes=" + std::to_string(nodes.size()) +
                                 " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    std::vector<int> tokens;
    tokens.reserve(static_cast<size_t>(end_pos));
    int pos = end_pos;
    int safety_counter = 0;
    
    while (pos > 0) {
        const UnigramViterbiNode& node = nodes[pos];
        if (node.prev_pos < 0 || node.prev_pos >= pos) {
            throw std::runtime_error(std::string(caller) +
                                     ": invalid Viterbi backpointer at pos=" + std::to_string(pos) +
                                     ", prev_pos=" + std::to_string(node.prev_pos) +
                                     " at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
        }
        if (node.token_id < 0) {
            throw std::runtime_error(std::string(caller) +
                                     ": invalid negative token_id during Viterbi backtrack at pos=" +
                                     std::to_string(pos) + " at " + std::string(__FILE__) + ":" +
                                     std::to_string(__LINE__));
        }

        tokens.push_back(node.token_id);
        pos = node.prev_pos;
        ++safety_counter;
        if (safety_counter > end_pos) {
            throw std::runtime_error(std::string(caller) +
                                     ": Viterbi backtrack exceeded end_pos steps; graph is cyclic or corrupt at " +
                                     std::string(__FILE__) + ":" + std::to_string(__LINE__));
        }
    }
    
    std::reverse(tokens.begin(), tokens.end());
    return tokens;
}

} // namespace Tokenizer
} // namespace GRIM
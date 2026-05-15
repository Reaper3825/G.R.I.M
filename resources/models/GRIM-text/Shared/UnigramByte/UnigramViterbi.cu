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
#include <cassert>
#include <stdexcept>
#include <string>

namespace GRIM {
namespace Tokenizer {

namespace {

// Punctuation isolation guard for Viterbi.
// Returns true if the character is punctuation that must be emitted as a
// standalone byte token rather than merged into adjacent letter/digit pieces.
__host__ __device__ static inline bool isPunctBoundary(unsigned char c) {
    return (c >= '!' && c <= '/') ||  // !"#$%&'()*+,-./
           (c >= ':' && c <= '@') ||  // :;<=>?@
           (c >= '[' && c <= '`') ||  // [\]^_`
           (c >= '{' && c <= '~');    // {|}~
}

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
    bool* __restrict__ needs_fallback,        // [length]
    int unk_id,
    float unk_score,
    bool enable_byte_fallback
) {
    // Single thread processes positions SEQUENTIALLY to maintain Viterbi invariants.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    assert(text != nullptr && "kernelViterbiForward text is NULL");
    assert(trie_children != nullptr && "kernelViterbiForward trie_children is NULL");
    assert(trie_token_ids != nullptr && "kernelViterbiForward trie_token_ids is NULL");
    assert(trie_scores != nullptr && "kernelViterbiForward trie_scores is NULL");
    assert(viterbi_scores != nullptr && "kernelViterbiForward viterbi_scores is NULL");
    assert(viterbi_prev != nullptr && "kernelViterbiForward viterbi_prev is NULL");
    assert(viterbi_tokens != nullptr && "kernelViterbiForward viterbi_tokens is NULL");
    assert(num_trie_nodes > 0 && "kernelViterbiForward requires non-empty trie");
    
    if (enable_byte_fallback) {
        assert(needs_fallback != nullptr && "kernelViterbiForward needs_fallback is NULL while byte fallback is enabled");
    }

    // Initialize all DP states before forward relaxation.
    for (size_t i = 0; i <= length; ++i) {
        viterbi_scores[i] = -1e30f;
        viterbi_prev[i] = -1;
        viterbi_tokens[i] = unk_id;
    }
    viterbi_scores[0] = 0.0f;
    viterbi_tokens[0] = -1;

    if (needs_fallback != nullptr) {
        for (size_t i = 0; i < length; ++i) {
            needs_fallback[i] = false;
        }
    }
    
    // Mirror the CPU Viterbi pass: from each reachable start position, walk
    // the forward trie over text[pos], text[pos + 1], ... and relax end states.
    for (size_t pos = 0; pos < length; ++pos) {
        if (viterbi_scores[pos] < -1e20f) continue;
        
        unsigned char cur_byte = static_cast<unsigned char>(text[pos]);
        if (isPunctBoundary(cur_byte)) {
            float byte_score = viterbi_scores[pos] + unk_score;
            if (byte_score > viterbi_scores[pos + 1]) {
                viterbi_scores[pos + 1] = byte_score;
                viterbi_prev[pos + 1] = static_cast<int>(pos);
                viterbi_tokens[pos + 1] = static_cast<int>(cur_byte) + BYTE_TOKEN_OFFSET;
            }
            continue;
        }
        
        int node = 0;
        for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= length; ++len) {
            unsigned char c = static_cast<unsigned char>(text[pos + len - 1]);
            if (isPunctBoundary(c)) break;
            
            int child = trie_children[node * 256 + c];
            if (child < 0) break;
            assert(child < num_trie_nodes && "kernelViterbiForward trie child index exceeds num_trie_nodes");
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
        
        float fallback_score = viterbi_scores[pos] + unk_score;
        if (fallback_score > viterbi_scores[pos + 1]) {
            viterbi_scores[pos + 1] = fallback_score;
            viterbi_prev[pos + 1] = static_cast<int>(pos);

            if (enable_byte_fallback) {
                viterbi_tokens[pos + 1] = static_cast<int>(cur_byte) + BYTE_TOKEN_OFFSET;
                needs_fallback[pos] = true;
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
    int max_tokens
) {
    // Single thread does backtracking.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    assert(viterbi_prev != nullptr && "kernelViterbiBacktrack viterbi_prev is NULL");
    assert(viterbi_tokens != nullptr && "kernelViterbiBacktrack viterbi_tokens is NULL");
    assert(output_tokens != nullptr && "kernelViterbiBacktrack output_tokens is NULL");
    assert(output_count != nullptr && "kernelViterbiBacktrack output_count is NULL");
    assert(max_tokens > 0 && "kernelViterbiBacktrack max_tokens must be positive");
    
    int count = 0;
    int pos = static_cast<int>(length);
    while (pos > 0) {
        count++;
        pos = viterbi_prev[pos];
    }
    
    if (count > max_tokens) {
        count = max_tokens;
    }
    
    *output_count = count;
    pos = static_cast<int>(length);
    int write_idx = count - 1;
    while (pos > 0 && write_idx >= 0) {
        output_tokens[write_idx] = viterbi_tokens[pos];
        pos = viterbi_prev[pos];
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
    int* __restrict__ match_token,
    int* __restrict__ match_length,
    float* __restrict__ match_score
) {
    // Single thread traverses trie from start_pos.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    assert(text != nullptr && "kernelTrieLookup text is NULL");
    assert(trie_children != nullptr && "kernelTrieLookup trie_children is NULL");
    assert(trie_token_ids != nullptr && "kernelTrieLookup trie_token_ids is NULL");
    assert(trie_scores != nullptr && "kernelTrieLookup trie_scores is NULL");
    assert(match_token != nullptr && "kernelTrieLookup match_token is NULL");
    assert(match_length != nullptr && "kernelTrieLookup match_length is NULL");
    assert(match_score != nullptr && "kernelTrieLookup match_score is NULL");
    
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
        if (isPunctBoundary(cur_byte)) {
            float byte_score = nodes[pos].score + UNKNOWN_SCORE;
            if (byte_score > nodes[pos + 1].score) {
                nodes[pos + 1].score = byte_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = static_cast<int>(cur_byte) + BYTE_TOKEN_OFFSET;
                nodes[pos + 1].piece_length = 1;
            }
            continue;
        }
        
        int node = 0;
        for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= n; ++len) {
            unsigned char c = static_cast<unsigned char>(normalized_text[pos + len - 1]);
            if (isPunctBoundary(c)) break;
            
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
        
        if (model.enable_byte_fallback_) {
            float byte_score = nodes[pos].score + UNKNOWN_SCORE;
            if (byte_score > nodes[pos + 1].score) {
                unsigned char byte_val = static_cast<unsigned char>(normalized_text[pos]);
                nodes[pos + 1].score = byte_score;
                nodes[pos + 1].prev_pos = static_cast<int>(pos);
                nodes[pos + 1].token_id = static_cast<int>(byte_val) + BYTE_TOKEN_OFFSET;
                nodes[pos + 1].piece_length = 1;
            }
        } else {
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
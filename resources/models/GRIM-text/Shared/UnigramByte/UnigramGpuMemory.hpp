//======================================================//
//  UnigramGpuMemory.hpp
//  CUDA memory owner for UnigramLM
//
//  Owns durable device buffers only. It must not perform
//  tokenization, Viterbi scoring, vocab I/O, or training.
//======================================================//

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mutex>

namespace GRIM {
namespace Tokenizer {

//======================================================//
//  UnigramGpuMemory
//======================================================//
class UnigramGpuMemory final {
public:
    UnigramGpuMemory() = default;
    ~UnigramGpuMemory() noexcept;

    UnigramGpuMemory(const UnigramGpuMemory&) = delete;
    UnigramGpuMemory& operator=(const UnigramGpuMemory&) = delete;

    UnigramGpuMemory(UnigramGpuMemory&& other) noexcept;
    UnigramGpuMemory& operator=(UnigramGpuMemory&& other) noexcept;

    void release() noexcept;

    // Trie on device
    int* d_trie_children = nullptr;       // [num_nodes * 256]
    int* d_trie_token_ids = nullptr;      // [num_nodes]
    float* d_trie_scores = nullptr;       // [num_nodes]
    int num_nodes = 0;

    // Piece data for decoding
    char* d_piece_data = nullptr;         // Concatenated piece strings
    int* d_piece_offsets = nullptr;       // Start offset of each piece
    int* d_piece_lengths = nullptr;       // Length of each piece

    // Viterbi workspace (pre-allocated, fixed capacity)
    char* d_viterbi_text = nullptr;
    float* d_viterbi_scores = nullptr;
    int* d_viterbi_prev = nullptr;
    int* d_viterbi_tokens = nullptr;
    int* d_viterbi_output_tokens = nullptr;
    int* d_viterbi_output_count = nullptr;
    bool* d_viterbi_selected_fallback = nullptr;
    int* d_viterbi_error_code = nullptr;
    size_t workspace_max_length = 0;      // Maximum sequence length supported
    std::uint64_t uploaded_trie_generation = 0;
    mutable std::mutex viterbi_workspace_mutex;

    bool initialized = false;
};

} // namespace Tokenizer
} // namespace GRIM

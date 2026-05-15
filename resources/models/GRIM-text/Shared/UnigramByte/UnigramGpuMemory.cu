//======================================================//
//  UnigramGpuMemory.cu
//  CUDA memory owner for UnigramLM
//
//  File boundary: this compilation unit owns durable GPU
//  buffer lifetime and upload transactions. Unigram.cu owns
//  vocab I/O, trie semantics, and CPU tokenization logic.
//======================================================//

#include "UnigramGpuMemory.hpp"
#include "Unigram.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace GRIM {
namespace Tokenizer {

namespace {

template <typename T>
void releaseDevicePointer(T*& ptr, const char* label) noexcept {
    if (ptr == nullptr) {
        return;
    }

    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "[UnigramGpuMemory] cudaFree failed for " << label
                  << ": " << cudaGetErrorString(err) << std::endl;
        std::terminate();
    }
    ptr = nullptr;
}

template <typename T>
bool allocateDevice(T*& ptr, size_t count, const char* label) {
    if (count == 0) {
        throw std::runtime_error(std::string("UnigramGpuMemory::allocateDevice count is zero for ") + label);
    }

    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T));
    if (err != cudaSuccess) {
        std::cerr << "[UnigramGpuMemory] cudaMalloc failed for " << label
                  << " (count=" << count << ", bytes=" << (count * sizeof(T))
                  << "): " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

template <typename T>
bool copyToDevice(T* dst, const T* src, size_t count, const char* label) {
    if (count == 0) {
        return true;
    }
    if (dst == nullptr) {
        throw std::runtime_error(std::string("UnigramGpuMemory::copyToDevice destination is NULL for ") + label);
    }
    if (src == nullptr) {
        throw std::runtime_error(std::string("UnigramGpuMemory::copyToDevice source is NULL for ") + label);
    }

    const cudaError_t err = cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[UnigramGpuMemory] cudaMemcpy failed for " << label
                  << " (count=" << count << ", bytes=" << (count * sizeof(T))
                  << "): " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

} // namespace

//======================================================//
//  UnigramGpuMemory Implementation
//======================================================//

UnigramGpuMemory::~UnigramGpuMemory() noexcept {
    release();
}

UnigramGpuMemory::UnigramGpuMemory(UnigramGpuMemory&& other) noexcept {
    *this = std::move(other);
}

UnigramGpuMemory& UnigramGpuMemory::operator=(UnigramGpuMemory&& other) noexcept {
    if (this != &other) {
        std::lock_guard<std::mutex> lock(viterbi_workspace_mutex);
        release();

        d_trie_children = std::exchange(other.d_trie_children, nullptr);
        d_trie_token_ids = std::exchange(other.d_trie_token_ids, nullptr);
        d_trie_scores = std::exchange(other.d_trie_scores, nullptr);
        num_nodes = std::exchange(other.num_nodes, 0);

        d_piece_data = std::exchange(other.d_piece_data, nullptr);
        d_piece_offsets = std::exchange(other.d_piece_offsets, nullptr);
        d_piece_lengths = std::exchange(other.d_piece_lengths, nullptr);

        d_viterbi_text = std::exchange(other.d_viterbi_text, nullptr);
        d_viterbi_scores = std::exchange(other.d_viterbi_scores, nullptr);
        d_viterbi_prev = std::exchange(other.d_viterbi_prev, nullptr);
        d_viterbi_tokens = std::exchange(other.d_viterbi_tokens, nullptr);
        d_viterbi_prev_is_fallback = std::exchange(other.d_viterbi_prev_is_fallback, nullptr);
        d_viterbi_output_tokens = std::exchange(other.d_viterbi_output_tokens, nullptr);
        d_viterbi_output_is_fallback = std::exchange(other.d_viterbi_output_is_fallback, nullptr);
        d_viterbi_output_count = std::exchange(other.d_viterbi_output_count, nullptr);
        d_viterbi_selected_fallback = std::exchange(other.d_viterbi_selected_fallback, nullptr);
        d_viterbi_error_code = std::exchange(other.d_viterbi_error_code, nullptr);
        workspace_max_length = std::exchange(other.workspace_max_length, 0);
        uploaded_trie_generation = std::exchange(other.uploaded_trie_generation, 0);

        initialized = std::exchange(other.initialized, false);
    }
    return *this;
}

void UnigramGpuMemory::release() noexcept {
    releaseDevicePointer(d_trie_children, "d_trie_children");
    releaseDevicePointer(d_trie_token_ids, "d_trie_token_ids");
    releaseDevicePointer(d_trie_scores, "d_trie_scores");
    releaseDevicePointer(d_piece_data, "d_piece_data");
    releaseDevicePointer(d_piece_offsets, "d_piece_offsets");
    releaseDevicePointer(d_piece_lengths, "d_piece_lengths");
    releaseDevicePointer(d_viterbi_text, "d_viterbi_text");
    releaseDevicePointer(d_viterbi_scores, "d_viterbi_scores");
    releaseDevicePointer(d_viterbi_prev, "d_viterbi_prev");
    releaseDevicePointer(d_viterbi_tokens, "d_viterbi_tokens");
    releaseDevicePointer(d_viterbi_prev_is_fallback, "d_viterbi_prev_is_fallback");
    releaseDevicePointer(d_viterbi_output_tokens, "d_viterbi_output_tokens");
    releaseDevicePointer(d_viterbi_output_is_fallback, "d_viterbi_output_is_fallback");
    releaseDevicePointer(d_viterbi_output_count, "d_viterbi_output_count");
    releaseDevicePointer(d_viterbi_selected_fallback, "d_viterbi_selected_fallback");
    releaseDevicePointer(d_viterbi_error_code, "d_viterbi_error_code");

    num_nodes = 0;
    workspace_max_length = 0;
    uploaded_trie_generation = 0;
    initialized = false;
}

//======================================================//
//  UnigramLM GPU Ownership Boundary
//======================================================//

UnigramLM::UnigramLM()
    : gpu_(std::make_unique<UnigramGpuMemory>())
{
    // Start with an empty learned vocabulary. Layout special tokens are saved
    // as vocab metadata records, not stored in pieces_ or the trie.
}

UnigramLM::~UnigramLM() = default;

UnigramLM::UnigramLM(UnigramLM&& other) noexcept
    : pieces_(std::move(other.pieces_))
    , piece_to_id_(std::move(other.piece_to_id_))
    , enable_byte_fallback_(other.enable_byte_fallback_)
    , trie_(std::move(other.trie_))
    , gpu_(std::move(other.gpu_))
    , trie_generation_(other.trie_generation_)
{
    other.trie_generation_ = 0;
}

UnigramLM& UnigramLM::operator=(UnigramLM&& other) noexcept {
    if (this != &other) {
        pieces_ = std::move(other.pieces_);
        piece_to_id_ = std::move(other.piece_to_id_);
        enable_byte_fallback_ = other.enable_byte_fallback_;
        trie_ = std::move(other.trie_);
        gpu_ = std::move(other.gpu_);
        trie_generation_ = other.trie_generation_;
        other.trie_generation_ = 0;
    }
    return *this;
}

bool UnigramLM::initGPU() {
    if (!gpu_) {
        throw std::runtime_error("UnigramLM::initGPU: gpu_ is NULL - object was moved from or not constructed");
    }
    if (gpu_->initialized && gpu_->uploaded_trie_generation == trie_generation_) {
        return true;
    }

    if (trie_.empty()) {
        buildTrie();
    }

    return uploadTrieToGPU();
}

bool UnigramLM::uploadTrieToGPU() {
    if (!gpu_) {
        throw std::runtime_error("UnigramLM::uploadTrieToGPU: gpu_ is NULL - object was moved from or not constructed");
    }
    if (pieces_.empty()) {
        throw std::runtime_error("UnigramLM::uploadTrieToGPU requires a non-empty learned vocabulary before GPU upload");
    }
    if (trie_.empty()) {
        throw std::runtime_error("UnigramLM::uploadTrieToGPU requires buildTrie() before GPU upload");
    }

    const size_t num_nodes = trie_.size();
    const size_t piece_count = pieces_.size();
    UnigramGpuMemory fresh;

    if (!allocateDevice(fresh.d_trie_children, num_nodes * 256, "d_trie_children")) return false;
    if (!allocateDevice(fresh.d_trie_token_ids, num_nodes, "d_trie_token_ids")) return false;
    if (!allocateDevice(fresh.d_trie_scores, num_nodes, "d_trie_scores")) return false;

    // Flatten and upload trie data.
    std::vector<int> children_flat(num_nodes * 256);
    std::vector<int> token_ids_flat(num_nodes);
    std::vector<float> scores_flat(num_nodes);

    for (size_t i = 0; i < num_nodes; ++i) {
        for (int c = 0; c < 256; ++c) {
            children_flat[i * 256 + c] = trie_[i].children[c];
        }
        token_ids_flat[i] = trie_[i].token_id;
        scores_flat[i] = trie_[i].score;
    }

    if (!copyToDevice(fresh.d_trie_children, children_flat.data(), children_flat.size(), "d_trie_children")) return false;
    if (!copyToDevice(fresh.d_trie_token_ids, token_ids_flat.data(), token_ids_flat.size(), "d_trie_token_ids")) return false;
    if (!copyToDevice(fresh.d_trie_scores, scores_flat.data(), scores_flat.size(), "d_trie_scores")) return false;

    fresh.num_nodes = static_cast<int>(num_nodes);

    // Upload piece data for decoding.
    size_t total_piece_length = 0;
    for (const auto& p : pieces_) {
        total_piece_length += p.text.size();
    }
    if (total_piece_length == 0) {
        throw std::runtime_error("UnigramLM::uploadTrieToGPU: all learned pieces are empty; vocab is invalid");
    }

    std::vector<char> piece_data(total_piece_length);
    std::vector<int> piece_offsets(piece_count);
    std::vector<int> piece_lengths(piece_count);

    size_t offset = 0;
    for (size_t i = 0; i < piece_count; ++i) {
        piece_offsets[i] = static_cast<int>(offset);
        piece_lengths[i] = static_cast<int>(pieces_[i].text.size());
        std::copy(pieces_[i].text.begin(), pieces_[i].text.end(), piece_data.begin() + offset);
        offset += pieces_[i].text.size();
    }

    if (!allocateDevice(fresh.d_piece_data, total_piece_length, "d_piece_data")) return false;
    if (!allocateDevice(fresh.d_piece_offsets, piece_count, "d_piece_offsets")) return false;
    if (!allocateDevice(fresh.d_piece_lengths, piece_count, "d_piece_lengths")) return false;

    constexpr size_t max_sequence_length = HyperParameters::UNIGRAM_MAX_SEQUENCE_LENGTH;
    fresh.workspace_max_length = max_sequence_length;

    if (!allocateDevice(fresh.d_viterbi_text, max_sequence_length, "d_viterbi_text")) return false;
    if (!allocateDevice(fresh.d_viterbi_scores, max_sequence_length + 1, "d_viterbi_scores")) return false;
    if (!allocateDevice(fresh.d_viterbi_prev, max_sequence_length + 1, "d_viterbi_prev")) return false;
    if (!allocateDevice(fresh.d_viterbi_tokens, max_sequence_length + 1, "d_viterbi_tokens")) return false;
    if (!allocateDevice(fresh.d_viterbi_prev_is_fallback, max_sequence_length + 1, "d_viterbi_prev_is_fallback")) return false;
    if (!allocateDevice(fresh.d_viterbi_output_tokens, max_sequence_length, "d_viterbi_output_tokens")) return false;
    if (!allocateDevice(fresh.d_viterbi_output_is_fallback, max_sequence_length, "d_viterbi_output_is_fallback")) return false;
    if (!allocateDevice(fresh.d_viterbi_output_count, 1, "d_viterbi_output_count")) return false;
    if (!allocateDevice(fresh.d_viterbi_selected_fallback, max_sequence_length, "d_viterbi_selected_fallback")) return false;
    if (!allocateDevice(fresh.d_viterbi_error_code, 1, "d_viterbi_error_code")) return false;

    if (!copyToDevice(fresh.d_piece_data, piece_data.data(), piece_data.size(), "d_piece_data")) return false;
    if (!copyToDevice(fresh.d_piece_offsets, piece_offsets.data(), piece_offsets.size(), "d_piece_offsets")) return false;
    if (!copyToDevice(fresh.d_piece_lengths, piece_lengths.data(), piece_lengths.size(), "d_piece_lengths")) return false;

    fresh.initialized = true;
    fresh.uploaded_trie_generation = trie_generation_;
    *gpu_ = std::move(fresh);

    std::cout << "[UnigramLM] GPU initialized with " << num_nodes << " trie nodes" << std::endl;
    return true;
}

} // namespace Tokenizer
} // namespace GRIM

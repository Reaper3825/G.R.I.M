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

struct UnigramGpuUploadTransaction final {
    UnigramGpuUploadTransaction() = default;
    ~UnigramGpuUploadTransaction() noexcept { release(); }

    UnigramGpuUploadTransaction(const UnigramGpuUploadTransaction&) = delete;
    UnigramGpuUploadTransaction& operator=(const UnigramGpuUploadTransaction&) = delete;
    UnigramGpuUploadTransaction(UnigramGpuUploadTransaction&&) = delete;
    UnigramGpuUploadTransaction& operator=(UnigramGpuUploadTransaction&&) = delete;

    void release() noexcept {
        releaseDevicePointer(d_trie_children, "upload.d_trie_children");
        releaseDevicePointer(d_trie_token_ids, "upload.d_trie_token_ids");
        releaseDevicePointer(d_trie_scores, "upload.d_trie_scores");
        releaseDevicePointer(d_piece_data, "upload.d_piece_data");
        releaseDevicePointer(d_piece_offsets, "upload.d_piece_offsets");
        releaseDevicePointer(d_piece_lengths, "upload.d_piece_lengths");
        releaseDevicePointer(d_viterbi_text, "upload.d_viterbi_text");
        releaseDevicePointer(d_viterbi_scores, "upload.d_viterbi_scores");
        releaseDevicePointer(d_viterbi_prev, "upload.d_viterbi_prev");
        releaseDevicePointer(d_viterbi_tokens, "upload.d_viterbi_tokens");
        releaseDevicePointer(d_viterbi_prev_is_fallback, "upload.d_viterbi_prev_is_fallback");
        releaseDevicePointer(d_viterbi_output_tokens, "upload.d_viterbi_output_tokens");
        releaseDevicePointer(d_viterbi_output_is_fallback, "upload.d_viterbi_output_is_fallback");
        releaseDevicePointer(d_viterbi_output_count, "upload.d_viterbi_output_count");
        releaseDevicePointer(d_viterbi_selected_fallback, "upload.d_viterbi_selected_fallback");
        releaseDevicePointer(d_viterbi_error_code, "upload.d_viterbi_error_code");

        num_nodes = 0;
        workspace_max_length = 0;
        uploaded_trie_generation = 0;
        initialized = false;
    }

    void commitTo(UnigramGpuMemory& target) noexcept {
        target.release();

        target.d_trie_children = d_trie_children;
        target.d_trie_token_ids = d_trie_token_ids;
        target.d_trie_scores = d_trie_scores;
        target.num_nodes = num_nodes;

        target.d_piece_data = d_piece_data;
        target.d_piece_offsets = d_piece_offsets;
        target.d_piece_lengths = d_piece_lengths;

        target.d_viterbi_text = d_viterbi_text;
        target.d_viterbi_scores = d_viterbi_scores;
        target.d_viterbi_prev = d_viterbi_prev;
        target.d_viterbi_tokens = d_viterbi_tokens;
        target.d_viterbi_prev_is_fallback = d_viterbi_prev_is_fallback;
        target.d_viterbi_output_tokens = d_viterbi_output_tokens;
        target.d_viterbi_output_is_fallback = d_viterbi_output_is_fallback;
        target.d_viterbi_output_count = d_viterbi_output_count;
        target.d_viterbi_selected_fallback = d_viterbi_selected_fallback;
        target.d_viterbi_error_code = d_viterbi_error_code;
        target.workspace_max_length = workspace_max_length;
        target.uploaded_trie_generation = uploaded_trie_generation;
        target.initialized = initialized;

        d_trie_children = nullptr;
        d_trie_token_ids = nullptr;
        d_trie_scores = nullptr;
        d_piece_data = nullptr;
        d_piece_offsets = nullptr;
        d_piece_lengths = nullptr;
        d_viterbi_text = nullptr;
        d_viterbi_scores = nullptr;
        d_viterbi_prev = nullptr;
        d_viterbi_tokens = nullptr;
        d_viterbi_prev_is_fallback = nullptr;
        d_viterbi_output_tokens = nullptr;
        d_viterbi_output_is_fallback = nullptr;
        d_viterbi_output_count = nullptr;
        d_viterbi_selected_fallback = nullptr;
        d_viterbi_error_code = nullptr;
        num_nodes = 0;
        workspace_max_length = 0;
        uploaded_trie_generation = 0;
        initialized = false;
    }

    int* d_trie_children = nullptr;
    int* d_trie_token_ids = nullptr;
    float* d_trie_scores = nullptr;
    int num_nodes = 0;

    char* d_piece_data = nullptr;
    int* d_piece_offsets = nullptr;
    int* d_piece_lengths = nullptr;

    char* d_viterbi_text = nullptr;
    float* d_viterbi_scores = nullptr;
    int* d_viterbi_prev = nullptr;
    int* d_viterbi_tokens = nullptr;
    bool* d_viterbi_prev_is_fallback = nullptr;
    int* d_viterbi_output_tokens = nullptr;
    bool* d_viterbi_output_is_fallback = nullptr;
    int* d_viterbi_output_count = nullptr;
    bool* d_viterbi_selected_fallback = nullptr;
    int* d_viterbi_error_code = nullptr;
    size_t workspace_max_length = 0;
    std::uint64_t uploaded_trie_generation = 0;
    bool initialized = false;
};

size_t resolveWorkspaceSequenceLength(size_t required_max_sequence_length, const char* caller) {
    if (required_max_sequence_length == 0) {
        throw std::runtime_error(std::string(caller) + " requires required_max_sequence_length > 0");
    }

    size_t workspace_sequence_length = required_max_sequence_length;
    if (workspace_sequence_length < HyperParameters::UNIGRAM_MAX_SEQUENCE_LENGTH) {
        workspace_sequence_length = HyperParameters::UNIGRAM_MAX_SEQUENCE_LENGTH;
    }
    return workspace_sequence_length;
}

} // namespace

//======================================================//
//  UnigramGpuMemory Implementation
//======================================================//

UnigramGpuMemory::~UnigramGpuMemory() noexcept {
    release();
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

UnigramLM::UnigramLM(bool enable_byte_fallback)
    : enable_byte_fallback_(enable_byte_fallback)
    , gpu_(std::make_unique<UnigramGpuMemory>())
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
    , last_training_runtime_report_(other.last_training_runtime_report_)
{
    other.trie_generation_ = 0;
    other.last_training_runtime_report_ = UnigramTrainingRuntimeReport{};
}

UnigramLM& UnigramLM::operator=(UnigramLM&& other) noexcept {
    if (this != &other) {
        pieces_ = std::move(other.pieces_);
        piece_to_id_ = std::move(other.piece_to_id_);
        enable_byte_fallback_ = other.enable_byte_fallback_;
        trie_ = std::move(other.trie_);
        gpu_ = std::move(other.gpu_);
        trie_generation_ = other.trie_generation_;
        last_training_runtime_report_ = other.last_training_runtime_report_;
        other.trie_generation_ = 0;
        other.last_training_runtime_report_ = UnigramTrainingRuntimeReport{};
    }
    return *this;
}

bool UnigramLM::initGPU() {
    return initGPUForMaxSequenceLength(HyperParameters::UNIGRAM_MAX_SEQUENCE_LENGTH);
}

bool UnigramLM::initGPUForMaxSequenceLength(size_t required_max_sequence_length) {
    if (!gpu_) {
        throw std::runtime_error("UnigramLM::initGPUForMaxSequenceLength: gpu_ is NULL - object was moved from or not constructed");
    }

    const size_t workspace_sequence_length = resolveWorkspaceSequenceLength(
        required_max_sequence_length,
        "UnigramLM::initGPUForMaxSequenceLength");

    if (trie_.empty()) {
        buildTrie();
    }

    std::lock_guard<std::mutex> lock(gpu_->viterbi_workspace_mutex);

    if (gpu_->initialized &&
        gpu_->uploaded_trie_generation == trie_generation_ &&
        gpu_->workspace_max_length >= workspace_sequence_length) {
        return true;
    }

    return uploadTrieToGPU(workspace_sequence_length);
}

bool UnigramLM::runtimeReadyForMaxSequenceLength(size_t required_max_sequence_length) const {
    if (!gpu_) {
        throw std::runtime_error("UnigramLM::runtimeReadyForMaxSequenceLength: gpu_ is NULL - object was moved from or not constructed");
    }
    const size_t workspace_sequence_length = resolveWorkspaceSequenceLength(
        required_max_sequence_length,
        "UnigramLM::runtimeReadyForMaxSequenceLength");

    std::lock_guard<std::mutex> lock(gpu_->viterbi_workspace_mutex);
    return gpu_->initialized &&
           gpu_->uploaded_trie_generation == trie_generation_ &&
           gpu_->workspace_max_length >= workspace_sequence_length;
}

void UnigramLM::requireRuntimeReadyForMaxSequenceLength(size_t required_max_sequence_length,
                                                       const char* caller) const {
    if (caller == nullptr || caller[0] == '\0') {
        throw std::runtime_error("UnigramLM::requireRuntimeReadyForMaxSequenceLength requires a non-empty caller label");
    }
    if (!gpu_) {
        throw std::runtime_error(std::string(caller) + ": UnigramLM.gpu_ is NULL - object was moved from or not constructed");
    }
    const size_t workspace_sequence_length = resolveWorkspaceSequenceLength(
        required_max_sequence_length,
        "UnigramLM::requireRuntimeReadyForMaxSequenceLength");

    std::lock_guard<std::mutex> lock(gpu_->viterbi_workspace_mutex);
    if (!gpu_->initialized ||
        gpu_->uploaded_trie_generation != trie_generation_ ||
        gpu_->workspace_max_length < workspace_sequence_length) {
        throw std::runtime_error(std::string(caller) +
                                 ": tokenizer runtime state is not ready for encoding: initialized=" +
                                 (gpu_->initialized ? std::string("true") : std::string("false")) +
                                 ", uploaded_generation=" + std::to_string(gpu_->uploaded_trie_generation) +
                                 ", live_generation=" + std::to_string(trie_generation_) +
                                 ", workspace_max_length=" + std::to_string(gpu_->workspace_max_length) +
                                 ", required_workspace_length=" + std::to_string(workspace_sequence_length));
    }
}

void UnigramLM::requireRuntimeReadyForLastTraining(const char* caller) const {
    if (last_training_runtime_report_.required_viterbi_workspace_length == 0) {
        throw std::runtime_error(std::string(caller) +
                                 ": tokenizer has no training runtime report; trainFromCorpus() must finalize runtime state before corpus encoding");
    }
    if (last_training_runtime_report_.finalized_trie_generation != trie_generation_) {
        throw std::runtime_error(std::string(caller) +
                                 ": tokenizer training runtime report is stale: report_generation=" +
                                 std::to_string(last_training_runtime_report_.finalized_trie_generation) +
                                 ", live_generation=" + std::to_string(trie_generation_));
    }
    requireRuntimeReadyForMaxSequenceLength(
        last_training_runtime_report_.required_viterbi_workspace_length,
        caller);
}

UnigramRuntimeStateSnapshot UnigramLM::runtimeStateSnapshot() const {
    if (!gpu_) {
        throw std::runtime_error("UnigramLM::runtimeStateSnapshot: gpu_ is NULL - object was moved from or not constructed");
    }

    std::lock_guard<std::mutex> lock(gpu_->viterbi_workspace_mutex);
    UnigramRuntimeStateSnapshot snapshot;
    snapshot.initialized = gpu_->initialized;
    snapshot.workspace_max_length = gpu_->workspace_max_length;
    snapshot.uploaded_trie_generation = gpu_->uploaded_trie_generation;
    snapshot.live_trie_generation = trie_generation_;
    snapshot.ready_for_live_trie = gpu_->initialized &&
                                   gpu_->uploaded_trie_generation == trie_generation_;
    return snapshot;
}

const UnigramTrainingRuntimeReport& UnigramLM::lastTrainingRuntimeReport() const {
    return last_training_runtime_report_;
}

bool UnigramLM::uploadTrieToGPU(size_t workspace_sequence_length) {
    // Caller must hold gpu_->viterbi_workspace_mutex. This function performs
    // the transactional replacement of runtime buffers; it is not a public
    // tokenizer operation and must not race a Viterbi encode.
    if (!gpu_) {
        throw std::runtime_error("UnigramLM::uploadTrieToGPU: gpu_ is NULL - object was moved from or not constructed");
    }
    if (workspace_sequence_length == 0) {
        throw std::runtime_error("UnigramLM::uploadTrieToGPU requires workspace_sequence_length > 0");
    }
    if (pieces_.empty()) {
        throw std::runtime_error("UnigramLM::uploadTrieToGPU requires a non-empty learned vocabulary before GPU upload");
    }
    if (trie_.empty()) {
        throw std::runtime_error("UnigramLM::uploadTrieToGPU requires buildTrie() before GPU upload");
    }

    const size_t num_nodes = trie_.size();
    const size_t piece_count = pieces_.size();
    UnigramGpuUploadTransaction upload;

    if (!allocateDevice(upload.d_trie_children, num_nodes * 256, "d_trie_children")) return false;
    if (!allocateDevice(upload.d_trie_token_ids, num_nodes, "d_trie_token_ids")) return false;
    if (!allocateDevice(upload.d_trie_scores, num_nodes, "d_trie_scores")) return false;

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

    if (!copyToDevice(upload.d_trie_children, children_flat.data(), children_flat.size(), "d_trie_children")) return false;
    if (!copyToDevice(upload.d_trie_token_ids, token_ids_flat.data(), token_ids_flat.size(), "d_trie_token_ids")) return false;
    if (!copyToDevice(upload.d_trie_scores, scores_flat.data(), scores_flat.size(), "d_trie_scores")) return false;

    upload.num_nodes = static_cast<int>(num_nodes);

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

    if (!allocateDevice(upload.d_piece_data, total_piece_length, "d_piece_data")) return false;
    if (!allocateDevice(upload.d_piece_offsets, piece_count, "d_piece_offsets")) return false;
    if (!allocateDevice(upload.d_piece_lengths, piece_count, "d_piece_lengths")) return false;

    upload.workspace_max_length = workspace_sequence_length;

    if (!allocateDevice(upload.d_viterbi_text, workspace_sequence_length, "d_viterbi_text")) return false;
    if (!allocateDevice(upload.d_viterbi_scores, workspace_sequence_length + 1, "d_viterbi_scores")) return false;
    if (!allocateDevice(upload.d_viterbi_prev, workspace_sequence_length + 1, "d_viterbi_prev")) return false;
    if (!allocateDevice(upload.d_viterbi_tokens, workspace_sequence_length + 1, "d_viterbi_tokens")) return false;
    if (!allocateDevice(upload.d_viterbi_prev_is_fallback, workspace_sequence_length + 1, "d_viterbi_prev_is_fallback")) return false;
    if (!allocateDevice(upload.d_viterbi_output_tokens, workspace_sequence_length, "d_viterbi_output_tokens")) return false;
    if (!allocateDevice(upload.d_viterbi_output_is_fallback, workspace_sequence_length, "d_viterbi_output_is_fallback")) return false;
    if (!allocateDevice(upload.d_viterbi_output_count, 1, "d_viterbi_output_count")) return false;
    if (!allocateDevice(upload.d_viterbi_selected_fallback, workspace_sequence_length, "d_viterbi_selected_fallback")) return false;
    if (!allocateDevice(upload.d_viterbi_error_code, 1, "d_viterbi_error_code")) return false;

    if (!copyToDevice(upload.d_piece_data, piece_data.data(), piece_data.size(), "d_piece_data")) return false;
    if (!copyToDevice(upload.d_piece_offsets, piece_offsets.data(), piece_offsets.size(), "d_piece_offsets")) return false;
    if (!copyToDevice(upload.d_piece_lengths, piece_lengths.data(), piece_lengths.size(), "d_piece_lengths")) return false;

    upload.initialized = true;
    upload.uploaded_trie_generation = trie_generation_;
    upload.commitTo(*gpu_);

    std::cout << "[UnigramLM] GPU initialized with " << num_nodes
              << " trie nodes, Viterbi workspace_max_length="
              << workspace_sequence_length << std::endl;
    return true;
}

} // namespace Tokenizer
} // namespace GRIM

//======================================================//
//  LocalAtomRetrievalInferenceState.cu
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "LocalAtomRetrievalInferenceState.hpp"

#include "../UnigramByte/SequenceLocalAtomTable.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace GRIM {

namespace {

int typeIndex(Tokenizer::AtomType type, const char* caller) {
    return Tokenizer::atomTypeIndexOrThrow(type, caller);
}

void checkCuda(cudaError_t status, const char* caller) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(caller) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

void LocalAtomRetrievalInferenceState::ensureAllocated(
    int d_model_in,
    int max_cached_seq_len,
    cudaStream_t stream) {
    constexpr const char* caller =
        "LocalAtomRetrievalInferenceState::ensureAllocated";
    if (!stream) {
        throw std::runtime_error(std::string(caller) + ": stream is NULL");
    }
    if (d_model_in <= 0 || max_cached_seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": invalid geometry");
    }
    // Every retained candidate contains OPEN, at least one content token, and
    // CLOSE. This is a strict upper bound for any one typed bank.
    const int capacity = max_cached_seq_len / 3 + 1;
    if (allocated && d_model == d_model_in &&
        candidate_capacity_per_type == capacity) {
        return;
    }

    for (Tensor& tensor : candidate_embeddings) {
        tensor = Tensor();
    }
    d_model = d_model_in;
    candidate_capacity_per_type = capacity;
    for (int type = 0; type < Tokenizer::kAtomTypeCount; ++type) {
        candidate_embeddings[static_cast<std::size_t>(type)] = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(capacity, d_model),
            false,
            stream,
            "local_atom_inference_candidate_bank");
    }
    allocated = true;
    resetSession();
}

void LocalAtomRetrievalInferenceState::resetSession() {
    candidate_counts.fill(0);
    for (auto& tokens_by_candidate : candidate_content_token_ids) {
        tokens_by_candidate.clear();
    }
    clearActiveSpan();
}

void LocalAtomRetrievalInferenceState::seedFromPrefill(
    const Tensor& compact_candidate_embeddings,
    const Batching::BatchPayload& payload,
    cudaStream_t stream) {
    constexpr const char* caller =
        "LocalAtomRetrievalInferenceState::seedFromPrefill";
    if (!allocated || !stream) {
        throw std::runtime_error(
            std::string(caller) + ": state/stream is unavailable");
    }
    if (!payload.isInferencePrefill() || payload.batch_size != 1) {
        throw std::runtime_error(
            std::string(caller) + ": payload must be single-row inference prefill");
    }
    payload.validate(caller);
    candidate_counts.fill(0);
    for (auto& tokens_by_candidate : candidate_content_token_ids) {
        tokens_by_candidate.clear();
    }
    clearActiveSpan();

    const int compact_count = payload.localAtomCandidateCount();
    if (compact_count == 0) {
        return;
    }
    compact_candidate_embeddings.require(caller);
    if (!compact_candidate_embeddings.shape.is_2d_layout()) {
        throw std::runtime_error(
            std::string(caller) + ": compact candidate embeddings must be 2D");
    }
    const auto shape = compact_candidate_embeddings.shape.as_2d();
    if (shape.rows != compact_count || shape.cols != d_model) {
        throw std::runtime_error(
            std::string(caller) + ": compact candidate embedding shape mismatch");
    }
    if (payload.seq_local_atom_tables.size() != 1 ||
        !payload.seq_local_atom_tables[0]) {
        throw std::runtime_error(
            std::string(caller) + ": prompt local atom table is unavailable");
    }

    const auto& bank_offsets = payload.local_atom_row_type_candidate_offsets;
    const auto& content_offsets = payload.local_atom_candidate_content_offsets;
    const auto& content_positions = payload.local_atom_candidate_content_positions;
    for (int type_index = 0; type_index < Tokenizer::kAtomTypeCount;
         ++type_index) {
        const auto type = static_cast<Tokenizer::AtomType>(type_index);
        const int begin = bank_offsets[static_cast<std::size_t>(type_index)];
        const int end = bank_offsets[static_cast<std::size_t>(type_index + 1)];
        const int count = end - begin;
        if (count < 0 || count > candidate_capacity_per_type) {
            throw std::runtime_error(
                std::string(caller) + ": typed candidate count exceeds capacity");
        }
        if (payload.seq_local_atom_tables[0]->size(type) !=
            static_cast<std::size_t>(count)) {
            throw std::runtime_error(
                std::string(caller) +
                ": typed prompt table size disagrees with candidate bank");
        }
        if (count == 0) {
            continue;
        }

        const std::size_t row_elements =
            static_cast<std::size_t>(count) * d_model;
        checkCuda(
            cudaMemcpyAsync(
                candidate_embeddings[static_cast<std::size_t>(type_index)].data,
                compact_candidate_embeddings.data +
                    static_cast<std::size_t>(begin) * d_model,
                row_elements * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream),
            caller);
        candidate_counts[static_cast<std::size_t>(type_index)] = count;

        auto& typed_tokens =
            candidate_content_token_ids[static_cast<std::size_t>(type_index)];
        typed_tokens.reserve(static_cast<std::size_t>(count));
        for (int candidate = begin; candidate < end; ++candidate) {
            const int content_begin =
                content_offsets[static_cast<std::size_t>(candidate)];
            const int content_end =
                content_offsets[static_cast<std::size_t>(candidate + 1)];
            std::vector<int> tokens;
            tokens.reserve(static_cast<std::size_t>(content_end - content_begin));
            for (int content = content_begin; content < content_end; ++content) {
                const int position =
                    content_positions[static_cast<std::size_t>(content)];
                tokens.push_back(payload.input_ids[static_cast<std::size_t>(position)]);
            }
            typed_tokens.push_back(std::move(tokens));
        }
    }
}

int LocalAtomRetrievalInferenceState::candidateCount(
    Tokenizer::AtomType type) const {
    return candidate_counts[static_cast<std::size_t>(
        typeIndex(type, "LocalAtomRetrievalInferenceState::candidateCount"))];
}

const Tensor& LocalAtomRetrievalInferenceState::candidateEmbeddingStorage(
    Tokenizer::AtomType type) const {
    if (!allocated) {
        throw std::runtime_error(
            "LocalAtomRetrievalInferenceState::candidateEmbeddingStorage: state is not allocated");
    }
    return candidate_embeddings[static_cast<std::size_t>(typeIndex(
        type,
        "LocalAtomRetrievalInferenceState::candidateEmbeddingStorage"))];
}

const std::vector<int>&
LocalAtomRetrievalInferenceState::candidateContentTokenIds(
    Tokenizer::AtomType type,
    int local_index) const {
    const int type_index = typeIndex(
        type,
        "LocalAtomRetrievalInferenceState::candidateContentTokenIds");
    const auto& candidates =
        candidate_content_token_ids[static_cast<std::size_t>(type_index)];
    if (local_index < 0 ||
        static_cast<std::size_t>(local_index) >= candidates.size()) {
        throw std::runtime_error(
            "LocalAtomRetrievalInferenceState::candidateContentTokenIds: local index is out of range");
    }
    return candidates[static_cast<std::size_t>(local_index)];
}

void LocalAtomRetrievalInferenceState::appendCandidate(
    Tokenizer::AtomType type,
    int local_index,
    const std::vector<float>& mean_embedding,
    const std::vector<int>& content_token_ids,
    cudaStream_t stream) {
    constexpr const char* caller =
        "LocalAtomRetrievalInferenceState::appendCandidate";
    if (!allocated || !stream) {
        throw std::runtime_error(
            std::string(caller) + ": state/stream is unavailable");
    }
    const int type_index = typeIndex(type, caller);
    int& count = candidate_counts[static_cast<std::size_t>(type_index)];
    auto& token_bank =
        candidate_content_token_ids[static_cast<std::size_t>(type_index)];
    if (local_index != count || token_bank.size() !=
            static_cast<std::size_t>(count)) {
        throw std::runtime_error(
            std::string(caller) + ": local index is not the next dense candidate");
    }
    if (count >= candidate_capacity_per_type) {
        throw std::runtime_error(
            std::string(caller) + ": typed candidate capacity exhausted");
    }
    if (mean_embedding.size() != static_cast<std::size_t>(d_model) ||
        content_token_ids.empty()) {
        throw std::runtime_error(
            std::string(caller) + ": candidate signal/content is invalid");
    }
    checkCuda(
        cudaMemcpyAsync(
            candidate_embeddings[static_cast<std::size_t>(type_index)].data +
                static_cast<std::size_t>(count) * d_model,
            mean_embedding.data(),
            static_cast<std::size_t>(d_model) * sizeof(float),
            cudaMemcpyHostToDevice,
            stream),
        caller);
    checkCuda(cudaStreamSynchronize(stream), caller);
    token_bank.push_back(content_token_ids);
    ++count;
}

int LocalAtomRetrievalInferenceState::popForcedToken() {
    if (!hasForcedToken()) {
        throw std::runtime_error(
            "LocalAtomRetrievalInferenceState::popForcedToken: queue is empty");
    }
    return forced_token_ids[forced_token_cursor++];
}

void LocalAtomRetrievalInferenceState::clearActiveSpan() {
    inside_atom = false;
    replaying_reference = false;
    active_type = Tokenizer::AtomType::ATOM_INT;
    active_open_sequence_index = 0;
    active_content_token_ids.clear();
    active_embedding_sum.clear();
    active_content_count = 0;
    forced_token_ids.clear();
    forced_token_cursor = 0;
}

} // namespace GRIM

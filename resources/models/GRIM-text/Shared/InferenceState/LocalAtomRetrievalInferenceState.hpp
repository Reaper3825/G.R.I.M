//======================================================//
//  LocalAtomRetrievalInferenceState.hpp
//  Session owner for incremental local-atom retrieval.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace GRIM {

// Owns only state that must survive across KV-cache forwards. Scoring borrows
// these tensors and never allocates or retains its own candidate memory.
struct LocalAtomRetrievalInferenceState {
    bool allocated = false;
    int d_model = 0;
    int candidate_capacity_per_type = 0;

    // Fixed-capacity typed banks. Row n is local_index n for that AtomType.
    std::array<Tensor, Tokenizer::kAtomTypeCount> candidate_embeddings;
    std::array<int, Tokenizer::kAtomTypeCount> candidate_counts{};
    std::array<std::vector<std::vector<int>>, Tokenizer::kAtomTypeCount>
        candidate_content_token_ids;

    // Host-side incremental span state. Encoder vectors are accumulated while a
    // novel atom is emitted and committed to the typed device bank on CLOSE.
    bool inside_atom = false;
    bool replaying_reference = false;
    Tokenizer::AtomType active_type = Tokenizer::AtomType::ATOM_INT;
    std::size_t active_open_sequence_index = 0;
    std::vector<int> active_content_token_ids;
    std::vector<float> active_embedding_sum;
    int active_content_count = 0;

    // Exact token replay selected by retrieval. The vector includes content and
    // the matching typed CLOSE token and is consumed in order by orchestration.
    std::vector<int> forced_token_ids;
    std::size_t forced_token_cursor = 0;

    void ensureAllocated(
        int d_model_in,
        int max_cached_seq_len,
        cudaStream_t stream);
    void resetSession();

    void seedFromPrefill(
        const Tensor& compact_candidate_embeddings,
        const Batching::BatchPayload& payload,
        cudaStream_t stream);

    int candidateCount(Tokenizer::AtomType type) const;
    const Tensor& candidateEmbeddingStorage(Tokenizer::AtomType type) const;
    const std::vector<int>& candidateContentTokenIds(
        Tokenizer::AtomType type,
        int local_index) const;

    void appendCandidate(
        Tokenizer::AtomType type,
        int local_index,
        const std::vector<float>& mean_embedding,
        const std::vector<int>& content_token_ids,
        cudaStream_t stream);

    bool hasForcedToken() const {
        return forced_token_cursor < forced_token_ids.size();
    }
    int popForcedToken();
    void clearActiveSpan();
};

} // namespace GRIM

#endif // USE_CUDA


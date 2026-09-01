#pragma once
//======================================================//
//  LocalAtomRetrievalForward.hpp
//  Scoring over pre-encoded sequence-local atom banks.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM::LocalAtomRetrieval {

struct LocalAtomRetrievalParameterTensors {
    // [kAtomTypeCount, retrieval_dim]. Candidate and query encoders own all
    // other projections; retrieval owns only the learned NO_REFERENCE key.
    Tensor& type_no_reference_key;
};

struct LocalAtomRetrievalForwardOutputs {
    // [query_count, class_count]. Column 0 is NO_REFERENCE. Column
    // local_index + 1 addresses the corresponding entry in the query's
    // payload-authored [sequence row, AtomType] bank.
    Tensor logits;

    int query_count = 0;
    int candidate_count = 0;
    int class_count = 0;
    int retrieval_dim = 0;
};

// Pre-encoded inputs use the compact payload ordering:
//
//   query_embeddings     [payload.localAtomQueryCount(), retrieval_dim]
//   candidate_embeddings [payload.localAtomCandidateCount(), retrieval_dim]
//
// Candidate row c corresponds directly to
// bindings.d_local_atom_candidate_first_close_positions[c]. Query banks are
// resolved from the payload-authored query position/type and row/type offsets;
// no second padded layout is accepted or synthesized here.
//
// The caller owns query_embeddings and candidate_embeddings through backward.
// LocalAtomRetrievalGradFn borrows their forward data and owns no saved copies.
LocalAtomRetrievalForwardOutputs LocalAtomRetrievalForward(
    Tensor& query_embeddings,
    Tensor& candidate_embeddings,
    const LocalAtomRetrievalParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

} // namespace GRIM::LocalAtomRetrieval

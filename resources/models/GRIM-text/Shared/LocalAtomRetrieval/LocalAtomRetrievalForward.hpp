#pragma once
//======================================================//
//  LocalAtomRetrievalForward.hpp
//  Scoring over pre-encoded sequence-local atom banks.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM::LocalAtomRetrieval {

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
// The learned NO_REFERENCE keys are resolved through the root parameter
// registry; this compute API does not accept or own a weight bundle.
//
// Returns logits [payload.localAtomQueryCount(), class_count]. AtomType selects
// the payload-authored row/type bank; it is not the class axis. Column 0 is
// NO_REFERENCE and column local_index + 1 selects that local bank candidate.
Tensor LocalAtomRetrievalForward(
    Tensor& query_embeddings,
    Tensor& candidate_embeddings,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

} // namespace GRIM::LocalAtomRetrieval

#pragma once
//======================================================//
//  LocalAtomRetrievalForward.hpp
//  Detached proposal for the sequence-local atom retrieval head.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

namespace GRIM::LocalAtomRetrieval {

// Parameter layouts intentionally keep AtomType as the outermost dimension:
//
//   type_query_projection [kAtomTypeCount * d_model, retrieval_dim]
//   type_key_projection   [kAtomTypeCount * d_model, retrieval_dim]
//   type_no_reference_key [kAtomTypeCount, retrieval_dim]
//
// A row range [type * d_model, (type + 1) * d_model) is therefore an entirely
// separate projection for that atom type. The head never mixes candidate banks
// belonging to different rows or types.
struct LocalAtomRetrievalParameterTensors {
    Tensor& type_query_projection;
    Tensor& type_key_projection;
    Tensor& type_no_reference_key;
};

struct LocalAtomRetrievalForwardOutputs {
    // [query_count, class_count]
    //
    // Column 0 is NO_REFERENCE. Column local_index + 1 addresses the candidate
    // in the query's own [sequence row, AtomType] bank. Slots outside that bank
    // and candidates that are not yet causally available contain a mask value.
    Tensor logits;

    int query_count = 0;
    int candidate_count = 0;
    int class_count = 0;
    int retrieval_dim = 0;
};

// Proposed detached head entry point. It is intentionally not called from
// ModelForward_GPU, loss, inference, or batching. The caller decides which
// contextual-state tensor to expose. Passing contextual_states.detach() makes
// the head train only its own parameters; passing the graph-connected tensor
// also lets retrieval loss train the shared representation.
LocalAtomRetrievalForwardOutputs LocalAtomRetrievalForward(
    const Tensor& contextual_states,
    const LocalAtomRetrievalParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

} // namespace GRIM::LocalAtomRetrieval

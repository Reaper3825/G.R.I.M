#pragma once
//======================================================//
//  LocalAtomRetrievalForward.hpp
//  Scoring over pre-encoded sequence-local atom banks.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../Batching/BatchPayload.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "../Forward/ModelForwardOutputs.hpp"
#include "../UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM {
struct LocalAtomRetrievalInferenceState;
}

namespace GRIM::LocalAtomRetrieval {

// Derives and commits the complete retrieval signal path to ModelForwardOutputs:
//
//   query embeddings     = gather encoder rows at local_atom_query_positions
//   candidate embeddings = mean encoder rows over each candidate's ragged
//                          local_atom_candidate_content_positions segment
//   logits               = causal dot-product selector scores
//
// BatchPayload owns host identities/geometry, BatchDeviceBindings owns no data
// and supplies the uploaded metadata addresses, StartupParameterRegistry owns
// the learned NO_REFERENCE keys, and ModelForwardOutputs is the sole active-step
// owner of the three produced tensors.
void LocalAtomRetrievalForward(
    Tensor& encoder_output,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool connect_parameter_graph,
    cudaStream_t stream,
    Forward::ModelForwardOutputs& forward_outputs);

// Read-only KV-decode assembly. The current encoder row supplies the query and
// GenerationState's retrieval state supplies the already-completed typed bank.
// The same registry key and scaled-dot scoring primitive are used as above.
void LocalAtomRetrievalDecodeForward(
    Tensor& current_encoder_output,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const LocalAtomRetrievalInferenceState& inference_state,
    Tokenizer::AtomType query_type,
    cudaStream_t stream,
    Forward::ModelForwardOutputs& forward_outputs);

} // namespace GRIM::LocalAtomRetrieval

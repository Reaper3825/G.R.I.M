#pragma once
//======================================================//
//  LocalAtomRetrievalBackwards.hpp
//  Internal autograd-node factory for retrieval scoring.
//======================================================//

#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

#include <memory>

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM::LocalAtomRetrieval {

// Creates the scoring GradFn without exposing its borrowed device views in the
// public header. Query/candidate data remain owned by ModelForwardOutputs and
// the learned key remains owned by StartupParameterRegistry.
std::shared_ptr<GradFn> makeLocalAtomRetrievalGradFn(
    Tensor& query_embeddings,
    Tensor& candidate_embeddings,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    int query_count,
    int candidate_count,
    int type_count,
    int sequence_length,
    int retrieval_dim,
    int candidate_slot_count,
    cudaStream_t stream);

} // namespace GRIM::LocalAtomRetrieval

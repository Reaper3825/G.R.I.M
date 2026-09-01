#pragma once
//======================================================//
//  LocalAtomRetrievalBackwards.hpp
//  Autograd node for retrieval scoring.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

#include <memory>

namespace ParameterRegistry {
struct StartupParameterRegistry;
}

namespace GRIM::LocalAtomRetrieval {

struct LocalAtomRetrievalGradFn final : public GradFn {
    LocalAtomRetrievalGradFn();

    void captureInputs(
        Tensor& query_embeddings,
        Tensor& candidate_embeddings,
        ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
        int query_count,
        int candidate_count,
        int type_count,
        int sequence_length,
        int retrieval_dim,
        int class_count,
        cudaStream_t stream);

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override;

    void release_saved() override;

private:
    // Host-side borrowed handles only. Device addresses are resolved inside
    // the CUDA implementation and never form part of this public header API.
    const Tensor* query_embeddings_ = nullptr;
    const Tensor* candidate_embeddings_ = nullptr;
    const ::ParameterRegistry::StartupParameterRegistry* parameter_registry_ = nullptr;

    // TensorContract-owned gradient carriers. Leaf parameter gradients remain
    // owned by the root registry tensor; non-leaf activation gradients remain
    // owned by the active autograd graph.
    std::shared_ptr<Tensor> query_embeddings_gradient_;
    std::shared_ptr<Tensor> candidate_embeddings_gradient_;
    std::shared_ptr<Tensor> type_no_reference_key_gradient_;

    int query_count_ = 0;
    int candidate_count_ = 0;
    int type_count_ = 0;
    int sequence_length_ = 0;
    int retrieval_dim_ = 0;
    int class_count_ = 0;
};

} // namespace GRIM::LocalAtomRetrieval

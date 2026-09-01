#pragma once
//======================================================//
//  LocalAtomRetrievalBackwards.hpp
//  Explicit backward primitive and GradFn for retrieval scoring.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

#include <memory>

namespace GRIM::LocalAtomRetrieval {

struct LocalAtomRetrievalBackwardsArgs {
    const Tensor& grad_logits;

    // Borrowed forward data. The forward-output owner keeps these buffers live
    // through backward; this primitive allocates and saves no private copies.
    const float* query_embeddings = nullptr;
    const float* candidate_embeddings = nullptr;
    const float* type_no_reference_key = nullptr;

    float* grad_query_embeddings = nullptr;
    float* grad_candidate_embeddings = nullptr;
    float* grad_type_no_reference_key = nullptr;

    int query_count = 0;
    int candidate_count = 0;
    int type_count = 0;
    int sequence_length = 0;
    int retrieval_dim = 0;
    int class_count = 0;
};

// Explicit Jacobian of LocalAtomRetrievalForward. Batch metadata remains owned
// by the active payload upload and is borrowed through BatchDeviceBindings.
void LocalAtomRetrievalBackwards(
    const LocalAtomRetrievalBackwardsArgs& args,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

struct LocalAtomRetrievalGradFn final : public GradFn {
    std::shared_ptr<Tensor> query_embeddings_gradient;
    std::shared_ptr<Tensor> candidate_embeddings_gradient;
    std::shared_ptr<Tensor> type_no_reference_key_gradient;

    const float* query_embeddings_data = nullptr;
    const float* candidate_embeddings_data = nullptr;
    const float* type_no_reference_key_data = nullptr;

    int query_count = 0;
    int candidate_count = 0;
    int type_count = 0;
    int sequence_length = 0;
    int retrieval_dim = 0;
    int class_count = 0;

    LocalAtomRetrievalGradFn();

    void captureInputs(
        Tensor& query_embeddings,
        Tensor& candidate_embeddings,
        Tensor& type_no_reference_key,
        cudaStream_t stream);

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override;

    void release_saved() override;
};

} // namespace GRIM::LocalAtomRetrieval

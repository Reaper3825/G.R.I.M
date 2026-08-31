#pragma once
//======================================================//
//  LocalAtomRetrievalBackwards.hpp
//  Explicit backward primitive and GradFn for local atom retrieval.
//======================================================//

#include "../Batching/BatchDeviceBindings.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"

#include <cuda_runtime.h>

#include <memory>

namespace GRIM::LocalAtomRetrieval {

// Saved forward values and gradient destinations consumed by the explicit
// backward primitive. Metadata remains batch-owned and is supplied through the
// backward bindings, matching the existing batch-aware GradFn contract.
struct LocalAtomRetrievalBackwardsArgs {
    const Tensor& grad_logits;

    const float* contextual_states = nullptr;       // [total_tokens, d_model]
    const float* type_query_projection = nullptr;   // [type_count*d_model, H]
    const float* type_key_projection = nullptr;     // [type_count*d_model, H]
    const float* type_no_reference_key = nullptr;   // [type_count, H]
    const float* pooled_candidate_states = nullptr; // [candidate_count, d_model]
    const float* query_embeddings = nullptr;        // [query_count, H]
    const float* candidate_embeddings = nullptr;    // [candidate_count, H]

    float* grad_contextual_states = nullptr;
    float* grad_type_query_projection = nullptr;
    float* grad_type_key_projection = nullptr;
    float* grad_type_no_reference_key = nullptr;

    int total_tokens = 0;
    int batch_size = 0;
    int sequence_length = 0;
    int type_count = 0;
    int d_model = 0;
    int retrieval_dim = 0;
    int query_count = 0;
    int candidate_count = 0;
    int class_count = 0;
};

// Explicit local derivative. This function knows no optimizer or loss policy;
// it only applies the Jacobian of LocalAtomRetrievalForward and accumulates into
// the supplied TensorContract gradient destinations.
void LocalAtomRetrievalBackwards(
    const LocalAtomRetrievalBackwardsArgs& args,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream);

// Autograd adapter around the explicit backward primitive. The forward op owns
// this node; the node captures graph edges and stable copies of every value the
// local derivative needs after forward temporaries have gone out of scope.
struct LocalAtomRetrievalGradFn final : public GradFn {
    std::shared_ptr<Tensor> contextual_states_gradient;
    std::shared_ptr<Tensor> type_query_projection_gradient;
    std::shared_ptr<Tensor> type_key_projection_gradient;
    std::shared_ptr<Tensor> type_no_reference_key_gradient;

    std::shared_ptr<float> saved_contextual_states;
    std::shared_ptr<float> saved_type_query_projection;
    std::shared_ptr<float> saved_type_key_projection;
    std::shared_ptr<float> saved_type_no_reference_key;
    std::shared_ptr<float> saved_pooled_candidate_states;
    std::shared_ptr<float> saved_query_embeddings;
    std::shared_ptr<float> saved_candidate_embeddings;

    int total_tokens = 0;
    int batch_size = 0;
    int sequence_length = 0;
    int type_count = 0;
    int d_model = 0;
    int retrieval_dim = 0;
    int query_count = 0;
    int candidate_count = 0;
    int class_count = 0;

    LocalAtomRetrievalGradFn();

    void captureInputs(
        Tensor& contextual_states,
        Tensor& type_query_projection,
        Tensor& type_key_projection,
        Tensor& type_no_reference_key,
        const Tensor& pooled_candidate_states,
        const Tensor& query_embeddings,
        const Tensor& candidate_embeddings,
        cudaStream_t stream);

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override;

    void release_saved() override;
};

} // namespace GRIM::LocalAtomRetrieval

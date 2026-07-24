#pragma once
//======================================================//
//  SlotRoutingGradFn.hpp
//  Differentiable routing over BatchDeviceBindings slot metadata.
//======================================================//

#include "../TensorContract_GPU.hpp"

#include <cuda_runtime.h>
#include <memory>

namespace GRIM {
namespace autograd {

struct SlotSeedInputGradFn : public GradFn {
    bool token_states_requires_grad = false;
    bool type_embeddings_requires_grad = false;
    bool type_embedding_enabled = false;
    float* token_states_grad = nullptr;
    float* type_embeddings_grad = nullptr;
    std::shared_ptr<float> owned_token_states_grad;
    std::shared_ptr<float> owned_type_embeddings_grad;
    TensorContract::TensorShape token_states_shape;
    TensorContract::TensorShape type_embeddings_shape;
    std::shared_ptr<GradFn> token_states_grad_fn;
    std::shared_ptr<GradFn> type_embeddings_grad_fn;
    int batch_size = 0;
    int max_seq_len = 0;
    int total_tokens = 0;
    int authored_atom_count = 0;
    int num_slots = 0;
    int d_model = 0;

    SlotSeedInputGradFn();

    void capture_inputs(
        Tensor& token_states,
        Tensor& type_embeddings,
        bool use_type_embeddings,
        const Batching::BatchPayload& payload,
        int slot_count,
        cudaStream_t stream);

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override;

    void release_saved() override;
};

struct AuthoredSlotMaskGradFn : public GradFn {
    bool input_requires_grad = false;
    float* input_grad = nullptr;
    std::shared_ptr<float> owned_input_grad;
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;
    int batch_size = 0;
    int max_seq_len = 0;
    int total_tokens = 0;
    int authored_atom_count = 0;
    int num_slots = 0;
    int d_model = 0;

    AuthoredSlotMaskGradFn();

    void capture_input(
        Tensor& input,
        const Batching::BatchPayload& payload,
        int slot_count,
        cudaStream_t stream);

    void apply_impl(
        const Tensor& grad_output,
        cudaStream_t stream,
        const Batching::BatchPayload* backward_payload,
        const Batching::BatchDeviceBindings* backward_bindings) override;

    void release_saved() override;
};

}  // namespace autograd
}  // namespace GRIM

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
    std::shared_ptr<Tensor> token_states_gradient;
    std::shared_ptr<Tensor> type_embeddings_gradient;
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
    std::shared_ptr<Tensor> input_gradient;
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

//======================================================//
//  SlotRoutingGradFn.cu
//======================================================//

#include "SlotRoutingGradFn.hpp"

#include "../../CudaAllocUtils.hpp"
#include "../../UnigramByte/TokenLayout.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

void trackKernelLaunch(const char* kernel_name, cudaStream_t stream);

namespace {

constexpr int kSlotRoutingBlockSize = 256;

__global__ void kernel_gather_slot_seed_inputs(
    const float* __restrict__ token_states,
    const float* __restrict__ type_embeddings,
    const int32_t* __restrict__ token_to_slot_index_map,
    const int* __restrict__ atom_positions,
    const int* __restrict__ atom_types,
    float* __restrict__ output,
    int authored_atom_count,
    int total_tokens,
    int max_seq_len,
    int batch_size,
    int num_slots,
    int d_model,
    bool type_embedding_enabled)
{
    const int atom = blockIdx.x;
    if (atom >= authored_atom_count) return;

    const int token = atom_positions[atom];
    if (token < 0 || token >= total_tokens) return;
    const int slot = token_to_slot_index_map[token];
    if (slot < 0 || slot >= num_slots) return;
    const int batch_row = token / max_seq_len;
    if (batch_row < 0 || batch_row >= batch_size) return;

    const int slot_row = batch_row * num_slots + slot;
    const int atom_type = atom_types[atom];
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        float value = token_states[static_cast<std::size_t>(token) * d_model + d];
        if (type_embedding_enabled &&
            (atom_type == static_cast<int>(GRIM::Tokenizer::AtomType::ATOM_INT) ||
             atom_type == static_cast<int>(GRIM::Tokenizer::AtomType::ATOM_FLOAT))) {
            value += type_embeddings[
                static_cast<std::size_t>(atom_type) * d_model + d];
        }
        output[static_cast<std::size_t>(slot_row) * d_model + d] = value;
    }
}

__global__ void kernel_gather_slot_seed_inputs_backward(
    const float* __restrict__ grad_output,
    const int32_t* __restrict__ token_to_slot_index_map,
    const int* __restrict__ atom_positions,
    const int* __restrict__ atom_types,
    float* __restrict__ token_states_grad,
    float* __restrict__ type_embeddings_grad,
    int authored_atom_count,
    int total_tokens,
    int max_seq_len,
    int batch_size,
    int num_slots,
    int d_model,
    bool token_states_requires_grad,
    bool type_embeddings_requires_grad)
{
    const int atom = blockIdx.x;
    if (atom >= authored_atom_count) return;

    const int token = atom_positions[atom];
    if (token < 0 || token >= total_tokens) return;
    const int slot = token_to_slot_index_map[token];
    if (slot < 0 || slot >= num_slots) return;
    const int batch_row = token / max_seq_len;
    if (batch_row < 0 || batch_row >= batch_size) return;

    const int slot_row = batch_row * num_slots + slot;
    const int atom_type = atom_types[atom];
    const bool valid_type =
        atom_type == static_cast<int>(GRIM::Tokenizer::AtomType::ATOM_INT) ||
        atom_type == static_cast<int>(GRIM::Tokenizer::AtomType::ATOM_FLOAT);
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        const float grad =
            grad_output[static_cast<std::size_t>(slot_row) * d_model + d];
        if (token_states_requires_grad) {
            atomicAdd(
                &token_states_grad[
                    static_cast<std::size_t>(token) * d_model + d],
                grad);
        }
        if (type_embeddings_requires_grad && valid_type) {
            atomicAdd(
                &type_embeddings_grad[
                    static_cast<std::size_t>(atom_type) * d_model + d],
                grad);
        }
    }
}

__global__ void kernel_mask_unauthored_slot_rows(
    const float* __restrict__ input,
    const int32_t* __restrict__ token_to_slot_index_map,
    const int* __restrict__ atom_positions,
    float* __restrict__ output,
    int authored_atom_count,
    int total_tokens,
    int max_seq_len,
    int batch_size,
    int num_slots,
    int d_model)
{
    const int atom = blockIdx.x;
    if (atom >= authored_atom_count) return;

    const int token = atom_positions[atom];
    if (token < 0 || token >= total_tokens) return;
    const int slot = token_to_slot_index_map[token];
    if (slot < 0 || slot >= num_slots) return;
    const int batch_row = token / max_seq_len;
    if (batch_row < 0 || batch_row >= batch_size) return;

    const int slot_row = batch_row * num_slots + slot;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        const std::size_t index =
            static_cast<std::size_t>(slot_row) * d_model + d;
        output[index] = input[index];
    }
}

__global__ void kernel_mask_unauthored_slot_rows_backward(
    const float* __restrict__ grad_output,
    const int32_t* __restrict__ token_to_slot_index_map,
    const int* __restrict__ atom_positions,
    float* __restrict__ input_grad,
    int authored_atom_count,
    int total_tokens,
    int max_seq_len,
    int batch_size,
    int num_slots,
    int d_model)
{
    const int atom = blockIdx.x;
    if (atom >= authored_atom_count) return;

    const int token = atom_positions[atom];
    if (token < 0 || token >= total_tokens) return;
    const int slot = token_to_slot_index_map[token];
    if (slot < 0 || slot >= num_slots) return;
    const int batch_row = token / max_seq_len;
    if (batch_row < 0 || batch_row >= batch_size) return;

    const int slot_row = batch_row * num_slots + slot;
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        const std::size_t index =
            static_cast<std::size_t>(slot_row) * d_model + d;
        atomicAdd(&input_grad[index], grad_output[index]);
    }
}

void requireRoutingBindings(
    const GRIM::Batching::BatchDeviceBindings& bindings,
    int authored_atom_count,
    const char* caller)
{
    if (!bindings.d_token_to_slot_index_map) {
        throw std::runtime_error(
            std::string(caller) + ": d_token_to_slot_index_map is NULL");
    }
    if (authored_atom_count > 0 &&
        (!bindings.d_atom_positions || !bindings.d_atom_types)) {
        throw std::runtime_error(
            std::string(caller) +
            ": compact authored atom bindings are NULL");
    }
}

void requireBackwardGeometry(
    const GRIM::Batching::BatchPayload& payload,
    int batch_size,
    int max_seq_len,
    int total_tokens,
    int authored_atom_count,
    const char* caller)
{
    if (payload.batch_size != batch_size ||
        payload.max_seq_len != max_seq_len ||
        payload.total_tokens != total_tokens ||
        static_cast<int>(payload.authoredAtomCount()) != authored_atom_count) {
        throw std::runtime_error(
            std::string(caller) +
            ": backward payload geometry differs from forward");
    }
}

}  // namespace

namespace GRIM {
namespace autograd {

SlotSeedInputGradFn::SlotSeedInputGradFn()
{
    op_name = "gather_slot_seed_inputs";
}

void SlotSeedInputGradFn::capture_inputs(
    Tensor& token_states,
    Tensor& type_embeddings,
    bool use_type_embeddings,
    const Batching::BatchPayload& payload,
    int slot_count,
    cudaStream_t stream)
{
    token_states_requires_grad = token_states.requires_grad;
    type_embedding_enabled = use_type_embeddings;
    type_embeddings_requires_grad =
        use_type_embeddings && type_embeddings.requires_grad;

    batch_size = payload.batch_size;
    max_seq_len = payload.max_seq_len;
    total_tokens = payload.total_tokens;
    authored_atom_count = static_cast<int>(payload.authoredAtomCount());
    num_slots = slot_count;
    d_model = token_states.shape.as_2d().cols;

    if (token_states_requires_grad) {
        token_states_gradient = capture_input_gradient(
            token_states, stream, "SlotSeedInputGradFn::capture_inputs token_states");
    }

    if (type_embeddings_requires_grad) {
        type_embeddings_gradient = capture_input_gradient(
            type_embeddings, stream, "SlotSeedInputGradFn::capture_inputs type_embeddings");
    }
}

void SlotSeedInputGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings)
{
    setCurrentGradFnOp("gather_slot_seed_inputs", this);
    if (applied) return;
    applied = true;
    if (!backward_payload || !backward_bindings) {
        throw std::runtime_error(
            "SlotSeedInputGradFn::apply: backward payload/bindings are NULL");
    }
    requireBackwardGeometry(
        *backward_payload,
        batch_size,
        max_seq_len,
        total_tokens,
        authored_atom_count,
        "SlotSeedInputGradFn::apply");
    requireRoutingBindings(
        *backward_bindings,
        authored_atom_count,
        "SlotSeedInputGradFn::apply");

    if ((token_states_requires_grad || type_embeddings_requires_grad) &&
        authored_atom_count > 0) {
        if ((token_states_requires_grad && !token_states_gradient) ||
            (type_embeddings_requires_grad && !type_embeddings_gradient)) {
            throw std::runtime_error(
                "SlotSeedInputGradFn::apply: captured input gradient is NULL");
        }
        float* token_states_grad = nullptr;
        float* type_embeddings_grad = nullptr;
        if (token_states_requires_grad) {
            token_states_grad = token_states_gradient->data;
        }
        if (type_embeddings_requires_grad) {
            type_embeddings_grad = type_embeddings_gradient->data;
        }
        kernel_gather_slot_seed_inputs_backward<<<
            authored_atom_count,
            kSlotRoutingBlockSize,
            0,
            stream>>>(
            grad_output.data,
            backward_bindings->d_token_to_slot_index_map,
            backward_bindings->d_atom_positions,
            backward_bindings->d_atom_types,
            token_states_grad,
            type_embeddings_grad,
            authored_atom_count,
            total_tokens,
            max_seq_len,
            batch_size,
            num_slots,
            d_model,
            token_states_requires_grad,
            type_embeddings_requires_grad);
        trackKernelLaunch(
            "kernel_gather_slot_seed_inputs_backward",
            stream);
    }

    if (token_states_requires_grad) {
        propagate_input_gradient(
            token_states_gradient, stream, backward_payload, backward_bindings,
            "SlotSeedInputGradFn::apply token_states");
    }
    if (type_embeddings_requires_grad) {
        propagate_input_gradient(
            type_embeddings_gradient, stream, backward_payload, backward_bindings,
            "SlotSeedInputGradFn::apply type_embeddings");
    }
}

void SlotSeedInputGradFn::release_saved()
{
    GradFn::release_saved();
    token_states_gradient.reset();
    type_embeddings_gradient.reset();
}

AuthoredSlotMaskGradFn::AuthoredSlotMaskGradFn()
{
    op_name = "mask_unauthored_slot_rows";
}

void AuthoredSlotMaskGradFn::capture_input(
    Tensor& input,
    const Batching::BatchPayload& payload,
    int slot_count,
    cudaStream_t stream)
{
    input_requires_grad = input.requires_grad;
    batch_size = payload.batch_size;
    max_seq_len = payload.max_seq_len;
    total_tokens = payload.total_tokens;
    authored_atom_count = static_cast<int>(payload.authoredAtomCount());
    num_slots = slot_count;
    d_model = input.shape.as_2d().cols;

    if (input_requires_grad) {
        input_gradient = capture_input_gradient(
            input, stream, "AuthoredSlotMaskGradFn::capture_input");
    }
}

void AuthoredSlotMaskGradFn::apply_impl(
    const Tensor& grad_output,
    cudaStream_t stream,
    const Batching::BatchPayload* backward_payload,
    const Batching::BatchDeviceBindings* backward_bindings)
{
    setCurrentGradFnOp("mask_unauthored_slot_rows", this);
    if (applied) return;
    applied = true;
    if (!input_requires_grad) return;
    if (!backward_payload || !backward_bindings) {
        throw std::runtime_error(
            "AuthoredSlotMaskGradFn::apply: backward payload/bindings are NULL");
    }
    requireBackwardGeometry(
        *backward_payload,
        batch_size,
        max_seq_len,
        total_tokens,
        authored_atom_count,
        "AuthoredSlotMaskGradFn::apply");
    requireRoutingBindings(
        *backward_bindings,
        authored_atom_count,
        "AuthoredSlotMaskGradFn::apply");

    if (!input_gradient) {
        throw std::runtime_error(
            "AuthoredSlotMaskGradFn::apply: captured input gradient Tensor is NULL");
    }
    if (authored_atom_count > 0) {
        kernel_mask_unauthored_slot_rows_backward<<<
            authored_atom_count,
            kSlotRoutingBlockSize,
            0,
            stream>>>(
            grad_output.data,
            backward_bindings->d_token_to_slot_index_map,
            backward_bindings->d_atom_positions,
            input_gradient->data,
            authored_atom_count,
            total_tokens,
            max_seq_len,
            batch_size,
            num_slots,
            d_model);
        trackKernelLaunch(
            "kernel_mask_unauthored_slot_rows_backward",
            stream);
    }

    propagate_input_gradient(
        input_gradient, stream, backward_payload, backward_bindings,
        "AuthoredSlotMaskGradFn::apply");
}

void AuthoredSlotMaskGradFn::release_saved()
{
    GradFn::release_saved();
    input_gradient.reset();
}

Tensor gather_slot_seed_inputs(
    const Tensor& token_states,
    const Tensor& type_embeddings,
    bool type_embedding_enabled,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int num_slots,
    cudaStream_t stream)
{
    if (!stream) {
        throw std::runtime_error(
            "autograd::gather_slot_seed_inputs: stream is NULL");
    }
    if (!token_states.data || !token_states.shape.is_2d_layout()) {
        throw std::runtime_error(
            "autograd::gather_slot_seed_inputs: token_states must be a live 2D tensor");
    }
    const auto shape = token_states.shape.as_2d();
    if (shape.rows != payload.total_tokens) {
        throw std::runtime_error(
            "autograd::gather_slot_seed_inputs: token-state rows do not match payload");
    }
    if (num_slots <= 0) {
        throw std::runtime_error(
            "autograd::gather_slot_seed_inputs: num_slots must be > 0");
    }
    if (type_embedding_enabled) {
        if (!type_embeddings.data ||
            !type_embeddings.shape.is_2d_layout() ||
            type_embeddings.shape.as_2d().rows != 2 ||
            type_embeddings.shape.as_2d().cols != shape.cols) {
            throw std::runtime_error(
                "autograd::gather_slot_seed_inputs: type_embeddings must be [2, d_model]");
        }
    }
    const int authored_atom_count =
        static_cast<int>(payload.authoredAtomCount());
    requireRoutingBindings(
        bindings,
        authored_atom_count,
        "autograd::gather_slot_seed_inputs");

    const bool requires_grad =
        token_states.requires_grad ||
        (type_embedding_enabled && type_embeddings.requires_grad);
    Tensor result = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(
            payload.batch_size * num_slots,
            shape.cols),
        requires_grad,
        stream,
        "slot_seed_contextual_input");

    if (authored_atom_count > 0) {
        kernel_gather_slot_seed_inputs<<<
            authored_atom_count,
            kSlotRoutingBlockSize,
            0,
            stream>>>(
            token_states.data,
            type_embeddings.data,
            bindings.d_token_to_slot_index_map,
            bindings.d_atom_positions,
            bindings.d_atom_types,
            result.data,
            authored_atom_count,
            payload.total_tokens,
            payload.max_seq_len,
            payload.batch_size,
            num_slots,
            shape.cols,
            type_embedding_enabled);
        trackKernelLaunch("kernel_gather_slot_seed_inputs", stream);
    }

    if (requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<SlotSeedInputGradFn>();
        grad_fn->capture_inputs(
            const_cast<Tensor&>(token_states),
            const_cast<Tensor&>(type_embeddings),
            type_embedding_enabled,
            payload,
            num_slots,
            stream);
        result.grad_fn = grad_fn;
    }
    return result;
}

Tensor mask_unauthored_slot_rows(
    const Tensor& input,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int num_slots,
    cudaStream_t stream)
{
    if (!stream) {
        throw std::runtime_error(
            "autograd::mask_unauthored_slot_rows: stream is NULL");
    }
    if (!input.data || !input.shape.is_2d_layout()) {
        throw std::runtime_error(
            "autograd::mask_unauthored_slot_rows: input must be a live 2D tensor");
    }
    const auto shape = input.shape.as_2d();
    if (shape.rows != payload.batch_size * num_slots || num_slots <= 0) {
        throw std::runtime_error(
            "autograd::mask_unauthored_slot_rows: slot geometry mismatch");
    }
    const int authored_atom_count =
        static_cast<int>(payload.authoredAtomCount());
    requireRoutingBindings(
        bindings,
        authored_atom_count,
        "autograd::mask_unauthored_slot_rows");

    Tensor result = Tensor::zeros(
        input.shape,
        input.requires_grad,
        stream,
        "slot_seed_masked");
    if (authored_atom_count > 0) {
        kernel_mask_unauthored_slot_rows<<<
            authored_atom_count,
            kSlotRoutingBlockSize,
            0,
            stream>>>(
            input.data,
            bindings.d_token_to_slot_index_map,
            bindings.d_atom_positions,
            result.data,
            authored_atom_count,
            payload.total_tokens,
            payload.max_seq_len,
            payload.batch_size,
            num_slots,
            shape.cols);
        trackKernelLaunch("kernel_mask_unauthored_slot_rows", stream);
    }

    if (input.requires_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<AuthoredSlotMaskGradFn>();
        grad_fn->capture_input(
            const_cast<Tensor&>(input),
            payload,
            num_slots,
            stream);
        result.grad_fn = grad_fn;
    }
    return result;
}

}  // namespace autograd
}  // namespace GRIM

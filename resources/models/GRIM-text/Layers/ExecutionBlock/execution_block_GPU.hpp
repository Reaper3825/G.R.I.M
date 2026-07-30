//======================================================//
//  execution_block_GPU.hpp
//  Retained execution-storage scaffold
//
//  The legacy gate/op/arg/write/stop/readback machine has been removed.
//  Parameter ownership remains in ParameterRegistry, and this lightweight
//  storage view remains available while the generator/candidate/verifier
//  replacement is introduced.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

struct ExecutionMemory {
    Tensor values;
    Tensor atom_embeds;
    Tensor state_embeds;
    Tensor valid_mask;
    Tensor usage;
    Tensor key_embeds;
    Tensor type_embed;
    Tensor recent_write_mask;

    void bind(
        Tensor& values_owner,
        Tensor& atom_embeds_owner,
        Tensor& state_embeds_owner,
        Tensor& valid_mask_owner,
        Tensor& usage_owner,
        Tensor& key_embeds_owner,
        Tensor& type_embed_owner,
        Tensor& recent_write_mask_owner)
    {
        values = Tensor::from_ptr(
            values_owner.data, values_owner.shape, false,
            values_owner.requires_grad, "execution_memory_values");
        atom_embeds = Tensor::from_ptr(
            atom_embeds_owner.data, atom_embeds_owner.shape, false,
            atom_embeds_owner.requires_grad, "execution_memory_atom_embeds");
        state_embeds = Tensor::from_ptr(
            state_embeds_owner.data, state_embeds_owner.shape, false,
            state_embeds_owner.requires_grad, "execution_memory_state_embeds");
        valid_mask = Tensor::from_ptr(
            valid_mask_owner.data, valid_mask_owner.shape, false,
            valid_mask_owner.requires_grad, "execution_memory_valid_mask");
        usage = Tensor::from_ptr(
            usage_owner.data, usage_owner.shape, false,
            usage_owner.requires_grad, "execution_memory_usage");
        key_embeds = Tensor::from_ptr(
            key_embeds_owner.data, key_embeds_owner.shape, false,
            key_embeds_owner.requires_grad, "execution_memory_key_embeds");
        type_embed = Tensor::from_ptr(
            type_embed_owner.data, type_embed_owner.shape, false,
            type_embed_owner.requires_grad, "execution_memory_type_embed");
        recent_write_mask = Tensor::from_ptr(
            recent_write_mask_owner.data, recent_write_mask_owner.shape, false,
            recent_write_mask_owner.requires_grad, "execution_memory_recent_write_mask");
    }
};

}  // namespace GRIM

#endif  // USE_CUDA

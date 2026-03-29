#include "execution_block_memory_stream_GPU.hpp"

#ifdef USE_CUDA

namespace GRIM {

using namespace ExecutionBlockInternal;

__global__ void kernelBootstrapSlotValues(
    float* __restrict__ M_values,
    float* __restrict__ M_valid_mask,
    const float* __restrict__ numeric_values,
    const int32_t* __restrict__ slot_map,
    int total_tokens, int V
) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total_tokens) return;
    const int slot = slot_map[pos];
    if (slot >= 0 && slot < V) {
        M_values[slot] = numeric_values[pos];
        M_valid_mask[slot] = 1.0f;
    }
}

__global__ void kernelBootstrapSlotEmbeddings(
    float* __restrict__ state_embeds,
    float* __restrict__ key_embeds,
    const float* __restrict__ values,
    const float* __restrict__ valid_mask,
    const float* __restrict__ W_val2emb,
    const float* __restrict__ b_val2emb,
    const float* __restrict__ W_key_proj,
    int V, int d_model, int d_key
) {
    const int slot = blockIdx.x;
    if (slot >= V || valid_mask[slot] == 0.0f) return;

    const float val = values[slot];
    extern __shared__ float smem[];
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        float emb_d = val * W_val2emb[d] + b_val2emb[d];
        state_embeds[static_cast<size_t>(slot) * d_model + d] = emb_d;
        smem[d] = emb_d;
    }
    __syncthreads();

    for (int k = threadIdx.x; k < d_key; k += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < d_model; ++d)
            sum += smem[d] * W_key_proj[d * d_key + k];
        key_embeds[static_cast<size_t>(slot) * d_key + k] = sum;
    }
}

__global__ void kernelBootstrapTypeEmbed(
    float* __restrict__ type_embed,
    const float* __restrict__ valid_mask,
    const float* __restrict__ type_num_emb,
    int V, int d_type
) {
    const int slot = blockIdx.x;
    if (slot >= V || valid_mask[slot] == 0.0f) return;
    for (int j = threadIdx.x; j < d_type; j += blockDim.x)
        type_embed[static_cast<size_t>(slot) * d_type + j] = type_num_emb[j];
}

void ExecutionMemory::allocate(int V, int atom_dim, int d_model, int d_key, int d_type, cudaStream_t stream) {
    EXEC_CHECK(V > 0, "ExecutionMemory::allocate: V must be positive");
    EXEC_CHECK(atom_dim > 0, "ExecutionMemory::allocate: atom_dim must be positive");
    EXEC_CHECK(d_model > 0, "ExecutionMemory::allocate: d_model must be positive");
    EXEC_CHECK(d_key > 0, "ExecutionMemory::allocate: d_key must be positive");
    EXEC_CHECK(d_type > 0, "ExecutionMemory::allocate: d_type must be positive");

    values            = Tensor::zeros({V, 1}, stream);
    atom_embeds       = Tensor::zeros({V, atom_dim}, stream);
    state_embeds      = Tensor::zeros({V, d_model}, stream);
    valid_mask        = Tensor::zeros({1, V}, stream);
    usage             = Tensor::zeros({1, V}, stream);
    key_embeds        = Tensor::zeros({V, d_key}, stream);
    type_embed        = Tensor::zeros({V, d_type}, stream);
    recent_write_mask = Tensor::zeros({1, V}, stream);
}

void ExecutionMemory::clear(cudaStream_t stream) {
    if (values.data)            cudaMemsetAsync(values.data, 0, values.size_bytes(), stream);
    if (atom_embeds.data)       cudaMemsetAsync(atom_embeds.data, 0, atom_embeds.size_bytes(), stream);
    if (state_embeds.data)      cudaMemsetAsync(state_embeds.data, 0, state_embeds.size_bytes(), stream);
    if (valid_mask.data)        cudaMemsetAsync(valid_mask.data, 0, valid_mask.size_bytes(), stream);
    if (usage.data)             cudaMemsetAsync(usage.data, 0, usage.size_bytes(), stream);
    if (key_embeds.data)        cudaMemsetAsync(key_embeds.data, 0, key_embeds.size_bytes(), stream);
    if (type_embed.data)        cudaMemsetAsync(type_embed.data, 0, type_embed.size_bytes(), stream);
    if (recent_write_mask.data) cudaMemsetAsync(recent_write_mask.data, 0, recent_write_mask.size_bytes(), stream);
}

void ExecutionBlockLayer::bootstrapMemoryFromSlotMap(
    ExecutionMemory& M,
    const float* device_numeric_values,
    const int32_t* device_slot_map,
    int total_tokens,
    cudaStream_t stream)
{
    EXEC_CHECK(device_numeric_values != nullptr, "bootstrapMemoryFromSlotMap: device_numeric_values is null");
    EXEC_CHECK(device_slot_map != nullptr, "bootstrapMemoryFromSlotMap: device_slot_map is null");
    EXEC_CHECK(total_tokens > 0, "bootstrapMemoryFromSlotMap: total_tokens must be positive");
    validateMemoryOrThrow(M);

    const int V = config_.num_slots;
    const int blocks = (total_tokens + kBlockSize - 1) / kBlockSize;
    kernelBootstrapSlotValues<<<blocks, kBlockSize, 0, stream>>>(
        M.values.data, M.valid_mask.data,
        device_numeric_values, device_slot_map,
        total_tokens, V);
    CUDA_CHECK_KERNEL();

    const int dm = config_.d_model;
    const int dk = config_.d_key;
    const size_t smem_bytes = dm * sizeof(float);
    kernelBootstrapSlotEmbeddings<<<V, kBlockSize, smem_bytes, stream>>>(
        M.state_embeds.data, M.key_embeds.data,
        M.values.data, M.valid_mask.data,
        W_value_to_emb_.data, b_value_to_emb_.data,
        W_key_proj_.data,
        V, dm, dk);
    CUDA_CHECK_KERNEL();

    const int dt = config_.d_type;
    kernelBootstrapTypeEmbed<<<V, kBlockSize, 0, stream>>>(
        M.type_embed.data, M.valid_mask.data,
        type_num_embed_.data, V, dt);
    CUDA_CHECK_KERNEL();
}

}  // namespace GRIM

#endif  // USE_CUDA

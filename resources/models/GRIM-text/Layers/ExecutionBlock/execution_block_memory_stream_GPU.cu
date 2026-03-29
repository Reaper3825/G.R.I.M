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

__global__ void kernelGatherSlotOnlyHidden(
    float* __restrict__ out,
    const float* __restrict__ mem_state,
    const float* __restrict__ mem_valid,
    int V_val, int S, int d_model
) {
    const int i = blockIdx.x;
    if (i >= V_val) return;
    const int slot = S + i;
    const float valid = mem_valid[slot];
    const float* src = mem_state + static_cast<size_t>(slot) * d_model;
    float* dst = out + static_cast<size_t>(i) * d_model;
    for (int j = threadIdx.x; j < d_model; j += blockDim.x)
        dst[j] = valid * src[j];
}

__global__ void kernelBuildSlotOnlyCandidateMask(
    float* __restrict__ mask,
    const float* __restrict__ mem_valid,
    int V_val, int S
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V_val) return;
    mask[i] = mem_valid[S + i];
}

__global__ void kernelGatherValueSlotScalars(
    float* __restrict__ out,
    const float* __restrict__ M_values,
    const float* __restrict__ M_valid,
    int V_val, int S
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V_val) return;
    const int slot = S + i;
    out[i] = M_valid[slot] * M_values[slot];
}

__global__ void kernelArgmax1DInt(
    int* __restrict__ out_idx,
    const float* __restrict__ probs,
    int N
) {
    if (threadIdx.x != 0) return;
    int best = 0;
    float bestv = (N > 0) ? probs[0] : 0.0f;
    for (int i = 1; i < N; ++i) {
        if (probs[i] > bestv) {
            bestv = probs[i];
            best = i;
        }
    }
    out_idx[0] = best;
}

__global__ void kernelReadSlotValueByRelIdx(
    float* __restrict__ out,
    const float* __restrict__ M_values,
    const float* __restrict__ M_valid,
    int S, int V, int V_val,
    const int* __restrict__ rel_idx,
    int* __restrict__ error_flag,
    int stage_invalid,
    int stage_uninit
) {
    if (threadIdx.x != 0) return;
    int r = rel_idx[0];
    if (r < 0 || r >= V_val) {
        atomicMax(error_flag, stage_invalid);
        out[0] = 0.0f;
        return;
    }
    int slot = S + r;
    if (slot < S || slot >= V) {
        atomicMax(error_flag, stage_invalid);
        out[0] = 0.0f;
        return;
    }
    if (M_valid[slot] == 0.0f) {
        atomicMax(error_flag, stage_uninit);
        out[0] = 0.0f;
        return;
    }
    out[0] = M_values[slot];
}

__global__ void kernelValidateWriteSlotDev(
    const int* __restrict__ d_write_slot,
    int S, int V,
    int* __restrict__ error_flag,
    int stage_id
) {
    if (threadIdx.x != 0) return;
    int ws = d_write_slot[0];
    if (ws < S || ws >= V)
        atomicMax(error_flag, stage_id);
}

__global__ void kernelHardWriteScalarDev(
    float* __restrict__ mem_values,
    const int* __restrict__ d_write_slot,
    const float* __restrict__ v_new
) {
    if (threadIdx.x != 0) return;
    mem_values[d_write_slot[0]] = v_new[0];
}

__global__ void kernelHardWriteRowDev(
    float* __restrict__ mem_mat,
    const int* __restrict__ d_write_slot, int D,
    const float* __restrict__ new_row
) {
    int ws = d_write_slot[0];
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        mem_mat[static_cast<size_t>(ws) * D + j] = new_row[j];
}

__global__ void kernelSetValidMaskDev(
    float* __restrict__ valid_mask,
    const int* __restrict__ d_write_slot
) {
    if (threadIdx.x != 0) return;
    valid_mask[d_write_slot[0]] = 1.0f;
}

__global__ void kernelSetRecentWriteOneHotDev(
    float* __restrict__ recent,
    int V,
    const int* __restrict__ d_write_slot
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    int ws = d_write_slot[0];
    recent[i] = (i == ws) ? 1.0f : 0.0f;
}

__global__ void kernelValidateAtomSlots(
    const int* __restrict__ atom_positions,
    const int32_t* __restrict__ slot_map,
    const float* __restrict__ M_valid_mask,
    int num_atoms, int row_tokens, int V, int S,
    int* __restrict__ error_flag,
    int stage_pos_invalid,
    int stage_missing,
    int stage_invalid,
    int stage_uninit
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms) return;

    int pos = atom_positions[i];
    if (pos < 0 || pos >= row_tokens) {
        atomicMax(error_flag, stage_pos_invalid);
        return;
    }
    int slot = slot_map[pos];
    if (slot == -1) {
        atomicMax(error_flag, stage_missing);
    } else if (slot < S || slot >= V) {
        atomicMax(error_flag, stage_invalid);
    } else if (M_valid_mask[slot] == 0.0f) {
        atomicMax(error_flag, stage_uninit);
    }
}

__global__ void kernelStateDeltaCheck(
    float* __restrict__ state_integrity_loss,
    int* __restrict__ changed_count,
    const float* __restrict__ before,
    const float* __restrict__ after,
    const int* __restrict__ d_write_slot,
    int V
) {
    if (threadIdx.x != 0) return;
    int ws = d_write_slot[0];
    float hinge_sum = 0.0f;
    int count = 0;
    for (int i = 0; i < V; ++i) {
        float delta = fabsf(after[i] - before[i]);
        float eps_i = fmaxf(1e-6f, 0.01f * fabsf(before[i]));
        if (delta > eps_i) {
            count++;
            if (i != ws)
                hinge_sum += delta - eps_i;
        }
    }
    state_integrity_loss[0] = hinge_sum;
    changed_count[0] = count;
}

__global__ void kernelCheckMultiSlotMutation(
    const int* __restrict__ changed_count,
    int* __restrict__ error_flag,
    int stage_id
) {
    if (threadIdx.x != 0) return;
    if (changed_count[0] > 1)
        atomicMax(error_flag, stage_id);
}

__global__ void kernelComputeGate(
    float* __restrict__ gate,
    const float* __restrict__ H,
    const float* __restrict__ w_gate,
    int total_tokens, int d_model
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total_tokens) return;
    const float* h = H + static_cast<size_t>(t) * d_model;
    float sum = 0.0f;
    for (int j = 0; j < d_model; ++j)
        sum += h[j] * w_gate[j];
    gate[t] = 1.0f / (1.0f + expf(-sum));
}

__global__ void kernelSmallMatmul(
    float* __restrict__ out,
    const float* __restrict__ A,
    const float* __restrict__ B,
    int M, int K, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        out[row * N + col] = sum;
    }
}

__global__ void kernelCrossAttnSharpScores(
    float* __restrict__ scores,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ valid,
    const float* __restrict__ tau_ptr,
    int total_tokens, int num_valid, int head_dim, int topk
) {
    const int t = blockIdx.x;
    if (t >= total_tokens) return;

    float tau = fmaxf(tau_ptr[0], 0.01f);
    float inv_sqrt_d_tau = 1.0f / (sqrtf(static_cast<float>(head_dim)) * tau);

    const float* q_row = Q + static_cast<size_t>(t) * head_dim;
    float* s_row = scores + static_cast<size_t>(t) * num_valid;
    for (int v = 0; v < num_valid; ++v) {
        float dot = 0.0f;
        const float* k_row = K + static_cast<size_t>(v) * head_dim;
        for (int d = 0; d < head_dim; ++d)
            dot += q_row[d] * k_row[d];
        s_row[v] = dot * inv_sqrt_d_tau;
        if (valid[v] < 1e-6f) s_row[v] = -FLT_MAX;
    }

    if (topk > 0 && topk < num_valid) {
        for (int pass = 0; pass < topk; ++pass) {
            float best = -FLT_MAX;
            int best_idx = -1;
            for (int v = 0; v < num_valid; ++v) {
                if (s_row[v] > best) { best = s_row[v]; best_idx = v; }
            }
            if (best_idx >= 0) s_row[best_idx] += 1e9f;
        }
        for (int v = 0; v < num_valid; ++v) {
            if (s_row[v] > 1e8f) s_row[v] -= 1e9f;
            else s_row[v] = -FLT_MAX;
        }
    }

    float max_s = -FLT_MAX;
    for (int v = 0; v < num_valid; ++v) max_s = fmaxf(max_s, s_row[v]);
    float sum_exp = 0.0f;
    for (int v = 0; v < num_valid; ++v) {
        float e = expf(s_row[v] - max_s);
        s_row[v] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / (sum_exp + kEps);
    for (int v = 0; v < num_valid; ++v) s_row[v] *= inv_sum;
}

__global__ void kernelCrossAttnWeightedValue(
    float* __restrict__ R,
    const float* __restrict__ attn,
    const float* __restrict__ V_proj,
    int total_tokens, int num_valid, int head_dim
) {
    const int t = blockIdx.x;
    if (t >= total_tokens) return;
    const float* a_row = attn + static_cast<size_t>(t) * num_valid;
    float* r_row = R + static_cast<size_t>(t) * head_dim;
    for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float sum = 0.0f;
        for (int v = 0; v < num_valid; ++v)
            sum += a_row[v] * V_proj[v * head_dim + j];
        r_row[j] = sum;
    }
}

__global__ void kernelCrossAttnGatedOutput(
    float* __restrict__ H,
    const float* __restrict__ R,
    const float* __restrict__ W_O,
    const float* __restrict__ gate,
    int total_tokens, int d_model, int head_dim
) {
    const int t = blockIdx.x;
    if (t >= total_tokens) return;
    float g = gate[t];
    const float* r_row = R + static_cast<size_t>(t) * head_dim;
    float* h_row = H + static_cast<size_t>(t) * d_model;
    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        float proj = 0.0f;
        for (int k = 0; k < head_dim; ++k)
            proj += r_row[k] * W_O[k * d_model + j];
        h_row[j] += g * proj;
    }
}

__global__ void kernelDecayedUsageUpdate(
    float* __restrict__ usage,
    const float* __restrict__ attn,
    float decay,
    int total_tokens, int num_valid, int V
) {
    const int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_valid) return;
    float sum = 0.0f;
    for (int t = 0; t < total_tokens; ++t)
        sum += attn[static_cast<size_t>(t) * num_valid + v];
    usage[v] = decay * usage[v] + sum;
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

static void ensureBootstrappedValueSlotsOrThrow(
    const ExecutionBlockLayer& layer,
    const ExecutionMemory& memory,
    cudaStream_t stream
) {
    const int V = layer.config().num_slots;
    const int S = layer.config().num_scratch_slots;
    const int V_val = V - S;
    EXEC_CHECK(V_val > 0, "executeStep: no value slots (V - S == 0)");

    float h_valid_sum = 0.0f;
    std::vector<float> h_mask(V_val);
    CUDA_CHECK(cudaMemcpyAsync(h_mask.data(), memory.valid_mask.data + S,
        V_val * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int i = 0; i < V_val; ++i)
        h_valid_sum += h_mask[i];
    EXEC_CHECK(h_valid_sum >= 0.5f,
        "executeStep: execution-active row reached ExecutionBlock with no bootstrapped value slots");
}

void prepareMemoryStepOrThrow(
    ExecutionBlockLayer& layer,
    const ExecutionMemory& memory,
    const int* atom_positions,
    const int32_t* token_to_slot_map,
    int num_atoms,
    int row_tokens,
    ExecutionBlockStepOutput* diag_out,
    cudaStream_t stream
) {
    ensureBootstrappedValueSlotsOrThrow(layer, memory, stream);

    const int V = layer.config().num_slots;
    if (diag_out) {
        diag_out->state_before_values = Tensor::zeros({V, 1}, stream, "state_before_values");
        diag_out->state_before_valid = Tensor::zeros({1, V}, stream, "state_before_valid");
        CUDA_CHECK(cudaMemcpyAsync(diag_out->state_before_values.data, memory.values.data,
            V * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(diag_out->state_before_valid.data, memory.valid_mask.data,
            V * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaMemsetAsync(LayerAccess::numericErrorFlag(layer), 0, sizeof(int), stream));
    if (num_atoms > 0) {
        kernelValidateAtomSlots<<<(num_atoms + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            atom_positions,
            token_to_slot_map,
            memory.valid_mask.data,
            num_atoms,
            row_tokens,
            layer.config().num_slots,
            layer.config().num_scratch_slots,
            LayerAccess::numericErrorFlag(layer),
            kStageAtomPosInvalid,
            kStageSlotMissing,
            kStageSlotInvalid,
            kStageSlotUninit);
        CUDA_CHECK_KERNEL();
    }
}

void buildValueSlotCandidates(
    ExecutionBlockLayer& layer,
    const ExecutionMemory& memory,
    cudaStream_t stream,
    StepWorkingSet& work
) {
    const int dm = layer.config().d_model;
    const int V_val = layer.config().num_slots - layer.config().num_scratch_slots;
    const int S = layer.config().num_scratch_slots;

    work.cand_hidden = Tensor::zeros({V_val, dm}, stream, "exec_cand_hidden");
    work.cand_hidden.requires_grad = false;
    work.cand_hidden.is_leaf = true;
    kernelGatherSlotOnlyHidden<<<V_val, kBlockSize, 0, stream>>>(
        work.cand_hidden.data, memory.state_embeds.data, memory.valid_mask.data, V_val, S, dm);
    CUDA_CHECK_KERNEL();

    work.cand_mask = Tensor::zeros({1, V_val}, stream, "exec_cand_mask");
    kernelBuildSlotOnlyCandidateMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.cand_mask.data, memory.valid_mask.data, V_val, S);
    CUDA_CHECK_KERNEL();

    work.slot_values = Tensor::zeros({V_val, 1}, stream, "exec_slot_values");
    kernelGatherValueSlotScalars<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.slot_values.data, memory.values.data, memory.valid_mask.data, V_val, S);
    CUDA_CHECK_KERNEL();
}

void materializeSelectedOperands(
    ExecutionBlockLayer& layer,
    const ExecutionMemory& memory,
    cudaStream_t stream,
    StepWorkingSet& work
) {
    const int V = layer.config().num_slots;
    const int S = layer.config().num_scratch_slots;
    const int V_val = V - S;
    int* d_exec_idx = LayerAccess::execIndices(layer);

    EXEC_CHECK(work.p_arg1.data != nullptr, "materializeSelectedOperands: p_arg1 is null");
    EXEC_CHECK(work.p_arg2.data != nullptr, "materializeSelectedOperands: p_arg2 is null");

    kernelArgmax1DInt<<<1, 1, 0, stream>>>(d_exec_idx, work.p_arg1.data, V_val);
    CUDA_CHECK_KERNEL();
    kernelArgmax1DInt<<<1, 1, 0, stream>>>(d_exec_idx + 1, work.p_arg2.data, V_val);
    CUDA_CHECK_KERNEL();

    work.v1 = Tensor::zeros({1, 1}, stream, "exec_v1");
    work.v1.requires_grad = work.p_arg1.requires_grad;
    work.v1.is_leaf = false;
    kernelReadSlotValueByRelIdx<<<1, 1, 0, stream>>>(
        work.v1.data, memory.values.data, memory.valid_mask.data, S, V, V_val,
        d_exec_idx, LayerAccess::numericErrorFlag(layer), kStageSlotInvalid, kStageSlotUninit);
    CUDA_CHECK_KERNEL();

    work.v2 = Tensor::zeros({1, 1}, stream, "exec_v2");
    work.v2.requires_grad = work.p_arg2.requires_grad;
    work.v2.is_leaf = false;
    kernelReadSlotValueByRelIdx<<<1, 1, 0, stream>>>(
        work.v2.data, memory.values.data, memory.valid_mask.data, S, V, V_val,
        d_exec_idx + 1, LayerAccess::numericErrorFlag(layer), kStageSlotInvalid, kStageSlotUninit);
    CUDA_CHECK_KERNEL();
}

void applyHardWriteback(
    ExecutionBlockLayer& layer,
    ExecutionMemory& memory,
    cudaStream_t stream,
    const StepWorkingSet& work
) {
    const int V = layer.config().num_slots;
    const int S = layer.config().num_scratch_slots;
    const int dm = layer.config().d_model;
    const int dk = layer.config().d_key;
    const int ae = layer.config().atom_embedding_dim;
    const int dt = layer.config().d_type;
    int* d_exec_idx = LayerAccess::execIndices(layer);

    kernelArgmax1DInt<<<1, 1, 0, stream>>>(d_exec_idx + 3, work.p_write.data, V);
    CUDA_CHECK_KERNEL();
    kernelValidateWriteSlotDev<<<1, 1, 0, stream>>>(
        d_exec_idx + 3, S, V, LayerAccess::numericErrorFlag(layer), kStageWriteSlotInvalid);
    CUDA_CHECK_KERNEL();

    kernelHardWriteScalarDev<<<1, 1, 0, stream>>>(memory.values.data, d_exec_idx + 3, work.v_out.data);
    CUDA_CHECK_KERNEL();
    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(memory.state_embeds.data, d_exec_idx + 3, dm, work.state_new.data);
    CUDA_CHECK_KERNEL();
    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(memory.key_embeds.data, d_exec_idx + 3, dk, work.key_new.data);
    CUDA_CHECK_KERNEL();
    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(memory.atom_embeds.data, d_exec_idx + 3, ae, work.atom_new.data);
    CUDA_CHECK_KERNEL();
    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(memory.type_embed.data, d_exec_idx + 3, dt, layer.type_num_embed().data);
    CUDA_CHECK_KERNEL();
    kernelSetValidMaskDev<<<1, 1, 0, stream>>>(memory.valid_mask.data, d_exec_idx + 3);
    CUDA_CHECK_KERNEL();
    kernelSetRecentWriteOneHotDev<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        memory.recent_write_mask.data, V, d_exec_idx + 3);
    CUDA_CHECK_KERNEL();
}

void captureStateAfterWriteAndCheckMutations(
    ExecutionBlockLayer& layer,
    const ExecutionMemory& memory,
    ExecutionBlockStepOutput* diag_out,
    cudaStream_t stream
) {
    if (!diag_out) return;

    const int V = layer.config().num_slots;
    diag_out->state_after_values = Tensor::zeros({V, 1}, stream, "state_after_values");
    diag_out->state_after_valid = Tensor::zeros({1, V}, stream, "state_after_valid");
    CUDA_CHECK(cudaMemcpyAsync(diag_out->state_after_values.data, memory.values.data,
        V * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(diag_out->state_after_valid.data, memory.valid_mask.data,
        V * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    float* d_hinge_discard = nullptr;
    int* d_changed_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hinge_discard, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_changed_count, sizeof(int)));
    kernelStateDeltaCheck<<<1, 1, 0, stream>>>(
        d_hinge_discard,
        d_changed_count,
        diag_out->state_before_values.data,
        diag_out->state_after_values.data,
        LayerAccess::execIndices(layer) + 3,
        V);
    CUDA_CHECK_KERNEL();
    cudaFreeAsync(d_hinge_discard, stream);

    kernelCheckMultiSlotMutation<<<1, 1, 0, stream>>>(
        d_changed_count,
        LayerAccess::numericErrorFlag(layer),
        kStageMultiSlotMutation);
    CUDA_CHECK_KERNEL();
    cudaFreeAsync(d_changed_count, stream);
}

void finalizeStepOrThrow(
    ExecutionBlockLayer& layer,
    ExecutionBlockStepOutput* diag_out,
    int step,
    cudaStream_t stream
) {
    int h_error = 0;
    int ri[3] = {0, 0, 0};
    float rf[3] = {0.0f, 0.0f, 0.0f};
    if (diag_out) {
        CUDA_CHECK(cudaMemcpyAsync(ri, LayerAccess::execRecordI(layer), 3 * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(rf, LayerAccess::execRecordF(layer), 3 * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(&h_error, LayerAccess::numericErrorFlag(layer), sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (diag_out) {
        diag_out->record.arg1_slot = ri[0];
        diag_out->record.arg2_slot = ri[1];
        diag_out->record.op_id = ri[2];
        diag_out->record.value_before_1 = rf[0];
        diag_out->record.value_before_2 = rf[1];
        diag_out->record.value_after = rf[2];
    }

    if (h_error > 0) {
        char buf[384];
        snprintf(buf, sizeof(buf),
            "ExecutionBlock FATAL: invalid state at step %d — %s (stage id=%d)",
            step, stageIdToName(h_error), h_error);
        if (layer.config().debug_mode)
            fprintf(stderr, "[ExecutionBlock debug] %s\n", buf);
        throw std::runtime_error(buf);
    }
}

void crossAttentionReadImpl(
    ExecutionBlockLayer& layer,
    Tensor& hidden_states,
    ExecutionMemory& memory,
    int total_tokens,
    cudaStream_t stream,
    int token_offset,
    int row_tokens
) {
    const int dm = layer.config().d_model;
    const int dk = layer.config().d_key;
    const int hd = layer.config().cross_attn_head_dim;
    const int V = layer.config().num_slots;
    const int nv = V;
    const int topk = layer.config().cross_attn_topk;
    float* H_row = hidden_states.data + static_cast<size_t>(token_offset) * dm;

    auto Q = Tensor::zeros({row_tokens, hd}, stream, "exec_read_Q");
    kernelSmallMatmul<<<row_tokens, hd, 0, stream>>>(Q.data, H_row, layer.W_Q_read().data, row_tokens, dm, hd);
    CUDA_CHECK_KERNEL();

    auto K_proj = Tensor::zeros({nv, hd}, stream, "exec_read_K");
    kernelSmallMatmul<<<nv, hd, 0, stream>>>(K_proj.data, memory.key_embeds.data, layer.W_K_read().data, nv, dk, hd);
    CUDA_CHECK_KERNEL();

    auto V_proj = Tensor::zeros({nv, hd}, stream, "exec_read_V");
    kernelSmallMatmul<<<nv, hd, 0, stream>>>(V_proj.data, memory.state_embeds.data, layer.W_V_read().data, nv, dm, hd);
    CUDA_CHECK_KERNEL();

    auto scores = Tensor::zeros({row_tokens, nv}, stream, "exec_read_scores");
    kernelCrossAttnSharpScores<<<row_tokens, 1, 0, stream>>>(
        scores.data, Q.data, K_proj.data, memory.valid_mask.data,
        layer.tau().data, row_tokens, nv, hd, topk);
    CUDA_CHECK_KERNEL();

    auto R = Tensor::zeros({row_tokens, hd}, stream, "exec_read_R");
    kernelCrossAttnWeightedValue<<<row_tokens, hd, 0, stream>>>(R.data, scores.data, V_proj.data, row_tokens, nv, hd);
    CUDA_CHECK_KERNEL();

    auto gate = Tensor::zeros({1, row_tokens}, stream, "exec_read_gate");
    kernelComputeGate<<<(row_tokens + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        gate.data, H_row, layer.W_gate_read().data, row_tokens, dm);
    CUDA_CHECK_KERNEL();

    kernelCrossAttnGatedOutput<<<row_tokens, kBlockSize, 0, stream>>>(
        H_row, R.data, layer.W_O_read().data, gate.data, row_tokens, dm, hd);
    CUDA_CHECK_KERNEL();

    kernelDecayedUsageUpdate<<<(nv + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        memory.usage.data, scores.data, layer.config().usage_decay, row_tokens, nv, V);
    CUDA_CHECK_KERNEL();
}

}  // namespace GRIM

#endif  // USE_CUDA

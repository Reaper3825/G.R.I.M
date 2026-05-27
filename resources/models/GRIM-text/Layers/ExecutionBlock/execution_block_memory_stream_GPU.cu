#include "execution_block_memory_stream_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

#ifdef USE_CUDA

namespace GRIM {

using namespace ExecutionBlockInternal;

// Bootstrap slots — last-token-wins semantics for duplicate slot mappings.
// Multiple tokens may map to the same slot; we use atomicExch on valid_mask
// (idempotent) and a two-pass approach: first mark valid, then a second kernel
// writes values for only the highest-position token per slot (deterministic).
__global__ void kernelBootstrapSlotMarkValid(
    float* __restrict__ M_valid_mask,
    int* __restrict__ slot_last_pos,
    const int32_t* __restrict__ slot_map,
    int total_tokens, int V
) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total_tokens) return;
    const int slot = slot_map[pos];
    if (slot >= 0 && slot < V) {
        M_valid_mask[slot] = 1.0f;  // idempotent — race-safe
        atomicMax(&slot_last_pos[slot], pos);  // deterministic: highest position wins
    }
}

__global__ void kernelBootstrapSlotWriteValues(
    float* __restrict__ M_values,
    const float* __restrict__ numeric_values,
    const int* __restrict__ slot_last_pos,
    int V
) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= V) return;
    int pos = slot_last_pos[slot];
    if (pos >= 0) {
        M_values[slot] = numeric_values[pos];
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
    const int tid = threadIdx.x;
    float best_val = -1e30f;
    int best_idx = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = probs[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
    }
    if (tid == 0) out_idx[0] = best_idx;
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

// Accumulate sum of gate values into a [2]-float accumulator: [sum, count]
// Launched with 1 block after kernelComputeGate to avoid extra sync.
__global__ void kernelAccumulateGateStats(
    float* __restrict__ accum,       // [2]: accum[0] = running sum, accum[1] = running count
    const float* __restrict__ gate,  // [n] gate values
    int n
) {
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        local_sum += gate[i];
    // Warp reduction
    for (int mask = warpSize / 2; mask > 0; mask >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    // Every warp leader contributes its partial sum
    if (threadIdx.x % warpSize == 0) {
        atomicAdd(&accum[0], local_sum);
    }
    // Only thread 0 adds count (once per kernel invocation)
    if (threadIdx.x == 0) {
        atomicAdd(&accum[1], static_cast<float>(n));
    }
}

// Non-differentiable masking kernel (like causal mask in standard attention).
// Applies valid_mask + top-k selection to pre-computed scores.
// Does NOT recompute dot products or softmax — those are handled by autograd ops.
__global__ void kernelApplyValidMaskAndTopK(
    float* __restrict__ scores,       // [total_tokens, num_valid] — modified in-place
    const float* __restrict__ valid,   // [num_valid]
    int total_tokens, int num_valid, int topk
) {
    const int t = blockIdx.x;
    if (t >= total_tokens) return;

    float* s_row = scores + static_cast<size_t>(t) * num_valid;

    // Apply valid mask: invalid slots → -FLT_MAX
    for (int v = threadIdx.x; v < num_valid; v += blockDim.x) {
        if (valid[v] < 1e-6f) s_row[v] = -FLT_MAX;
    }
    __syncthreads();

    // Top-k selection (serial, thread 0 — num_valid is small)
    if (threadIdx.x == 0 && topk > 0 && topk < num_valid) {
        for (int pass = 0; pass < topk; ++pass) {
            float best = -FLT_MAX;
            int best_idx = -1;
            for (int v = 0; v < num_valid; ++v) {
                if (s_row[v] >= 1e30f) continue;
                if (s_row[v] > best) { best = s_row[v]; best_idx = v; }
            }
            if (best_idx >= 0) s_row[best_idx] = 1e30f + best;
        }
        for (int v = 0; v < num_valid; ++v) {
            if (s_row[v] >= 1e30f) s_row[v] -= 1e30f;
            else s_row[v] = -FLT_MAX;
        }
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
    int row_tokens,
    cudaStream_t stream)
{
    EXEC_CHECK(device_numeric_values != nullptr, "bootstrapMemoryFromSlotMap: device_numeric_values is null");
    EXEC_CHECK(device_slot_map != nullptr, "bootstrapMemoryFromSlotMap: device_slot_map is null");
    EXEC_CHECK(row_tokens > 0, "bootstrapMemoryFromSlotMap: row_tokens must be positive");
    validateMemoryOrThrow(M);

    const int V = hp_.num_slots;

    // Two-pass bootstrap: resolve race condition when multiple tokens map to same slot.
    // Pass 1: mark valid + find highest-position token per slot (deterministic last-writer-wins).
    int* d_slot_last_pos = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_slot_last_pos), V * sizeof(int), "bootstrap_slot_last_pos");
    CUDA_CHECK(cudaMemsetAsync(d_slot_last_pos, 0xFF, V * sizeof(int), stream));  // fill with -1

    const int blocks = (row_tokens + kBlockSize - 1) / kBlockSize;
    kernelBootstrapSlotMarkValid<<<blocks, kBlockSize, 0, stream>>>(
        M.valid_mask.data, d_slot_last_pos,
        device_slot_map, row_tokens, V);
    CUDA_CHECK_KERNEL();

    // Pass 2: write values — only the highest-position token per slot writes.
    const int slot_blocks = (V + kBlockSize - 1) / kBlockSize;
    kernelBootstrapSlotWriteValues<<<slot_blocks, kBlockSize, 0, stream>>>(
        M.values.data, device_numeric_values, d_slot_last_pos, V);
    CUDA_CHECK_KERNEL();
    cudaFreeAsync(d_slot_last_pos, stream);

    const int dm = hp_.d_model;
    const int dk = hp_.d_key;
    const size_t smem_bytes = dm * sizeof(float);
    kernelBootstrapSlotEmbeddings<<<V, kBlockSize, smem_bytes, stream>>>(
        M.state_embeds.data, M.key_embeds.data,
        M.values.data, M.valid_mask.data,
        W_value_to_emb_.data, b_value_to_emb_.data,
        W_key_proj_.data,
        V, dm, dk);
    CUDA_CHECK_KERNEL();

    const int dt = hp_.d_type;
    kernelBootstrapTypeEmbed<<<V, kBlockSize, 0, stream>>>(
        M.type_embed.data, M.valid_mask.data,
        type_num_embed_.data, V, dt);
    CUDA_CHECK_KERNEL();
}

void ExecutionBlockLayer::prepareForwardRuntime(
    const Batching::BatchPayload& payload,
    bool connect_parameter_graph,
    cudaStream_t stream,
    std::vector<ExecutionMemory>& exec_memories,
    std::vector<ExecutionBlockOutput>& exec_outputs_per_row,
    std::vector<Tensor>& exec_expected_target_tensors,
    std::vector<std::vector<ExecutionRecord>>& execution_trace_by_row,
    std::vector<Tensor>& trace_state_by_row
) const {
    validateConfigOrThrow();
    EXEC_CHECK(stream != nullptr, "prepareForwardRuntime: stream is NULL");
    EXEC_CHECK(payload.batch_size > 0, "prepareForwardRuntime: payload.batch_size must be positive");
    if (!payload.execution_active.empty()) {
        EXEC_CHECK(static_cast<int>(payload.execution_active.size()) == payload.batch_size,
                   "prepareForwardRuntime: payload.execution_active size must equal payload.batch_size");
    }

    exec_memories.resize(payload.batch_size);
    exec_outputs_per_row.resize(payload.batch_size);
    execution_trace_by_row.resize(payload.batch_size);
    trace_state_by_row.resize(payload.batch_size);
    exec_expected_target_tensors.clear();
    exec_expected_target_tensors.reserve(
        static_cast<size_t>(payload.batch_size) * static_cast<size_t>(hp_.num_exec_steps));

    for (int b = 0; b < payload.batch_size; ++b) {
        execution_trace_by_row[static_cast<size_t>(b)].clear();
        exec_outputs_per_row[static_cast<size_t>(b)].steps.clear();

        const bool row_exec_active = !payload.execution_active.empty()
            && payload.execution_active[static_cast<size_t>(b)];
        if (!row_exec_active) {
            trace_state_by_row[static_cast<size_t>(b)] = Tensor();
            continue;
        }

        auto& row_memory = exec_memories[static_cast<size_t>(b)];
        row_memory.allocate(
            hp_.num_slots,
            hp_.atom_embedding_dim,
            hp_.d_model,
            hp_.d_key,
            hp_.d_type,
            stream);
        row_memory.clear(stream);

        trace_state_by_row[static_cast<size_t>(b)] = Tensor::zeros({1, hp_.d_model}, stream, "trace_state_row");
        if (connect_parameter_graph) {
            trace_state_by_row[static_cast<size_t>(b)].requires_grad_();
            trace_state_by_row[static_cast<size_t>(b)].ensure_grad();
        } else {
            trace_state_by_row[static_cast<size_t>(b)].requires_grad = false;
        }
    }
}

// Device kernel: check if any value slot [S..S+V_val) has valid_mask >= 0.5.
// If none valid, sets error_flag via atomicMax to fail in finalizeStepOrThrow.
__global__ void kernelCheckAnyValueSlotValid(
    const float* __restrict__ valid_mask,
    int* __restrict__ error_flag,
    int S, int V_val, int stage_id
) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < V_val; ++i) {
        if (valid_mask[S + i] >= 0.5f) return;  // at least one valid — OK
    }
    atomicMax(error_flag, stage_id);  // no valid value slots
}

static void ensureBootstrappedValueSlotsOrThrow(
    ExecutionBlockLayer& layer,
    const ExecutionMemory& memory,
    cudaStream_t stream
) {
    const int V = layer.hp().num_slots;
    const int S = layer.hp().num_scratch_slots;
    const int V_val = V - S;
    EXEC_CHECK(V_val > 0, "executeStep: no value slots (V - S == 0)");

    // Uses existing persistent error flag — defers sync to finalizeStepOrThrow.
    // No per-step allocation or pipeline drain.
    kernelCheckAnyValueSlotValid<<<1, 1, 0, stream>>>(
        memory.valid_mask.data,
        LayerAccess::numericErrorFlag(layer),
        S, V_val, kStageSlotUninit);
    CUDA_CHECK_KERNEL();
}

namespace ExecutionBlockInternal {

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

    const int V = layer.hp().num_slots;
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
            layer.hp().num_slots,
            layer.hp().num_scratch_slots,
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
    const int dm = layer.hp().d_model;
    const int V_val = layer.hp().num_slots - layer.hp().num_scratch_slots;
    const int S = layer.hp().num_scratch_slots;

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
    const int V = layer.hp().num_slots;
    const int S = layer.hp().num_scratch_slots;
    const int V_val = V - S;
    int* d_exec_idx = LayerAccess::execIndices(layer);

    EXEC_CHECK(work.p_arg1.data != nullptr, "materializeSelectedOperands: p_arg1 is null");
    EXEC_CHECK(work.p_arg2.data != nullptr, "materializeSelectedOperands: p_arg2 is null");

    kernelArgmax1DInt<<<1, 32, 0, stream>>>(d_exec_idx, work.p_arg1.data, V_val);
    CUDA_CHECK_KERNEL();
    kernelArgmax1DInt<<<1, 32, 0, stream>>>(d_exec_idx + 1, work.p_arg2.data, V_val);
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
    const int V = layer.hp().num_slots;
    const int S = layer.hp().num_scratch_slots;
    const int dm = layer.hp().d_model;
    const int dk = layer.hp().d_key;
    const int ae = layer.hp().atom_embedding_dim;
    const int dt = layer.hp().d_type;
    int* d_exec_idx = LayerAccess::execIndices(layer);

    kernelArgmax1DInt<<<1, 32, 0, stream>>>(d_exec_idx + 3, work.p_write.data, V);
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

    const int V = layer.hp().num_slots;
    diag_out->state_after_values = Tensor::zeros({V, 1}, stream, "state_after_values");
    diag_out->state_after_valid = Tensor::zeros({1, V}, stream, "state_after_valid");
    CUDA_CHECK(cudaMemcpyAsync(diag_out->state_after_values.data, memory.values.data,
        V * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(diag_out->state_after_valid.data, memory.valid_mask.data,
        V * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Use Tensor::zeros (async cudaMemsetAsync) instead of synchronous cudaMalloc.
    auto hinge_discard = Tensor::zeros({1, 1}, stream, "memstream_hinge_discard");
    auto changed_count_tensor = Tensor::zeros({1, 1}, stream, "memstream_changed_count");
    // Reinterpret float* as int* for changed_count (single element, aligned)
    int* d_changed_count = reinterpret_cast<int*>(changed_count_tensor.data);

    kernelStateDeltaCheck<<<1, 1, 0, stream>>>(
        hinge_discard.data,
        d_changed_count,
        diag_out->state_before_values.data,
        diag_out->state_after_values.data,
        LayerAccess::execIndices(layer) + 3,
        V);
    CUDA_CHECK_KERNEL();

    kernelCheckMultiSlotMutation<<<1, 1, 0, stream>>>(
        d_changed_count,
        LayerAccess::numericErrorFlag(layer),
        kStageMultiSlotMutation);
    CUDA_CHECK_KERNEL();
    // hinge_discard and changed_count_tensor freed by Tensor RAII destructor
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

        // Emit execution record via module log system
        static const char* op_names[] = {"+", "-", "*", "/"};
        const char* op_str = (ri[2] >= 0 && ri[2] < 4) ? op_names[ri[2]] : "?";
        char msg[256];
        snprintf(msg, sizeof(msg),
            "[EXEC_RECORD_EQUATION] step=%d: slot[%d](%.4f) %s slot[%d](%.4f) = %.4f",
            step, ri[0], rf[0], op_str, ri[1], rf[1], rf[2]);
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::ExecutionBlock, msg);
    }

    if (h_error > 0) {
        char buf[384];
        snprintf(buf, sizeof(buf),
            "ExecutionBlock FATAL: invalid state at step %d — %s (stage id=%d)",
            step, stageIdToName(h_error), h_error);
        if (layer.hp().debug_mode)
            GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::ExecutionBlock,
                std::string("[ExecutionBlock debug] ") + buf);
        throw std::runtime_error(buf);
    }
}

Tensor crossAttentionReadImpl(
    ExecutionBlockLayer& layer,
    const Tensor& hidden_states,
    ExecutionMemory& memory,
    cudaStream_t stream,
    int token_offset,
    int row_tokens,
    float* d_gate_accum  // [2] device accumulator: [sum, count]. nullptr = skip.
) {
    using namespace autograd;

    const int dm = layer.hp().d_model;
    const int dk = layer.hp().d_key;
    const int hd = layer.hp().cross_attn_head_dim;
    const int V = layer.hp().num_slots;
    const int nv = V;
    const int topk = layer.hp().cross_attn_topk;
    const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(hd));

    // Non-owning view of H[token_offset : token_offset + row_tokens, :].
    // requires_grad=false: gradient w.r.t. H flows through the add() at the call site,
    // not through Q/gate projections (standard residual detach pattern).
    float* H_row = hidden_states.data + static_cast<size_t>(token_offset) * dm;
    auto H_view = Tensor::from_ptr(H_row, {row_tokens, dm}, stream, "exec_read_H_view");

    // Q = H_view @ W_Q_read  [row_tokens, hd]
    Tensor Q = matmul(H_view, layer.W_Q_read(), stream, nullptr, nullptr, false);

    // K_proj = key_embeds @ W_K_read  [nv, hd]
    Tensor K_proj = matmul(memory.key_embeds, layer.W_K_read(), stream, nullptr, nullptr, false);

    // V_proj = state_embeds @ W_V_read  [nv, hd]
    Tensor V_proj = matmul(memory.state_embeds, layer.W_V_read(), stream, nullptr, nullptr, false);

    // raw_scores = Q @ K_proj^T  [row_tokens, nv]
    Tensor raw_scores = matmul(Q, K_proj, stream, nullptr, nullptr, true);

    // Scale by 1/sqrt(hd) (constant — no gradient needed)
    Tensor scaled_scores = mul_scalar(raw_scores, inv_sqrt_d, stream);

    // Apply valid_mask + top-k masking (non-differentiable, like causal mask in attention).
    // Read tau to host for softmax temperature — matches original behavior.
    // (tau gradient can be added later with broadcast_mul primitive if needed.)
    float h_tau = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_tau, layer.tau().data, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (h_tau < 0.01f) h_tau = 0.01f;

    // Non-differentiable masking (like causal mask): valid_mask + top-k.
    // In-place on scaled_scores.data is safe — masked positions get 0 gradient
    // naturally from softmax (output ≈ 0 for -FLT_MAX inputs).
    kernelApplyValidMaskAndTopK<<<row_tokens, kBlockSize, 0, stream>>>(
        scaled_scores.data, memory.valid_mask.data,
        row_tokens, nv, topk);
    CUDA_CHECK_KERNEL();

    // Softmax over attention scores with tau temperature
    Tensor attn_weights = softmax(scaled_scores, h_tau, stream);

    // R = attn_weights @ V_proj  [row_tokens, hd]
    Tensor R = matmul(attn_weights, V_proj, stream, nullptr, nullptr, false);

    // proj = R @ W_O_read  [row_tokens, dm]
    Tensor proj = matmul(R, layer.W_O_read(), stream, nullptr, nullptr, false);

    // gate = sigmoid(H_view @ W_gate_read) [row_tokens, 1]
    // Composed from primitives: sigmoid(x) = reciprocal(add_scalar(exp(mul_scalar(x, -1)), 1))
    Tensor gate_logits = matmul(H_view, layer.W_gate_read(), stream, nullptr, nullptr, false);
    Tensor neg_logits = mul_scalar(gate_logits, -1.0f, stream);
    Tensor exp_neg = exp(neg_logits, stream);
    Tensor one_plus_exp = add_scalar(exp_neg, 1.0f, stream);
    Tensor gate = reciprocal(one_plus_exp, stream);

    // Accumulate gate statistics for telemetry (detached, no grad flow needed)
    if (d_gate_accum) {
        kernelAccumulateGateStats<<<1, kBlockSize, 0, stream>>>(
            d_gate_accum, gate.data, row_tokens);
        CUDA_CHECK_KERNEL();
    }

    // delta = gate * proj  [row_tokens, dm]  (broadcast [row_tokens,1] * [row_tokens,dm])
    Tensor delta = broadcast_row_mul(gate, proj, stream);

    // Usage update (detached telemetry — no autograd)
    kernelDecayedUsageUpdate<<<(nv + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        memory.usage.data, attn_weights.data, layer.hp().usage_decay, row_tokens, nv, V);
    CUDA_CHECK_KERNEL();

    return delta;
}

}  // namespace ExecutionBlockInternal

}  // namespace GRIM

#endif  // USE_CUDA

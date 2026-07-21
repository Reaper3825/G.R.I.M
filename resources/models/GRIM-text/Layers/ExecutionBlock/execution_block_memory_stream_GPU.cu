#include "execution_block_memory_stream_GPU.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;



namespace GRIM {

using namespace ExecutionBlockInternal;
using Forward::ExecutionBlockOutput;
using Forward::ExecutionBlockStepOutput;
using Forward::ExecutionRecord;

constexpr float kResidualFusionScale = 0.7071067811865475f; // 1 / sqrt(2)

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
    const float* __restrict__ selector_candidate_keys,
    const int* __restrict__ bootstrap_slot_to_pool_index,
    int num_pool_atoms,
    int* __restrict__ bridge_error,
    int V, int d_model, int d_key
) {
    const int slot = blockIdx.x;
    if (slot >= V || valid_mask[slot] == 0.0f) return;

    const float val = values[slot];
    int pool_index = -1;
    const bool bridge_enabled = selector_candidate_keys != nullptr;
    if (bridge_enabled) {
        pool_index = bootstrap_slot_to_pool_index[slot];
        if (pool_index < 0 || pool_index >= num_pool_atoms) {
            if (threadIdx.x == 0 && bridge_error) atomicCAS(bridge_error, 0, 1);
            pool_index = -1;
        }
    }
    extern __shared__ float smem[];
    for (int d = threadIdx.x; d < d_model; d += blockDim.x) {
        float emb_d = val * W_val2emb[d] + (b_val2emb ? b_val2emb[d] : 0.0f);
        if (pool_index >= 0) {
            emb_d = kResidualFusionScale *
                (emb_d + selector_candidate_keys[
                    static_cast<size_t>(pool_index) * d_model + d]);
        }
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

__global__ void kernelBuildValidSlotRoute(
    float* __restrict__ route,
    const float* __restrict__ mem_valid,
    int V_val, int S, int V
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V_val) return;
    const int slot = S + i;
    route[static_cast<size_t>(i) * V + slot] = mem_valid[slot];
}

__global__ void kernelMaskDenseRouteRows(
    float* __restrict__ route,
    const float* __restrict__ mem_valid,
    int rows, int cols, int S
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * cols;
    if (index >= count) return;
    const int row = static_cast<int>(index / cols);
    route[index] *= mem_valid[S + row];
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
        recordFirstExecutionError(error_flag, stage_invalid);
        out[0] = 0.0f;
        return;
    }
    int slot = S + r;
    if (slot < S || slot >= V) {
        recordFirstExecutionError(error_flag, stage_invalid);
        out[0] = 0.0f;
        return;
    }
    if (M_valid[slot] == 0.0f) {
        recordFirstExecutionError(error_flag, stage_uninit);
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
        recordFirstExecutionError(error_flag, stage_id);
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
        recordFirstExecutionError(error_flag, stage_id);
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

void ExecutionMemory::bind(
    Tensor& values_owner,
    Tensor& atom_embeds_owner,
    Tensor& state_embeds_owner,
    Tensor& valid_mask_owner,
    Tensor& usage_owner,
    Tensor& key_embeds_owner,
    Tensor& type_embed_owner,
    Tensor& recent_write_mask_owner)
{
    auto borrow = [](Tensor& owner) {
        EXEC_CHECK(owner.data != nullptr, "ExecutionMemory::bind: owner tensor is empty");
        Tensor view = Tensor::from_ptr(
            owner.data,
            owner.shape,
            false,
            owner.requires_grad,
            owner.name);
        view.stream = owner.stream;
        view.is_leaf = owner.is_leaf;
        return view;
    };

    values            = borrow(values_owner);
    atom_embeds       = borrow(atom_embeds_owner);
    state_embeds      = borrow(state_embeds_owner);
    valid_mask        = borrow(valid_mask_owner);
    usage             = borrow(usage_owner);
    key_embeds        = borrow(key_embeds_owner);
    type_embed        = borrow(type_embed_owner);
    recent_write_mask = borrow(recent_write_mask_owner);
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

void executionBlockBootstrapMemoryFromSlotMap(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionMemory& M,
    ExecutionBlockParameterTensors& parameters,
    const float* device_numeric_values,
    const int32_t* device_slot_map,
    const float* selector_candidate_keys,
    const int* bootstrap_slot_to_pool_index,
    int num_pool_atoms,
    int row_tokens,
    cudaStream_t stream)
{
    EXEC_CHECK(device_numeric_values != nullptr, "bootstrapMemoryFromSlotMap: device_numeric_values is null");
    EXEC_CHECK(device_slot_map != nullptr, "bootstrapMemoryFromSlotMap: device_slot_map is null");
    EXEC_CHECK(row_tokens > 0, "bootstrapMemoryFromSlotMap: row_tokens must be positive");
    const bool has_selector_keys = selector_candidate_keys != nullptr;
    const bool has_bridge_map = bootstrap_slot_to_pool_index != nullptr;
    EXEC_CHECK(has_selector_keys == has_bridge_map,
               "bootstrapMemoryFromSlotMap: selector keys and slot-to-pool map must be supplied together");
    if (has_selector_keys) {
        EXEC_CHECK(num_pool_atoms > 0,
                   "bootstrapMemoryFromSlotMap: selector bridge requires num_pool_atoms > 0");
    } else {
        EXEC_CHECK(num_pool_atoms == 0,
                   "bootstrapMemoryFromSlotMap: num_pool_atoms must be zero when selector bridge is absent");
    }
    auto& params = parameters;

    const int V = hp.num_slots;

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

    const int dm = hp.d_model;
    const int dk = hp.d_key;
    const size_t smem_bytes = dm * sizeof(float);
    int* d_bridge_error = nullptr;
    if (has_selector_keys) {
        cudaMallocOrThrow(
            reinterpret_cast<void**>(&d_bridge_error),
            sizeof(int),
            "bootstrap_selector_bridge_error");
        CUDA_CHECK(cudaMemsetAsync(d_bridge_error, 0, sizeof(int), stream));
    }
    kernelBootstrapSlotEmbeddings<<<V, kBlockSize, smem_bytes, stream>>>(
        M.state_embeds.data, M.key_embeds.data,
        M.values.data, M.valid_mask.data,
        params.W_value_to_emb.data, params.b_value_to_emb.data,
        params.W_key_proj.data,
        selector_candidate_keys,
        bootstrap_slot_to_pool_index,
        num_pool_atoms,
        d_bridge_error,
        V, dm, dk);
    CUDA_CHECK_KERNEL();
    if (d_bridge_error) {
        int h_bridge_error = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &h_bridge_error,
            d_bridge_error,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_bridge_error));
        if (h_bridge_error != 0) {
            throw std::runtime_error(
                "bootstrapMemoryFromSlotMap: a valid authored execution slot has no "
                "in-range selector-pool identity");
        }
    }

    const int dt = hp.d_type;
    kernelBootstrapTypeEmbed<<<V, kBlockSize, 0, stream>>>(
        M.type_embed.data, M.valid_mask.data,
        params.type_num_embed.data, V, dt);
    CUDA_CHECK_KERNEL();
}

namespace ExecutionBlockInternal {

void buildValueSlotCandidates(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    const ExecutionMemory& memory,
    const ExecutionBlockParameterTensors& parameters,
    const Batching::BatchPayload& payload,
    int batch_row,
    const std::vector<Forward::ExecutionRecord>& prior_records,
    const Tensor* selector_candidate_keys,
    cudaStream_t stream,
    StepWorkingSet& work
) {
    const int dm = hp.d_model;
    const int V = hp.num_slots;
    const int V_val = hp.num_slots - hp.num_scratch_slots;
    const int S = hp.num_scratch_slots;

    // Runtime register contents are hard state, so gather them as a detached
    // operand. Trainable identity is then composed around that state with
    // ordinary autograd primitives: route @ E_slot supplies absolute slot
    // identity, and the zero-valued key branch below restores the authored
    // selector-key edge without changing the forward value.
    Tensor runtime_state = Tensor::zeros(
        {V_val, dm}, stream, "exec_candidate_runtime_state");
    kernelGatherSlotOnlyHidden<<<V_val, kBlockSize, 0, stream>>>(
        runtime_state.data,
        memory.state_embeds.data,
        memory.valid_mask.data,
        V_val,
        S,
        dm);
    CUDA_CHECK_KERNEL();

    Tensor slot_route = Tensor::zeros(
        {V_val, V}, stream, "exec_candidate_slot_route");
    kernelBuildValidSlotRoute<<<
        (V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        slot_route.data, memory.valid_mask.data, V_val, S, V);
    CUDA_CHECK_KERNEL();

    Tensor slot_rows = autograd::matmul(
        slot_route, parameters.E_slot, stream);
    Tensor state_with_slot = autograd::add(
        runtime_state, slot_rows, stream);
    Tensor base_candidate = autograd::mul_scalar(
        state_with_slot, kResidualFusionScale, stream);

    if (selector_candidate_keys) {
        EXEC_CHECK(selector_candidate_keys->data != nullptr,
            "buildValueSlotCandidates: selector_candidate_keys has null data");
        EXEC_CHECK_SHAPE2(
            *selector_candidate_keys,
            "selector_candidate_keys (execution operand bridge)",
            selector_candidate_keys->shape.flat.rows,
            dm);
        const int num_pool_atoms = selector_candidate_keys->shape.flat.rows;
        EXEC_CHECK(num_pool_atoms > 0,
            "buildValueSlotCandidates: selector candidate pool is empty");
        EXEC_CHECK(batch_row >= 0 && batch_row < payload.batch_size,
            "buildValueSlotCandidates: batch_row out of range");
        EXEC_CHECK(payload.execution_slot_count == V,
            "buildValueSlotCandidates: payload execution-slot geometry mismatch");
        const size_t expected_map_size =
            static_cast<size_t>(payload.batch_size) * V;
        EXEC_CHECK(payload.bootstrap_slot_to_pool_index.size() == expected_map_size,
            "buildValueSlotCandidates: selector-to-slot map geometry mismatch");

        std::vector<uint8_t> overwritten(static_cast<size_t>(V), 0);
        for (const auto& record : prior_records) {
            EXEC_CHECK(record.write_slot >= S && record.write_slot < V,
                "buildValueSlotCandidates: prior write slot is outside value-slot range");
            overwritten[static_cast<size_t>(record.write_slot)] = 1;
        }

        std::vector<float> host_pool_route(
            static_cast<size_t>(V_val) * num_pool_atoms, 0.0f);
        const size_t row_map_offset = static_cast<size_t>(batch_row) * V;
        for (int i = 0; i < V_val; ++i) {
            const int slot = S + i;
            if (overwritten[static_cast<size_t>(slot)] != 0) continue;
            const int pool_index = payload.bootstrap_slot_to_pool_index[
                row_map_offset + static_cast<size_t>(slot)];
            if (pool_index < 0) continue;
            EXEC_CHECK(pool_index < num_pool_atoms,
                "buildValueSlotCandidates: selector pool index out of range");
            host_pool_route[
                static_cast<size_t>(i) * num_pool_atoms + pool_index] = 1.0f;
        }

        Tensor active_pool_route = Tensor::zeros(
            {V_val, num_pool_atoms}, stream, "exec_candidate_pool_route");
        CUDA_CHECK(cudaMemcpyAsync(
            active_pool_route.data,
            host_pool_route.data(),
            host_pool_route.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream));
        const size_t route_count = host_pool_route.size();
        kernelMaskDenseRouteRows<<<
            static_cast<int>((route_count + kBlockSize - 1) / kBlockSize),
            kBlockSize,
            0,
            stream>>>(
            active_pool_route.data,
            memory.valid_mask.data,
            V_val,
            num_pool_atoms,
            S);
        CUDA_CHECK_KERNEL();

        Tensor mapped_keys = autograd::matmul(
            active_pool_route, *selector_candidate_keys, stream);
        Tensor detached_keys = mapped_keys.detach(stream);
        Tensor neg_detached_keys = autograd::mul_scalar(
            detached_keys, -1.0f, stream);
        Tensor zero_value_key_edge = autograd::add(
            mapped_keys, neg_detached_keys, stream);
        Tensor scaled_key_edge = autograd::mul_scalar(
            zero_value_key_edge, 0.5f, stream);
        work.cand_hidden = autograd::add(
            base_candidate, scaled_key_edge, stream);
    } else {
        work.cand_hidden = std::move(base_candidate);
    }

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
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    const ExecutionMemory& memory,
    cudaStream_t stream,
    StepWorkingSet& work
) {
    const int V = hp.num_slots;
    const int S = hp.num_scratch_slots;
    const int V_val = V - S;
    int* d_exec_idx = diag.execIndices();

    work.v1 = Tensor::zeros({1, 1}, stream, "exec_v1");
    work.v1.requires_grad = work.p_arg1.requires_grad;
    work.v1.is_leaf = false;
    kernelReadSlotValueByRelIdx<<<1, 1, 0, stream>>>(
        work.v1.data, memory.values.data, memory.valid_mask.data, S, V, V_val,
        d_exec_idx, diag.numericErrorFlag(), kStageSlotInvalid, kStageSlotUninit);
    CUDA_CHECK_KERNEL();

    work.v2 = Tensor::zeros({1, 1}, stream, "exec_v2");
    work.v2.requires_grad = work.p_arg2.requires_grad;
    work.v2.is_leaf = false;
    kernelReadSlotValueByRelIdx<<<1, 1, 0, stream>>>(
        work.v2.data, memory.values.data, memory.valid_mask.data, S, V, V_val,
        d_exec_idx + 1, diag.numericErrorFlag(), kStageSlotInvalid, kStageSlotUninit);
    CUDA_CHECK_KERNEL();
}

void applyHardWriteback(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    ExecutionBlockParameterTensors& parameters,
    ExecutionMemory& memory,
    cudaStream_t stream,
    const StepWorkingSet& work
) {
    auto& params = parameters;
    const int V = hp.num_slots;
    const int S = hp.num_scratch_slots;
    const int dm = hp.d_model;
    const int dk = hp.d_key;
    const int ae = hp.atom_embedding_dim;
    const int dt = hp.d_type;
    int* d_exec_idx = diag.execIndices();

    kernelValidateWriteSlotDev<<<1, 1, 0, stream>>>(
        d_exec_idx + 3, S, V, diag.numericErrorFlag(), kStageWriteSlotInvalid);
    CUDA_CHECK_KERNEL();

    kernelHardWriteScalarDev<<<1, 1, 0, stream>>>(memory.values.data, d_exec_idx + 3, work.v_out.data);
    CUDA_CHECK_KERNEL();
    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(memory.state_embeds.data, d_exec_idx + 3, dm, work.state_new.data);
    CUDA_CHECK_KERNEL();
    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(memory.key_embeds.data, d_exec_idx + 3, dk, work.key_new.data);
    CUDA_CHECK_KERNEL();
    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(memory.atom_embeds.data, d_exec_idx + 3, ae, work.atom_new.data);
    CUDA_CHECK_KERNEL();
    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(memory.type_embed.data, d_exec_idx + 3, dt, params.type_num_embed.data);
    CUDA_CHECK_KERNEL();
    kernelSetValidMaskDev<<<1, 1, 0, stream>>>(memory.valid_mask.data, d_exec_idx + 3);
    CUDA_CHECK_KERNEL();
    kernelSetRecentWriteOneHotDev<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        memory.recent_write_mask.data, V, d_exec_idx + 3);
    CUDA_CHECK_KERNEL();
}

void captureStateAfterWriteAndCheckMutations(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    const ExecutionMemory& memory,
    ExecutionBlockStepOutput& forward_output,
    cudaStream_t stream
) {
    const int V = hp.num_slots;
    forward_output.state_after_values = Tensor::zeros({V, 1}, stream, "state_after_values");
    forward_output.state_after_valid = Tensor::zeros({1, V}, stream, "state_after_valid");
    CUDA_CHECK(cudaMemcpyAsync(forward_output.state_after_values.data, memory.values.data,
        V * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(forward_output.state_after_valid.data, memory.valid_mask.data,
        V * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Use Tensor::zeros (async cudaMemsetAsync) instead of synchronous cudaMalloc.
    auto hinge_discard = Tensor::zeros({1, 1}, stream, "memstream_hinge_discard");
    auto changed_count_tensor = Tensor::zeros({1, 1}, stream, "memstream_changed_count");
    // Reinterpret float* as int* for changed_count (single element, aligned)
    int* d_changed_count = reinterpret_cast<int*>(changed_count_tensor.data);

    kernelStateDeltaCheck<<<1, 1, 0, stream>>>(
        hinge_discard.data,
        d_changed_count,
        forward_output.state_before_values.data,
        forward_output.state_after_values.data,
        diag.execIndices() + 3,
        V);
    CUDA_CHECK_KERNEL();

    kernelCheckMultiSlotMutation<<<1, 1, 0, stream>>>(
        d_changed_count,
        diag.numericErrorFlag(),
        kStageMultiSlotMutation);
    CUDA_CHECK_KERNEL();
    // hinge_discard and changed_count_tensor freed by Tensor RAII destructor
}

void finalizeStepOrThrow(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    ExecutionBlockStepOutput& forward_output,
    int step,
    cudaStream_t stream
) {
    int h_error = 0;
    int write_slot = -1;
    int ri[3] = {0, 0, 0};
    float rf[3] = {0.0f, 0.0f, 0.0f};
    CUDA_CHECK(cudaMemcpyAsync(ri, diag.execRecordI(), 3 * sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(rf, diag.execRecordF(), 3 * sizeof(float),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&write_slot, diag.execIndices() + 3, sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&h_error, diag.numericErrorFlag(), sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    forward_output.record.arg1_slot = ri[0];
    forward_output.record.arg2_slot = ri[1];
    forward_output.record.op_id = ri[2];
    forward_output.record.write_slot = write_slot;
    forward_output.record.value_before_1 = rf[0];
    forward_output.record.value_before_2 = rf[1];
    forward_output.record.value_after = rf[2];

    // Emit execution record via module log system
    static const char* op_names[] = {"+", "-", "*", "/"};
    const char* op_str = (ri[2] >= 0 && ri[2] < 4) ? op_names[ri[2]] : "?";
    char msg[256];
    snprintf(msg, sizeof(msg),
        "[EXEC_RECORD_EQUATION] step=%d: slot[%d](%.4f) %s slot[%d](%.4f) -> slot[%d]=%.4f",
        step, ri[0], rf[0], op_str, ri[1], rf[1], write_slot, rf[2]);
    GRIM::Logging::EmitModuleInfo(
        GRIM::Logging::ModuleId::ExecutionBlock, msg);

    if (h_error > 0) {
        char buf[640];
        snprintf(buf, sizeof(buf),
            "ExecutionBlock FATAL: invalid state at step %d - %s (stage id=%d); "
            "decision=%s; equation=slot[%d](%.9g) %s slot[%d](%.9g) -> "
            "slot[%d]=%.9g; magnitude_limit=%.9g",
            step, stageIdToName(h_error), h_error,
            forward_output.teacher_forced_transition ? "teacher" : "model",
            ri[0], static_cast<double>(rf[0]), op_str,
            ri[1], static_cast<double>(rf[1]), write_slot,
            static_cast<double>(rf[2]), static_cast<double>(hp.magnitude_limit));
        if (hp.debug_mode)
            GRIM::Logging::EmitModuleError(GRIM::Logging::ModuleId::ExecutionBlock,
                std::string("[ExecutionBlock debug] ") + buf);
        throw std::runtime_error(buf);
    }
}

Tensor crossAttentionReadImpl(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockParameterTensors& parameters,
    const Tensor& hidden_states,
    ExecutionMemory& memory,
    cudaStream_t stream,
    int token_offset,
    int row_tokens,
    float* d_gate_accum  // [2] device accumulator: [sum, count]. nullptr = skip.
) {
    using namespace autograd;
    auto& params = parameters;

    const int dm = hp.d_model;
    const int hd = hp.cross_attn_head_dim;
    const int V = hp.num_slots;
    const int nv = V;
    const int topk = hp.cross_attn_topk;
    const float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(hd));

    // Non-owning view of H[token_offset : token_offset + row_tokens, :].
    // requires_grad=false: gradient w.r.t. H flows through the add() at the call site,
    // not through Q/gate projections (standard residual detach pattern).
    float* H_row = hidden_states.data + static_cast<size_t>(token_offset) * dm;
    auto H_view = Tensor::from_ptr(H_row, {row_tokens, dm}, stream, "exec_read_H_view");

    // Q = H_view @ W_Q_read  [row_tokens, hd]
    Tensor Q = matmul(H_view, params.W_Q_read, stream);

    // K_proj = key_embeds @ W_K_read  [nv, hd]
    Tensor K_proj = matmul(memory.key_embeds, params.W_K_read, stream);

    // V_proj = state_embeds @ W_V_read  [nv, hd]
    Tensor V_proj = matmul(memory.state_embeds, params.W_V_read, stream);

    // raw_scores = Q @ K_proj^T  [row_tokens, nv]
    Tensor raw_scores = matmul(Q, K_proj, stream, true);

    // Scale by 1/sqrt(hd) (constant — no gradient needed)
    Tensor scaled_scores = mul_scalar(raw_scores, inv_sqrt_d, stream);

    // Apply valid_mask + top-k masking (non-differentiable, like causal mask in attention).
    // Read tau to host for softmax temperature — matches original behavior.
    // (tau gradient can be added later with broadcast_mul primitive if needed.)
    float h_tau = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_tau, params.tau.data, sizeof(float), cudaMemcpyDeviceToHost, stream));
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
    Tensor R = matmul(attn_weights, V_proj, stream);

    // proj = R @ W_O_read  [row_tokens, dm]
    Tensor proj = matmul(R, params.W_O_read, stream);

    // gate = sigmoid(H_view @ W_gate_read) [row_tokens, 1]
    // Composed from primitives: sigmoid(x) = reciprocal(add_scalar(exp(mul_scalar(x, -1)), 1))
    Tensor gate_logits = matmul(H_view, params.W_gate_read, stream);
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
        memory.usage.data, attn_weights.data, hp.usage_decay, row_tokens, nv, V);
    CUDA_CHECK_KERNEL();

    return delta;
}

}  // namespace ExecutionBlockInternal

}  // namespace GRIM

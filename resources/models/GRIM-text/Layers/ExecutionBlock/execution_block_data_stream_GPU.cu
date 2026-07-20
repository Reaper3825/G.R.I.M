#include "execution_block_data_stream_GPU.hpp"
#include "execution_block_memory_stream_GPU.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/TensorContract/GradFns/ExecutionBlockInjectGradFn.hpp"
#include "../../Shared/TensorContract/GradFns/GatedTraceUpdateGradFn.hpp"
#include "../../Shared/TensorContract/GradFns/RecordEncodeGradFn.hpp"
#include "../../Shared/TensorContract/GradFns/ReduceMeanGradFn.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM::ExecutionBlockInternal {

using GRIM::Forward::ExecutionBlockStepOutput;
using GRIM::Forward::ExecutionGateOutput;
using GRIM::Forward::ExecutionRecord;
using GRIM::autograd::ExecutionBlockInjectGradFn;
using GRIM::autograd::GatedTraceUpdateGradFn;
using GRIM::autograd::RecordEncodeGradFn;
using GRIM::autograd::ReduceMeanGradFn;

// Check that execution begins with at least one initialized value slot.
__global__ void kernelCheckAnyValueSlotValid(
    const float* __restrict__ valid_mask,
    int* __restrict__ error_flag,
    int S, int V_val, int stage_id
) {
    if (threadIdx.x != 0) return;
    for (int i = 0; i < V_val; ++i) {
        if (valid_mask[S + i] >= 0.5f) return;
    }
    recordFirstExecutionError(error_flag, stage_id);
}

__global__ void kernelValidateStateBearingSlots(
    const uint8_t* __restrict__ atom_mask,
    const int32_t* __restrict__ slot_map,
    const float* __restrict__ valid_mask,
    int row_tokens, int V, int S,
    int* __restrict__ error_flag,
    int stage_invalid,
    int stage_uninit
) {
    // Only compiled bootstrap literals are state-bearing. Ordinary numeric
    // atoms keep slot=-1 and remain LM-owned.
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= row_tokens) return;

    const int slot = slot_map[pos];
    if (slot == -1) return;
    if (atom_mask[pos] == 0 || slot < S || slot >= V) {
        recordFirstExecutionError(error_flag, stage_invalid);
    } else if (valid_mask[slot] == 0.0f) {
        recordFirstExecutionError(error_flag, stage_uninit);
    }
}

// ── Warp-level reduction primitives (all 32 threads must participate) ──
__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__device__ __forceinline__ float warpBroadcast(float val) {
    return __shfl_sync(0xffffffff, val, 0);
}
__device__ __forceinline__ int warpReduceOr(int val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        val |= __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void kernelEntropy(
    float* __restrict__ out,
    const float* __restrict__ probs,
    int N
) {
    const int tid = threadIdx.x;
    float ent = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float p = probs[i];
        if (p > 1e-10f)
            ent -= p * logf(p + 1e-10f);
    }
    ent = warpReduceSum(ent);
    if (tid == 0) out[0] = ent;
}

__global__ void kernelAccumScalar(
    float* __restrict__ out,
    const float* __restrict__ in
) {
    if (threadIdx.x == 0)
        out[0] += in[0];
}

__global__ void kernelCheckFinite(
    const float* __restrict__ data,
    int N,
    int* __restrict__ error_flag,
    int stage_id,
    float magnitude_limit
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float val = data[i];
    if (isnan(val) || isinf(val) || fabsf(val) > magnitude_limit)
        recordFirstExecutionError(error_flag, stage_id);
}

__global__ void kernelValidateSoftmax(
    const float* __restrict__ probs,
    int N,
    int* __restrict__ error_flag,
    int stage_id
) {
    const int tid = threadIdx.x;
    float local_sum = 0.0f;
    int local_bad = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float p = probs[i];
        if (isnan(p) || isinf(p) || p < 0.0f) local_bad = 1;
        local_sum += p;
    }
    local_sum = warpReduceSum(local_sum);
    local_bad = warpReduceOr(local_bad);
    if (tid == 0 && (local_bad || fabsf(local_sum - 1.0f) > 1e-3f))
        recordFirstExecutionError(error_flag, stage_id);
}

// Masked reduce-mean: exclude atom positions from the mean so that numeric
// surface features in H cannot leak into execution decision context.
// When atom_mask is nullptr the kernel falls back to the unmasked path.
__global__ void kernelReduceMeanForward(
    float* __restrict__ out,
    const float* __restrict__ H,
    int total_tokens, int d_model,
    const uint8_t* __restrict__ atom_mask  // [total_tokens] or nullptr
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_model) return;
    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < total_tokens; ++i) {
        if (atom_mask && atom_mask[i] != 0) continue;  // skip atom positions
        sum += H[static_cast<size_t>(i) * d_model + j];
        ++count;
    }
    // Guard: if every position is an atom, fall back to full mean rather than
    // dividing by zero.  This can happen during single-token decode of an atom.
    if (count == 0) {
        for (int i = 0; i < total_tokens; ++i)
            sum += H[static_cast<size_t>(i) * d_model + j];
        count = total_tokens;
    }
    out[j] = sum / static_cast<float>(count);
}

__global__ void kernelComputeEntropyScalar(
    float* __restrict__ out,
    const float* __restrict__ probs,
    int N
) {
    const int tid = threadIdx.x;
    float ent = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float p = probs[i];
        if (p > 1e-10f)
            ent -= p * logf(p + 1e-10f);
    }
    ent = warpReduceSum(ent);
    // Normalize: ent / log(N) ∈ [0,1] so diagnostics are comparable across N.
    if (tid == 0) {
        float max_ent = logf(static_cast<float>(N));
        out[0] = (max_ent > 1e-10f) ? (ent / max_ent) : 0.0f;
    }
}

__global__ void kernelComputeMax(
    float* __restrict__ out,
    const float* __restrict__ data,
    int N
) {
    const int tid = threadIdx.x;
    float mx = -1e30f;
    for (int i = tid; i < N; i += blockDim.x)
        mx = fmaxf(mx, data[i]);
    mx = warpReduceMax(mx);
    if (tid == 0) out[0] = mx;
}

__global__ void kernelApplyLogitMask(
    float* __restrict__ masked_logits,
    const float* __restrict__ logits,
    const float* __restrict__ mask,
    int C
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    masked_logits[i] = logits[i] + (1.0f - mask[i]) * (-1e9f);
}

// [DELETED] kernelOperandConsistencyGrad — removed per Fix #1.
// Value-path gradients must NOT leak into arg selection via any route.
// Selection CE from TeacherSelectionTargets is the ONLY arg selection signal.

__global__ void kernelFourOps(
    float* __restrict__ results,
    const float* __restrict__ pv1,
    const float* __restrict__ pv2,
    float eps,
    int* __restrict__ div_clamp_counter,
    int* __restrict__ div_invalid_flag    // [1] per-step: set to 1 if division was clamped
) {
    const int k = threadIdx.x;
    if (k >= 4) return;
    float v1 = pv1[0];
    float v2 = pv2[0];
    switch (k) {
        case 0: results[0] = v1 + v2; break;
        case 1: results[1] = v1 - v2; break;
        case 2: results[2] = v1 * v2; break;
        case 3: {
            float abs_v2 = fabsf(v2);
            float denom = (abs_v2 > eps) ? v2 : copysignf(eps, v2);
            if (abs_v2 <= eps) {
                if (div_clamp_counter) atomicAdd(div_clamp_counter, 1);
                if (div_invalid_flag) div_invalid_flag[0] = 1;
            }
            results[3] = v1 / denom;
            break;
        }
    }
}

__global__ void kernelArgmax1DIntData(
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
    // Warp-parallel argmax: exchange (val, idx) pairs
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
        if (other_val > best_val) { best_val = other_val; best_idx = other_idx; }
    }
    if (tid == 0) out_idx[0] = best_idx;
}

__global__ void kernelSetHardReadAndOpDecision(
    int* __restrict__ exec_indices,
    int arg1_rel,
    int arg2_rel,
    int op_id
) {
    if (threadIdx.x != 0) return;
    exec_indices[0] = arg1_rel;
    exec_indices[1] = arg2_rel;
    exec_indices[2] = op_id;
}

__global__ void kernelSetHardWriteDecision(
    int* __restrict__ exec_indices,
    int write_slot
) {
    if (threadIdx.x != 0) return;
    exec_indices[3] = write_slot;
}

__global__ void kernelHardPickOpForward(
    float* __restrict__ v_out,
    const float* __restrict__ results,
    const int* __restrict__ op_idx,
    int num_ops
) {
    if (threadIdx.x != 0) return;
    int k = op_idx[0];
    if (k < 0 || k >= num_ops)
        k = 0;
    v_out[0] = results[k];
}

__global__ void kernelAssembleExecRecord(
    int* __restrict__ out_i,
    float* __restrict__ out_f,
    const int* __restrict__ rel_arg1,
    const int* __restrict__ rel_arg2,
    const int* __restrict__ op_idx,
    const float* __restrict__ v1,
    const float* __restrict__ v2,
    const float* __restrict__ v_after,
    int S
) {
    if (threadIdx.x != 0) return;
    out_i[0] = S + rel_arg1[0];
    out_i[1] = S + rel_arg2[0];
    out_i[2] = op_idx[0];
    out_f[0] = v1[0];
    out_f[1] = v2[0];
    out_f[2] = v_after[0];
}

__global__ void kernelInjectResultSlot(
    float* __restrict__ H,
    const float* __restrict__ result_emb,
    const float* __restrict__ w_gate,
    float inv_sqrt_d,
    int result_slot, int d_model,
    float gate_temp,
    float* __restrict__ save_gate,
    float* __restrict__ save_H_pre
) {
    float* h_slot = H + static_cast<size_t>(result_slot) * d_model;
    for (int j = threadIdx.x; j < d_model; j += blockDim.x)
        save_H_pre[j] = h_slot[j];

    __shared__ float s_gate;
    if (threadIdx.x == 0) {
        float logit = 0.0f;
        for (int j = 0; j < d_model; ++j)
            logit += h_slot[j] * w_gate[j];
        logit *= gate_temp;
        float g = 1.0f / (1.0f + expf(-logit));
        s_gate = g;
        save_gate[0] = g;
    }
    __syncthreads();

    float scale = inv_sqrt_d * s_gate;
    for (int j = threadIdx.x; j < d_model; j += blockDim.x)
        h_slot[j] += scale * result_emb[j];
}

__global__ void kernelEncodeRecords(
    float* __restrict__ out,
    const float* __restrict__ E_slot,
    const float* __restrict__ E_op,
    const float* __restrict__ W_scal,
    const float* __restrict__ b_scal,
    const int* __restrict__ slot1_ids,
    const int* __restrict__ slot2_ids,
    const int* __restrict__ op_ids,
    const float* __restrict__ scalars,
    int N, int num_slots, int num_ops, int d_model
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * d_model;
    if (idx >= total) return;
    int i = idx / d_model;
    int j = idx % d_model;

    float val = b_scal ? b_scal[j] : 0.0f;
    int s1 = slot1_ids[i];
    if (s1 >= 0 && s1 < num_slots) val += E_slot[s1 * d_model + j];
    int s2 = slot2_ids[i];
    if (s2 >= 0 && s2 < num_slots) val += E_slot[s2 * d_model + j];
    int op = op_ids[i];
    if (op >= 0 && op < num_ops) val += E_op[op * d_model + j];
    for (int k = 0; k < 3; ++k)
        val += scalars[i * 3 + k] * W_scal[k * d_model + j];
    out[idx] = val;
}

__global__ void kernelAddVectors(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a[i] + b[i];
}

// ─── Fix #5: Gated Trace Update ─────────────────────────────────────────────
// Forward: out[i] = sigmoid(gate_logits[i]) * old_trace[i]
//                  + (1 - sigmoid(gate_logits[i])) * candidate[i]
// Saves gate_vals for backward.
__global__ void kernelGatedTraceUpdate(
    float* __restrict__ out,
    float* __restrict__ gate_vals,
    const float* __restrict__ old_trace,
    const float* __restrict__ candidate,
    const float* __restrict__ gate_logits,
    int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = 1.0f / (1.0f + expf(-gate_logits[i]));
    gate_vals[i] = g;
    out[i] = g * old_trace[i] + (1.0f - g) * candidate[i];
}

__global__ void kernelEncodeScalarToAtomEmbed(
    float* __restrict__ out,
    const float* __restrict__ scalar,
    int atom_dim
) {
    int d = threadIdx.x;
    if (d >= atom_dim) return;

    float v = *scalar;
    float result = 0.0f;
    if (d >= 16 && d < 32) {
        int bit = d - 16;
        float log_mag = log2f(fabsf(v) + 1.0f);
        float freq = static_cast<float>(bit + 1) * 0.5f;
        result = 0.5f * sinf(log_mag * freq);
    } else if (d >= 32 && d < 40) {
        int feat = d - 32;
        if (feat == 0) {
            result = (v < 0.0f) ? 0.5f : -0.5f;
        } else {
            int int_val = static_cast<int>(fabsf(v));
            result = ((int_val >> (feat - 1)) & 1) ? 0.3f : -0.3f;
        }
    }
    out[d] = result;
}

__global__ void kernelNormalizeUsage(
    float* __restrict__ out,
    const float* __restrict__ usage,
    int V
) {
    const int tid = threadIdx.x;
    // Pass 1: warp-parallel max
    float mx = 0.0f;
    for (int i = tid; i < V; i += blockDim.x)
        mx = fmaxf(mx, usage[i]);
    mx = warpReduceMax(mx);
    mx = warpBroadcast(mx);  // all threads need max for normalization
    float inv = 1.0f / (mx + kEps);
    // Pass 2: parallel normalize
    for (int i = tid; i < V; i += blockDim.x)
        out[i] = usage[i] * inv;
}

// Element-wise negate: out[i] = -in[i]
__global__ void kernelNegateVec(float* __restrict__ out, const float* __restrict__ in, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = -in[i];
}

// [DELETED] kernelComputeWriteBias — removed per Fix #4.
// Non-differentiable bonus_bias created forward/backward mismatch.
// Write slot selection is now pure CE classification.

__global__ void kernelMaskScratchSlots(
    float* __restrict__ logits,
    int S
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S) logits[i] = -1e9f;
}

static void copyStepDiagnostics(const StepWorkingSet& work,
                                ExecutionBlockStepOutput& forward_output,
                                int V_val,
                                int nop,
                                int V,
                                int dm,
                                cudaStream_t stream) {
    forward_output.p_arg1 = Tensor::zeros({1, V_val}, stream);
    cudaMemcpyAsync(forward_output.p_arg1.data, work.p_arg1.data, V_val * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    forward_output.p_arg2 = Tensor::zeros({1, V_val}, stream);
    cudaMemcpyAsync(forward_output.p_arg2.data, work.p_arg2.data, V_val * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    forward_output.p_op = Tensor::zeros({1, nop}, stream);
    cudaMemcpyAsync(forward_output.p_op.data, work.p_op.data, nop * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    forward_output.p_write = Tensor::zeros({1, V}, stream);
    cudaMemcpyAsync(forward_output.p_write.data, work.p_write.data, V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    forward_output.v_out = Tensor::zeros({1, 1}, stream);
    cudaMemcpyAsync(forward_output.v_out.data, work.v_out.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);
    forward_output.result_emb = Tensor::zeros({1, dm}, stream);
    cudaMemcpyAsync(forward_output.result_emb.data, work.result_emb.data, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

static void collectStepMetrics(const HyperParameters::ExecutionBlockConstructionHP& hp,
                               ExecutionBlockDiagnosticsBuffers& diag,
                               const StepWorkingSet& work,
                               ExecutionBlockStepOutput& forward_output,
                               int V_val,
                               int nop,
                               int V,
                               cudaStream_t stream) {
    if (!hp.debug_mode) return;

    float d_metrics[7];
    float* d_buf = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_buf), 7 * sizeof(float), "datastream_diag_buf");

    kernelComputeEntropyScalar<<<1, kWarpSize, 0, stream>>>(d_buf + 0, work.p_arg1.data, V_val);
    kernelComputeEntropyScalar<<<1, kWarpSize, 0, stream>>>(d_buf + 1, work.p_arg2.data, V_val);
    kernelComputeEntropyScalar<<<1, kWarpSize, 0, stream>>>(d_buf + 2, work.p_op.data, nop);
    kernelComputeEntropyScalar<<<1, kWarpSize, 0, stream>>>(d_buf + 3, work.p_write.data, V);
    kernelComputeMax<<<1, kWarpSize, 0, stream>>>(d_buf + 4, work.p_write.data, V);
    CUDA_CHECK(cudaMemcpyAsync(d_metrics, d_buf, 5 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    int div_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&div_count, diag.divClampCount(), sizeof(int), cudaMemcpyDeviceToHost, stream));

    float op_dist[4] = {};
    int op_copy = (nop <= 4) ? nop : 4;
    CUDA_CHECK(cudaMemcpyAsync(op_dist, work.p_op.data, op_copy * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    forward_output.metrics.arg1_entropy = d_metrics[0];
    forward_output.metrics.arg2_entropy = d_metrics[1];
    forward_output.metrics.op_entropy = d_metrics[2];
    forward_output.metrics.write_entropy = d_metrics[3];
    forward_output.metrics.max_p_write = d_metrics[4];
    forward_output.metrics.div_clamp_count = div_count;
    for (int k = 0; k < 4; ++k)
        forward_output.metrics.op_distribution[k] = (k < op_copy) ? op_dist[k] : 0.0f;

    const auto normalized_entropy = [](float entropy, int choices) {
        const float max_entropy = logf(static_cast<float>(choices));
        return max_entropy > 1e-10f ? entropy / max_entropy : 0.0f;
    };
    const float arg1_entropy_norm = normalized_entropy(d_metrics[0], V_val);
    const float arg2_entropy_norm = normalized_entropy(d_metrics[1], V_val);
    const float op_entropy_norm = normalized_entropy(d_metrics[2], nop);
    const bool low_selection_entropy =
        arg1_entropy_norm < hp.entropy_collapse_threshold ||
        arg2_entropy_norm < hp.entropy_collapse_threshold ||
        op_entropy_norm < hp.entropy_collapse_threshold;
    const bool high_write_confidence = d_metrics[4] > hp.write_collapse_threshold;
    if (low_selection_entropy || high_write_confidence) {
        char warning[384];
        snprintf(warning, sizeof(warning),
            "[ExecutionBlock diagnostic] confident selection; continuing training | "
            "H_norm(arg1)=%.4f H_norm(arg2)=%.4f H_norm(op)=%.4f "
            "max_p_write=%.4f thresholds(entropy=%.4f,write=%.4f)",
            arg1_entropy_norm, arg2_entropy_norm, op_entropy_norm,
            d_metrics[4], hp.entropy_collapse_threshold, hp.write_collapse_threshold);
        GRIM::Logging::EmitModuleWarning(
            GRIM::Logging::ModuleId::ExecutionBlock, warning);
    }

    // Emit per-step metrics via module log system
    {
        char msg[384];
        snprintf(msg, sizeof(msg),
            "[EXEC_STEP_EQUATION] entropy: H(arg1)=%.4f H(arg2)=%.4f H(op)=%.4f H(write)=%.4f "
            "max_p_write=%.4f div_clamps=%d op_dist=[%.3f,%.3f,%.3f,%.3f]",
            d_metrics[0], d_metrics[1], d_metrics[2], d_metrics[3],
            d_metrics[4], div_count,
            forward_output.metrics.op_distribution[0],
            forward_output.metrics.op_distribution[1],
            forward_output.metrics.op_distribution[2],
            forward_output.metrics.op_distribution[3]);
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::ExecutionBlock, msg);
    }

    cudaFree(d_buf);
}

void predictExecutionGateImpl(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockParameterTensors& parameters,
    Tensor& H,
    const Batching::BatchPayload& payload,
    int batch_row,
    cudaStream_t stream,
    ExecutionGateOutput* output)
{
    if (!output) {
        throw std::runtime_error("predictExecutionGateImpl: output is NULL");
    }
    if (batch_row < 0 || batch_row >= payload.batch_size) {
        throw std::runtime_error("predictExecutionGateImpl: batch_row out of range");
    }
    if (payload.execution_prompt_end_positions.empty()) {
        throw std::runtime_error("predictExecutionGateImpl: execution_prompt_end_positions is empty");
    }

    const int query_pos = payload.execution_prompt_end_positions[static_cast<size_t>(batch_row)];
    const int row_tokens = payload.seq_lengths[static_cast<size_t>(batch_row)];
    if (query_pos < 0 || query_pos >= row_tokens) {
        throw std::runtime_error(
            "predictExecutionGateImpl: planner query position out of row bounds");
    }

    const int dm = hp.d_model;
    const int absolute_query = batch_row * payload.max_seq_len + query_pos;
    Tensor query = Tensor::zeros({1, dm}, stream, "exec_gate_query");
    query.requires_grad = true;
    query.is_leaf = false;
    kernelReduceMeanForward<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        query.data,
        H.data + static_cast<size_t>(absolute_query) * dm,
        1,
        dm,
        nullptr);
    CUDA_CHECK_KERNEL();
    {
        auto gather_fn = std::make_shared<ReduceMeanGradFn>();
        gather_fn->capture(H, payload.total_tokens, dm, stream, absolute_query, 1, nullptr);
        query.grad_fn = gather_fn;
    }

    Tensor logits = autograd::matmul(query, parameters.W_execute, stream);
    logits = autograd::add(logits, parameters.b_execute, stream);
    output->probabilities = autograd::softmax(logits, 1.0f, stream);
    output->logits = std::move(logits);
}

namespace {

const Execution::TeacherStep& requireTeacherForcedStep(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    const Batching::BatchPayload& payload,
    int batch_row,
    int step
) {
    EXEC_CHECK(payload.isTraining(),
        "teacher-forced transition requested outside Training mode");
    EXEC_CHECK(batch_row >= 0 && batch_row < payload.batch_size,
        "teacher-forced transition batch row is out of range");
    EXEC_CHECK(static_cast<int>(payload.teacher_steps.size()) == payload.batch_size,
        "teacher-forced transition requires one BatchPayload.teacher_steps row per batch row");
    EXEC_CHECK(static_cast<int>(payload.teacher_step_mask.size()) == payload.batch_size,
        "teacher-forced transition requires one BatchPayload.teacher_step_mask row per batch row");

    const auto& teacher_row = payload.teacher_steps[static_cast<size_t>(batch_row)];
    const auto& mask_row = payload.teacher_step_mask[static_cast<size_t>(batch_row)];
    EXEC_CHECK(step >= 0 && step < static_cast<int>(teacher_row.size()),
        "teacher-forced transition step is absent from BatchPayload.teacher_steps");
    EXEC_CHECK(step < static_cast<int>(mask_row.size())
        && mask_row[static_cast<size_t>(step)] != 0,
        "teacher-forced transition cannot consume a padded teacher step");

    const auto& teacher = teacher_row[static_cast<size_t>(step)];
    EXEC_CHECK(teacher.arg1_slot >= hp.num_scratch_slots
        && teacher.arg1_slot < hp.num_slots,
        "teacher arg1 slot is outside the configured value-slot range [S,V)");
    EXEC_CHECK(teacher.arg2_slot >= hp.num_scratch_slots
        && teacher.arg2_slot < hp.num_slots,
        "teacher arg2 slot is outside the configured value-slot range [S,V)");
    EXEC_CHECK(teacher.op_id >= 0 && teacher.op_id < hp.num_ops,
        "teacher op id is outside the configured operation range");
    EXEC_CHECK(teacher.write_slot >= hp.num_scratch_slots
        && teacher.write_slot < hp.num_slots,
        "teacher write slot is outside the configured value-slot range [S,V)");
    return teacher;
}

}  // namespace

void materializeHardReadAndOpDecision(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    const Batching::BatchPayload& payload,
    int batch_row,
    int step,
    cudaStream_t stream,
    const StepWorkingSet& work
) {
    EXEC_CHECK(work.p_arg1.data != nullptr,
        "materializeHardReadAndOpDecision: p_arg1 is null");
    EXEC_CHECK(work.p_arg2.data != nullptr,
        "materializeHardReadAndOpDecision: p_arg2 is null");
    EXEC_CHECK(work.p_op.data != nullptr,
        "materializeHardReadAndOpDecision: p_op is null");

    int* d_exec_idx = diag.execIndices();
    if (payload.isTraining() && hp.teacher_force_transitions) {
        const auto& teacher = requireTeacherForcedStep(hp, payload, batch_row, step);
        kernelSetHardReadAndOpDecision<<<1, 1, 0, stream>>>(
            d_exec_idx,
            teacher.arg1_slot - hp.num_scratch_slots,
            teacher.arg2_slot - hp.num_scratch_slots,
            teacher.op_id);
        CUDA_CHECK_KERNEL();
        return;
    }

    const int value_slots = hp.num_slots - hp.num_scratch_slots;
    kernelArgmax1DIntData<<<1, kWarpSize, 0, stream>>>(
        d_exec_idx, work.p_arg1.data, value_slots);
    CUDA_CHECK_KERNEL();
    kernelArgmax1DIntData<<<1, kWarpSize, 0, stream>>>(
        d_exec_idx + 1, work.p_arg2.data, value_slots);
    CUDA_CHECK_KERNEL();
    kernelArgmax1DIntData<<<1, kWarpSize, 0, stream>>>(
        d_exec_idx + 2, work.p_op.data, hp.num_ops);
    CUDA_CHECK_KERNEL();
}

void materializeHardWriteDecision(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    const Batching::BatchPayload& payload,
    int batch_row,
    int step,
    cudaStream_t stream,
    const StepWorkingSet& work
) {
    EXEC_CHECK(work.p_write.data != nullptr,
        "materializeHardWriteDecision: p_write is null");

    int* d_exec_idx = diag.execIndices();
    if (payload.isTraining() && hp.teacher_force_transitions) {
        const auto& teacher = requireTeacherForcedStep(hp, payload, batch_row, step);
        kernelSetHardWriteDecision<<<1, 1, 0, stream>>>(
            d_exec_idx, teacher.write_slot);
        CUDA_CHECK_KERNEL();
        return;
    }

    kernelArgmax1DIntData<<<1, kWarpSize, 0, stream>>>(
        d_exec_idx + 3, work.p_write.data, hp.num_slots);
    CUDA_CHECK_KERNEL();
}

void executeStepCoordinatorImpl(
    const HyperParameters::ExecutionBlockConstructionHP& hp,
    ExecutionBlockDiagnosticsBuffers& diag,
    ExecutionBlockParameterTensors& parameters,
    Tensor& H,
    ExecutionMemory& memory,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int batch_row,
    int step,
    float temperature,
    cudaStream_t stream,
    ExecutionBlockStepOutput& forward_output,
    GRIM::Forward::RecordEncodeBackwardStaging& record_encode_backward_staging,
    Tensor& trace_state,
    const std::vector<ExecutionRecord>& prior_records
) {
    // ModelForwardOutputs owns the supplied backward staging. The step output
    // remains diagnostics/live-loss output and is not the staging owner.
    auto& params = parameters;
    const int dm = hp.d_model;
    const int V = hp.num_slots;
    const int S = hp.num_scratch_slots;
    const int V_val = V - S;
    const int dk = hp.d_key;
    const int nop = hp.num_ops;
    const int ae = hp.atom_embedding_dim;
    const int vid = hp.value_decode_input_dim;
    const int vhd = hp.value_decode_hidden_dim;
    const float operand_selection_scale =
        HyperParameters::computeExecutionOperandSelectionScale(
            dm, "executeStepCoordinatorImpl");
    int* d_exec_idx = diag.execIndices();
    int* d_exec_record_i = diag.execRecordI();
    float* d_exec_record_f = diag.execRecordF();
    forward_output.teacher_forced_transition =
        payload.isTraining() && hp.teacher_force_transitions;

    // Row-local device slot map: derived from BatchDeviceBindings (host BatchPayload no
    // longer carries device pointers — see Shared/Batching/BatchDeviceBindings.hpp).
    const int32_t* d_slot_map_row = bindings.d_token_to_slot_map
        + static_cast<size_t>(batch_row) * payload.max_seq_len;
    const int row_tokens = payload.seq_lengths[static_cast<size_t>(batch_row)];
    if (payload.execution_prompt_lengths.empty()) {
        throw std::runtime_error(
            "executeStepCoordinatorImpl: execution_prompt_lengths is empty");
    }
    const int prompt_tokens =
        payload.execution_prompt_lengths[static_cast<size_t>(batch_row)];
    if (prompt_tokens <= 0 || prompt_tokens > row_tokens) {
        throw std::runtime_error(
            "executeStepCoordinatorImpl: execution prompt length out of row bounds");
    }

    // Row slice of the GLOBAL atom mask (BatchDeviceBindings). This is the
    // authoritative numeric-atom annotation for this row. ReduceMean atom
    // exclusion reads it directly, while state-bearing positions come only from
    // the compiled slot map. payload.atom_mask is validated at build time to
    // equal token_layout.isAtom(token_id), so it matches ScratchBlock detection.
    if (!bindings.d_atom_mask) {
        throw std::runtime_error(
            "executeStepCoordinatorImpl: bindings.d_atom_mask is NULL - execution "
            "requires global atom annotations to validate state-bearing slot bindings");
    }
    const uint8_t* d_atom_mask_row = bindings.d_atom_mask
        + static_cast<size_t>(batch_row) * payload.max_seq_len;

    // One flag is shared by all validation kernels in this step. Reset it
    // before the first kernel so earlier failures are not erased below.
    CUDA_CHECK(cudaMemsetAsync(diag.numericErrorFlag(), 0, sizeof(int), stream));
    EXEC_CHECK(V_val > 0, "executeStep: no value slots (V - S == 0)");

    // Defer validation failure synchronization to finalizeStepOrThrow.
    kernelCheckAnyValueSlotValid<<<1, 1, 0, stream>>>(
        memory.valid_mask.data,
        diag.numericErrorFlag(),
        S, V_val, kStageSlotUninit);
    CUDA_CHECK_KERNEL();

    forward_output.state_before_values = Tensor::zeros({V, 1}, stream, "state_before_values");
    forward_output.state_before_valid = Tensor::zeros({1, V}, stream, "state_before_valid");
    CUDA_CHECK(cudaMemcpyAsync(forward_output.state_before_values.data, memory.values.data,
        V * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(forward_output.state_before_valid.data, memory.valid_mask.data,
        V * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    if (row_tokens > 0) {
        kernelValidateStateBearingSlots<<<(row_tokens + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            d_atom_mask_row,
            d_slot_map_row,
            memory.valid_mask.data,
            row_tokens,
            hp.num_slots,
            hp.num_scratch_slots,
            diag.numericErrorFlag(),
            kStageSlotInvalid,
            kStageSlotUninit);
        CUDA_CHECK_KERNEL();
    }

    StepWorkingSet work;
    buildValueSlotCandidates(hp, memory, stream, work);
    kernelCheckFinite<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.slot_values.data, V_val, diag.numericErrorFlag(), kStageV1, hp.magnitude_limit);

    // Row-local device atom mask (computed above) excludes atom positions from
    // the decision context, preventing numeric surface leakage into execution
    // op/arg/write selection.
    work.context = Tensor::zeros({1, dm}, stream, "exec_context");
    work.context.requires_grad = true;
    work.context.is_leaf = false;
    kernelReduceMeanForward<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.context.data,
        H.data + static_cast<size_t>(batch_row) * payload.max_seq_len * dm,
        prompt_tokens,
        dm,
        d_atom_mask_row);
    CUDA_CHECK_KERNEL();
    {
        auto mean_fn = std::make_shared<ReduceMeanGradFn>();
        mean_fn->capture(H, payload.total_tokens, dm, stream, batch_row * payload.max_seq_len, prompt_tokens,
                         d_atom_mask_row);
        work.context.grad_fn = mean_fn;
    }

    const int K = hp.num_exec_steps;
    const int N_prior = static_cast<int>(prior_records.size());
    if (N_prior > 0) {
        std::vector<int> h_slot1(N_prior), h_slot2(N_prior), h_ops(N_prior);
        std::vector<float> h_scalars(N_prior * 3);
        for (int i = 0; i < N_prior; ++i) {
            h_slot1[i] = prior_records[i].arg1_slot;
            h_slot2[i] = prior_records[i].arg2_slot;
            h_ops[i] = prior_records[i].op_id;
            h_scalars[i * 3 + 0] = prior_records[i].value_before_1;
            h_scalars[i * 3 + 1] = prior_records[i].value_before_2;
            h_scalars[i * 3 + 2] = prior_records[i].value_after;
        }

        int *d_slot1 = nullptr, *d_slot2 = nullptr, *d_ops = nullptr;
        float* d_scalars = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_slot1), N_prior * sizeof(int), "datastream_prior_slot1");
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_slot2), N_prior * sizeof(int), "datastream_prior_slot2");
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_ops), N_prior * sizeof(int), "datastream_prior_ops");
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_scalars), N_prior * 3 * sizeof(float), "datastream_prior_scalars");
        CUDA_CHECK(cudaMemcpyAsync(d_slot1, h_slot1.data(), N_prior * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_slot2, h_slot2.data(), N_prior * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ops, h_ops.data(), N_prior * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_scalars, h_scalars.data(), N_prior * 3 * sizeof(float), cudaMemcpyHostToDevice, stream));

        auto encoded_records = Tensor::zeros({N_prior, dm}, stream, "exec_encoded_records");
        int total_enc = N_prior * dm;
        kernelEncodeRecords<<<(total_enc + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            encoded_records.data,
            params.E_slot.data,
            params.E_op.data,
            params.W_scal.data,
            params.b_scal.data,
            d_slot1,
            d_slot2,
            d_ops,
            d_scalars,
            N_prior,
            V,
            nop,
            dm);
        CUDA_CHECK_KERNEL();

        auto padded = Tensor::zeros({K, dm}, stream, "exec_trace_padded");
        int copy_rows = (N_prior < K) ? N_prior : K;
        int src_offset = (N_prior > K) ? (N_prior - K) : 0;
        CUDA_CHECK(cudaMemcpyAsync(padded.data,
            encoded_records.data + static_cast<size_t>(src_offset) * dm,
            copy_rows * dm * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        auto flattened = Tensor::zeros({1, K * dm}, stream, "exec_trace_flat");
        CUDA_CHECK(cudaMemcpyAsync(flattened.data, padded.data, K * dm * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // Prior records are discrete host history. The raw padding/flattening
        // copies produce a detached trace input; attaching a
        // RecordEncodeGradFn to encoded_records here created an unreachable
        // backward node because neither copy has an autograd edge.

        work.trace_vec = autograd::matmul(flattened, params.W_trace, stream);
        if (hp.trace_bias_enabled) {
            work.trace_vec = autograd::add(work.trace_vec, params.b_trace, stream);
        }

        cudaFreeAsync(d_slot1, stream);
        cudaFreeAsync(d_slot2, stream);
        cudaFreeAsync(d_ops, stream);
        cudaFreeAsync(d_scalars, stream);
    } else {
        work.trace_vec = Tensor::zeros({1, dm}, stream, "exec_trace_vec_zero");
    }

    work.context_enriched = autograd::add(work.context, work.trace_vec, stream);
    work.step_emb = Tensor::zeros({1, dm}, stream, "exec_step_emb");
    work.step_emb.requires_grad = false;
    work.step_emb.is_leaf = true;
    cudaMemcpyAsync(work.step_emb.data,
        params.step_embeddings.data + static_cast<size_t>(step) * dm,
        dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto dec_12 = autograd::concat(work.context_enriched, trace_state, stream);
    auto decision_input = autograd::concat(dec_12, work.step_emb, stream);

    auto query1 = autograd::matmul(decision_input, params.w_arg1_select, stream);
    auto arg1_raw_scores = autograd::matmul(query1, work.cand_hidden, stream, true);
    auto arg1_logits = autograd::mul_scalar(
        arg1_raw_scores, operand_selection_scale, stream);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg1_logits.data, arg1_logits.data, work.cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    work.p_arg1 = autograd::softmax(arg1_logits, temperature, stream);
    forward_output.arg1_logits_tensor = std::move(arg1_logits);
    forward_output.selection_temperature = temperature;
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_arg1.data, V_val, diag.numericErrorFlag(), kStagePArg1);

    auto query2 = autograd::matmul(decision_input, params.w_arg2_select, stream);
    auto arg2_raw_scores = autograd::matmul(query2, work.cand_hidden, stream, true);
    auto arg2_logits = autograd::mul_scalar(
        arg2_raw_scores, operand_selection_scale, stream);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg2_logits.data, arg2_logits.data, work.cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    work.p_arg2 = autograd::softmax(arg2_logits, temperature, stream);
    forward_output.arg2_logits_tensor = std::move(arg2_logits);
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_arg2.data, V_val, diag.numericErrorFlag(), kStagePArg2);

    // The op head is independent of the selected operand values. Compute all
    // three read/op distributions before materializing the detached decision.
    auto op_logits = autograd::matmul(decision_input, params.W_op_select, stream);
    work.p_op = autograd::softmax(op_logits, temperature, stream);
    forward_output.op_logits_tensor = std::move(op_logits);
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(
        work.p_op.data, nop, diag.numericErrorFlag(), kStagePOp);

    materializeHardReadAndOpDecision(
        hp, diag, payload, batch_row, step, stream, work);

    materializeSelectedOperands(hp, diag, memory, stream, work);
    kernelCheckFinite<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.slot_values.data, V_val, diag.numericErrorFlag(), kStageV1, hp.magnitude_limit);
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v1.data, 1, diag.numericErrorFlag(), kStageV1, hp.magnitude_limit);
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v2.data, 1, diag.numericErrorFlag(), kStageV2, hp.magnitude_limit);

    // v1/v2 are DETACHED from p_arg.  No gradient path from execution
    // value loss back into arg selection.  Only selection CE trains arg selection.
    // v1/v2 grad_fn are left as nullptr (set by materializeSelectedOperands).

    // [DELETED] h_arg1/h_arg2 = matmul(p_arg, cand_hidden)
    // These leaked arg selection into op and write heads. All three selection
    // heads (arg, op, write) are now fully independent classification tasks.

    // Op selection is DETACHED from arg selection (Option B).
    // op sees (context_enriched, trace_state, step_emb) only — same as decision_input.
    // arg selection cannot influence op selection through the gradient path.

    work.op_results = Tensor::zeros({1, nop}, stream, "exec_op_results");
    // Fix #6: Reset per-step division invalid flag before FourOps
    int* d_div_flag = diag.divInvalidFlag();
    int h_div_flag = 0;
    CUDA_CHECK(cudaMemsetAsync(d_div_flag, 0, sizeof(int), stream));
    kernelFourOps<<<1, kWarpSize, 0, stream>>>(work.op_results.data, work.v1.data, work.v2.data, kEps, diag.divClampCount(), d_div_flag);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpyAsync(&h_div_flag, d_div_flag, sizeof(int), cudaMemcpyDeviceToHost, stream));

    // Hard op application (clean classification).
    // The mode-specific decision above already materialized hard_op; select
    // exactly one discrete result. v_out = results[hard_op].
    // The hard result is detached. Op selection is supervised from retained logits
    // at the loss boundary; execution values do not create an autograd edge.
    // Blending +/-/*/÷ results is FUCKING STUPID! for a register machine.
    work.v_out = Tensor::zeros({1, 1}, stream, "exec_v_out");
    work.v_out.requires_grad = false;
    work.v_out.is_leaf = true;
    kernelHardPickOpForward<<<1, 1, 0, stream>>>(work.v_out.data, work.op_results.data, d_exec_idx + 2, nop);
    CUDA_CHECK_KERNEL();
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v_out.data, 1, diag.numericErrorFlag(), kStageVOut, hp.magnitude_limit);

    kernelAssembleExecRecord<<<1, 1, 0, stream>>>(
        d_exec_record_i, d_exec_record_f,
        d_exec_idx, d_exec_idx + 1, d_exec_idx + 2,
        work.v1.data, work.v2.data, work.v_out.data, S);
    CUDA_CHECK_KERNEL();

    {
        auto cur_encoded = Tensor::zeros({1, dm}, stream, "exec_cur_step_enc");
        cur_encoded.requires_grad = true;
        cur_encoded.is_leaf = false;
        kernelEncodeRecords<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            cur_encoded.data,
            params.E_slot.data,
            params.E_op.data,
            params.W_scal.data,
            params.b_scal.data,
            d_exec_record_i,
            d_exec_record_i + 1,
            d_exec_record_i + 2,
            d_exec_record_f,
            1,
            V,
            nop,
            dm);
        CUDA_CHECK_KERNEL();

        {
            if (record_encode_backward_staging.record_count != 1 ||
                !record_encode_backward_staging.saved_ids ||
                !record_encode_backward_staging.saved_scalars) {
                throw std::runtime_error(
                    "executeStepCoordinatorImpl: invalid ModelForwardOutputs "
                    "RecordEncode backward staging");
            }
            auto enc_fn = std::make_shared<RecordEncodeGradFn>();
            enc_fn->capture(1, V, nop, dm,
                            d_exec_record_i, d_exec_record_i + 1,
                            d_exec_record_i + 2, d_exec_record_f,
                            record_encode_backward_staging.saved_ids.get(),
                            record_encode_backward_staging.saved_scalars.get(),
                            params.E_slot, params.E_op, params.W_scal, params.b_scal, stream);
            cur_encoded.grad_fn = enc_fn;
        }

        auto update_input = autograd::concat(trace_state, cur_encoded, stream);
        auto candidate = autograd::matmul(update_input, params.W_reason_gate, stream);
        auto gate_logits = autograd::matmul(update_input, params.W_trace_gate, stream);

        // Fix #5: Gated trace update — bounded interpolation replaces unbounded accumulation.
        // trace_state = sigmoid(gate_logits) * trace_state + (1 - sigmoid(gate_logits)) * candidate
        float* gate_vals_buf = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&gate_vals_buf), dm * sizeof(float), "gated_trace_gate_vals");

        Tensor new_trace = Tensor::zeros({1, dm}, stream, "gated_trace_out");
        new_trace.requires_grad = true;
        new_trace.is_leaf = false;

        kernelGatedTraceUpdate<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            new_trace.data, gate_vals_buf,
            trace_state.data, candidate.data, gate_logits.data, dm);
        CUDA_CHECK_KERNEL();

        {
            auto gfn = std::make_shared<GatedTraceUpdateGradFn>();
            gfn->capture(trace_state, candidate, gate_logits, gate_vals_buf, dm, stream);
            new_trace.grad_fn = gfn;
        }
        trace_state = std::move(new_trace);
    }

    work.atom_new = Tensor::zeros({1, ae}, stream, "exec_atom_new");
    kernelEncodeScalarToAtomEmbed<<<1, ae, 0, stream>>>(work.atom_new.data, work.v_out.data, ae);
    CUDA_CHECK_KERNEL();

    auto decode_input = Tensor::zeros({1, vid}, stream, "exec_decode_input");
    // This direct slice is intentionally detached. w_decode_1 still receives
    // grad_B because the durable registry-owned weight requires grad, and
    // MatMulGradFn owns the saved A copy needed for that calculation. Giving
    // this function-local tensor a leaf grad buffer would violate the leaf
    // contract: MatMulGradFn borrows leaf grads under the assumption that their
    // ParameterRegistry owner outlives backward.
    decode_input.requires_grad = false;
    decode_input.is_leaf = true;
    cudaMemcpyAsync(decode_input.data, work.atom_new.data + 16, vid * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto decode_h = autograd::matmul(decode_input, params.w_decode_1, stream);
    if (hp.decode_bias_enabled) {
        decode_h = autograd::add(decode_h, params.b_decode_1, stream);
    }

    auto decode_act = autograd::silu(decode_h, stream, decode_h.data);

    // SiluGradFn deliberately borrows its forward input cache. Move the
    // pre-activation into the ModelForwardOutputs-owned step payload before the
    // local working set can release it. The payload is cleared only after the
    // active forward/loss/backward window completes.
    forward_output.decoder_silu_input_tensor = std::move(decode_h);

    work.v_decoded = autograd::matmul(decode_act, params.w_decode_2, stream);
    work.result_emb = autograd::matmul(work.v_decoded, params.W_value_to_emb, stream);
    if (hp.value_embedding_bias_enabled) {
        work.result_emb = autograd::add(work.result_emb, params.b_value_to_emb, stream);
    }
    kernelCheckFinite<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.result_emb.data, dm, diag.numericErrorFlag(), kStageResultEmb, hp.magnitude_limit);

    const int row_final_slot = batch_row * payload.max_seq_len + row_tokens - 1;
    if (hp.result_slot_mode == 1) {
        if (hp.result_slot_index != row_final_slot) {
            throw std::runtime_error(
                "ExecutionBlock: fixed result_slot_index=" + std::to_string(hp.result_slot_index) +
                " would inject row-final execution memory into token " +
                std::to_string(hp.result_slot_index) +
                " for batch_row=" + std::to_string(batch_row) +
                "; causal shared forward requires row-final slot " +
                std::to_string(row_final_slot));
        }
        work.result_slot = hp.result_slot_index;
    } else {
        work.result_slot = row_final_slot;
    }
    EXEC_CHECK(work.result_slot >= 0 && work.result_slot < payload.total_tokens, "result_slot out of bounds");
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(dm));

    float* save_H_slot_buf = nullptr;
    float h_inject_gate_value = 0.0f;  // host-side readback for telemetry
    forward_output.inject_gate_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, 1),
        false,
        stream,
        "exec_inject_gate");
    cudaMallocOrThrow(reinterpret_cast<void**>(&save_H_slot_buf), dm * sizeof(float), "datastream_save_H_slot");
    kernelInjectResultSlot<<<1, kBlockSize, 0, stream>>>(
        H.data, work.result_emb.data, params.w_inject_gate.data,
        inv_sqrt_d, work.result_slot, dm, hp.inject_gate_temp,
        forward_output.inject_gate_tensor.data, save_H_slot_buf);
    CUDA_CHECK_KERNEL();
    // Queue async readback of inject gate for telemetry (completes at next sync)
    if (hp.debug_mode) {
        CUDA_CHECK(cudaMemcpyAsync(&h_inject_gate_value, forward_output.inject_gate_tensor.data, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }

    {
        auto inject_fn = std::make_shared<ExecutionBlockInjectGradFn>();
        inject_fn->capture(H, work.result_emb, params.w_inject_gate,
                           forward_output.inject_gate_tensor, save_H_slot_buf,
                           inv_sqrt_d, hp.inject_gate_temp,
                           work.result_slot, payload.total_tokens, dm, stream);
        H.grad_fn = inject_fn;
        H.is_leaf = false;
        H.requires_grad = true;
    }

    // Write context: DETACHED from arg selection.
    // Sees (context, result, trace, step) only — no h_arg1/h_arg2 coupling.
    auto write_ctx_12 = autograd::concat(work.context_enriched, work.result_emb, stream);
    auto write_ctx_123 = autograd::concat(write_ctx_12, trace_state, stream);
    auto write_ctx = autograd::concat(write_ctx_123, work.step_emb, stream);

    Tensor stop_logits = autograd::matmul(write_ctx, params.W_stop, stream);
    stop_logits = autograd::add(stop_logits, params.b_stop, stream);
    forward_output.stop_probabilities = autograd::softmax(stop_logits, 1.0f, stream);
    forward_output.stop_logits_tensor = std::move(stop_logits);

    auto q_write = autograd::matmul(write_ctx, params.W_write_query, stream);

    auto K_proj = autograd::matmul(memory.key_embeds, params.W_write_key, stream);

    auto usage_norm = Tensor::zeros({1, V}, stream, "exec_usage_norm");
    kernelNormalizeUsage<<<1, kWarpSize, 0, stream>>>(usage_norm.data, memory.usage.data, V);
    CUDA_CHECK_KERNEL();

    // ── Write logits via autograd ops (tape-connected to W_write_query, W_write_key, alpha, beta) ──
    // Formula: logits[i] = alpha * dot(q, K_proj[i]) / sqrt(d_key)
    //                    + beta * (-usage[i])
    // Pure CE classification — no non-differentiable bias terms (Fix #4).
    //
    // Scale query/key content scores exactly as the execution read-attention
    // path does. Without this, initialization variance grows with d_key and
    // saturates p_write before structured CE can train the selector.
    auto raw_content_scores = autograd::matmul(q_write, K_proj, stream,
        /*transpose_b=*/true);
    const float inv_sqrt_dk = 1.0f / sqrtf(static_cast<float>(dk));
    auto content_scores = autograd::mul_scalar(raw_content_scores, inv_sqrt_dk, stream);

    // alpha_content = alpha [1,1] @ content_scores [1,V] → [1,V] (broadcast scalar multiply)
    auto alpha_content = autograd::matmul(params.alpha, content_scores, stream);

    // neg_usage: detached [1,V] = -usage_norm
    auto neg_usage = Tensor::zeros({1, V}, stream, "exec_neg_usage");
    kernelNegateVec<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        neg_usage.data, usage_norm.data, V);
    CUDA_CHECK_KERNEL();

    // beta_usage = beta [1,1] @ neg_usage [1,V] → [1,V] (broadcast scalar multiply)
    auto beta_usage = autograd::matmul(params.beta, neg_usage, stream);

    // Fix #4: Pure CE classification for write slot.
    // No non-differentiable bonus_bias — forward and backward see the same function.
    // write_logits = alpha*content + beta*(-usage) → [1,V]
    auto write_logits = autograd::add(alpha_content, beta_usage, stream);

    if (S > 0) {
        kernelMaskScratchSlots<<<(S + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(write_logits.data, S);
        CUDA_CHECK_KERNEL();
    }
    work.p_write = autograd::softmax(write_logits, temperature, stream);
    forward_output.write_logits_tensor = std::move(write_logits);
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_write.data, V, diag.numericErrorFlag(), kStagePWrite);

    work.state_new = Tensor::zeros({1, dm}, stream, "exec_state_new");
    const float* step_emb_ptr = params.step_embeddings.data + static_cast<size_t>(step) * dm;
    kernelAddVectors<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.state_new.data, work.result_emb.data, step_emb_ptr, dm);
    CUDA_CHECK_KERNEL();

    work.key_new = autograd::matmul(work.result_emb, params.W_key_proj, stream);
    materializeHardWriteDecision(
        hp, diag, payload, batch_row, step, stream, work);
    applyHardWriteback(hp, diag, parameters, memory, stream, work);
    copyStepDiagnostics(work, forward_output, V_val, nop, V, dm, stream);
    captureStateAfterWriteAndCheckMutations(hp, diag, memory, forward_output, stream);
    collectStepMetrics(hp, diag, work, forward_output, V_val, nop, V, stream);
    // collectStepMetrics synced the stream — h_inject_gate_value is now valid
    if (hp.debug_mode) {
        forward_output.metrics.inject_gate_value = h_inject_gate_value;
    }
    finalizeStepOrThrow(hp, diag, forward_output, step, stream);
    forward_output.div_was_clamped = (h_div_flag != 0);
    forward_output.v_out_tensor = std::move(work.v_out);
}

}  // namespace GRIM::ExecutionBlockInternal

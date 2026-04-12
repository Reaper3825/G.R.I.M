#include "execution_block_data_stream_GPU.hpp"
#include "execution_block_memory_stream_GPU.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

#ifdef USE_CUDA

namespace GRIM::ExecutionBlockInternal {

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

// Fused: compute NORMALIZED entropy [0,1] and atomicAdd to accumulator.
// Normalized by log(N) so distributions of different sizes contribute equally.
__global__ void kernelEntropyAccum(
    float* __restrict__ accum,
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
    if (tid == 0) {
        float max_ent = logf(static_cast<float>(N));
        float ent_norm = (max_ent > 1e-10f) ? (ent / max_ent) : 0.0f;
        atomicAdd(accum, ent_norm);
    }
}

__global__ void kernelScaleNegAvg(
    float* __restrict__ out,
    const float* __restrict__ in,
    float weight,
    int count
) {
    if (threadIdx.x == 0)
        out[0] = -weight * (in[0] / fmaxf(static_cast<float>(count), 1.0f));
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
        atomicMax(error_flag, stage_id);
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
        atomicMax(error_flag, stage_id);
}

__global__ void kernelCheckEntropyCollapse(
    const float* __restrict__ probs,
    int N,
    int* __restrict__ error_flag,
    int stage_id,
    float threshold   // compared against NORMALIZED entropy in [0,1]
) {
    const int tid = threadIdx.x;
    float ent = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float p = probs[i];
        if (p > 1e-10f)
            ent -= p * logf(p + 1e-10f);
    }
    ent = warpReduceSum(ent);
    // Normalize: ent_norm = ent / log(N) ∈ [0,1] regardless of distribution size.
    // Without this, threshold means different things for N=4 vs N=128.
    if (tid == 0) {
        float max_ent = logf(static_cast<float>(N));
        float ent_norm = (max_ent > 1e-10f) ? (ent / max_ent) : 0.0f;
        if (ent_norm < threshold)
            atomicMax(error_flag, stage_id);
    }
}

__global__ void kernelCheckWriteCollapse(
    const float* __restrict__ probs,
    int N,
    int* __restrict__ error_flag,
    int stage_id,
    float threshold
) {
    const int tid = threadIdx.x;
    float mx = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)
        mx = fmaxf(mx, probs[i]);
    mx = warpReduceMax(mx);
    if (tid == 0 && mx > threshold)
        atomicMax(error_flag, stage_id);
}

__global__ void kernelReduceMeanForward(
    float* __restrict__ out,
    const float* __restrict__ H,
    int total_tokens, int d_model
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_model) return;
    float sum = 0.0f;
    for (int i = 0; i < total_tokens; ++i)
        sum += H[static_cast<size_t>(i) * d_model + j];
    out[j] = sum / static_cast<float>(total_tokens);
}

__global__ void kernelReduceMeanBackward(
    float* __restrict__ grad_H,
    const float* __restrict__ grad_out,
    int total_tokens, int d_model
) {
    const int i = blockIdx.x;
    if (i >= total_tokens) return;
    float scale = 1.0f / static_cast<float>(total_tokens);
    float* dst = grad_H + static_cast<size_t>(i) * d_model;
    for (int j = threadIdx.x; j < d_model; j += blockDim.x)
        dst[j] += grad_out[j] * scale;
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

__global__ void kernelAccumulateScalar(
    float* __restrict__ a,
    const float* __restrict__ b
) {
    if (threadIdx.x != 0) return;
    a[0] += b[0];
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

// Fix #6: Division invalid penalty.
// penalty = div_invalid_flag * weight * p_op[div_op_idx]
// Teaches: "don't select ÷ when |v2| < eps."
// The clamped division is kept for numerical stability, but the model learns
// that division is an invalid operation for this (v1, v2) pair.
__global__ void kernelDivInvalidPenalty(
    float* __restrict__ out_loss,        // [1] scalar penalty
    const float* __restrict__ p_op,      // [num_ops] softmax probs
    const int* __restrict__ div_flag,    // [1] 0 or 1
    float weight,
    int div_op_idx                       // = 3
) {
    if (threadIdx.x != 0) return;
    out_loss[0] = div_flag[0] ? (weight * p_op[div_op_idx]) : 0.0f;
}

__global__ void kernelDivInvalidPenaltyBackward(
    float* __restrict__ grad_p_op,       // [num_ops] output gradient
    const float* __restrict__ grad_out,  // [1] incoming gradient
    const int* __restrict__ div_flag,    // [1] saved flag
    float weight,
    int div_op_idx,
    int num_ops
) {
    if (threadIdx.x != 0) return;
    // Only select division slot gets gradient — all others stay zero
    for (int k = 0; k < num_ops; ++k)
        grad_p_op[k] = 0.0f;
    if (div_flag[0])
        grad_p_op[div_op_idx] = weight * grad_out[0];
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

__global__ void kernelInjectSlotBackward(
    float* __restrict__ grad_result,
    float* __restrict__ grad_w_gate,
    float* __restrict__ mod_grad_slot,
    const float* __restrict__ saved_result,
    const float* __restrict__ saved_H_slot,
    const float* __restrict__ w_gate,
    const float* __restrict__ saved_gate,
    float inv_sqrt_d,
    float gate_temp,
    int d_model
) {
    float gate_val = saved_gate[0];

    __shared__ float s_d_logit;
    if (threadIdx.x == 0) {
        float dot = 0.0f;
        for (int j = 0; j < d_model; ++j)
            dot += mod_grad_slot[j] * saved_result[j];
        s_d_logit = dot * inv_sqrt_d * gate_val * (1.0f - gate_val) * gate_temp;
    }
    __syncthreads();
    float d_logit = s_d_logit;

    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        float go = mod_grad_slot[j];
        grad_result[j] += inv_sqrt_d * gate_val * go;
        grad_w_gate[j] += saved_H_slot[j] * d_logit;
        mod_grad_slot[j] = go + w_gate[j] * d_logit;
    }
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

    float val = b_scal[j];
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

__global__ void kernelEncodeRecordsBackward(
    const float* __restrict__ grad_out,
    float* __restrict__ E_slot_grad,
    float* __restrict__ E_op_grad,
    float* __restrict__ W_scal_grad,
    float* __restrict__ b_scal_grad,
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
    float g = grad_out[idx];

    int s1 = slot1_ids[i];
    if (s1 >= 0 && s1 < num_slots)
        atomicAdd(&E_slot_grad[s1 * d_model + j], g);
    int s2 = slot2_ids[i];
    if (s2 >= 0 && s2 < num_slots)
        atomicAdd(&E_slot_grad[s2 * d_model + j], g);
    int op = op_ids[i];
    if (op >= 0 && op < num_ops)
        atomicAdd(&E_op_grad[op * d_model + j], g);
    for (int k = 0; k < 3; ++k)
        atomicAdd(&W_scal_grad[k * d_model + j], scalars[i * 3 + k] * g);
    atomicAdd(&b_scal_grad[j], g);
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

// Backward: d_old[i]  += upstream[i] * gate[i]
//           d_cand[i] += upstream[i] * (1 - gate[i])
//           d_gate_logits[i] += upstream[i] * (old[i] - cand[i]) * gate[i] * (1 - gate[i])
__global__ void kernelGatedTraceUpdateBackward(
    float* __restrict__ d_old_trace,
    float* __restrict__ d_candidate,
    float* __restrict__ d_gate_logits,
    const float* __restrict__ upstream,
    const float* __restrict__ old_trace,
    const float* __restrict__ candidate,
    const float* __restrict__ gate_vals,
    int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = gate_vals[i];
    float u = upstream[i];
    d_old_trace[i]   += u * g;
    d_candidate[i]   += u * (1.0f - g);
    d_gate_logits[i] += u * (old_trace[i] - candidate[i]) * g * (1.0f - g);
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

__global__ void kernelAbsDiff(
    float* __restrict__ out,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int* __restrict__ error_flag,
    int stage_id,
    float hard_threshold
) {
    if (threadIdx.x != 0) return;
    float diff = fabsf(a[0] - b[0]);
    out[0] = diff;
    if (hard_threshold > 0.0f && diff > hard_threshold && error_flag)
        atomicMax(error_flag, stage_id);
}

__global__ void kernelRecomputeOpResult(
    float* __restrict__ expected,
    const float* __restrict__ M_values,
    const int* __restrict__ d_arg1_rel,
    const int* __restrict__ d_arg2_rel,
    const int* __restrict__ d_op_id,
    int S, int V, int V_val, float eps
) {
    if (threadIdx.x != 0) return;
    int r1 = d_arg1_rel[0];
    int r2 = d_arg2_rel[0];
    int op = d_op_id[0];
    if (r1 < 0 || r1 >= V_val || r2 < 0 || r2 >= V_val) {
        expected[0] = 0.0f;
        return;
    }
    int s1 = S + r1;
    int s2 = S + r2;
    float v1 = M_values[s1];
    float v2 = M_values[s2];
    switch (op) {
        case 0: expected[0] = v1 + v2; break;
        case 1: expected[0] = v1 - v2; break;
        case 2: expected[0] = v1 * v2; break;
        case 3: {
            float abs_v2 = fabsf(v2);
            float denom = (abs_v2 > eps) ? v2 : copysignf(eps, v2);
            expected[0] = v1 / denom;
            break;
        }
        default: expected[0] = 0.0f; break;
    }
}

__global__ void kernelL1LossBackward(
    float* __restrict__ grad_input,
    const float* __restrict__ grad_output,
    const float* __restrict__ a,
    const float* __restrict__ b
) {
    if (threadIdx.x != 0) return;
    float diff = a[0] - b[0];
    float s = (fabsf(diff) > 1e-10f) ? copysignf(1.0f, diff) : 0.0f;
    grad_input[0] += grad_output[0] * s;
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

__global__ void kernelFourOpMixBackward(
    float* __restrict__ grad_v1,
    float* __restrict__ grad_v2,
    float* __restrict__ grad_p_op,
    const float* __restrict__ grad_v_out_ptr,
    const float* __restrict__ pv1,
    const float* __restrict__ pv2,
    const int* __restrict__ op_idx,    // hard-selected op from forward argmax
    float eps,
    int num_ops
) {
    const int tid = threadIdx.x;
    float grad_v_out = grad_v_out_ptr[0];
    float v1 = pv1[0];
    float v2 = pv2[0];
    float abs_v2 = fabsf(v2);
    float denom = (abs_v2 > eps) ? v2 : copysignf(eps, v2);

    // ── FULL HARD backward (Fix #2 hardened) ──
    // Forward: v_out = results[argmax(p_op)]   (hard pick)
    // Backward: use ONLY the selected op's derivative for v1/v2.
    //
    // Previous code used p_op as soft weights for v1/v2 derivatives.
    // That leaked: p_op influenced v1/v2 gradient magnitude/direction,
    // so the model could shape p_op to get "better" value gradients.
    // This violated true symbolic separation.
    //
    // Now: backward matches forward exactly — only the hard-selected op
    // contributes derivative signal.  p_op has ZERO influence on any
    // gradient path.  The ONLY training signal for op selection is
    // selection_ce_op (CE classification).

    if (tid == 0) {
        constexpr float kDerivClip = 10.0f;
        int k = op_idx[0];
        if (k < 0 || k >= num_ops) k = 0;

        // Derivative of the selected op only
        float dv1_k = 0.0f, dv2_k = 0.0f;
        switch (k) {
            case 0:  // add: v1 + v2
                dv1_k = 1.0f;
                dv2_k = 1.0f;
                break;
            case 1:  // sub: v1 - v2
                dv1_k = 1.0f;
                dv2_k = -1.0f;
                break;
            case 2:  // mul: v1 * v2
                dv1_k = fminf(fmaxf(v2, -kDerivClip), kDerivClip);
                dv2_k = fminf(fmaxf(v1, -kDerivClip), kDerivClip);
                break;
            case 3: { // div: v1 / denom
                float dv1_raw = 1.0f / denom;
                dv1_k = fminf(fmaxf(dv1_raw, -kDerivClip), kDerivClip);
                float dv2_raw = -v1 / (denom * denom);
                dv2_k = fminf(fmaxf(dv2_raw, -kDerivClip), kDerivClip);
                break;
            }
        }

        if (grad_v1) grad_v1[0] += dv1_k * grad_v_out;
        if (grad_v2) grad_v2[0] += dv2_k * grad_v_out;

        // grad_p_op intentionally left at zero — CE only
    }
}

// [DELETED] OperandConsistencyGradFn — removed per Fix #1.
// Value-path gradients leaked back into arg selection (p_arg) through
// v1/v2 → FourOpMixGradFn → v1/v2 → OperandConsistencyGradFn → p_arg.
// This taught "pick slots that give good answers" instead of "pick the
// correct symbolic slot".  Selection CE is now the ONLY arg selection signal.
// v1/v2 grad_fn are left as nullptr (detached from p_arg).

// Placeholder struct removed — OperandConsistencyGradFn no longer exists.
// The code below continues with FourOpMixGradFn.
struct _OperandConsistencyGradFn_DELETED;
// (end of deleted OperandConsistencyGradFn)

struct FourOpMixGradFn : public GradFn {
    float* saved_v1 = nullptr;
    float* saved_v2 = nullptr;
    int* saved_op_idx = nullptr;       // hard-selected op from forward argmax
    int num_ops_ = 4;

    float* grad_v1 = nullptr;
    float* grad_v2 = nullptr;
    float* grad_p_op = nullptr;
    std::shared_ptr<float> owned_grad_v1;
    std::shared_ptr<float> owned_grad_v2;
    std::shared_ptr<float> owned_grad_p_op;
    std::shared_ptr<GradFn> v1_grad_fn;
    std::shared_ptr<GradFn> v2_grad_fn;
    std::shared_ptr<GradFn> p_op_grad_fn;
    TensorContract::TensorShape v1_shape;
    TensorContract::TensorShape v2_shape;
    TensorContract::TensorShape p_op_shape;
    bool v1_requires_grad = false;
    bool v2_requires_grad = false;
    bool p_op_requires_grad = false;

    FourOpMixGradFn() { op_name = "four_op_mix"; }

    ~FourOpMixGradFn() override {
        if (saved_v1) cudaFree(saved_v1);
        if (saved_v2) cudaFree(saved_v2);
        if (saved_op_idx) cudaFree(saved_op_idx);
    }

    void capture(Tensor& v1_t, Tensor& v2_t, Tensor& p_op_t,
                 const int* d_op_idx,
                 int num_ops, cudaStream_t stream) {
        num_ops_ = num_ops;

        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_v1), sizeof(float), "datastream_saved_v1");
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_v2), sizeof(float), "datastream_saved_v2");
        cudaMemcpyAsync(saved_v1, v1_t.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(saved_v2, v2_t.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // Save the hard-selected op index — backward uses ONLY this op's derivative.
        // No p_op or results saved: p_op must have ZERO influence on value gradients.
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_op_idx), sizeof(int), "datastream_saved_op_idx");
        cudaMemcpyAsync(saved_op_idx, d_op_idx, sizeof(int), cudaMemcpyDeviceToDevice, stream);

        v1_requires_grad = v1_t.requires_grad;
        v2_requires_grad = v2_t.requires_grad;
        p_op_requires_grad = p_op_t.requires_grad;
        v1_shape = v1_t.shape;
        v2_shape = v2_t.shape;
        p_op_shape = p_op_t.shape;
        v1_grad_fn = v1_t.grad_fn;
        v2_grad_fn = v2_t.grad_fn;
        p_op_grad_fn = p_op_t.grad_fn;

        auto alloc_grad = [&](Tensor& t, float*& gp, std::shared_ptr<float>& owned, size_t n) {
            if (!t.requires_grad) return;
            t.ensure_grad();
            if (t.is_leaf) {
                gp = t.grad_data();
            } else {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), n * sizeof(float), "datastream_alu_grad_buf");
                cudaMemsetAsync(buf, 0, n * sizeof(float), stream);
                owned = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
                gp = owned.get();
            }
        };

        alloc_grad(v1_t, grad_v1, owned_grad_v1, 1);
        alloc_grad(v2_t, grad_v2, owned_grad_v2, 1);
        alloc_grad(p_op_t, grad_p_op, owned_grad_p_op, num_ops);
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;

        kernelFourOpMixBackward<<<1, kWarpSize, 0, stream>>>(
            grad_v1, grad_v2, grad_p_op,
            grad_output.data,
            saved_v1, saved_v2,
            saved_op_idx,
            kEps, num_ops_);
        CUDA_CHECK_KERNEL();

        if (v2_requires_grad && v2_grad_fn && v2_grad_fn == v1_grad_fn) {
            kernelAccumulateScalar<<<1, 1, 0, stream>>>(grad_v1, grad_v2);
            Tensor view;
            view.data = grad_v1;
            view.shape = v1_shape;
            view.owns_data = false;
            view.stream = stream;
            v1_grad_fn->apply(view, stream);
        } else {
            if (v1_requires_grad && v1_grad_fn) {
                Tensor view;
                view.data = grad_v1;
                view.shape = v1_shape;
                view.owns_data = false;
                view.stream = stream;
                v1_grad_fn->apply(view, stream);
            }
            if (v2_requires_grad && v2_grad_fn) {
                Tensor view;
                view.data = grad_v2;
                view.shape = v2_shape;
                view.owns_data = false;
                view.stream = stream;
                v2_grad_fn->apply(view, stream);
            }
        }

        if (p_op_requires_grad && p_op_grad_fn) {
            Tensor view;
            view.data = grad_p_op;
            view.shape = p_op_shape;
            view.owns_data = false;
            view.stream = stream;
            p_op_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_v1) { cudaFree(saved_v1); saved_v1 = nullptr; }
        if (saved_v2) { cudaFree(saved_v2); saved_v2 = nullptr; }
        if (saved_op_idx) { cudaFree(saved_op_idx); saved_op_idx = nullptr; }
        grad_v1 = nullptr;
        grad_v2 = nullptr;
        grad_p_op = nullptr;
        v1_grad_fn.reset();
        v2_grad_fn.reset();
        p_op_grad_fn.reset();
    }
};

// Fix #6: Penalty for selecting division when |v2| < eps.
// Forward:  penalty = flag * weight * p_op[div_idx]
// Backward: d_p_op[div_idx] = flag * weight * grad_out   (others zero)
// Connects through p_op's softmax grad_fn → op_logits → W_op_select.
struct DivInvalidPenaltyGradFn : public GradFn {
    int* saved_div_flag = nullptr;    // device [1] — 0 or 1
    float weight = 0.0f;
    int div_op_idx = 3;
    int num_ops = 4;

    float* grad_p_op = nullptr;
    std::shared_ptr<float> owned_grad_p_op;
    std::shared_ptr<GradFn> p_op_grad_fn;
    TensorContract::TensorShape p_op_shape;
    bool p_op_requires_grad = false;

    DivInvalidPenaltyGradFn() { op_name = "div_invalid_penalty"; }
    ~DivInvalidPenaltyGradFn() override {
        if (saved_div_flag) { cudaFree(saved_div_flag); saved_div_flag = nullptr; }
    }

    void capture(const int* d_div_flag, float w, int op_idx, int nop,
                 Tensor& p_op_t, cudaStream_t stream) {
        weight = w;
        div_op_idx = op_idx;
        num_ops = nop;

        // Save a copy of the flag (it gets reset each step)
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_div_flag), sizeof(int), "div_penalty_saved_flag");
        cudaMemcpyAsync(saved_div_flag, d_div_flag, sizeof(int), cudaMemcpyDeviceToDevice, stream);

        p_op_requires_grad = p_op_t.requires_grad;
        p_op_shape = p_op_t.shape;
        p_op_grad_fn = p_op_t.grad_fn;

        // Allocate gradient buffer for p_op
        float* buf = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&buf), nop * sizeof(float), "div_penalty_grad_p_op");
        cudaMemsetAsync(buf, 0, nop * sizeof(float), stream);
        owned_grad_p_op = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
        grad_p_op = owned_grad_p_op.get();
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;

        kernelDivInvalidPenaltyBackward<<<1, 1, 0, stream>>>(
            grad_p_op, grad_output.data, saved_div_flag,
            weight, div_op_idx, num_ops);
        CUDA_CHECK_KERNEL();

        if (p_op_requires_grad && p_op_grad_fn) {
            Tensor view;
            view.data = grad_p_op;
            view.shape = p_op_shape;
            view.owns_data = false;
            view.stream = stream;
            p_op_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_div_flag) { cudaFree(saved_div_flag); saved_div_flag = nullptr; }
        grad_p_op = nullptr;
        p_op_grad_fn.reset();
    }
};

// ─── Fix #8: Division Magnitude Penalty ─────────────────────────────────────
// Penalizes large |v_out| after clamped division to prevent division exploit.
// Forward:  loss = div_flag * weight * |v_out|
// Backward: d_v_out = div_flag * weight * sign(v_out) * grad_out
// Gradient flows through v_out → FourOpMixGradFn → v1, v2.

__global__ void kernelDivMagnitudePenalty(
    float* __restrict__ out_loss,
    const float* __restrict__ v_out,
    const int* __restrict__ div_flag,
    float weight
) {
    if (threadIdx.x != 0) return;
    if (div_flag[0]) {
        out_loss[0] = weight * fabsf(v_out[0]);
    } else {
        out_loss[0] = 0.0f;
    }
}

__global__ void kernelDivMagnitudePenaltyBackward(
    float* __restrict__ grad_v_out,
    const float* __restrict__ grad_out,
    const float* __restrict__ saved_v_out,
    const int* __restrict__ saved_div_flag,
    float weight
) {
    if (threadIdx.x != 0) return;
    if (saved_div_flag[0]) {
        float sign_v = (saved_v_out[0] >= 0.0f) ? 1.0f : -1.0f;
        grad_v_out[0] += grad_out[0] * weight * sign_v;
    }
}

struct DivMagnitudePenaltyGradFn : public GradFn {
    float* saved_v_out = nullptr;
    int* saved_div_flag = nullptr;
    float weight_ = 0.0f;

    float* grad_v_out = nullptr;
    std::shared_ptr<float> owned_grad_v_out;
    std::shared_ptr<GradFn> v_out_grad_fn;
    TensorContract::TensorShape v_out_shape;
    bool v_out_requires_grad = false;

    DivMagnitudePenaltyGradFn() { op_name = "div_magnitude_penalty"; }
    ~DivMagnitudePenaltyGradFn() override {
        if (saved_v_out) cudaFree(saved_v_out);
        if (saved_div_flag) cudaFree(saved_div_flag);
    }

    void capture(Tensor& v_out_t, const int* d_div_flag, float w, cudaStream_t stream) {
        weight_ = w;
        v_out_requires_grad = v_out_t.requires_grad;
        v_out_shape = v_out_t.shape;
        v_out_grad_fn = v_out_t.grad_fn;

        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_v_out), sizeof(float), "div_mag_saved_vout");
        cudaMemcpyAsync(saved_v_out, v_out_t.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);

        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_div_flag), sizeof(int), "div_mag_saved_flag");
        cudaMemcpyAsync(saved_div_flag, d_div_flag, sizeof(int), cudaMemcpyDeviceToDevice, stream);

        if (v_out_requires_grad) {
            v_out_t.ensure_grad();
            if (v_out_t.is_leaf) {
                grad_v_out = v_out_t.grad_data();
            } else {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), sizeof(float), "div_mag_grad_vout");
                cudaMemsetAsync(buf, 0, sizeof(float), stream);
                owned_grad_v_out = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
                grad_v_out = owned_grad_v_out.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        if (!v_out_requires_grad || !grad_v_out) return;

        kernelDivMagnitudePenaltyBackward<<<1, 1, 0, stream>>>(
            grad_v_out, grad_output.data, saved_v_out, saved_div_flag, weight_);
        CUDA_CHECK_KERNEL();

        if (v_out_grad_fn) {
            Tensor view;
            view.data = grad_v_out;
            view.shape = v_out_shape;
            view.owns_data = false;
            view.stream = stream;
            v_out_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_v_out) { cudaFree(saved_v_out); saved_v_out = nullptr; }
        if (saved_div_flag) { cudaFree(saved_div_flag); saved_div_flag = nullptr; }
        grad_v_out = nullptr;
        owned_grad_v_out.reset();
        v_out_grad_fn.reset();
    }
};

// ─── Fix #7: Arg REINFORCE Loss ─────────────────────────────────────────────
// REINFORCE policy gradient for discrete arg selection.
// Forward:  loss = weight * detached_err * (-log p_arg1[k1] - log p_arg2[k2])
// Backward: d_p_arg1[k1] = -weight * detached_err / p_arg1[k1]  (all others 0)
//           d_p_arg2[k2] = -weight * detached_err / p_arg2[k2]  (all others 0)
// Then propagated through softmax grad_fn → arg_logits → W_arg_select.
//
// transition_err is DETACHED — just a scalar multiplier, no gradient into v_out/p_op.
// This is standard policy gradient: -reward * log π(action).

__global__ void kernelArgReinforceLossForward(
    float* __restrict__ out_loss,
    float* __restrict__ out_advantage,   // [1] written for backward to read
    const float* __restrict__ p_arg1,
    const float* __restrict__ p_arg2,
    const int* __restrict__ idx1,       // d_exec_idx[0]
    const int* __restrict__ idx2,       // d_exec_idx[1]
    const float* __restrict__ transition_err,  // detached scalar |v_out - target|
    float* __restrict__ baseline,       // [1] EMA baseline (read-modify-write)
    float baseline_decay,               // EMA decay rate (e.g. 0.99)
    float weight
) {
    if (threadIdx.x != 0) return;
    int k1 = idx1[0];
    int k2 = idx2[0];
    float err = transition_err[0];
    // Variance-reduction baseline: advantage = err - E[err]
    float b = baseline[0];
    float advantage = err - b;
    // Clamp advantage to prevent gradient explosions from error spikes
    constexpr float kAdvClamp = 5.0f;
    advantage = fminf(fmaxf(advantage, -kAdvClamp), kAdvClamp);
    // Update EMA baseline ONLY when weight > 0 (op matches teacher target).
    // Wrong-op cases must not pollute baseline statistics — they would distort
    // the advantage term on subsequent valid arg updates.
    if (weight > 0.0f) {
        float new_b = baseline_decay * b + (1.0f - baseline_decay) * err;
        constexpr float kBaselineMin = 0.0f;
        constexpr float kBaselineMax = 1e6f;
        baseline[0] = fminf(fmaxf(new_b, kBaselineMin), kBaselineMax);
    }
    // Save advantage for backward pass
    out_advantage[0] = advantage;
    // Clamp log for numerical safety (p near 0 → large -log)
    constexpr float kMinProb = 1e-7f;
    float nll1 = -logf(fmaxf(p_arg1[k1], kMinProb));
    float nll2 = -logf(fmaxf(p_arg2[k2], kMinProb));
    out_loss[0] = weight * advantage * (nll1 + nll2);
}

__global__ void kernelArgReinforceLossBackward(
    float* __restrict__ grad_p_arg1,    // [N] output — sparse write at k1 only
    float* __restrict__ grad_p_arg2,    // [N] output — sparse write at k2 only
    const float* __restrict__ grad_out, // [1]
    const int* __restrict__ idx1,
    const int* __restrict__ idx2,
    const float* __restrict__ saved_p_arg1,
    const float* __restrict__ saved_p_arg2,
    const float* __restrict__ saved_advantage,
    float weight,
    int N                               // distribution size
) {
    if (threadIdx.x != 0) return;
    int k1 = idx1[0];
    int k2 = idx2[0];
    float adv = saved_advantage[0];
    float g = grad_out[0];
    constexpr float kMinProb = 1e-7f;
    // d(loss)/d(p_arg1[k1]) = weight * advantage * (-1 / p_arg1[k1])
    if (grad_p_arg1) {
        float p1 = fmaxf(saved_p_arg1[k1], kMinProb);
        grad_p_arg1[k1] += g * weight * adv * (-1.0f / p1);
    }
    if (grad_p_arg2) {
        float p2 = fmaxf(saved_p_arg2[k2], kMinProb);
        grad_p_arg2[k2] += g * weight * adv * (-1.0f / p2);
    }
}

struct ArgReinforceLossGradFn : public GradFn {
    int* saved_idx1 = nullptr;
    int* saved_idx2 = nullptr;
    float* saved_p_arg1 = nullptr;
    float* saved_p_arg2 = nullptr;
    float* saved_advantage = nullptr;  // written by forward kernel, read by backward
    float weight_ = 0.0f;
    int N_ = 0;

    float* grad_p_arg1 = nullptr;
    float* grad_p_arg2 = nullptr;
    std::shared_ptr<float> owned_grad_p_arg1;
    std::shared_ptr<float> owned_grad_p_arg2;
    std::shared_ptr<GradFn> p_arg1_grad_fn;
    std::shared_ptr<GradFn> p_arg2_grad_fn;
    TensorContract::TensorShape p_arg1_shape;
    TensorContract::TensorShape p_arg2_shape;
    bool p_arg1_requires_grad = false;
    bool p_arg2_requires_grad = false;

    ArgReinforceLossGradFn() { op_name = "arg_reinforce_loss"; }

    ~ArgReinforceLossGradFn() override {
        if (saved_idx1) cudaFree(saved_idx1);
        if (saved_idx2) cudaFree(saved_idx2);
        if (saved_p_arg1) cudaFree(saved_p_arg1);
        if (saved_p_arg2) cudaFree(saved_p_arg2);
        if (saved_advantage) cudaFree(saved_advantage);
    }

    // Allocates saved_advantage buffer BEFORE kernel launch.
    // Forward kernel writes advantage into this buffer.
    // Backward reads it back.
    float* allocate_advantage_buffer(cudaStream_t stream) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_advantage), sizeof(float), "reinforce_saved_advantage");
        cudaMemsetAsync(saved_advantage, 0, sizeof(float), stream);
        return saved_advantage;
    }

    void capture(const int* d_idx1, const int* d_idx2,
                 Tensor& p_arg1_t, Tensor& p_arg2_t,
                 float weight, int N,
                 cudaStream_t stream) {
        weight_ = weight;
        N_ = N;

        // Save copies of indices and probabilities (they change each step)
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_idx1), sizeof(int), "reinforce_saved_idx1");
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_idx2), sizeof(int), "reinforce_saved_idx2");
        cudaMemcpyAsync(saved_idx1, d_idx1, sizeof(int), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(saved_idx2, d_idx2, sizeof(int), cudaMemcpyDeviceToDevice, stream);

        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_p_arg1), N * sizeof(float), "reinforce_saved_p_arg1");
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_p_arg2), N * sizeof(float), "reinforce_saved_p_arg2");
        cudaMemcpyAsync(saved_p_arg1, p_arg1_t.data, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(saved_p_arg2, p_arg2_t.data, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // saved_advantage was already allocated and written by the forward kernel
        // (via allocate_advantage_buffer + kernelArgReinforceLossForward)

        p_arg1_requires_grad = p_arg1_t.requires_grad;
        p_arg2_requires_grad = p_arg2_t.requires_grad;
        p_arg1_shape = p_arg1_t.shape;
        p_arg2_shape = p_arg2_t.shape;
        p_arg1_grad_fn = p_arg1_t.grad_fn;
        p_arg2_grad_fn = p_arg2_t.grad_fn;

        auto alloc_grad = [&](Tensor& t, float*& gp, std::shared_ptr<float>& owned, size_t n) {
            if (!t.requires_grad) return;
            t.ensure_grad();
            if (t.is_leaf) {
                gp = t.grad_data();
            } else {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), n * sizeof(float), "reinforce_grad_buf");
                cudaMemsetAsync(buf, 0, n * sizeof(float), stream);
                owned = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
                gp = owned.get();
            }
        };
        alloc_grad(p_arg1_t, grad_p_arg1, owned_grad_p_arg1, N);
        alloc_grad(p_arg2_t, grad_p_arg2, owned_grad_p_arg2, N);
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;

        kernelArgReinforceLossBackward<<<1, kWarpSize, 0, stream>>>(
            grad_p_arg1, grad_p_arg2,
            grad_output.data,
            saved_idx1, saved_idx2,
            saved_p_arg1, saved_p_arg2,
            saved_advantage, weight_, N_);
        CUDA_CHECK_KERNEL();

        if (p_arg1_requires_grad && p_arg1_grad_fn) {
            Tensor view;
            view.data = grad_p_arg1;
            view.shape = p_arg1_shape;
            view.owns_data = false;
            view.stream = stream;
            p_arg1_grad_fn->apply(view, stream);
        }
        if (p_arg2_requires_grad && p_arg2_grad_fn) {
            Tensor view;
            view.data = grad_p_arg2;
            view.shape = p_arg2_shape;
            view.owns_data = false;
            view.stream = stream;
            p_arg2_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_idx1) { cudaFree(saved_idx1); saved_idx1 = nullptr; }
        if (saved_idx2) { cudaFree(saved_idx2); saved_idx2 = nullptr; }
        if (saved_p_arg1) { cudaFree(saved_p_arg1); saved_p_arg1 = nullptr; }
        if (saved_p_arg2) { cudaFree(saved_p_arg2); saved_p_arg2 = nullptr; }
        if (saved_advantage) { cudaFree(saved_advantage); saved_advantage = nullptr; }
        grad_p_arg1 = nullptr;
        grad_p_arg2 = nullptr;
        p_arg1_grad_fn.reset();
        p_arg2_grad_fn.reset();
    }
};

struct ExecutionBlockInjectGradFn : public GradFn {
    float* saved_result_emb = nullptr;
    float* saved_H_slot = nullptr;
    float* saved_gate = nullptr;
    float* mod_grad_buf = nullptr;
    float inv_sqrt_d = 0.0f;
    float gate_temp = 0.5f;
    int result_slot = 0;
    int total_tokens = 0;
    int d_model = 0;

    float* grad_result_emb = nullptr;
    float* w_gate_data = nullptr;
    float* w_gate_grad = nullptr;
    std::shared_ptr<float> owned_grad_result;

    std::shared_ptr<GradFn> H_grad_fn;
    std::shared_ptr<GradFn> result_grad_fn;
    TensorContract::TensorShape H_shape;
    TensorContract::TensorShape result_shape;
    bool H_requires_grad = false;
    bool result_requires_grad = false;

    ExecutionBlockInjectGradFn() { op_name = "exec_inject_slot"; }

    ~ExecutionBlockInjectGradFn() override {
        if (saved_result_emb) cudaFree(saved_result_emb);
        if (saved_H_slot) cudaFree(saved_H_slot);
        if (saved_gate) cudaFree(saved_gate);
        if (mod_grad_buf) cudaFree(mod_grad_buf);
    }

    void capture(Tensor& H_t, Tensor& result_t, Tensor& w_gate_t,
                 float* gate_device, float* H_slot_device,
                 float inv_sqrt_d_, float gate_temp_,
                 int result_slot_, int total_tokens_, int d_model_,
                 cudaStream_t stream) {
        inv_sqrt_d = inv_sqrt_d_;
        gate_temp = gate_temp_;
        result_slot = result_slot_;
        total_tokens = total_tokens_;
        d_model = d_model_;

        saved_gate = gate_device;
        saved_H_slot = H_slot_device;

        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_result_emb), d_model_ * sizeof(float), "datastream_saved_result_emb");
        cudaMemcpyAsync(saved_result_emb, result_t.data, d_model_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        size_t total_size = static_cast<size_t>(total_tokens_) * d_model_ * sizeof(float);
        cudaMallocOrThrow(reinterpret_cast<void**>(&mod_grad_buf), total_size, "datastream_mod_grad_buf");

        w_gate_data = w_gate_t.data;
        w_gate_t.ensure_grad();
        w_gate_grad = w_gate_t.grad_data();

        H_requires_grad = H_t.requires_grad;
        result_requires_grad = result_t.requires_grad;
        H_shape = H_t.shape;
        result_shape = result_t.shape;
        H_grad_fn = H_t.grad_fn;
        result_grad_fn = result_t.grad_fn;

        if (result_requires_grad) {
            result_t.ensure_grad();
            if (result_t.is_leaf) {
                grad_result_emb = result_t.grad_data();
            } else {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), d_model_ * sizeof(float), "datastream_grad_result_emb");
                cudaMemsetAsync(buf, 0, d_model_ * sizeof(float), stream);
                owned_grad_result = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
                grad_result_emb = owned_grad_result.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;

        size_t total_size = static_cast<size_t>(total_tokens) * d_model * sizeof(float);
        cudaMemcpyAsync(mod_grad_buf, grad_output.data, total_size, cudaMemcpyDeviceToDevice, stream);
        float* slot_grad = mod_grad_buf + static_cast<size_t>(result_slot) * d_model;

        if (!saved_gate || !saved_result_emb || !saved_H_slot)
            throw std::runtime_error("ExecutionBlockInjectGradFn::apply: saved state is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));

        // The backward kernel MUST always run — it computes d_logit, writes
        // grad_w_gate, and corrects the slot gradient in mod_grad_buf.
        // When result_emb is frozen/detached, provide a scratch buffer for
        // grad_result so the kernel can still write there (output is discarded).
        float* kernel_grad_result = grad_result_emb;
        std::shared_ptr<float> discard_buf;
        if (!kernel_grad_result) {
            float* tmp = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&tmp), d_model * sizeof(float), "inject_discard_grad_result");
            cudaMemsetAsync(tmp, 0, d_model * sizeof(float), stream);
            discard_buf = std::shared_ptr<float>(tmp, [](float* p) { cudaFree(p); });
            kernel_grad_result = discard_buf.get();
        }

        kernelInjectSlotBackward<<<1, kBlockSize, 0, stream>>>(
            kernel_grad_result,
            w_gate_grad,
            slot_grad,
            saved_result_emb,
            saved_H_slot,
            w_gate_data,
            saved_gate,
            inv_sqrt_d,
            gate_temp,
            d_model);
        CUDA_CHECK_KERNEL();

        if (result_requires_grad && result_grad_fn && grad_result_emb) {
            Tensor view;
            view.data = grad_result_emb;
            view.shape = result_shape;
            view.owns_data = false;
            view.stream = stream;
            result_grad_fn->apply(view, stream);
        }

        if (H_requires_grad && H_grad_fn && mod_grad_buf) {
            Tensor view;
            view.data = mod_grad_buf;
            view.shape = H_shape;
            view.owns_data = false;
            view.stream = stream;
            H_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_result_emb) { cudaFree(saved_result_emb); saved_result_emb = nullptr; }
        if (saved_H_slot) { cudaFree(saved_H_slot); saved_H_slot = nullptr; }
        if (saved_gate) { cudaFree(saved_gate); saved_gate = nullptr; }
        if (mod_grad_buf) { cudaFree(mod_grad_buf); mod_grad_buf = nullptr; }
        grad_result_emb = nullptr;
        w_gate_data = nullptr;
        w_gate_grad = nullptr;
        H_grad_fn.reset();
        result_grad_fn.reset();
    }
};

struct ReduceMeanGradFn : public GradFn {
    int total_tokens_ = 0;
    int d_model_ = 0;
    int token_offset_ = 0;
    int row_tokens_ = 0;

    std::shared_ptr<GradFn> H_grad_fn;
    TensorContract::TensorShape H_shape;
    bool H_requires_grad = false;
    bool H_is_leaf_ = false;
    float* grad_H_buf = nullptr;

    ReduceMeanGradFn() { op_name = "reduce_mean"; }

    ~ReduceMeanGradFn() override {
        if (!H_is_leaf_ && grad_H_buf) cudaFree(grad_H_buf);
    }

    void capture(Tensor& H, int total_tokens, int d_model, cudaStream_t stream,
                 int token_offset = 0, int row_tokens = -1) {
        total_tokens_ = total_tokens;
        d_model_ = d_model;
        token_offset_ = token_offset;
        row_tokens_ = (row_tokens == -1) ? total_tokens : row_tokens;
        H_requires_grad = H.requires_grad;
        H_shape = H.shape;
        H_grad_fn = H.grad_fn;
        H_is_leaf_ = H.is_leaf;

        if (H_requires_grad) {
            H.ensure_grad();
            if (H.is_leaf) {
                grad_H_buf = H.grad_data();
            } else {
                size_t total = static_cast<size_t>(total_tokens) * d_model;
                cudaMallocOrThrow(reinterpret_cast<void**>(&grad_H_buf), total * sizeof(float), "datastream_grad_H_buf");
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        if (!H_requires_grad) return;

        if (!H_is_leaf_) {
            size_t total = static_cast<size_t>(total_tokens_) * d_model_;
            CUDA_CHECK(cudaMemsetAsync(grad_H_buf, 0, total * sizeof(float), stream));
        }
        kernelReduceMeanBackward<<<row_tokens_, kBlockSize, 0, stream>>>(
            grad_H_buf + static_cast<size_t>(token_offset_) * d_model_,
            grad_output.data,
            row_tokens_,
            d_model_);
        CUDA_CHECK_KERNEL();

        if (H_grad_fn) {
            Tensor view;
            view.data = grad_H_buf;
            view.shape = H_shape;
            view.owns_data = false;
            view.stream = stream;
            H_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (!H_is_leaf_ && grad_H_buf) { cudaFree(grad_H_buf); }
        grad_H_buf = nullptr;
        H_grad_fn.reset();
    }
};

struct RecordEncodeGradFn : public GradFn {
    int N_ = 0;
    int num_slots_ = 0;
    int num_ops_ = 0;
    int d_model_ = 0;

    int* saved_slot1_ = nullptr;
    int* saved_slot2_ = nullptr;
    int* saved_ops_ = nullptr;
    float* saved_scalars_ = nullptr;

    float* E_slot_grad_ = nullptr;
    float* E_op_grad_ = nullptr;
    float* W_scal_grad_ = nullptr;
    float* b_scal_grad_ = nullptr;

    RecordEncodeGradFn() { op_name = "record_encode"; }

    ~RecordEncodeGradFn() override {
        if (saved_slot1_) cudaFree(saved_slot1_);
        if (saved_slot2_) cudaFree(saved_slot2_);
        if (saved_ops_) cudaFree(saved_ops_);
        if (saved_scalars_) cudaFree(saved_scalars_);
    }

    void capture(int N, int num_slots, int num_ops, int d_model,
                 const int* d_slot1, const int* d_slot2,
                 const int* d_ops, const float* d_scalars,
                 Tensor& E_slot, Tensor& E_op,
                 Tensor& W_scal, Tensor& b_scal,
                 cudaStream_t stream) {
        N_ = N;
        num_slots_ = num_slots;
        num_ops_ = num_ops;
        d_model_ = d_model;

        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_slot1_), N * sizeof(int), "datastream_saved_slot1");
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_slot2_), N * sizeof(int), "datastream_saved_slot2");
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_ops_), N * sizeof(int), "datastream_saved_ops");
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_scalars_), N * 3 * sizeof(float), "datastream_saved_scalars");
        CUDA_CHECK(cudaMemcpyAsync(saved_slot1_, d_slot1, N * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(saved_slot2_, d_slot2, N * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(saved_ops_, d_ops, N * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(saved_scalars_, d_scalars, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        E_slot_grad_ = E_slot.grad_data();
        E_op_grad_ = E_op.grad_data();
        W_scal_grad_ = W_scal.grad_data();
        b_scal_grad_ = b_scal.grad_data();
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        int total = N_ * d_model_;
        int blocks = (total + kBlockSize - 1) / kBlockSize;
        kernelEncodeRecordsBackward<<<blocks, kBlockSize, 0, stream>>>(
            grad_output.data,
            E_slot_grad_,
            E_op_grad_,
            W_scal_grad_,
            b_scal_grad_,
            saved_slot1_,
            saved_slot2_,
            saved_ops_,
            saved_scalars_,
            N_,
            num_slots_,
            num_ops_,
            d_model_);
        CUDA_CHECK_KERNEL();
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_slot1_) { cudaFree(saved_slot1_); saved_slot1_ = nullptr; }
        if (saved_slot2_) { cudaFree(saved_slot2_); saved_slot2_ = nullptr; }
        if (saved_ops_) { cudaFree(saved_ops_); saved_ops_ = nullptr; }
        if (saved_scalars_) { cudaFree(saved_scalars_); saved_scalars_ = nullptr; }
    }
};

struct L1ScalarLossGradFn : public GradFn {
    float* saved_a_ = nullptr;
    float* saved_b_ = nullptr;
    float* grad_a_ = nullptr;
    std::shared_ptr<float> owned_grad_a_;

    std::shared_ptr<GradFn> a_grad_fn_;
    TensorContract::TensorShape a_shape_;
    bool a_requires_grad_ = false;

    L1ScalarLossGradFn() { op_name = "l1_scalar_loss"; }

    ~L1ScalarLossGradFn() override {
        if (saved_a_) cudaFree(saved_a_);
        if (saved_b_) cudaFree(saved_b_);
    }

    void capture(Tensor& a_tensor, const float* target_device_ptr, cudaStream_t stream) {
        a_requires_grad_ = a_tensor.requires_grad;
        a_shape_ = a_tensor.shape;
        a_grad_fn_ = a_tensor.grad_fn;

        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_a_), sizeof(float), "datastream_saved_a");
        CUDA_CHECK(cudaMemcpyAsync(saved_a_, a_tensor.data, sizeof(float), cudaMemcpyDeviceToDevice, stream));
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_b_), sizeof(float), "datastream_saved_b");
        CUDA_CHECK(cudaMemcpyAsync(saved_b_, target_device_ptr, sizeof(float), cudaMemcpyDeviceToDevice, stream));

        if (a_requires_grad_) {
            a_tensor.ensure_grad();
            if (a_tensor.is_leaf) {
                grad_a_ = a_tensor.grad_data();
            } else {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), sizeof(float), "datastream_grad_a");
                CUDA_CHECK(cudaMemsetAsync(buf, 0, sizeof(float), stream));
                owned_grad_a_ = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
                grad_a_ = owned_grad_a_.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        if (!a_requires_grad_ || !grad_a_) return;

        kernelL1LossBackward<<<1, 1, 0, stream>>>(
            grad_a_, grad_output.data, saved_a_, saved_b_);
        CUDA_CHECK_KERNEL();

        if (a_grad_fn_) {
            Tensor view;
            view.data = grad_a_;
            view.shape = a_shape_;
            view.owns_data = false;
            view.stream = stream;
            a_grad_fn_->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_a_) { cudaFree(saved_a_); saved_a_ = nullptr; }
        if (saved_b_) { cudaFree(saved_b_); saved_b_ = nullptr; }
        grad_a_ = nullptr;
        owned_grad_a_.reset();
        a_grad_fn_.reset();
    }
};

// Fix #5: Gated trace update GradFn.
// Forward: out = gate * old_trace + (1 - gate) * candidate,  gate = sigmoid(gate_logits)
// Backward chains to old_trace, candidate (from W_reason_gate matmul), gate_logits (from W_trace_gate matmul).
struct GatedTraceUpdateGradFn : public GradFn {
    float* saved_old_trace = nullptr;   // [dm]
    float* saved_candidate = nullptr;   // [dm]
    float* saved_gate_vals = nullptr;   // [dm] sigmoid outputs — ownership transferred at capture
    int dm_ = 0;

    float* grad_old_trace = nullptr;
    float* grad_candidate = nullptr;
    float* grad_gate_logits = nullptr;
    std::shared_ptr<float> owned_grad_old_trace;
    std::shared_ptr<float> owned_grad_candidate;
    std::shared_ptr<float> owned_grad_gate_logits;
    std::shared_ptr<GradFn> old_trace_grad_fn;
    std::shared_ptr<GradFn> candidate_grad_fn;
    std::shared_ptr<GradFn> gate_logits_grad_fn;
    TensorContract::TensorShape old_trace_shape;
    TensorContract::TensorShape candidate_shape;
    TensorContract::TensorShape gate_logits_shape;
    bool old_trace_requires_grad = false;
    bool candidate_requires_grad = false;
    bool gate_logits_requires_grad = false;

    GatedTraceUpdateGradFn() { op_name = "gated_trace_update"; }

    ~GatedTraceUpdateGradFn() override {
        if (saved_old_trace) cudaFree(saved_old_trace);
        if (saved_candidate) cudaFree(saved_candidate);
        if (saved_gate_vals) cudaFree(saved_gate_vals);
    }

    void capture(Tensor& old_trace_t, Tensor& candidate_t, Tensor& gate_logits_t,
                 float* gate_vals_buf, int dm, cudaStream_t stream) {
        dm_ = dm;
        saved_gate_vals = gate_vals_buf;  // ownership transferred

        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_old_trace), dm * sizeof(float), "gated_trace_saved_old");
        cudaMallocOrThrow(reinterpret_cast<void**>(&saved_candidate), dm * sizeof(float), "gated_trace_saved_cand");
        cudaMemcpyAsync(saved_old_trace, old_trace_t.data, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(saved_candidate, candidate_t.data, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        old_trace_requires_grad = old_trace_t.requires_grad;
        candidate_requires_grad = candidate_t.requires_grad;
        gate_logits_requires_grad = gate_logits_t.requires_grad;
        old_trace_shape = old_trace_t.shape;
        candidate_shape = candidate_t.shape;
        gate_logits_shape = gate_logits_t.shape;
        old_trace_grad_fn = old_trace_t.grad_fn;
        candidate_grad_fn = candidate_t.grad_fn;
        gate_logits_grad_fn = gate_logits_t.grad_fn;

        auto alloc_grad = [&](Tensor& t, float*& gp, std::shared_ptr<float>& owned, size_t n) {
            if (!t.requires_grad) return;
            t.ensure_grad();
            if (t.is_leaf) {
                gp = t.grad_data();
            } else {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), n * sizeof(float), "gated_trace_grad_buf");
                cudaMemsetAsync(buf, 0, n * sizeof(float), stream);
                owned = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
                gp = owned.get();
            }
        };

        alloc_grad(old_trace_t, grad_old_trace, owned_grad_old_trace, dm);
        alloc_grad(candidate_t, grad_candidate, owned_grad_candidate, dm);
        alloc_grad(gate_logits_t, grad_gate_logits, owned_grad_gate_logits, dm);
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;

        int blocks = (dm_ + kBlockSize - 1) / kBlockSize;
        kernelGatedTraceUpdateBackward<<<blocks, kBlockSize, 0, stream>>>(
            grad_old_trace, grad_candidate, grad_gate_logits,
            grad_output.data,
            saved_old_trace, saved_candidate, saved_gate_vals,
            dm_);
        CUDA_CHECK_KERNEL();

        if (old_trace_requires_grad && old_trace_grad_fn) {
            Tensor view;
            view.data = grad_old_trace;
            view.shape = old_trace_shape;
            view.owns_data = false;
            view.stream = stream;
            old_trace_grad_fn->apply(view, stream);
        }
        if (candidate_requires_grad && candidate_grad_fn) {
            Tensor view;
            view.data = grad_candidate;
            view.shape = candidate_shape;
            view.owns_data = false;
            view.stream = stream;
            candidate_grad_fn->apply(view, stream);
        }
        if (gate_logits_requires_grad && gate_logits_grad_fn) {
            Tensor view;
            view.data = grad_gate_logits;
            view.shape = gate_logits_shape;
            view.owns_data = false;
            view.stream = stream;
            gate_logits_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_old_trace) { cudaFree(saved_old_trace); saved_old_trace = nullptr; }
        if (saved_candidate) { cudaFree(saved_candidate); saved_candidate = nullptr; }
        if (saved_gate_vals) { cudaFree(saved_gate_vals); saved_gate_vals = nullptr; }
        grad_old_trace = nullptr;
        grad_candidate = nullptr;
        grad_gate_logits = nullptr;
        old_trace_grad_fn.reset();
        candidate_grad_fn.reset();
        gate_logits_grad_fn.reset();
    }
};

static void copyStepDiagnostics(const StepWorkingSet& work,
                                ExecutionBlockStepOutput* diag_out,
                                int V_val,
                                int nop,
                                int V,
                                int dm,
                                cudaStream_t stream) {
    if (!diag_out) return;

    diag_out->p_arg1 = Tensor::zeros({1, V_val}, stream);
    cudaMemcpyAsync(diag_out->p_arg1.data, work.p_arg1.data, V_val * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    diag_out->p_arg2 = Tensor::zeros({1, V_val}, stream);
    cudaMemcpyAsync(diag_out->p_arg2.data, work.p_arg2.data, V_val * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    diag_out->p_op = Tensor::zeros({1, nop}, stream);
    cudaMemcpyAsync(diag_out->p_op.data, work.p_op.data, nop * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    diag_out->p_write = Tensor::zeros({1, V}, stream);
    cudaMemcpyAsync(diag_out->p_write.data, work.p_write.data, V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    diag_out->v_out = Tensor::zeros({1, 1}, stream);
    cudaMemcpyAsync(diag_out->v_out.data, work.v_out.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);
    diag_out->result_emb = Tensor::zeros({1, dm}, stream);
    cudaMemcpyAsync(diag_out->result_emb.data, work.result_emb.data, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

static void collectStepMetrics(ExecutionBlockLayer& layer,
                               const StepWorkingSet& work,
                               ExecutionBlockStepOutput* diag_out,
                               int V_val,
                               int nop,
                               int V,
                               cudaStream_t stream) {
    if (!diag_out || !layer.config().debug_mode) return;

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
    CUDA_CHECK(cudaMemcpyAsync(&div_count, LayerAccess::divClampCount(layer), sizeof(int), cudaMemcpyDeviceToHost, stream));

    float op_dist[4] = {};
    int op_copy = (nop <= 4) ? nop : 4;
    CUDA_CHECK(cudaMemcpyAsync(op_dist, work.p_op.data, op_copy * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    diag_out->metrics.arg1_entropy = d_metrics[0];
    diag_out->metrics.arg2_entropy = d_metrics[1];
    diag_out->metrics.op_entropy = d_metrics[2];
    diag_out->metrics.write_entropy = d_metrics[3];
    diag_out->metrics.max_p_write = d_metrics[4];
    diag_out->metrics.div_clamp_count = div_count;
    for (int k = 0; k < 4; ++k)
        diag_out->metrics.op_distribution[k] = (k < op_copy) ? op_dist[k] : 0.0f;

    // Emit per-step metrics via module log system
    {
        char msg[384];
        snprintf(msg, sizeof(msg),
            "[EXEC_STEP_EQUATION] entropy: H(arg1)=%.4f H(arg2)=%.4f H(op)=%.4f H(write)=%.4f "
            "max_p_write=%.4f div_clamps=%d op_dist=[%.3f,%.3f,%.3f,%.3f]",
            d_metrics[0], d_metrics[1], d_metrics[2], d_metrics[3],
            d_metrics[4], div_count,
            diag_out->metrics.op_distribution[0],
            diag_out->metrics.op_distribution[1],
            diag_out->metrics.op_distribution[2],
            diag_out->metrics.op_distribution[3]);
        GRIM::Logging::EmitModuleInfo(
            GRIM::Logging::ModuleId::ExecutionBlock, msg);
    }

    cudaFree(d_buf);
}

void executeStepCoordinatorImpl(
    ExecutionBlockLayer& layer,
    Tensor& H,
    ExecutionMemory& memory,
    const int* atom_positions,
    int num_atoms,
    const Batching::BatchPayload& payload,
    int batch_row,
    int step,
    float temperature,
    cudaStream_t stream,
    ExecutionBlockStepOutput* diag_out,
    Tensor& trace_state,
    const std::vector<ExecutionRecord>& prior_records,
    const float* expected_target,
    const TeacherSelectionTargets* selection_targets
) {
    const int dm = layer.config().d_model;
    const int V = layer.config().num_slots;
    const int S = layer.config().num_scratch_slots;
    const int V_val = V - S;
    const int dk = layer.config().d_key;
    const int nop = layer.config().num_ops;
    const int ae = layer.config().atom_embedding_dim;
    const int vid = layer.config().value_decode_input_dim;
    const int vhd = layer.config().value_decode_hidden_dim;
    int* d_exec_idx = LayerAccess::execIndices(layer);
    int* d_exec_record_i = LayerAccess::execRecordI(layer);
    float* d_exec_record_f = LayerAccess::execRecordF(layer);

    // Row-local device slot map: derived from payload's device mirror
    const int32_t* d_slot_map_row = payload.d_token_to_slot_map
        + static_cast<size_t>(batch_row) * payload.max_seq_len;

    StepWorkingSet work;
    prepareMemoryStepOrThrow(layer, memory, atom_positions, d_slot_map_row, num_atoms, payload.max_seq_len, diag_out, stream);
    buildValueSlotCandidates(layer, memory, stream, work);
    kernelCheckFinite<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.slot_values.data, V_val, LayerAccess::numericErrorFlag(layer), kStageV1, layer.config().magnitude_limit);

    work.context = Tensor::zeros({1, dm}, stream, "exec_context");
    work.context.requires_grad = true;
    work.context.is_leaf = false;
    kernelReduceMeanForward<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.context.data,
        H.data + static_cast<size_t>(batch_row) * payload.max_seq_len * dm,
        payload.max_seq_len,
        dm);
    CUDA_CHECK_KERNEL();
    {
        auto mean_fn = std::make_shared<ReduceMeanGradFn>();
        mean_fn->capture(H, payload.total_tokens, dm, stream, batch_row * payload.max_seq_len, payload.max_seq_len);
        work.context.grad_fn = mean_fn;
    }

    const int K = layer.config().num_exec_steps;
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
        encoded_records.requires_grad = true;
        encoded_records.is_leaf = false;
        int total_enc = N_prior * dm;
        kernelEncodeRecords<<<(total_enc + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            encoded_records.data,
            layer.E_slot().data,
            layer.E_op().data,
            layer.W_scal().data,
            layer.b_scal().data,
            d_slot1,
            d_slot2,
            d_ops,
            d_scalars,
            N_prior,
            V,
            nop,
            dm);
        CUDA_CHECK_KERNEL();

        {
            auto enc_fn = std::make_shared<RecordEncodeGradFn>();
            enc_fn->capture(N_prior, V, nop, dm, d_slot1, d_slot2, d_ops, d_scalars,
                            layer.E_slot(), layer.E_op(), layer.W_scal(), layer.b_scal(), stream);
            encoded_records.grad_fn = enc_fn;
        }

        auto padded = Tensor::zeros({K, dm}, stream, "exec_trace_padded");
        int copy_rows = (N_prior < K) ? N_prior : K;
        int src_offset = (N_prior > K) ? (N_prior - K) : 0;
        CUDA_CHECK(cudaMemcpyAsync(padded.data,
            encoded_records.data + static_cast<size_t>(src_offset) * dm,
            copy_rows * dm * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        padded.requires_grad = encoded_records.requires_grad;
        padded.is_leaf = false;

        auto flattened = Tensor::zeros({1, K * dm}, stream, "exec_trace_flat");
        CUDA_CHECK(cudaMemcpyAsync(flattened.data, padded.data, K * dm * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        flattened.requires_grad = padded.requires_grad;
        flattened.is_leaf = false;

        work.trace_vec = autograd::matmul(flattened, layer.W_trace(), stream, flattened.data, nullptr);
        work.trace_vec = autograd::add(work.trace_vec, layer.b_trace(), stream);

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
        layer.step_embeddings().data + static_cast<size_t>(step) * dm,
        dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto dec_12 = autograd::concat(work.context_enriched, trace_state, stream);
    auto decision_input = autograd::concat(dec_12, work.step_emb, stream);

    auto query1 = autograd::matmul(decision_input, layer.w_arg1_select(), stream, decision_input.data, nullptr);
    auto arg1_logits = autograd::matmul(query1, work.cand_hidden, stream, nullptr, nullptr, true);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg1_logits.data, arg1_logits.data, work.cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    // ── Selection CE for arg1 (logits-space, before softmax) ──
    if (selection_targets && selection_targets->valid) {
        if (!diag_out) throw std::runtime_error("selection_targets->valid requires diag_out != nullptr");
        diag_out->selection_ce_arg1 = autograd::cross_entropy_logits(
            arg1_logits, selection_targets->arg1_target, stream);
    }
    work.p_arg1 = autograd::softmax(arg1_logits, temperature, stream);
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_arg1.data, V_val, LayerAccess::numericErrorFlag(layer), kStagePArg1);
    kernelCheckEntropyCollapse<<<1, kWarpSize, 0, stream>>>(
        work.p_arg1.data, V_val, LayerAccess::numericErrorFlag(layer), kStageEntropyArg1, layer.config().entropy_collapse_threshold);

    auto query2 = autograd::matmul(decision_input, layer.w_arg2_select(), stream, decision_input.data, nullptr);
    auto arg2_logits = autograd::matmul(query2, work.cand_hidden, stream, nullptr, nullptr, true);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg2_logits.data, arg2_logits.data, work.cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    // ── Selection CE for arg2 (logits-space, before softmax) ──
    if (selection_targets && selection_targets->valid) {
        if (!diag_out) throw std::runtime_error("selection_targets->valid requires diag_out != nullptr");
        diag_out->selection_ce_arg2 = autograd::cross_entropy_logits(
            arg2_logits, selection_targets->arg2_target, stream);
    }
    work.p_arg2 = autograd::softmax(arg2_logits, temperature, stream);
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_arg2.data, V_val, LayerAccess::numericErrorFlag(layer), kStagePArg2);
    kernelCheckEntropyCollapse<<<1, kWarpSize, 0, stream>>>(
        work.p_arg2.data, V_val, LayerAccess::numericErrorFlag(layer), kStageEntropyArg2, layer.config().entropy_collapse_threshold);

    materializeSelectedOperands(layer, memory, stream, work);
    kernelCheckFinite<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.slot_values.data, V_val, LayerAccess::numericErrorFlag(layer), kStageV1, layer.config().magnitude_limit);
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v1.data, 1, LayerAccess::numericErrorFlag(layer), kStageV1, layer.config().magnitude_limit);
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v2.data, 1, LayerAccess::numericErrorFlag(layer), kStageV2, layer.config().magnitude_limit);

    // Fix #1: v1/v2 are DETACHED from p_arg.  No gradient path from execution
    // value loss back into arg selection.  Only selection CE trains arg selection.
    // v1/v2 grad_fn are left as nullptr (set by materializeSelectedOperands).

    // [DELETED] h_arg1/h_arg2 = matmul(p_arg, cand_hidden)
    // These leaked arg selection into op and write heads. All three selection
    // heads (arg, op, write) are now fully independent classification tasks.

    // Op selection is DETACHED from arg selection (Option B).
    // op sees (context_enriched, trace_state, step_emb) only — same as decision_input.
    // arg selection cannot influence op selection through the gradient path.
    auto op_logits = autograd::matmul(decision_input, layer.W_op_select(), stream, decision_input.data, nullptr);
    // ── Selection CE for op (logits-space, before softmax) ──
    if (selection_targets && selection_targets->valid) {
        if (!diag_out) throw std::runtime_error("selection_targets->valid requires diag_out != nullptr");
        diag_out->selection_ce_op = autograd::cross_entropy_logits(
            op_logits, selection_targets->op_target, stream);
    }
    work.p_op = autograd::softmax(op_logits, temperature, stream);
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_op.data, nop, LayerAccess::numericErrorFlag(layer), kStagePOp);
    kernelCheckEntropyCollapse<<<1, kWarpSize, 0, stream>>>(
        work.p_op.data, nop, LayerAccess::numericErrorFlag(layer), kStageEntropyOp, layer.config().entropy_collapse_threshold);

    work.op_results = Tensor::zeros({1, nop}, stream, "exec_op_results");
    // Fix #6: Reset per-step division invalid flag before FourOps
    int* d_div_flag = LayerAccess::divInvalidFlag(layer);
    CUDA_CHECK(cudaMemsetAsync(d_div_flag, 0, sizeof(int), stream));
    kernelFourOps<<<1, kWarpSize, 0, stream>>>(work.op_results.data, work.v1.data, work.v2.data, kEps, LayerAccess::divClampCount(layer), d_div_flag);
    CUDA_CHECK_KERNEL();

    // Fix #2, Option A: Hard op selection (clean classification).
    // Forward: argmax(p_op) → pick discrete result.  v_out = results[hard_op].
    // Backward: grad_p_op is ZERO — no execution-value gradient into op selection.
    // The ONLY training signal for op is selection_ce_op (CE classification).
    // Blending +/-/*/÷ results is semantically meaningless for a register machine.
    kernelArgmax1DIntData<<<1, kWarpSize, 0, stream>>>(d_exec_idx + 2, work.p_op.data, nop);
    CUDA_CHECK_KERNEL();

    work.v_out = Tensor::zeros({1, 1}, stream, "exec_v_out");
    work.v_out.requires_grad = true;
    work.v_out.is_leaf = false;
    kernelHardPickOpForward<<<1, 1, 0, stream>>>(work.v_out.data, work.op_results.data, d_exec_idx + 2, nop);
    CUDA_CHECK_KERNEL();
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v_out.data, 1, LayerAccess::numericErrorFlag(layer), kStageVOut, layer.config().magnitude_limit);

    {
        auto grad_fn = std::make_shared<FourOpMixGradFn>();
        grad_fn->capture(work.v1, work.v2, work.p_op, d_exec_idx + 2,
                         nop, stream);
        work.v_out.grad_fn = grad_fn;
    }

    // Fix #6: Division invalid penalty — penalize p_op[3] when division was clamped.
    // Gradient: d_p_op[3] = flag * weight * grad_out → softmax → op_logits → W_op_select.
    // This teaches: "don't select ÷ when |v2| < eps."
    if (diag_out && layer.config().div_invalid_penalty_weight > 0.0f) {
        constexpr int kDivOpIdx = 3;
        const float penalty_w = layer.config().div_invalid_penalty_weight;

        diag_out->div_invalid_penalty = Tensor::zeros({1, 1}, stream, "exec_div_penalty");
        kernelDivInvalidPenalty<<<1, 1, 0, stream>>>(
            diag_out->div_invalid_penalty.data, work.p_op.data, d_div_flag, penalty_w, kDivOpIdx);
        CUDA_CHECK_KERNEL();

        auto pen_fn = std::make_shared<DivInvalidPenaltyGradFn>();
        pen_fn->capture(d_div_flag, penalty_w, kDivOpIdx, nop, work.p_op, stream);
        diag_out->div_invalid_penalty.grad_fn = pen_fn;
        diag_out->div_invalid_penalty.requires_grad = true;
        diag_out->div_invalid_penalty.is_leaf = false;
    }

    // Fix #8: Division magnitude penalty — penalize large |v_out| after clamped division.
    // Gradient: d_v_out = div_flag * weight * sign(v_out) → FourOpMixGradFn → v1, v2.
    // This teaches: "produce smaller operands so division doesn't blow up."
    if (diag_out && layer.config().div_magnitude_penalty_weight > 0.0f) {
        const float mag_w = layer.config().div_magnitude_penalty_weight;

        diag_out->div_magnitude_penalty = Tensor::zeros({1, 1}, stream, "exec_div_mag_penalty");
        kernelDivMagnitudePenalty<<<1, 1, 0, stream>>>(
            diag_out->div_magnitude_penalty.data, work.v_out.data, d_div_flag, mag_w);
        CUDA_CHECK_KERNEL();

        auto mag_fn = std::make_shared<DivMagnitudePenaltyGradFn>();
        mag_fn->capture(work.v_out, d_div_flag, mag_w, stream);
        diag_out->div_magnitude_penalty.grad_fn = mag_fn;
        diag_out->div_magnitude_penalty.requires_grad = true;
        diag_out->div_magnitude_penalty.is_leaf = false;
    }

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
            layer.E_slot().data,
            layer.E_op().data,
            layer.W_scal().data,
            layer.b_scal().data,
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
            auto enc_fn = std::make_shared<RecordEncodeGradFn>();
            enc_fn->capture(1, V, nop, dm,
                            d_exec_record_i, d_exec_record_i + 1,
                            d_exec_record_i + 2, d_exec_record_f,
                            layer.E_slot(), layer.E_op(), layer.W_scal(), layer.b_scal(), stream);
            cur_encoded.grad_fn = enc_fn;
        }

        auto update_input = autograd::concat(trace_state, cur_encoded, stream);
        auto candidate = autograd::matmul(update_input, layer.W_reason_gate(), stream, update_input.data, nullptr);
        auto gate_logits = autograd::matmul(update_input, layer.W_trace_gate(), stream, update_input.data, nullptr);

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
        trace_state = new_trace;
    }

    work.atom_new = Tensor::zeros({1, ae}, stream, "exec_atom_new");
    kernelEncodeScalarToAtomEmbed<<<1, ae, 0, stream>>>(work.atom_new.data, work.v_out.data, ae);
    CUDA_CHECK_KERNEL();

    auto decode_input = Tensor::zeros({1, vid}, stream, "exec_decode_input");
    decode_input.requires_grad = true;
    decode_input.is_leaf = true;
    cudaMemcpyAsync(decode_input.data, work.atom_new.data + 16, vid * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto decode_h = autograd::matmul(decode_input, layer.w_decode_1(), stream, decode_input.data, nullptr);
    decode_h = autograd::add(decode_h, layer.b_decode_1(), stream);

    auto decode_act = autograd::silu(decode_h, stream, decode_h.data);

    work.v_decoded = autograd::matmul(decode_act, layer.w_decode_2(), stream, decode_act.data, nullptr);
    work.result_emb = autograd::matmul(work.v_decoded, layer.W_value_to_emb(), stream, work.v_decoded.data, nullptr);
    work.result_emb = autograd::add(work.result_emb, layer.b_value_to_emb(), stream);
    kernelCheckFinite<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.result_emb.data, dm, LayerAccess::numericErrorFlag(layer), kStageResultEmb, layer.config().magnitude_limit);

    if (layer.config().result_slot_mode == 1 && layer.config().result_slot_index >= 0 &&
        layer.config().result_slot_index < payload.total_tokens) {
        work.result_slot = layer.config().result_slot_index;
    } else {
        work.result_slot = batch_row * payload.max_seq_len + payload.max_seq_len - 1;
    }
    EXEC_CHECK(work.result_slot >= 0 && work.result_slot < payload.total_tokens, "result_slot out of bounds");
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(dm));

    float* save_gate_buf = nullptr;
    float* save_H_slot_buf = nullptr;
    float h_inject_gate_value = 0.0f;  // host-side readback for telemetry
    cudaMallocOrThrow(reinterpret_cast<void**>(&save_gate_buf), sizeof(float), "datastream_save_gate");
    cudaMallocOrThrow(reinterpret_cast<void**>(&save_H_slot_buf), dm * sizeof(float), "datastream_save_H_slot");
    kernelInjectResultSlot<<<1, kBlockSize, 0, stream>>>(
        H.data, work.result_emb.data, layer.w_inject_gate().data,
        inv_sqrt_d, work.result_slot, dm, layer.config().inject_gate_temp,
        save_gate_buf, save_H_slot_buf);
    CUDA_CHECK_KERNEL();
    // Queue async readback of inject gate for telemetry (completes at next sync)
    if (diag_out && layer.config().debug_mode) {
        CUDA_CHECK(cudaMemcpyAsync(&h_inject_gate_value, save_gate_buf, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }

    {
        auto inject_fn = std::make_shared<ExecutionBlockInjectGradFn>();
        inject_fn->capture(H, work.result_emb, layer.w_inject_gate(),
                           save_gate_buf, save_H_slot_buf,
                           inv_sqrt_d, layer.config().inject_gate_temp,
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

    auto q_write = autograd::matmul(write_ctx, layer.W_write_query(), stream, write_ctx.data, nullptr);

    auto K_proj = autograd::matmul(memory.key_embeds, layer.W_write_key(), stream, memory.key_embeds.data, nullptr);

    auto usage_norm = Tensor::zeros({1, V}, stream, "exec_usage_norm");
    kernelNormalizeUsage<<<1, kWarpSize, 0, stream>>>(usage_norm.data, memory.usage.data, V);
    CUDA_CHECK_KERNEL();

    // ── Write logits via autograd ops (tape-connected to W_write_query, W_write_key, alpha, beta) ──
    // Formula: logits[i] = alpha * dot(q, K_proj[i]) + beta * (-usage[i])
    // Pure CE classification — no non-differentiable bias terms (Fix #4).
    //
    // content_scores = q_write @ K_proj^T → [1, V]
    auto content_scores = autograd::matmul(q_write, K_proj, stream,
        q_write.data, K_proj.data, /*transpose_b=*/true);

    // alpha_content = alpha [1,1] @ content_scores [1,V] → [1,V] (broadcast scalar multiply)
    auto alpha_content = autograd::matmul(layer.alpha(), content_scores, stream,
        layer.alpha().data, content_scores.data, /*transpose_b=*/false);

    // neg_usage: detached [1,V] = -usage_norm
    auto neg_usage = Tensor::zeros({1, V}, stream, "exec_neg_usage");
    kernelNegateVec<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        neg_usage.data, usage_norm.data, V);
    CUDA_CHECK_KERNEL();

    // beta_usage = beta [1,1] @ neg_usage [1,V] → [1,V] (broadcast scalar multiply)
    auto beta_usage = autograd::matmul(layer.beta(), neg_usage, stream,
        layer.beta().data, neg_usage.data, /*transpose_b=*/false);

    // Fix #4: Pure CE classification for write slot.
    // No non-differentiable bonus_bias — forward and backward see the same function.
    // write_logits = alpha*content + beta*(-usage) → [1,V]
    auto write_logits = autograd::add(alpha_content, beta_usage, stream);

    if (S > 0) {
        kernelMaskScratchSlots<<<(S + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(write_logits.data, S);
        CUDA_CHECK_KERNEL();
    }

    // ── Selection CE for write (logits-space, before softmax) ──
    if (selection_targets && selection_targets->valid) {
        if (!diag_out) throw std::runtime_error("selection_targets->valid requires diag_out != nullptr");
        diag_out->selection_ce_write = autograd::cross_entropy_logits(
            write_logits, selection_targets->write_target, stream);
        diag_out->has_selection_ce = true;
    }

    work.p_write = autograd::softmax(write_logits, temperature, stream);
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_write.data, V, LayerAccess::numericErrorFlag(layer), kStagePWrite);
    kernelCheckWriteCollapse<<<1, kWarpSize, 0, stream>>>(
        work.p_write.data, V, LayerAccess::numericErrorFlag(layer), kStageWriteCollapse, layer.config().write_collapse_threshold);

    if (diag_out) {
        // Hard recomputed result: re-evaluate the argmax-selected op from memory values
        auto recomputed_result = Tensor::zeros({1, 1}, stream, "exec_recomputed");
        kernelRecomputeOpResult<<<1, 1, 0, stream>>>(
            recomputed_result.data, memory.values.data, d_exec_idx, d_exec_idx + 1, d_exec_idx + 2,
            S, V, V_val, kEps);
        CUDA_CHECK_KERNEL();

        // transition_error_hard: |hard_recomputed - target|
        // Measures how far the argmax-selected op result is from the teacher target.
        const float* target_ptr = expected_target ? expected_target : work.v_out.data;
        diag_out->used_expected_target = (expected_target != nullptr);

        diag_out->transition_error_hard = Tensor::zeros({1, 1}, stream, "exec_te_hard");
        kernelAbsDiff<<<1, 1, 0, stream>>>(
            diag_out->transition_error_hard.data, recomputed_result.data, target_ptr, nullptr, 0, 0.0f);
        CUDA_CHECK_KERNEL();

        // transition_loss: |v_out - target| (autograd-connected through FourOpMixGradFn)
        // v_out is hard-picked results[argmax(p_op)]; backward uses hard derivative for v1/v2.
        // Gradient chain: transition_loss → L1ScalarLossGradFn → v_out → FourOpMixGradFn
        //   → v1,v2 (dead end — detached from p_arg per Fix #1)
        // Arg alignment is handled by Fix #7 (REINFORCE) below.
        diag_out->transition_loss = Tensor::zeros({1, 1}, stream, "exec_trans_loss");
        kernelAbsDiff<<<1, 1, 0, stream>>>(
            diag_out->transition_loss.data, work.v_out.data, target_ptr, nullptr, 0, 0.0f);
        CUDA_CHECK_KERNEL();

        {
            auto l1_fn = std::make_shared<L1ScalarLossGradFn>();
            l1_fn->capture(work.v_out, target_ptr, stream);
            diag_out->transition_loss.grad_fn = l1_fn;
            diag_out->transition_loss.requires_grad = true;
            diag_out->transition_loss.is_leaf = false;
        }

        // Fix #7: Arg REINFORCE — use advantage (err - baseline) as DETACHED reward.
        // loss = λ * (|v_out-target| - baseline) * (-log p_arg1[k1] - log p_arg2[k2])
        // Gradient: ONLY into arg logits (through p_arg softmax grad_fn).
        // Baseline is an EMA of transition_err for variance reduction.
        // Op-correctness gate: only train arg selection when op matches teacher target.
        // Wrong op → reward is meaningless for arg credit assignment.
        if (layer.config().arg_reinforce_weight > 0.0f) {
            float op_gate = 1.0f;
            if (selection_targets && selection_targets->valid) {
                int op_target = selection_targets->op_target;
                int h_op_idx = 0;
                CUDA_CHECK(cudaMemcpyAsync(&h_op_idx, d_exec_idx + 2, sizeof(int),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                op_gate = (h_op_idx == op_target) ? 1.0f : 0.0f;
            }

            int* d_exec_idx_local = LayerAccess::execIndices(layer);
            const float arf_w = layer.config().arg_reinforce_weight * op_gate;
            const float bld   = layer.config().arg_reinforce_baseline_decay;

            auto arf_fn = std::make_shared<ArgReinforceLossGradFn>();
            // Allocate advantage buffer BEFORE kernel launch — kernel writes advantage here
            float* d_advantage = arf_fn->allocate_advantage_buffer(stream);

            diag_out->arg_reinforce_loss = Tensor::zeros({1, 1}, stream, "exec_arg_reinforce");
            // transition_loss.data already contains |v_out - target| — use as detached reward
            kernelArgReinforceLossForward<<<1, 1, 0, stream>>>(
                diag_out->arg_reinforce_loss.data,
                d_advantage,
                work.p_arg1.data, work.p_arg2.data,
                d_exec_idx_local, d_exec_idx_local + 1,
                diag_out->transition_loss.data,  // DETACHED: just the scalar value
                LayerAccess::reinforceBaseline(layer),  // persistent EMA baseline
                bld,
                arf_w);
            CUDA_CHECK_KERNEL();

            arf_fn->capture(d_exec_idx_local, d_exec_idx_local + 1,
                           work.p_arg1, work.p_arg2,
                           arf_w, V_val, stream);
            diag_out->arg_reinforce_loss.grad_fn = arf_fn;
            diag_out->arg_reinforce_loss.requires_grad = true;
            diag_out->arg_reinforce_loss.is_leaf = false;
        }

        if (layer.config().transition_hard_threshold > 0.0f) {
            kernelAbsDiff<<<1, 1, 0, stream>>>(
                diag_out->transition_error_hard.data,
                recomputed_result.data,
                target_ptr,
                LayerAccess::numericErrorFlag(layer),
                kStageTransitionInvalid,
                layer.config().transition_hard_threshold);
            CUDA_CHECK_KERNEL();
        }
    }

    work.state_new = Tensor::zeros({1, dm}, stream, "exec_state_new");
    const float* step_emb_ptr = layer.step_embeddings().data + static_cast<size_t>(step) * dm;
    kernelAddVectors<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.state_new.data, work.result_emb.data, step_emb_ptr, dm);
    CUDA_CHECK_KERNEL();

    work.key_new = autograd::matmul(work.result_emb, layer.W_key_proj(), stream, work.result_emb.data, nullptr);
    applyHardWriteback(layer, memory, stream, work);
    copyStepDiagnostics(work, diag_out, V_val, nop, V, dm, stream);
    captureStateAfterWriteAndCheckMutations(layer, memory, diag_out, stream);
    collectStepMetrics(layer, work, diag_out, V_val, nop, V, stream);
    // collectStepMetrics synced the stream — h_inject_gate_value is now valid
    if (diag_out && layer.config().debug_mode) {
        diag_out->metrics.inject_gate_value = h_inject_gate_value;
    }
    finalizeStepOrThrow(layer, diag_out, step, stream);
}

}  // namespace GRIM::ExecutionBlockInternal

GRIM::Tensor GRIM::ExecutionBlockLayer::computeEntropyLoss(
    const std::vector<const GRIM::ExecutionBlockStepOutput*>& steps,
    float weight,
    cudaStream_t stream) const
{
    if (steps.empty() || weight <= 0.0f) {
        return GRIM::Tensor::zeros({1, 1}, stream, "exec_entropy_zero");
    }

    auto accum = GRIM::Tensor::zeros({1, 1}, stream, "exec_entropy_accum");
    int count = 0;

    for (const auto* s : steps) {
        auto accum_ent = [&](const GRIM::Tensor& probs, int n) {
            if (!probs.data || n <= 0) return;
            GRIM::ExecutionBlockInternal::kernelEntropyAccum<<<1, 32, 0, stream>>>(accum.data, probs.data, n);
            count++;
        };
        if (s->p_arg1.data) accum_ent(s->p_arg1, s->p_arg1.shape.flat.cols);
        if (s->p_arg2.data) accum_ent(s->p_arg2, s->p_arg2.shape.flat.cols);
        if (s->p_op.data) accum_ent(s->p_op, s->p_op.shape.flat.cols);
        if (s->p_write.data) accum_ent(s->p_write, s->p_write.shape.flat.cols);
    }

    auto result = GRIM::Tensor::zeros({1, 1}, stream, "exec_entropy_loss");
    GRIM::ExecutionBlockInternal::kernelScaleNegAvg<<<1, 1, 0, stream>>>(result.data, accum.data, weight, count);
    return result;
}

#endif  // USE_CUDA

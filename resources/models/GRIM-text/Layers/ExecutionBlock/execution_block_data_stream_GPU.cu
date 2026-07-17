#include "execution_block_data_stream_GPU.hpp"
#include "execution_block_memory_stream_GPU.hpp"
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM::ExecutionBlockInternal {

using GRIM::Forward::ExecutionBlockStepOutput;
using GRIM::Forward::ExecutionGateOutput;
using GRIM::Forward::ExecutionRecord;

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

// Backward: scatter gradient only to non-atom positions (matching forward mask).
__global__ void kernelReduceMeanBackward(
    float* __restrict__ grad_H,
    const float* __restrict__ grad_out,
    int total_tokens, int d_model,
    const uint8_t* __restrict__ atom_mask  // [total_tokens] or nullptr
) {
    const int i = blockIdx.x;
    if (i >= total_tokens) return;
    // Skip atom positions — they were excluded from the forward mean,
    // so they receive zero gradient from this path.
    if (atom_mask && atom_mask[i] != 0) return;
    // Count non-atom tokens to match forward scale.
    int count = 0;
    if (atom_mask) {
        for (int t = 0; t < total_tokens; ++t)
            if (atom_mask[t] == 0) ++count;
    } else {
        count = total_tokens;
    }
    // Guard: same fallback as forward — if all atoms, include all.
    if (count == 0) count = total_tokens;
    float scale = 1.0f / static_cast<float>(count);
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
    if (b_scal_grad) atomicAdd(&b_scal_grad[j], g);
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
        register_input(H_t.grad_fn);
        register_input(result_t.grad_fn);

        if (result_requires_grad) {
            if (result_t.is_leaf) {
                result_t.ensure_grad();
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

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
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
            result_grad_fn->apply(view, stream, backward_payload, backward_bindings);
        }

        if (H_requires_grad && H_grad_fn && mod_grad_buf) {
            Tensor view;
            view.data = mod_grad_buf;
            view.shape = H_shape;
            view.owns_data = false;
            view.stream = stream;
            H_grad_fn->apply(view, stream, backward_payload, backward_bindings);
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

    // Row-local atom mask (non-owning, points into batch-global device buffer).
    // When non-null, backward excludes atom positions matching the forward mask.
    const uint8_t* atom_mask_row_ = nullptr;

    ReduceMeanGradFn() { op_name = "reduce_mean"; }

    ~ReduceMeanGradFn() override {
        if (!H_is_leaf_ && grad_H_buf) cudaFree(grad_H_buf);
    }

    void capture(Tensor& H, int total_tokens, int d_model, cudaStream_t stream,
                 int token_offset = 0, int row_tokens = -1,
                 const uint8_t* atom_mask_row = nullptr) {
        total_tokens_ = total_tokens;
        d_model_ = d_model;
        token_offset_ = token_offset;
        row_tokens_ = (row_tokens == -1) ? total_tokens : row_tokens;
        atom_mask_row_ = atom_mask_row;
        H_requires_grad = H.requires_grad;
        H_shape = H.shape;
        H_grad_fn = H.grad_fn;
        register_input(H.grad_fn);
        H_is_leaf_ = H.is_leaf;

        if (H_requires_grad) {
            if (H.is_leaf) {
                H.ensure_grad();
                grad_H_buf = H.grad_data();
            } else {
                size_t total = static_cast<size_t>(total_tokens) * d_model;
                cudaMallocOrThrow(reinterpret_cast<void**>(&grad_H_buf), total * sizeof(float), "datastream_grad_H_buf");
            }
        }
    }

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
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
            d_model_,
            atom_mask_row_);
        CUDA_CHECK_KERNEL();

        if (H_grad_fn) {
            Tensor view;
            view.data = grad_H_buf;
            view.shape = H_shape;
            view.owns_data = false;
            view.stream = stream;
            H_grad_fn->apply(view, stream, backward_payload, backward_bindings);
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
        b_scal_grad_ = b_scal.data ? b_scal.grad_data() : nullptr;
    }

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
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
        register_input(old_trace_t.grad_fn);
        register_input(candidate_t.grad_fn);
        register_input(gate_logits_t.grad_fn);

        auto setup_grad_buf = [&](Tensor& t, float*& gp, std::shared_ptr<float>& owned, size_t n) {
            if (!t.requires_grad) return;
            if (t.is_leaf) {
                t.ensure_grad();
                gp = t.grad_data();
            } else {
                float* buf = nullptr;
                cudaMallocOrThrow(reinterpret_cast<void**>(&buf), n * sizeof(float), "gated_trace_grad_buf");
                cudaMemsetAsync(buf, 0, n * sizeof(float), stream);
                owned = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
                gp = owned.get();
            }
        };

        setup_grad_buf(old_trace_t, grad_old_trace, owned_grad_old_trace, dm);
        setup_grad_buf(candidate_t, grad_candidate, owned_grad_candidate, dm);
        setup_grad_buf(gate_logits_t, grad_gate_logits, owned_grad_gate_logits, dm);
    }

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
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
            old_trace_grad_fn->apply(view, stream, backward_payload, backward_bindings);
        }
        if (candidate_requires_grad && candidate_grad_fn) {
            Tensor view;
            view.data = grad_candidate;
            view.shape = candidate_shape;
            view.owns_data = false;
            view.stream = stream;
            candidate_grad_fn->apply(view, stream, backward_payload, backward_bindings);
        }
        if (gate_logits_requires_grad && gate_logits_grad_fn) {
            Tensor view;
            view.data = grad_gate_logits;
            view.shape = gate_logits_shape;
            view.owns_data = false;
            view.stream = stream;
            gate_logits_grad_fn->apply(view, stream, backward_payload, backward_bindings);
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

static void collectStepMetrics(const HyperParameters::ExecutionBlockConstructionHP& hp,
                               ExecutionBlockDiagnosticsBuffers& diag,
                               const StepWorkingSet& work,
                               ExecutionBlockStepOutput* diag_out,
                               int V_val,
                               int nop,
                               int V,
                               cudaStream_t stream) {
    if (!diag_out || !hp.debug_mode) return;

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

    diag_out->metrics.arg1_entropy = d_metrics[0];
    diag_out->metrics.arg2_entropy = d_metrics[1];
    diag_out->metrics.op_entropy = d_metrics[2];
    diag_out->metrics.write_entropy = d_metrics[3];
    diag_out->metrics.max_p_write = d_metrics[4];
    diag_out->metrics.div_clamp_count = div_count;
    for (int k = 0; k < 4; ++k)
        diag_out->metrics.op_distribution[k] = (k < op_copy) ? op_dist[k] : 0.0f;

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
            diag_out->metrics.op_distribution[0],
            diag_out->metrics.op_distribution[1],
            diag_out->metrics.op_distribution[2],
            diag_out->metrics.op_distribution[3]);
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
    ExecutionBlockStepOutput* diag_out,
    Tensor& trace_state,
    const std::vector<ExecutionRecord>& prior_records
) {
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
    int* d_exec_idx = diag.execIndices();
    int* d_exec_record_i = diag.execRecordI();
    float* d_exec_record_f = diag.execRecordF();

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

    StepWorkingSet work;
    prepareMemoryStepOrThrow(hp, diag, memory, d_atom_mask_row, d_slot_map_row, row_tokens, diag_out, stream);
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
        encoded_records.requires_grad = true;
        encoded_records.is_leaf = false;
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

        {
            auto enc_fn = std::make_shared<RecordEncodeGradFn>();
            enc_fn->capture(N_prior, V, nop, dm, d_slot1, d_slot2, d_ops, d_scalars,
                            params.E_slot, params.E_op, params.W_scal, params.b_scal, stream);
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
    auto arg1_logits = autograd::matmul(query1, work.cand_hidden, stream, true);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg1_logits.data, arg1_logits.data, work.cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    work.p_arg1 = autograd::softmax(arg1_logits, temperature, stream);
    if (diag_out) {
        diag_out->arg1_logits_tensor = std::move(arg1_logits);
        diag_out->selection_temperature = temperature;
    }
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_arg1.data, V_val, diag.numericErrorFlag(), kStagePArg1);

    auto query2 = autograd::matmul(decision_input, params.w_arg2_select, stream);
    auto arg2_logits = autograd::matmul(query2, work.cand_hidden, stream, true);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg2_logits.data, arg2_logits.data, work.cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    work.p_arg2 = autograd::softmax(arg2_logits, temperature, stream);
    if (diag_out) {
        diag_out->arg2_logits_tensor = std::move(arg2_logits);
    }
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_arg2.data, V_val, diag.numericErrorFlag(), kStagePArg2);

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
    auto op_logits = autograd::matmul(decision_input, params.W_op_select, stream);
    work.p_op = autograd::softmax(op_logits, temperature, stream);
    if (diag_out) {
        diag_out->op_logits_tensor = std::move(op_logits);
    }
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_op.data, nop, diag.numericErrorFlag(), kStagePOp);

    work.op_results = Tensor::zeros({1, nop}, stream, "exec_op_results");
    // Fix #6: Reset per-step division invalid flag before FourOps
    int* d_div_flag = diag.divInvalidFlag();
    int h_div_flag = 0;
    CUDA_CHECK(cudaMemsetAsync(d_div_flag, 0, sizeof(int), stream));
    kernelFourOps<<<1, kWarpSize, 0, stream>>>(work.op_results.data, work.v1.data, work.v2.data, kEps, diag.divClampCount(), d_div_flag);
    CUDA_CHECK_KERNEL();
    if (diag_out) {
        CUDA_CHECK(cudaMemcpyAsync(&h_div_flag, d_div_flag, sizeof(int), cudaMemcpyDeviceToHost, stream));
    }

    // Hard op selection (clean classification).
    // Forward: argmax(p_op) → pick discrete result.  v_out = results[hard_op].
    // The hard result is detached. Op selection is supervised from retained logits
    // at the loss boundary; execution values do not create an autograd edge.
    // Blending +/-/*/÷ results is FUCKING STUPID! for a register machine.
    kernelArgmax1DIntData<<<1, kWarpSize, 0, stream>>>(d_exec_idx + 2, work.p_op.data, nop);
    CUDA_CHECK_KERNEL();

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
            auto enc_fn = std::make_shared<RecordEncodeGradFn>();
            enc_fn->capture(1, V, nop, dm,
                            d_exec_record_i, d_exec_record_i + 1,
                            d_exec_record_i + 2, d_exec_record_f,
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
    // This direct slice is a detached leaf. MatMulGradFn therefore consumes its
    // persistent leaf-gradient buffer rather than owning a transient non-leaf
    // buffer; provision that storage at the leaf's creation boundary.
    decode_input.requires_grad_();
    decode_input.is_leaf = true;
    decode_input.alloc_grad();
    cudaMemcpyAsync(decode_input.data, work.atom_new.data + 16, vid * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    auto decode_h = autograd::matmul(decode_input, params.w_decode_1, stream);
    if (hp.decode_bias_enabled) {
        decode_h = autograd::add(decode_h, params.b_decode_1, stream);
    }

    auto decode_act = autograd::silu(decode_h, stream, decode_h.data);

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

    float* save_gate_buf = nullptr;
    float* save_H_slot_buf = nullptr;
    float h_inject_gate_value = 0.0f;  // host-side readback for telemetry
    cudaMallocOrThrow(reinterpret_cast<void**>(&save_gate_buf), sizeof(float), "datastream_save_gate");
    cudaMallocOrThrow(reinterpret_cast<void**>(&save_H_slot_buf), dm * sizeof(float), "datastream_save_H_slot");
    kernelInjectResultSlot<<<1, kBlockSize, 0, stream>>>(
        H.data, work.result_emb.data, params.w_inject_gate.data,
        inv_sqrt_d, work.result_slot, dm, hp.inject_gate_temp,
        save_gate_buf, save_H_slot_buf);
    CUDA_CHECK_KERNEL();
    // Queue async readback of inject gate for telemetry (completes at next sync)
    if (diag_out && hp.debug_mode) {
        CUDA_CHECK(cudaMemcpyAsync(&h_inject_gate_value, save_gate_buf, sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }

    {
        auto inject_fn = std::make_shared<ExecutionBlockInjectGradFn>();
        inject_fn->capture(H, work.result_emb, params.w_inject_gate,
                           save_gate_buf, save_H_slot_buf,
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

    if (diag_out) {
        Tensor stop_logits = autograd::matmul(write_ctx, params.W_stop, stream);
        stop_logits = autograd::add(stop_logits, params.b_stop, stream);
        diag_out->stop_probabilities = autograd::softmax(stop_logits, 1.0f, stream);
        diag_out->stop_logits_tensor = std::move(stop_logits);
    }

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
    if (diag_out) {
        diag_out->write_logits_tensor = std::move(write_logits);
    }
    kernelValidateSoftmax<<<1, kWarpSize, 0, stream>>>(work.p_write.data, V, diag.numericErrorFlag(), kStagePWrite);

    work.state_new = Tensor::zeros({1, dm}, stream, "exec_state_new");
    const float* step_emb_ptr = params.step_embeddings.data + static_cast<size_t>(step) * dm;
    kernelAddVectors<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.state_new.data, work.result_emb.data, step_emb_ptr, dm);
    CUDA_CHECK_KERNEL();

    work.key_new = autograd::matmul(work.result_emb, params.W_key_proj, stream);
    applyHardWriteback(hp, diag, parameters, memory, stream, work);
    copyStepDiagnostics(work, diag_out, V_val, nop, V, dm, stream);
    captureStateAfterWriteAndCheckMutations(hp, diag, memory, diag_out, stream);
    collectStepMetrics(hp, diag, work, diag_out, V_val, nop, V, stream);
    // collectStepMetrics synced the stream — h_inject_gate_value is now valid
    if (diag_out && hp.debug_mode) {
        diag_out->metrics.inject_gate_value = h_inject_gate_value;
    }
    finalizeStepOrThrow(hp, diag, diag_out, step, stream);
    if (diag_out) {
        diag_out->div_was_clamped = (h_div_flag != 0);
        diag_out->v_out_tensor = std::move(work.v_out);
    }
}

}  // namespace GRIM::ExecutionBlockInternal

GRIM::Tensor GRIM::executionBlockComputeEntropyLoss(
    const std::vector<const GRIM::Forward::ExecutionBlockStepOutput*>& steps,
    float weight,
    cudaStream_t stream)
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

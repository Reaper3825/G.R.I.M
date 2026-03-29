#include "execution_block_data_stream_GPU.hpp"
#include "execution_block_memory_stream_GPU.hpp"

#ifdef USE_CUDA

namespace GRIM {

using namespace ExecutionBlockInternal;

__global__ void kernelEntropy(
    float* __restrict__ out,
    const float* __restrict__ probs,
    int N
) {
    if (threadIdx.x != 0) return;
    float ent = 0.0f;
    for (int i = 0; i < N; ++i) {
        float p = probs[i];
        if (p > 1e-10f)
            ent -= p * logf(p + 1e-10f);
    }
    out[0] = ent;
}

__global__ void kernelAccumScalar(
    float* __restrict__ out,
    const float* __restrict__ in
) {
    if (threadIdx.x == 0)
        out[0] += in[0];
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

Tensor ExecutionBlockLayer::computeEntropyLoss(
    const std::vector<ExecutionBlockStepOutput>& steps,
    float weight,
    cudaStream_t stream) const
{
    if (steps.empty() || weight <= 0.0f) {
        return Tensor::zeros({1, 1}, stream, "exec_entropy_zero");
    }

    auto accum = Tensor::zeros({1, 1}, stream, "exec_entropy_accum");
    auto tmp   = Tensor::zeros({1, 1}, stream, "exec_entropy_tmp");
    int count = 0;

    for (const auto& s : steps) {
        auto accum_ent = [&](const Tensor& probs, int n) {
            if (!probs.data || n <= 0) return;
            kernelEntropy<<<1, 1, 0, stream>>>(tmp.data, probs.data, n);
            kernelAccumScalar<<<1, 1, 0, stream>>>(accum.data, tmp.data);
            count++;
        };
        if (s.p_arg1.data) accum_ent(s.p_arg1, s.p_arg1.shape.flat.cols);
        if (s.p_arg2.data) accum_ent(s.p_arg2, s.p_arg2.shape.flat.cols);
        if (s.p_op.data) accum_ent(s.p_op, s.p_op.shape.flat.cols);
        if (s.p_write.data) accum_ent(s.p_write, s.p_write.shape.flat.cols);
    }

    auto result = Tensor::zeros({1, 1}, stream, "exec_entropy_loss");
    kernelScaleNegAvg<<<1, 1, 0, stream>>>(result.data, accum.data, weight, count);
    return result;
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
    if (threadIdx.x != 0) return;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float p = probs[i];
        if (isnan(p) || isinf(p) || p < 0.0f) {
            atomicMax(error_flag, stage_id);
            return;
        }
        sum += p;
    }
    if (fabsf(sum - 1.0f) > 1e-3f)
        atomicMax(error_flag, stage_id);
}

__global__ void kernelCheckEntropyCollapse(
    const float* __restrict__ probs,
    int N,
    int* __restrict__ error_flag,
    int stage_id,
    float threshold
) {
    if (threadIdx.x != 0) return;
    float ent = 0.0f;
    for (int i = 0; i < N; ++i) {
        float p = probs[i];
        if (p > 1e-10f)
            ent -= p * logf(p + 1e-10f);
    }
    if (ent < threshold)
        atomicMax(error_flag, stage_id);
}

__global__ void kernelCheckWriteCollapse(
    const float* __restrict__ probs,
    int N,
    int* __restrict__ error_flag,
    int stage_id,
    float threshold
) {
    if (threadIdx.x != 0) return;
    float mx = 0.0f;
    for (int i = 0; i < N; ++i)
        mx = fmaxf(mx, probs[i]);
    if (mx > threshold)
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
    if (threadIdx.x != 0) return;
    float ent = 0.0f;
    for (int i = 0; i < N; ++i) {
        float p = probs[i];
        if (p > 1e-10f)
            ent -= p * logf(p + 1e-10f);
    }
    out[0] = ent;
}

__global__ void kernelComputeMax(
    float* __restrict__ out,
    const float* __restrict__ data,
    int N
) {
    if (threadIdx.x != 0) return;
    float mx = -1e30f;
    for (int i = 0; i < N; ++i)
        mx = fmaxf(mx, data[i]);
    out[0] = mx;
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

__global__ void kernelSTSlotValueGradToP(
    float* __restrict__ grad_p,
    const float* __restrict__ grad_v,
    const float* __restrict__ slot_vals_saved,
    int N
) {
    if (threadIdx.x != 0) return;
    float gv = grad_v[0];
    for (int j = 0; j < N; ++j)
        grad_p[j] = gv * slot_vals_saved[j];
}

__global__ void kernelFourOps(
    float* __restrict__ results,
    const float* __restrict__ pv1,
    const float* __restrict__ pv2,
    float eps,
    int* __restrict__ div_clamp_counter
) {
    if (threadIdx.x != 0) return;
    float v1 = pv1[0];
    float v2 = pv2[0];
    results[0] = v1 + v2;
    results[1] = v1 - v2;
    results[2] = v1 * v2;
    float abs_v2 = fabsf(v2);
    float denom = (abs_v2 > eps) ? v2 : copysignf(eps, v2);
    if (abs_v2 <= eps && div_clamp_counter)
        atomicAdd(div_clamp_counter, 1);
    results[3] = v1 / denom;
}

__global__ void kernelArgmax1DIntData(
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

__global__ void kernelReluForward(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
}

__global__ void kernelReluBackward(
    float* grad_input,
    const float* grad_output,
    const float* fwd_input,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_input[i] += (fwd_input[i] > 0.0f) ? grad_output[i] : 0.0f;
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
    grad_input[0] = grad_output[0] * s;
}

__global__ void kernelNormalizeUsage(
    float* __restrict__ out,
    const float* __restrict__ usage,
    int V
) {
    if (threadIdx.x != 0) return;
    float mx = 0.0f;
    for (int i = 0; i < V; ++i) mx = fmaxf(mx, usage[i]);
    float inv = 1.0f / (mx + kEps);
    for (int i = 0; i < V; ++i) out[i] = usage[i] * inv;
}

__global__ void kernelL2Normalize(float* __restrict__ v, int d) {
    if (threadIdx.x != 0) return;
    float sq = 0.0f;
    for (int j = 0; j < d; ++j) sq += v[j] * v[j];
    float inv = rsqrtf(sq + kEps);
    for (int j = 0; j < d; ++j) v[j] *= inv;
}

__global__ void kernelComputeWriteLogits(
    float* __restrict__ logits,
    const float* __restrict__ q_norm,
    const float* __restrict__ keys,
    const float* __restrict__ W_wk,
    const float* __restrict__ usage,
    const float* __restrict__ valid_mask,
    const float* __restrict__ recent_wr,
    const float* __restrict__ alpha_ptr,
    const float* __restrict__ beta_ptr,
    float empty_bonus, float kappa,
    int V, int d_key
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;

    float alpha_val = alpha_ptr[0];
    float beta_val = beta_ptr[0];
    float k_buf[64];
    for (int j = 0; j < d_key; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < d_key; ++k)
            sum += keys[i * d_key + k] * W_wk[k * d_key + j];
        k_buf[j] = sum;
    }
    float k_norm_sq = 0.0f;
    for (int j = 0; j < d_key; ++j) k_norm_sq += k_buf[j] * k_buf[j];
    float k_inv_norm = rsqrtf(k_norm_sq + kEps);
    for (int j = 0; j < d_key; ++j) k_buf[j] *= k_inv_norm;

    float content_score = 0.0f;
    for (int j = 0; j < d_key; ++j) content_score += q_norm[j] * k_buf[j];
    float usage_penalty = -usage[i];
    float logit = alpha_val * content_score + beta_val * usage_penalty;
    logit += (1.0f - valid_mask[i]) * empty_bonus;
    logit -= kappa * recent_wr[i];
    logits[i] = logit;
}

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
    const float* __restrict__ p_op,
    const float* __restrict__ results,
    float eps,
    int num_ops
) {
    if (threadIdx.x != 0) return;
    float grad_v_out = grad_v_out_ptr[0];
    float v1 = pv1[0];
    float v2 = pv2[0];
    float abs_v2 = fabsf(v2);
    float denom = (abs_v2 > eps) ? v2 : copysignf(eps, v2);

    if (grad_p_op) {
        for (int k = 0; k < num_ops; ++k)
            grad_p_op[k] += results[k] * grad_v_out;
    }

    if (grad_v1) {
        float dv1 = 0.0f;
        if (num_ops > 0) dv1 += p_op[0] * 1.0f;
        if (num_ops > 1) dv1 += p_op[1] * 1.0f;
        if (num_ops > 2) dv1 += p_op[2] * v2;
        if (num_ops > 3) dv1 += p_op[3] * (1.0f / denom);
        grad_v1[0] += dv1 * grad_v_out;
    }

    if (grad_v2) {
        float dv2 = 0.0f;
        if (num_ops > 0) dv2 += p_op[0] * 1.0f;
        if (num_ops > 1) dv2 += p_op[1] * (-1.0f);
        if (num_ops > 2) dv2 += p_op[2] * v1;
        if (num_ops > 3) {
            float div_grad = (abs_v2 >= eps) ? (-v1 / (denom * denom)) : 0.0f;
            dv2 += p_op[3] * div_grad;
        }
        grad_v2[0] += dv2 * grad_v_out;
    }
}

struct ReluGradFn : public GradFn {
    std::shared_ptr<GradFn> input_grad_fn;
    TensorContract::TensorShape input_shape;
    bool input_requires_grad = false;
    float* grad_input = nullptr;
    std::shared_ptr<float> owned_grad_input;
    std::shared_ptr<float> owned_fwd_input;
    const float* fwd_input = nullptr;
    int count = 0;

    ReluGradFn() { op_name = "relu"; }

    void capture(Tensor& input, int n, cudaStream_t stream) {
        input_requires_grad = input.requires_grad;
        input_shape = input.shape;
        input_grad_fn = input.grad_fn;
        count = n;

        float* buf = nullptr;
        cudaMalloc(&buf, n * sizeof(float));
        cudaMemcpyAsync(buf, input.data, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        owned_fwd_input = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
        fwd_input = owned_fwd_input.get();

        if (input_requires_grad) {
            input.ensure_grad();
            if (input.is_leaf) {
                grad_input = input.grad_data();
            } else {
                float* gbuf = nullptr;
                cudaMalloc(&gbuf, n * sizeof(float));
                cudaMemsetAsync(gbuf, 0, n * sizeof(float), stream);
                owned_grad_input = std::shared_ptr<float>(gbuf, [](float* p) { cudaFree(p); });
                grad_input = owned_grad_input.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        if (!input_requires_grad || !grad_input || !fwd_input) return;

        kernelReluBackward<<<(count + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            grad_input, grad_output.data, fwd_input, count);

        if (input_grad_fn) {
            Tensor view;
            view.data = grad_input;
            view.shape = input_shape;
            view.owns_data = false;
            view.stream = stream;
            input_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        owned_fwd_input.reset();
        fwd_input = nullptr;
        grad_input = nullptr;
        owned_grad_input.reset();
        input_grad_fn.reset();
    }
};

struct SlotValueSTGradFn : public GradFn {
    std::shared_ptr<GradFn> p_softmax_grad_fn;
    TensorContract::TensorShape p_shape;
    bool p_requires = false;
    std::shared_ptr<float> saved_slot_vals_owned;
    const float* saved_slot_vals = nullptr;
    int N = 0;
    float* grad_p = nullptr;
    std::shared_ptr<float> owned_grad_p;

    SlotValueSTGradFn() { op_name = "slot_value_ste"; }

    void capture(Tensor& p_arg_softmax, const float* d_slot_vals, int n, cudaStream_t stream) {
        p_softmax_grad_fn = p_arg_softmax.grad_fn;
        p_shape = p_arg_softmax.shape;
        p_requires = p_arg_softmax.requires_grad;
        N = n;
        float* buf = nullptr;
        cudaMalloc(&buf, n * sizeof(float));
        cudaMemcpyAsync(buf, d_slot_vals, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        saved_slot_vals_owned = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
        saved_slot_vals = saved_slot_vals_owned.get();

        if (p_requires && p_softmax_grad_fn) {
            float* grad_buf = nullptr;
            cudaMalloc(&grad_buf, n * sizeof(float));
            cudaMemsetAsync(grad_buf, 0, n * sizeof(float), stream);
            owned_grad_p = std::shared_ptr<float>(grad_buf, [](float* p) { cudaFree(p); });
            grad_p = owned_grad_p.get();
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        if (!p_requires || !p_softmax_grad_fn || !grad_p || !saved_slot_vals) return;

        kernelSTSlotValueGradToP<<<1, 1, 0, stream>>>(
            grad_p, grad_output.data, saved_slot_vals, N);
        CUDA_CHECK_KERNEL();

        Tensor view;
        view.data = grad_p;
        view.shape = p_shape;
        view.owns_data = false;
        view.stream = stream;
        p_softmax_grad_fn->apply(view, stream);
    }

    void release_saved() override {
        GradFn::release_saved();
        owned_grad_p.reset();
        grad_p = nullptr;
        saved_slot_vals_owned.reset();
        saved_slot_vals = nullptr;
        p_softmax_grad_fn.reset();
        N = 0;
    }
};

struct FourOpMixGradFn : public GradFn {
    float* saved_v1 = nullptr;
    float* saved_v2 = nullptr;
    float* saved_p_op = nullptr;
    float* saved_results = nullptr;
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
        if (saved_p_op) cudaFree(saved_p_op);
        if (saved_results) cudaFree(saved_results);
    }

    void capture(Tensor& v1_t, Tensor& v2_t, Tensor& p_op_t,
                 const float* d_p_op, const float* d_results,
                 int num_ops, cudaStream_t stream) {
        num_ops_ = num_ops;

        cudaMalloc(&saved_v1, sizeof(float));
        cudaMalloc(&saved_v2, sizeof(float));
        cudaMemcpyAsync(saved_v1, v1_t.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(saved_v2, v2_t.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);

        cudaMalloc(&saved_p_op, num_ops * sizeof(float));
        cudaMalloc(&saved_results, num_ops * sizeof(float));
        cudaMemcpyAsync(saved_p_op, d_p_op, num_ops * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(saved_results, d_results, num_ops * sizeof(float), cudaMemcpyDeviceToDevice, stream);

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
                cudaMalloc(&buf, n * sizeof(float));
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

        kernelFourOpMixBackward<<<1, 1, 0, stream>>>(
            grad_v1, grad_v2, grad_p_op,
            grad_output.data,
            saved_v1, saved_v2,
            saved_p_op, saved_results,
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
        if (saved_p_op) { cudaFree(saved_p_op); saved_p_op = nullptr; }
        if (saved_results) { cudaFree(saved_results); saved_results = nullptr; }
        grad_v1 = nullptr;
        grad_v2 = nullptr;
        grad_p_op = nullptr;
        v1_grad_fn.reset();
        v2_grad_fn.reset();
        p_op_grad_fn.reset();
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

        cudaMalloc(&saved_result_emb, d_model_ * sizeof(float));
        cudaMemcpyAsync(saved_result_emb, result_t.data, d_model_ * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        size_t total_size = static_cast<size_t>(total_tokens_) * d_model_ * sizeof(float);
        cudaMalloc(&mod_grad_buf, total_size);

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
                cudaMalloc(&buf, d_model_ * sizeof(float));
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
        if (!grad_result_emb)
            throw std::runtime_error("ExecutionBlockInjectGradFn::apply: grad_result_emb is NULL — result_emb MUST require grad");

        kernelInjectSlotBackward<<<1, kBlockSize, 0, stream>>>(
            grad_result_emb,
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
    float* grad_H_buf = nullptr;

    ReduceMeanGradFn() { op_name = "reduce_mean"; }

    ~ReduceMeanGradFn() override {
        if (grad_H_buf) cudaFree(grad_H_buf);
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

        if (H_requires_grad) {
            size_t total = static_cast<size_t>(total_tokens) * d_model;
            CUDA_CHECK(cudaMalloc(&grad_H_buf, total * sizeof(float)));
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        if (!H_requires_grad) return;

        size_t total = static_cast<size_t>(total_tokens_) * d_model_;
        CUDA_CHECK(cudaMemsetAsync(grad_H_buf, 0, total * sizeof(float), stream));
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
        if (grad_H_buf) { cudaFree(grad_H_buf); grad_H_buf = nullptr; }
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

        CUDA_CHECK(cudaMalloc(&saved_slot1_, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&saved_slot2_, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&saved_ops_, N * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&saved_scalars_, N * 3 * sizeof(float)));
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

        CUDA_CHECK(cudaMalloc(&saved_a_, sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(saved_a_, a_tensor.data, sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMalloc(&saved_b_, sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(saved_b_, target_device_ptr, sizeof(float), cudaMemcpyDeviceToDevice, stream));

        if (a_requires_grad_) {
            CUDA_CHECK(cudaMalloc(&grad_a_, sizeof(float)));
            CUDA_CHECK(cudaMemsetAsync(grad_a_, 0, sizeof(float), stream));
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;
        if (!a_requires_grad_ || !a_grad_fn_ || !grad_a_) return;

        kernelL1LossBackward<<<1, 1, 0, stream>>>(
            grad_a_, grad_output.data, saved_a_, saved_b_);
        CUDA_CHECK_KERNEL();

        Tensor view;
        view.data = grad_a_;
        view.shape = a_shape_;
        view.owns_data = false;
        view.stream = stream;
        a_grad_fn_->apply(view, stream);
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_a_) { cudaFree(saved_a_); saved_a_ = nullptr; }
        if (saved_b_) { cudaFree(saved_b_); saved_b_ = nullptr; }
        grad_a_ = nullptr;
        a_grad_fn_.reset();
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
    CUDA_CHECK(cudaMalloc(&d_buf, 7 * sizeof(float)));

    kernelComputeEntropyScalar<<<1, 1, 0, stream>>>(d_buf + 0, work.p_arg1.data, V_val);
    kernelComputeEntropyScalar<<<1, 1, 0, stream>>>(d_buf + 1, work.p_arg2.data, V_val);
    kernelComputeEntropyScalar<<<1, 1, 0, stream>>>(d_buf + 2, work.p_op.data, nop);
    kernelComputeEntropyScalar<<<1, 1, 0, stream>>>(d_buf + 3, work.p_write.data, V);
    kernelComputeMax<<<1, 1, 0, stream>>>(d_buf + 4, work.p_write.data, V);
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

    cudaFree(d_buf);
}

void executeStepCoordinatorImpl(
    ExecutionBlockLayer& layer,
    Tensor& H,
    ExecutionMemory& memory,
    const float* atom_embeddings,
    const int* atom_positions,
    const int32_t* token_to_slot_map,
    int num_atoms,
    int total_tokens,
    int step,
    float temperature,
    cudaStream_t stream,
    ExecutionBlockStepOutput* diag_out,
    int token_offset,
    int row_tokens,
    Tensor& trace_state,
    const std::vector<ExecutionRecord>& prior_records,
    const float* expected_target
) {
    (void)atom_embeddings;

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

    StepWorkingSet work;
    prepareMemoryStepOrThrow(layer, memory, atom_positions, token_to_slot_map, num_atoms, row_tokens, diag_out, stream);
    buildValueSlotCandidates(layer, memory, stream, work);
    kernelCheckFinite<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.slot_values.data, V_val, LayerAccess::numericErrorFlag(layer), kStageV1, layer.config().magnitude_limit);

    work.context = Tensor::zeros({1, dm}, stream, "exec_context");
    work.context.requires_grad = true;
    work.context.is_leaf = false;
    kernelReduceMeanForward<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.context.data,
        H.data + static_cast<size_t>(token_offset) * dm,
        row_tokens,
        dm);
    CUDA_CHECK_KERNEL();
    {
        auto mean_fn = std::make_shared<ReduceMeanGradFn>();
        mean_fn->capture(H, total_tokens, dm, stream, token_offset, row_tokens);
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
        CUDA_CHECK(cudaMalloc(&d_slot1, N_prior * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_slot2, N_prior * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ops, N_prior * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scalars, N_prior * 3 * sizeof(float)));
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
    work.p_arg1 = autograd::softmax(arg1_logits, temperature, stream);
    kernelValidateSoftmax<<<1, 1, 0, stream>>>(work.p_arg1.data, V_val, LayerAccess::numericErrorFlag(layer), kStagePArg1);
    kernelCheckEntropyCollapse<<<1, 1, 0, stream>>>(
        work.p_arg1.data, V_val, LayerAccess::numericErrorFlag(layer), kStageEntropyArg1, layer.config().entropy_collapse_threshold);

    auto query2 = autograd::matmul(decision_input, layer.w_arg2_select(), stream, decision_input.data, nullptr);
    auto arg2_logits = autograd::matmul(query2, work.cand_hidden, stream, nullptr, nullptr, true);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg2_logits.data, arg2_logits.data, work.cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    work.p_arg2 = autograd::softmax(arg2_logits, temperature, stream);
    kernelValidateSoftmax<<<1, 1, 0, stream>>>(work.p_arg2.data, V_val, LayerAccess::numericErrorFlag(layer), kStagePArg2);
    kernelCheckEntropyCollapse<<<1, 1, 0, stream>>>(
        work.p_arg2.data, V_val, LayerAccess::numericErrorFlag(layer), kStageEntropyArg2, layer.config().entropy_collapse_threshold);

    materializeSelectedOperands(layer, memory, stream, work);
    kernelCheckFinite<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.slot_values.data, V_val, LayerAccess::numericErrorFlag(layer), kStageV1, layer.config().magnitude_limit);
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v1.data, 1, LayerAccess::numericErrorFlag(layer), kStageV1, layer.config().magnitude_limit);
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v2.data, 1, LayerAccess::numericErrorFlag(layer), kStageV2, layer.config().magnitude_limit);

    if (work.p_arg1.requires_grad) {
        auto ste = std::make_shared<SlotValueSTGradFn>();
        ste->capture(work.p_arg1, work.slot_values.data, V_val, stream);
        work.v1.grad_fn = ste;
    }
    if (work.p_arg2.requires_grad) {
        auto ste = std::make_shared<SlotValueSTGradFn>();
        ste->capture(work.p_arg2, work.slot_values.data, V_val, stream);
        work.v2.grad_fn = ste;
    }

    work.h_arg1 = autograd::matmul(work.p_arg1, work.cand_hidden, stream);
    work.h_arg2 = autograd::matmul(work.p_arg2, work.cand_hidden, stream);

    auto pool_12 = autograd::concat(work.h_arg1, work.h_arg2, stream);
    auto pool_123 = autograd::concat(pool_12, work.context_enriched, stream);
    auto pool_1234 = autograd::concat(pool_123, trace_state, stream);
    auto pool = autograd::concat(pool_1234, work.step_emb, stream);

    auto op_logits = autograd::matmul(pool, layer.W_op_select(), stream, pool.data, nullptr);
    work.p_op = autograd::softmax(op_logits, temperature, stream);
    kernelValidateSoftmax<<<1, 1, 0, stream>>>(work.p_op.data, nop, LayerAccess::numericErrorFlag(layer), kStagePOp);
    kernelCheckEntropyCollapse<<<1, 1, 0, stream>>>(
        work.p_op.data, nop, LayerAccess::numericErrorFlag(layer), kStageEntropyOp, layer.config().entropy_collapse_threshold);

    work.op_results = Tensor::zeros({1, nop}, stream, "exec_op_results");
    kernelFourOps<<<1, 1, 0, stream>>>(work.op_results.data, work.v1.data, work.v2.data, kEps, LayerAccess::divClampCount(layer));
    CUDA_CHECK_KERNEL();

    kernelArgmax1DIntData<<<1, 1, 0, stream>>>(d_exec_idx + 2, work.p_op.data, nop);
    CUDA_CHECK_KERNEL();

    work.v_out = Tensor::zeros({1, 1}, stream, "exec_v_out");
    work.v_out.requires_grad = true;
    work.v_out.is_leaf = false;
    kernelHardPickOpForward<<<1, 1, 0, stream>>>(work.v_out.data, work.op_results.data, d_exec_idx + 2, nop);
    CUDA_CHECK_KERNEL();
    kernelCheckFinite<<<1, 1, 0, stream>>>(work.v_out.data, 1, LayerAccess::numericErrorFlag(layer), kStageVOut, layer.config().magnitude_limit);

    {
        auto grad_fn = std::make_shared<FourOpMixGradFn>();
        grad_fn->capture(work.v1, work.v2, work.p_op, work.p_op.data, work.op_results.data, nop, stream);
        work.v_out.grad_fn = grad_fn;
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
        auto update = autograd::matmul(update_input, layer.W_reason_gate(), stream, update_input.data, nullptr);
        trace_state = autograd::add(trace_state, update, stream);
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

    auto decode_relu = Tensor::zeros({1, vhd}, stream, "exec_decode_relu");
    decode_relu.requires_grad = true;
    decode_relu.is_leaf = false;
    kernelReluForward<<<(vhd + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(decode_relu.data, decode_h.data, vhd);
    CUDA_CHECK_KERNEL();
    {
        auto relu_fn = std::make_shared<ReluGradFn>();
        relu_fn->capture(decode_h, vhd, stream);
        decode_relu.grad_fn = relu_fn;
    }

    work.v_decoded = autograd::matmul(decode_relu, layer.w_decode_2(), stream, decode_relu.data, nullptr);
    work.result_emb = autograd::matmul(work.v_decoded, layer.W_value_to_emb(), stream, work.v_decoded.data, nullptr);
    work.result_emb = autograd::add(work.result_emb, layer.b_value_to_emb(), stream);
    kernelCheckFinite<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        work.result_emb.data, dm, LayerAccess::numericErrorFlag(layer), kStageResultEmb, layer.config().magnitude_limit);

    if (layer.config().result_slot_mode == 1 && layer.config().result_slot_index >= 0 &&
        layer.config().result_slot_index < total_tokens) {
        work.result_slot = layer.config().result_slot_index;
    } else {
        work.result_slot = token_offset + row_tokens - 1;
    }
    EXEC_CHECK(work.result_slot >= 0 && work.result_slot < total_tokens, "result_slot out of bounds");
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(dm));

    float* save_gate_buf = nullptr;
    float* save_H_slot_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&save_gate_buf, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&save_H_slot_buf, dm * sizeof(float)));
    kernelInjectResultSlot<<<1, kBlockSize, 0, stream>>>(
        H.data, work.result_emb.data, layer.w_inject_gate().data,
        inv_sqrt_d, work.result_slot, dm, layer.config().inject_gate_temp,
        save_gate_buf, save_H_slot_buf);
    CUDA_CHECK_KERNEL();

    {
        auto inject_fn = std::make_shared<ExecutionBlockInjectGradFn>();
        inject_fn->capture(H, work.result_emb, layer.w_inject_gate(),
                           save_gate_buf, save_H_slot_buf,
                           inv_sqrt_d, layer.config().inject_gate_temp,
                           work.result_slot, total_tokens, dm, stream);
        H.grad_fn = inject_fn;
        H.is_leaf = false;
        H.requires_grad = true;
    }

    auto write_ctx_12 = autograd::concat(work.h_arg1, work.h_arg2, stream);
    auto write_ctx_123 = autograd::concat(write_ctx_12, work.context_enriched, stream);
    auto write_ctx_1234 = autograd::concat(write_ctx_123, work.result_emb, stream);
    auto write_ctx_12345 = autograd::concat(write_ctx_1234, trace_state, stream);
    auto write_ctx = autograd::concat(write_ctx_12345, work.step_emb, stream);

    auto q_write = autograd::matmul(write_ctx, layer.W_write_query(), stream, write_ctx.data, nullptr);
    kernelL2Normalize<<<1, 1, 0, stream>>>(q_write.data, dk);
    CUDA_CHECK_KERNEL();

    auto usage_norm = Tensor::zeros({1, V}, stream, "exec_usage_norm");
    kernelNormalizeUsage<<<1, 1, 0, stream>>>(usage_norm.data, memory.usage.data, V);
    CUDA_CHECK_KERNEL();

    auto write_logits = Tensor::zeros({1, V}, stream, "exec_write_logits");
    write_logits.requires_grad = true;
    write_logits.is_leaf = false;
    kernelComputeWriteLogits<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        write_logits.data,
        q_write.data,
        memory.key_embeds.data,
        layer.W_write_key().data,
        usage_norm.data,
        memory.valid_mask.data,
        memory.recent_write_mask.data,
        layer.alpha().data,
        layer.beta().data,
        layer.config().empty_slot_bonus,
        layer.config().diversity_kappa,
        V,
        dk);
    CUDA_CHECK_KERNEL();
    if (S > 0) {
        kernelMaskScratchSlots<<<(S + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(write_logits.data, S);
        CUDA_CHECK_KERNEL();
    }

    work.p_write = autograd::softmax(write_logits, temperature, stream);
    kernelValidateSoftmax<<<1, 1, 0, stream>>>(work.p_write.data, V, LayerAccess::numericErrorFlag(layer), kStagePWrite);
    kernelCheckWriteCollapse<<<1, 1, 0, stream>>>(
        work.p_write.data, V, LayerAccess::numericErrorFlag(layer), kStageWriteCollapse, layer.config().write_collapse_threshold);

    if (diag_out) {
        auto op_results_col = Tensor::zeros({nop, 1}, stream, "exec_op_res_col");
        cudaMemcpyAsync(op_results_col.data, work.op_results.data, nop * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        auto v_soft = autograd::matmul(work.p_op, op_results_col, stream);

        auto recomputed_result = Tensor::zeros({1, 1}, stream, "exec_recomputed");
        kernelRecomputeOpResult<<<1, 1, 0, stream>>>(
            recomputed_result.data, memory.values.data, d_exec_idx, d_exec_idx + 1, d_exec_idx + 2,
            S, V, V_val, kEps);
        CUDA_CHECK_KERNEL();

        diag_out->transition_error_hard = Tensor::zeros({1, 1}, stream, "exec_te_hard");
        kernelAbsDiff<<<1, 1, 0, stream>>>(
            diag_out->transition_error_hard.data, work.v_out.data, recomputed_result.data, nullptr, 0, 0.0f);
        CUDA_CHECK_KERNEL();

        const float* target_ptr = expected_target ? expected_target : work.v_out.data;
        diag_out->used_expected_target = (expected_target != nullptr);
        diag_out->transition_loss = Tensor::zeros({1, 1}, stream, "exec_trans_loss");
        kernelAbsDiff<<<1, 1, 0, stream>>>(
            diag_out->transition_loss.data, v_soft.data, target_ptr, nullptr, 0, 0.0f);
        CUDA_CHECK_KERNEL();

        {
            auto l1_fn = std::make_shared<L1ScalarLossGradFn>();
            l1_fn->capture(v_soft, target_ptr, stream);
            diag_out->transition_loss.grad_fn = l1_fn;
            diag_out->transition_loss.requires_grad = true;
            diag_out->transition_loss.is_leaf = false;
        }

        if (layer.config().transition_hard_threshold > 0.0f) {
            kernelAbsDiff<<<1, 1, 0, stream>>>(
                diag_out->transition_error_hard.data,
                work.v_out.data,
                recomputed_result.data,
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
    finalizeStepOrThrow(layer, diag_out, step, stream);
}

}  // namespace GRIM

#endif  // USE_CUDA

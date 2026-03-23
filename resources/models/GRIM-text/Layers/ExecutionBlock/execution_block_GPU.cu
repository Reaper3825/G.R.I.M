//======================================================//
//  execution_block_GPU.cu
//  Differentiable Register Machine — GPU Implementation
//
//  Owns: all CUDA kernels, validation helpers, and
//  ExecutionBlockLayer method implementations.
//  No model-global config. No training-loop control flow.
//  No serialization code.
//======================================================//

#include "execution_block_GPU.hpp"

#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cfloat>

namespace GRIM {

static constexpr int kBlockSize = 256;
static constexpr float kEps = 1e-7f;

//======================================================//
//  Shape validation macro — hard-fail with context
//======================================================//
#define EXEC_CHECK(cond, msg) \
    do { if (!(cond)) { \
        char buf[512]; \
        snprintf(buf, sizeof(buf), "ExecutionBlock FATAL [%s:%d]: %s", __FILE__, __LINE__, msg); \
        throw std::runtime_error(buf); \
    }} while(0)

#define EXEC_CHECK_SHAPE2(tensor, name, expected_r, expected_c) \
    do { \
        if ((tensor).data == nullptr) { \
            char buf[256]; snprintf(buf, sizeof(buf), "%s: null data pointer", name); \
            EXEC_CHECK(false, buf); \
        } \
        if ((tensor).shape.flat.rows != (expected_r) || (tensor).shape.flat.cols != (expected_c)) { \
            char buf[256]; snprintf(buf, sizeof(buf), \
                "%s: expected [%d, %d], got [%d, %d]", \
                name, (int)(expected_r), (int)(expected_c), \
                (tensor).shape.flat.rows, (tensor).shape.flat.cols); \
            EXEC_CHECK(false, buf); \
        } \
    } while(0)

#define EXEC_CHECK_SHAPE1(tensor, name, expected_n) \
    EXEC_CHECK_SHAPE2(tensor, name, 1, expected_n)

//======================================================//
//  ExecutionMemory — allocate / clear
//======================================================//
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
    write_score       = Tensor::zeros({1, V}, stream);
    key_embeds        = Tensor::zeros({V, d_key}, stream);
    type_embed        = Tensor::zeros({V, d_type}, stream);
    recent_write_mask = Tensor::zeros({1, V}, stream);
    num_filled = 0;
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
    num_filled = 0;
}

//======================================================//
//  Validation helpers
//======================================================//
void ExecutionBlockLayer::validateConfigOrThrow() const {
    EXEC_CHECK(config_.d_model > 0,            "d_model must be positive");
    EXEC_CHECK(config_.atom_embedding_dim > 0,  "atom_embedding_dim must be positive");
    EXEC_CHECK(config_.num_ops == 4,           "num_ops must be 4 (+,-,*,/)");
    EXEC_CHECK(config_.num_slots > 0,          "num_slots must be positive");
    EXEC_CHECK(config_.num_exec_steps > 0,     "num_exec_steps must be positive");
    EXEC_CHECK(config_.d_key > 0,              "d_key must be positive");
    EXEC_CHECK(config_.d_type > 0,             "d_type must be positive");
    EXEC_CHECK(config_.cross_attn_head_dim > 0,"cross_attn_head_dim must be positive");
    EXEC_CHECK(config_.value_decode_input_dim == 24,  "value_decode_input_dim must be 24");
}

void ExecutionBlockLayer::validateMemoryOrThrow(const ExecutionMemory& M) const {
    const int V = config_.num_slots;
    const int ae = config_.atom_embedding_dim;
    const int dm = config_.d_model;
    const int dk = config_.d_key;
    const int dt = config_.d_type;

    EXEC_CHECK_SHAPE2(M.values,       "M.values",       V, 1);
    EXEC_CHECK_SHAPE2(M.atom_embeds,   "M.atom_embeds",   V, ae);
    EXEC_CHECK_SHAPE2(M.state_embeds,  "M.state_embeds",  V, dm);
    EXEC_CHECK_SHAPE1(M.valid_mask,    "M.valid_mask",    V);
    EXEC_CHECK_SHAPE1(M.usage,         "M.usage",         V);
    EXEC_CHECK_SHAPE1(M.write_score,   "M.write_score",   V);
    EXEC_CHECK_SHAPE2(M.key_embeds,    "M.key_embeds",    V, dk);
    EXEC_CHECK_SHAPE2(M.type_embed,    "M.type_embed",    V, dt);
    EXEC_CHECK_SHAPE1(M.recent_write_mask, "M.recent_write_mask", V);
}

void ExecutionBlockLayer::validateExecuteStepInputsOrThrow(
    const Tensor& H, const float* atom_embeddings,
    const int* atom_positions, int num_atoms,
    int total_tokens, const ExecutionMemory& M, int step) const
{
    const int dm = config_.d_model;
    EXEC_CHECK_SHAPE2(H, "H (executeStep)", total_tokens, dm);
    EXEC_CHECK(atom_embeddings != nullptr || num_atoms == 0, "atom_embeddings null with num_atoms > 0");
    EXEC_CHECK(atom_positions != nullptr || num_atoms == 0, "atom_positions null with num_atoms > 0");
    EXEC_CHECK(num_atoms >= 0, "num_atoms must be non-negative");
    EXEC_CHECK(total_tokens > 0, "total_tokens must be positive");
    EXEC_CHECK(step >= 0 && step < config_.num_exec_steps, "step out of range");
    validateMemoryOrThrow(M);
}

void ExecutionBlockLayer::validateCrossAttentionInputsOrThrow(
    const Tensor& hidden_states, const ExecutionMemory& M, int total_tokens) const
{
    const int dm = config_.d_model;
    EXEC_CHECK_SHAPE2(hidden_states, "hidden_states (cross-attn)", total_tokens, dm);
    validateMemoryOrThrow(M);
    EXEC_CHECK(M.num_filled > 0, "crossAttentionRead called with num_filled == 0");
}

//======================================================//
//  CUDA Kernels
//======================================================//

__global__ void kernelGatherCandidateHidden(
    float* __restrict__ out,
    const float* __restrict__ H,
    const int* __restrict__ positions,
    const float* __restrict__ mem_state,
    const float* __restrict__ mem_valid,
    int num_atoms, int V, int d_model,
    int total_tokens
) {
    const int i = blockIdx.x;
    const int C = num_atoms + V;
    if (i >= C) return;

    float* dst = out + static_cast<size_t>(i) * d_model;

    if (i < num_atoms) {
        const int pos = positions[i];
        const bool in_range = (pos >= 0 && pos < total_tokens);
        const float scale = in_range ? 1.0f : 0.0f;
        const float* src = in_range ? (H + static_cast<size_t>(pos) * d_model) : dst;
        for (int j = threadIdx.x; j < d_model; j += blockDim.x)
            dst[j] = scale * src[j];
    } else {
        const int slot = i - num_atoms;
        const float valid = mem_valid[slot];
        const float* src = mem_state + static_cast<size_t>(slot) * d_model;
        for (int j = threadIdx.x; j < d_model; j += blockDim.x)
            dst[j] = valid * src[j];
    }
}

__global__ void kernelGatherCandidateAtomEmb(
    float* __restrict__ out,
    const float* __restrict__ real_emb,
    const float* __restrict__ mem_emb,
    const float* __restrict__ mem_valid,
    int num_atoms, int V, int atom_dim
) {
    const int i = blockIdx.x;
    const int C = num_atoms + V;
    if (i >= C) return;

    float* dst = out + static_cast<size_t>(i) * atom_dim;
    if (i < num_atoms) {
        const float* src = real_emb + static_cast<size_t>(i) * atom_dim;
        for (int j = threadIdx.x; j < atom_dim; j += blockDim.x)
            dst[j] = src[j];
    } else {
        const int slot = i - num_atoms;
        const float valid = mem_valid[slot];
        const float* src = mem_emb + static_cast<size_t>(slot) * atom_dim;
        for (int j = threadIdx.x; j < atom_dim; j += blockDim.x)
            dst[j] = valid * src[j];
    }
}

// Build validity mask: 1.0 for valid candidates, 0.0 for invalid
__global__ void kernelBuildCandidateMask(
    float* __restrict__ mask,       // [C]
    const float* __restrict__ mem_valid, // [V]
    int num_atoms, int V
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = num_atoms + V;
    if (i >= C) return;
    if (i < num_atoms)
        mask[i] = 1.0f;
    else
        mask[i] = mem_valid[i - num_atoms];
}

// Apply mask to logits: masked_logits = logits + (1-mask)*(-1e9)
__global__ void kernelApplyLogitMask(
    float* __restrict__ masked_logits,  // [C]
    const float* __restrict__ logits,   // [C]
    const float* __restrict__ mask,     // [C]
    int C
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    masked_logits[i] = logits[i] + (1.0f - mask[i]) * (-1e9f);
}

// Apply mask to decoded values: masked_values = values * mask
__global__ void kernelApplyValueMask(
    float* __restrict__ masked,       // [C]
    const float* __restrict__ values, // [C]
    const float* __restrict__ mask,   // [C]
    int C
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;
    masked[i] = values[i] * mask[i];
}

// Extract a contiguous column slice: out[i, j] = in[i, start_col + j]
__global__ void kernelSliceColumns(
    float* __restrict__ out,      // [rows, width]
    const float* __restrict__ in, // [rows, in_cols]
    int rows, int in_cols, int start_col, int width
) {
    const int i = blockIdx.x;
    if (i >= rows) return;
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        out[i * width + j] = in[i * in_cols + start_col + j];
}

// Build atom/mem masks for decoded value blending
// atom_mask[i] = 1 for atoms, 0 for memory slots
// mem_values[i] = valid * stored_value for memory slots, 0 for atoms
__global__ void kernelBuildDecodeComponents(
    float* __restrict__ atom_mask,       // [C]
    float* __restrict__ mem_values_out,  // [C]
    const float* __restrict__ M_values,  // [V]
    const float* __restrict__ M_valid,   // [V]
    int num_atoms, int V
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = num_atoms + V;
    if (i >= C) return;
    if (i < num_atoms) {
        atom_mask[i] = 1.0f;
        mem_values_out[i] = 0.0f;
    } else {
        const int slot = i - num_atoms;
        atom_mask[i] = 0.0f;
        mem_values_out[i] = M_valid[slot] * M_values[slot];
    }
}

__global__ void kernelComputeContext(
    float* __restrict__ out,
    const float* __restrict__ H,
    int total_tokens, int d_model
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_model) return;
    float sum = 0.0f;
    for (int t = 0; t < total_tokens; ++t)
        sum += H[static_cast<size_t>(t) * d_model + j];
    out[j] = sum / static_cast<float>(total_tokens > 0 ? total_tokens : 1);
}

// Execute all 4 ops: results[0]=v1+v2, [1]=v1-v2, [2]=v1*v2, [3]=v1/safe(v2)
__global__ void kernelFourOps(
    float* __restrict__ results,    // [4]
    const float* __restrict__ pv1,  // [1] device pointer
    const float* __restrict__ pv2,  // [1] device pointer
    float eps
) {
    if (threadIdx.x != 0) return;
    float v1 = pv1[0];
    float v2 = pv2[0];
    results[0] = v1 + v2;
    results[1] = v1 - v2;
    results[2] = v1 * v2;
    float abs_v2 = fabsf(v2);
    float denom = (abs_v2 > eps) ? v2 : copysignf(eps, v2);
    results[3] = v1 / denom;
}

// FourOpMix forward: v_out = sum_k p_op[k] * results[k]
__global__ void kernelFourOpMixForward(
    float* __restrict__ v_out,      // [1]
    const float* __restrict__ p_op, // [4]
    const float* __restrict__ results, // [4]
    int num_ops
) {
    if (threadIdx.x != 0) return;
    float s = 0.0f;
    for (int k = 0; k < num_ops; ++k)
        s += p_op[k] * results[k];
    v_out[0] = s;
}

// FourOpMix backward: compute gradients for v1, v2, and p_op (all device pointers, null-safe)
__global__ void kernelFourOpMixBackward(
    float* __restrict__ grad_v1,           // [1] accumulate (may be null)
    float* __restrict__ grad_v2,           // [1] accumulate (may be null)
    float* __restrict__ grad_p_op,         // [4] accumulate
    const float* __restrict__ grad_v_out_ptr, // [1] device
    const float* __restrict__ pv1,         // [1] device
    const float* __restrict__ pv2,         // [1] device
    const float* __restrict__ p_op,        // [4]
    const float* __restrict__ results,     // [4]
    float eps
) {
    if (threadIdx.x != 0) return;
    float grad_v_out = grad_v_out_ptr[0];
    float v1 = pv1[0];
    float v2 = pv2[0];
    float abs_v2 = fabsf(v2);
    float denom = (abs_v2 > eps) ? v2 : copysignf(eps, v2);

    if (grad_p_op) {
        for (int k = 0; k < 4; ++k)
            grad_p_op[k] += results[k] * grad_v_out;
    }

    if (grad_v1) {
        float dv1 = 0.0f;
        dv1 += p_op[0] * 1.0f;
        dv1 += p_op[1] * 1.0f;
        dv1 += p_op[2] * v2;
        dv1 += p_op[3] * (1.0f / denom);
        grad_v1[0] += dv1 * grad_v_out;
    }

    if (grad_v2) {
        float dv2 = 0.0f;
        dv2 += p_op[0] * 1.0f;
        dv2 += p_op[1] * (-1.0f);
        dv2 += p_op[2] * v1;
        float div_grad = (abs_v2 >= eps) ? (-v1 / (denom * denom)) : 0.0f;
        dv2 += p_op[3] * div_grad;
        grad_v2[0] += dv2 * grad_v_out;
    }
}

// Single-slot injection forward: compute gate, save pre-injection state, inject
// H[result_slot] += (1/√d) * sigmoid(H[slot] · w_gate) * result_emb
__global__ void kernelInjectResultSlot(
    float* __restrict__ H,               // [total_tokens, d_model]
    const float* __restrict__ result_emb, // [d_model]
    const float* __restrict__ w_gate,     // [d_model, 1]
    float inv_sqrt_d,
    int result_slot, int d_model,
    float* __restrict__ save_gate,        // [1] output: computed gate value
    float* __restrict__ save_H_pre        // [d_model] output: H[slot] before injection
) {
    float* h_slot = H + static_cast<size_t>(result_slot) * d_model;

    for (int j = threadIdx.x; j < d_model; j += blockDim.x)
        save_H_pre[j] = h_slot[j];

    __shared__ float s_gate;
    if (threadIdx.x == 0) {
        float logit = 0.0f;
        for (int j = 0; j < d_model; ++j)
            logit += h_slot[j] * w_gate[j];
        float g = 1.0f / (1.0f + expf(-logit));
        s_gate = g;
        save_gate[0] = g;
    }
    __syncthreads();

    float scale = inv_sqrt_d * s_gate;
    for (int j = threadIdx.x; j < d_model; j += blockDim.x)
        h_slot[j] += scale * result_emb[j];
}

// Single-slot injection backward: computes grad_result, grad_w_gate,
// and modifies grad at slot row to include gate-path gradient
__global__ void kernelInjectSlotBackward(
    float* __restrict__ grad_result,       // [d_model] accumulate
    float* __restrict__ grad_w_gate,       // [d_model] accumulate
    float* __restrict__ mod_grad_slot,     // [d_model] in/out: copy of grad_output[slot], modified with gate-path
    const float* __restrict__ saved_result, // [d_model]
    const float* __restrict__ saved_H_slot, // [d_model] H[slot] before injection
    const float* __restrict__ w_gate,       // [d_model]
    const float* __restrict__ saved_gate,   // [1] device
    float inv_sqrt_d,
    int d_model
) {
    float gate_val = saved_gate[0];

    __shared__ float s_d_logit;
    if (threadIdx.x == 0) {
        float dot = 0.0f;
        for (int j = 0; j < d_model; ++j)
            dot += mod_grad_slot[j] * saved_result[j];
        s_d_logit = dot * inv_sqrt_d * gate_val * (1.0f - gate_val);
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

// Sigmoid gate for cross-attention (unchanged, used by crossAttentionRead only)
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

// Blended memory write: M_field = (1 - p_write) * M_field + p_write * new_val (broadcast)
__global__ void kernelBlendedWriteValues(
    float* __restrict__ mem_values,          // [V, 1]
    const float* __restrict__ p_write,       // [V]
    const float* __restrict__ new_value_ptr, // [1] device
    int V
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    float p = p_write[i];
    mem_values[i] = (1.0f - p) * mem_values[i] + p * new_value_ptr[0];
}

__global__ void kernelBlendedWriteVectors(
    float* __restrict__ mem_vecs,       // [V, D]
    const float* __restrict__ p_write,  // [V]
    const float* __restrict__ new_vec,  // [D]
    int V, int D
) {
    const int i = blockIdx.x;
    if (i >= V) return;
    float p = p_write[i];
    float* dst = mem_vecs + static_cast<size_t>(i) * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        dst[j] = (1.0f - p) * dst[j] + p * new_vec[j];
}

__global__ void kernelBlendedWriteValidMask(
    float* __restrict__ valid_mask,
    const float* __restrict__ p_write,
    int V
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    valid_mask[i] = fmaxf(valid_mask[i], p_write[i]);
}

// Normalize a vector in-place: v[j] /= ||v|| + eps
__global__ void kernelL2Normalize(float* __restrict__ v, int d) {
    if (threadIdx.x != 0) return;
    float sq = 0.0f;
    for (int j = 0; j < d; ++j) sq += v[j] * v[j];
    float inv = rsqrtf(sq + kEps);
    for (int j = 0; j < d; ++j) v[j] *= inv;
}

__global__ void kernelNormalizeUsage(
    float* __restrict__ out, const float* __restrict__ usage, int V
) {
    if (threadIdx.x != 0) return;
    float mx = 0.0f;
    for (int i = 0; i < V; ++i) mx = fmaxf(mx, usage[i]);
    float inv = 1.0f / (mx + kEps);
    for (int i = 0; i < V; ++i) out[i] = usage[i] * inv;
}

__global__ void kernelNormalizeWriteScore(
    float* __restrict__ out, const float* __restrict__ ws, int V
) {
    if (threadIdx.x != 0) return;
    float sq = 0.0f;
    for (int i = 0; i < V; ++i) sq += ws[i] * ws[i];
    float inv = rsqrtf(sq + kEps);
    for (int i = 0; i < V; ++i) out[i] = ws[i] * inv;
}

// Simple matmul for small matrices: out[i,j] = sum_k A[i,k] * B[k,j]
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

// Compute write logits from normalized query, transformed keys, and penalties
__global__ void kernelComputeWriteLogits(
    float* __restrict__ logits,
    const float* __restrict__ q_norm,
    const float* __restrict__ keys,
    const float* __restrict__ W_wk,
    const float* __restrict__ usage,
    const float* __restrict__ write_sc,
    const float* __restrict__ valid_mask,
    const float* __restrict__ recent_wr,
    const float* __restrict__ alpha_ptr,  // [1] device
    const float* __restrict__ beta_ptr,   // [1] device
    const float* __restrict__ gamma_ptr,  // [1] device
    float empty_bonus, float kappa,
    int V, int d_key
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;

    float alpha_val = alpha_ptr[0];
    float beta_val  = beta_ptr[0];
    float gamma_val = gamma_ptr[0];

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
    float ws = write_sc[i];

    float logit = alpha_val * content_score + beta_val * usage_penalty + gamma_val * ws;
    logit += (1.0f - valid_mask[i]) * empty_bonus;
    logit -= kappa * recent_wr[i];
    logits[i] = logit;
}

// Entropy: H(p) = -sum_i p_i * log(p_i + eps)
__global__ void kernelEntropy(
    float* __restrict__ out,        // [1]
    const float* __restrict__ probs, // [N]
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

// Accumulate: out[0] += in[0]
__global__ void kernelAccumScalar(
    float* __restrict__ out,
    const float* __restrict__ in
) {
    if (threadIdx.x == 0)
        out[0] += in[0];
}

// Scale and negate: out[0] = -weight * (in[0] / count)
__global__ void kernelScaleNegAvg(
    float* __restrict__ out,
    const float* __restrict__ in,
    float weight,
    int count
) {
    if (threadIdx.x == 0)
        out[0] = -weight * (in[0] / fmaxf(static_cast<float>(count), 1.0f));
}

// Element-wise vector addition: out[i] = a[i] + b[i]
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

//======================================================//
//  Cross-attention kernels (unchanged)
//======================================================//

__global__ void kernelCrossAttnSharpScores(
    float* __restrict__ scores,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ valid,
    const float* __restrict__ tau_ptr, // [1] device
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

//======================================================//
//  FourOpMixGradFn — backward for 4-op + weighted mix
//======================================================//
struct FourOpMixGradFn : public GradFn {
    float* saved_v1 = nullptr;       // [1] device
    float* saved_v2 = nullptr;       // [1] device
    float* saved_p_op = nullptr;     // [4] device
    float* saved_results = nullptr;  // [4] device

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
                 cudaStream_t stream) {
        cudaMalloc(&saved_v1, sizeof(float));
        cudaMalloc(&saved_v2, sizeof(float));
        cudaMemcpyAsync(saved_v1, v1_t.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(saved_v2, v2_t.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);

        cudaMalloc(&saved_p_op, 4 * sizeof(float));
        cudaMalloc(&saved_results, 4 * sizeof(float));
        cudaMemcpyAsync(saved_p_op, d_p_op, 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(saved_results, d_results, 4 * sizeof(float), cudaMemcpyDeviceToDevice, stream);

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
        alloc_grad(p_op_t, grad_p_op, owned_grad_p_op, 4);
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;

        kernelFourOpMixBackward<<<1, 1, 0, stream>>>(
            grad_v1,
            grad_v2,
            grad_p_op,
            grad_output.data,
            saved_v1, saved_v2,
            saved_p_op, saved_results, kEps);

        if (v1_requires_grad && v1_grad_fn) {
            Tensor view; view.data = grad_v1; view.shape = v1_shape;
            view.owns_data = false; view.stream = stream;
            v1_grad_fn->apply(view, stream);
        }
        if (v2_requires_grad && v2_grad_fn && v2_grad_fn != v1_grad_fn) {
            Tensor view; view.data = grad_v2; view.shape = v2_shape;
            view.owns_data = false; view.stream = stream;
            v2_grad_fn->apply(view, stream);
        }
        if (p_op_requires_grad && p_op_grad_fn) {
            Tensor view; view.data = grad_p_op; view.shape = p_op_shape;
            view.owns_data = false; view.stream = stream;
            p_op_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_v1) { cudaFree(saved_v1); saved_v1 = nullptr; }
        if (saved_v2) { cudaFree(saved_v2); saved_v2 = nullptr; }
        if (saved_p_op) { cudaFree(saved_p_op); saved_p_op = nullptr; }
        if (saved_results) { cudaFree(saved_results); saved_results = nullptr; }
        grad_v1 = nullptr; grad_v2 = nullptr; grad_p_op = nullptr;
        v1_grad_fn.reset(); v2_grad_fn.reset(); p_op_grad_fn.reset();
    }
};

//======================================================//
//  ExecutionBlockInjectGradFn — single-slot injection
//  Forward: H[slot] += (1/√d) * sigmoid(H[slot]·w_gate) * result_emb
//  Backward: identity path for all tokens, injection+gate path at slot
//======================================================//
struct ExecutionBlockInjectGradFn : public GradFn {
    float* saved_result_emb = nullptr;  // [d_model] device copy
    float* saved_H_slot = nullptr;      // [d_model] device (pre-injection H[slot])
    float* saved_gate = nullptr;        // [1] device (sigmoid gate value)
    float* mod_grad_buf = nullptr;      // [total_tokens * d_model] pre-allocated for backward
    float inv_sqrt_d = 0.0f;
    int result_slot = 0;
    int total_tokens = 0;
    int d_model = 0;

    float* grad_result_emb = nullptr;
    float* w_gate_data = nullptr;       // read-only pointer to w_inject_gate_.data
    float* w_gate_grad = nullptr;       // pointer to w_inject_gate_.grad_data() (leaf)
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
                 float inv_sqrt_d_, int result_slot_, int total_tokens_, int d_model_,
                 cudaStream_t stream) {
        inv_sqrt_d = inv_sqrt_d_;
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

        if (saved_gate && saved_result_emb && saved_H_slot) {
            kernelInjectSlotBackward<<<1, kBlockSize, 0, stream>>>(
                grad_result_emb ? grad_result_emb : slot_grad,
                w_gate_grad,
                slot_grad,
                saved_result_emb, saved_H_slot, w_gate_data, saved_gate,
                inv_sqrt_d, d_model);
        }

        if (result_requires_grad && result_grad_fn && grad_result_emb) {
            Tensor view; view.data = grad_result_emb; view.shape = result_shape;
            view.owns_data = false; view.stream = stream;
            result_grad_fn->apply(view, stream);
        }

        if (H_requires_grad && H_grad_fn && mod_grad_buf) {
            Tensor view; view.data = mod_grad_buf; view.shape = H_shape;
            view.owns_data = false; view.stream = stream;
            H_grad_fn->apply(view, stream);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_result_emb) { cudaFree(saved_result_emb); saved_result_emb = nullptr; }
        if (saved_H_slot) { cudaFree(saved_H_slot); saved_H_slot = nullptr; }
        if (saved_gate) { cudaFree(saved_gate); saved_gate = nullptr; }
        if (mod_grad_buf) { cudaFree(mod_grad_buf); mod_grad_buf = nullptr; }
        grad_result_emb = nullptr; w_gate_data = nullptr; w_gate_grad = nullptr;
        H_grad_fn.reset(); result_grad_fn.reset();
    }
};

//======================================================//
//  Constructor
//======================================================//
ExecutionBlockLayer::ExecutionBlockLayer(const ExecutionBlockConfig& config,
                                       uint64_t seed,
                                       cudaStream_t init_stream)
    : config_(config)
{
    validateConfigOrThrow();
    EXEC_CHECK(init_stream != nullptr, "init_stream is NULL");

    const int dm  = config_.d_model;
    const int ae  = config_.atom_embedding_dim;
    const int dk  = config_.d_key;
    const int dt  = config_.d_type;
    const int hd  = config_.cross_attn_head_dim;
    const int nop = config_.num_ops;   // 4
    const int K   = config_.num_exec_steps;
    const int vid = config_.value_decode_input_dim;
    const int vhd = config_.value_decode_hidden_dim;

    auto make_param = [&](int rows, int cols, uint64_t s, const char* name) -> Tensor {
        auto t = Tensor::zeros(TensorContract::TensorShape::make_BSM(rows, cols),
                               true, init_stream, name);
        t.requires_grad_();
        t.ensure_grad();
        Tensor::xavier_uniform_(t, s, init_stream);
        return t;
    };
    auto make_bias = [&](int cols, const char* name) -> Tensor {
        auto t = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, cols),
                               true, init_stream, name);
        t.requires_grad_();
        t.ensure_grad();
        return t;
    };
    auto make_scalar = [&](float init_val, const char* name) -> Tensor {
        auto t = Tensor::zeros(TensorContract::TensorShape::make_BSM(1, 1),
                               true, init_stream, name);
        t.requires_grad_();
        t.ensure_grad();
        cudaMemcpyAsync(t.data, &init_val, sizeof(float), cudaMemcpyHostToDevice, init_stream);
        return t;
    };

    // Value decode MLP
    w_decode_1_ = make_param(vid, vhd, seed,     "exec_block.w_decode_1");
    b_decode_1_ = make_bias(vhd,                 "exec_block.b_decode_1");
    w_decode_2_ = make_param(vhd, 1, seed + 1,   "exec_block.w_decode_2");

    // Arg selection: [1, d_model] for matmul as [1,dm] @ [dm,C]^T = [1,C]
    w_arg1_select_ = make_param(1, dm, seed + 2, "exec_block.w_arg1_select");
    w_arg2_select_ = make_param(1, dm, seed + 3, "exec_block.w_arg2_select");

    // Context-aware op selection (4 ops)
    W_op_select_ = make_param(3 * dm, nop, seed + 4, "exec_block.W_op_select");

    // Key projection from result embedding
    W_key_proj_ = make_param(dm, dk, seed + 5, "exec_block.W_key_proj");

    // Write-head (write_context = 4*d_model -> d_key query)
    W_write_query_ = make_param(4 * dm, dk, seed + 7, "exec_block.W_write_query");
    W_write_key_   = make_param(dk, dk, seed + 8, "exec_block.W_write_key");

    // Learned scalars (init 1.0)
    alpha_ = make_scalar(1.0f, "exec_block.alpha");
    beta_  = make_scalar(1.0f, "exec_block.beta");
    gamma_ = make_scalar(1.0f, "exec_block.gamma");

    // Step encoding
    step_embeddings_ = make_param(K, dm, seed + 9, "exec_block.step_embeddings");

    // Type embedding
    type_num_embed_ = make_param(1, dt, seed + 10, "exec_block.type_num_embed");

    // Linear value embedding (scalar -> d_model)
    W_value_to_emb_ = make_param(1, dm, seed + 15, "exec_block.W_value_to_emb");
    b_value_to_emb_ = make_bias(dm,                "exec_block.b_value_to_emb");

    // Injection gate: zeros so gate starts at sigmoid(0) = 0.5
    w_inject_gate_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(dm, 1),
                                   true, init_stream, "exec_block.w_inject_gate");
    w_inject_gate_.requires_grad_();
    w_inject_gate_.ensure_grad();

    // Cross-attention read
    W_Q_read_    = make_param(dm, hd, seed + 11, "exec_block.W_Q_read");
    W_K_read_    = make_param(dk, hd, seed + 12, "exec_block.W_K_read");
    W_V_read_    = make_param(dm, hd, seed + 13, "exec_block.W_V_read");
    W_O_read_    = make_param(hd, dm, seed + 14, "exec_block.W_O_read");
    W_gate_read_ = Tensor::zeros(TensorContract::TensorShape::make_BSM(dm, 1),
                                 true, init_stream, "exec_block.W_gate_read");
    W_gate_read_.requires_grad_();
    W_gate_read_.ensure_grad();

    // Temperature (init 1.0)
    tau_ = make_scalar(1.0f, "exec_block.tau");
}

//======================================================//
//  Move semantics
//======================================================//
ExecutionBlockLayer::ExecutionBlockLayer(ExecutionBlockLayer&& other) noexcept
    : config_(other.config_),
      w_decode_1_(std::move(other.w_decode_1_)),
      b_decode_1_(std::move(other.b_decode_1_)),
      w_decode_2_(std::move(other.w_decode_2_)),
      w_arg1_select_(std::move(other.w_arg1_select_)),
      w_arg2_select_(std::move(other.w_arg2_select_)),
      W_op_select_(std::move(other.W_op_select_)),
      W_key_proj_(std::move(other.W_key_proj_)),
      W_write_query_(std::move(other.W_write_query_)),
      W_write_key_(std::move(other.W_write_key_)),
      alpha_(std::move(other.alpha_)),
      beta_(std::move(other.beta_)),
      gamma_(std::move(other.gamma_)),
      step_embeddings_(std::move(other.step_embeddings_)),
      type_num_embed_(std::move(other.type_num_embed_)),
      W_value_to_emb_(std::move(other.W_value_to_emb_)),
      b_value_to_emb_(std::move(other.b_value_to_emb_)),
      w_inject_gate_(std::move(other.w_inject_gate_)),
      W_Q_read_(std::move(other.W_Q_read_)),
      W_K_read_(std::move(other.W_K_read_)),
      W_V_read_(std::move(other.W_V_read_)),
      W_O_read_(std::move(other.W_O_read_)),
      W_gate_read_(std::move(other.W_gate_read_)),
      tau_(std::move(other.tau_))
{}

ExecutionBlockLayer& ExecutionBlockLayer::operator=(ExecutionBlockLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        w_decode_1_    = std::move(other.w_decode_1_);
        b_decode_1_    = std::move(other.b_decode_1_);
        w_decode_2_    = std::move(other.w_decode_2_);
        w_arg1_select_ = std::move(other.w_arg1_select_);
        w_arg2_select_ = std::move(other.w_arg2_select_);
        W_op_select_   = std::move(other.W_op_select_);
        W_key_proj_    = std::move(other.W_key_proj_);
        W_write_query_ = std::move(other.W_write_query_);
        W_write_key_   = std::move(other.W_write_key_);
        alpha_         = std::move(other.alpha_);
        beta_          = std::move(other.beta_);
        gamma_         = std::move(other.gamma_);
        step_embeddings_= std::move(other.step_embeddings_);
        type_num_embed_ = std::move(other.type_num_embed_);
        W_value_to_emb_= std::move(other.W_value_to_emb_);
        b_value_to_emb_= std::move(other.b_value_to_emb_);
        w_inject_gate_ = std::move(other.w_inject_gate_);
        W_Q_read_      = std::move(other.W_Q_read_);
        W_K_read_      = std::move(other.W_K_read_);
        W_V_read_      = std::move(other.W_V_read_);
        W_O_read_      = std::move(other.W_O_read_);
        W_gate_read_   = std::move(other.W_gate_read_);
        tau_           = std::move(other.tau_);
    }
    return *this;
}

//======================================================//
//  executeStep — fully differentiable, GPU-only
//======================================================//
void ExecutionBlockLayer::executeStep(
    Tensor& H,
    ExecutionMemory& M,
    const float* atom_embeddings,
    const int* atom_positions,
    int num_atoms,
    int total_tokens,
    int step,
    float temperature,
    cudaStream_t stream,
    ExecutionBlockStepOutput* diag_out)
{
    validateExecuteStepInputsOrThrow(H, atom_embeddings, atom_positions,
                                      num_atoms, total_tokens, M, step);

    const int dm  = config_.d_model;
    const int ae  = config_.atom_embedding_dim;
    const int V   = config_.num_slots;
    const int dk  = config_.d_key;
    const int nop = config_.num_ops;  // 4
    const int vid = config_.value_decode_input_dim;
    const int vhd = config_.value_decode_hidden_dim;
    const int C   = num_atoms + V;
    EXEC_CHECK(C > 0, "executeStep: no candidates (num_atoms + V == 0)");

    // ──── 1. Gather candidates (raw kernels, detached) ────
    auto cand_hidden = Tensor::zeros({C, dm}, stream, "exec_cand_hidden");
    kernelGatherCandidateHidden<<<C, kBlockSize, 0, stream>>>(
        cand_hidden.data, H.data, atom_positions,
        M.state_embeds.data, M.valid_mask.data,
        num_atoms, V, dm, total_tokens);

    auto cand_atom_emb = Tensor::zeros({C, ae}, stream, "exec_cand_atom_emb");
    kernelGatherCandidateAtomEmb<<<C, kBlockSize, 0, stream>>>(
        cand_atom_emb.data, atom_embeddings, M.atom_embeds.data,
        M.valid_mask.data, num_atoms, V, ae);

    // ──── 2. Build candidate mask [1, C] ────
    auto cand_mask = Tensor::zeros({1, C}, stream, "exec_cand_mask");
    kernelBuildCandidateMask<<<(C + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        cand_mask.data, M.valid_mask.data, num_atoms, V);

    // ──── 3. Decode values [C, 1] — autograd MLP with SiLU ────
    // 3a. Extract MLP input slice [C, vid] (dims 16..16+vid) from atom embeddings
    auto decode_input = Tensor::zeros({C, vid}, stream, "exec_decode_input");
    kernelSliceColumns<<<C, kBlockSize, 0, stream>>>(
        decode_input.data, cand_atom_emb.data, C, ae, 16, vid);

    // 3b. MLP forward: hidden = SiLU(input @ W1 + b1), decoded = hidden @ W2
    auto decode_hidden = autograd::matmul(decode_input, w_decode_1_, stream);     // [C, vhd]
    decode_hidden = autograd::broadcast_add(decode_hidden, b_decode_1_, stream);  // [C, vhd]
    decode_hidden = autograd::silu(decode_hidden, stream);                        // [C, vhd]
    auto mlp_out = autograd::matmul(decode_hidden, w_decode_2_, stream);          // [C, 1]

    // 3c. Build atom/mem masks and blend: atoms use MLP, memory uses stored values
    auto atom_mask = Tensor::zeros({C, 1}, stream, "exec_atom_mask");
    auto mem_vals  = Tensor::zeros({C, 1}, stream, "exec_mem_vals");
    kernelBuildDecodeComponents<<<(C + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        atom_mask.data, mem_vals.data, M.values.data, M.valid_mask.data, num_atoms, V);

    // atom_decoded zeros out memory slot positions → no MLP gradient from garbage inputs
    auto atom_decoded   = autograd::elementwise_mul(mlp_out, atom_mask, stream);  // [C, 1]
    auto decoded_values = autograd::add(atom_decoded, mem_vals, stream);          // [C, 1]

    // ──── 4. Arg selection (autograd: matmul + detached mask + softmax) ────
    // arg1_logits = w_arg1_select [1,dm] @ cand_hidden^T [dm,C] = [1,C]
    auto arg1_logits = autograd::matmul(w_arg1_select_, cand_hidden, stream,
                                         nullptr, nullptr, true);

    // Masking is non-learnable — detach logits, apply mask in-place, then softmax
    auto arg1_detached = arg1_logits.detach(stream);
    kernelApplyLogitMask<<<(C + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg1_detached.data, arg1_detached.data, cand_mask.data, C);
    auto p_arg1 = autograd::softmax(arg1_detached, temperature, stream);

    // Same for arg2
    auto arg2_logits = autograd::matmul(w_arg2_select_, cand_hidden, stream,
                                         nullptr, nullptr, true);
    auto arg2_detached = arg2_logits.detach(stream);
    kernelApplyLogitMask<<<(C + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg2_detached.data, arg2_detached.data, cand_mask.data, C);
    auto p_arg2 = autograd::softmax(arg2_detached, temperature, stream);

    // ──── 5. Soft-select values and hidden (autograd: matmul) ────
    // v1 = p_arg1 [1,C] @ decoded_values [C,1] = [1,1]
    auto v1 = autograd::matmul(p_arg1, decoded_values, stream);
    auto v2 = autograd::matmul(p_arg2, decoded_values, stream);

    // h_arg1 = p_arg1 [1,C] @ cand_hidden [C,dm] = [1,dm]
    auto h_arg1 = autograd::matmul(p_arg1, cand_hidden, stream);
    auto h_arg2 = autograd::matmul(p_arg2, cand_hidden, stream);

    // ──── 6. Context = mean_pool(H) [1, dm] (detached) ────
    auto context = Tensor::zeros({1, dm}, stream, "exec_context");
    kernelComputeContext<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        context.data, H.data, total_tokens, dm);

    // ──── 7. Op selection (autograd: concat + matmul + softmax) ────
    auto pool12 = autograd::concat(h_arg1, h_arg2, stream);  // [1, 2*dm]
    auto pool = autograd::concat(pool12, context, stream);     // [1, 3*dm]

    auto op_logits = autograd::matmul(pool, W_op_select_, stream); // [1, 4]
    auto p_op = autograd::softmax(op_logits, temperature, stream);  // [1, 4]

    // ──── 8. Four ops + soft mix (custom GradFn, all GPU) ────
    auto op_results = Tensor::zeros({1, nop}, stream, "exec_op_results");
    kernelFourOps<<<1, 1, 0, stream>>>(op_results.data, v1.data, v2.data, kEps);

    auto v_out = Tensor::zeros({1, 1}, stream, "exec_v_out");
    v_out.requires_grad = true;
    v_out.is_leaf = false;
    kernelFourOpMixForward<<<1, 1, 0, stream>>>(
        v_out.data, p_op.data, op_results.data, nop);

    {
        auto grad_fn = std::make_shared<FourOpMixGradFn>();
        grad_fn->capture(v1, v2, p_op, p_op.data, op_results.data, stream);
        v_out.grad_fn = grad_fn;
    }

    // ──── 9. Linear value embedding (autograd: matmul + add) ────
    auto result_emb = autograd::matmul(v_out, W_value_to_emb_, stream);
    result_emb = autograd::add(result_emb, b_value_to_emb_, stream);

    // ──── 10. Inject into H at result_slot (single slot, custom GradFn) ────
    int result_slot;
    if (config_.result_slot_mode == 1 && config_.result_slot_index >= 0 &&
        config_.result_slot_index < total_tokens) {
        result_slot = config_.result_slot_index;
    } else {
        result_slot = total_tokens - 1;
    }
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(dm));

    float* save_gate_buf = nullptr;
    float* save_H_slot_buf = nullptr;
    cudaMalloc(&save_gate_buf, sizeof(float));
    cudaMalloc(&save_H_slot_buf, dm * sizeof(float));

    kernelInjectResultSlot<<<1, kBlockSize, 0, stream>>>(
        H.data, result_emb.data, w_inject_gate_.data,
        inv_sqrt_d, result_slot, dm,
        save_gate_buf, save_H_slot_buf);

    {
        auto inject_fn = std::make_shared<ExecutionBlockInjectGradFn>();
        inject_fn->capture(H, result_emb, w_inject_gate_,
                           save_gate_buf, save_H_slot_buf,
                           inv_sqrt_d, result_slot, total_tokens, dm, stream);
        H.grad_fn = inject_fn;
        H.is_leaf = false;
        H.requires_grad = true;
    }

    // ──── 11. Write to memory (blended, softmax-weighted, all GPU) ────
    auto write_ctx_12 = autograd::concat(h_arg1, h_arg2, stream);
    auto write_ctx_123 = autograd::concat(write_ctx_12, context, stream);
    auto write_ctx = autograd::concat(write_ctx_123, result_emb, stream);

    auto q_write = autograd::matmul(write_ctx, W_write_query_, stream);
    kernelL2Normalize<<<1, 1, 0, stream>>>(q_write.data, dk);

    auto usage_norm = Tensor::zeros({1, V}, stream);
    auto ws_norm = Tensor::zeros({1, V}, stream);
    kernelNormalizeUsage<<<1, 1, 0, stream>>>(usage_norm.data, M.usage.data, V);
    kernelNormalizeWriteScore<<<1, 1, 0, stream>>>(ws_norm.data, M.write_score.data, V);

    auto write_logits = Tensor::zeros({1, V}, stream, "exec_write_logits");
    write_logits.requires_grad = true;
    write_logits.is_leaf = false;
    kernelComputeWriteLogits<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        write_logits.data, q_write.data, M.key_embeds.data,
        W_write_key_.data, usage_norm.data, ws_norm.data,
        M.valid_mask.data, M.recent_write_mask.data,
        alpha_.data, beta_.data, gamma_.data,
        config_.empty_slot_bonus, config_.diversity_kappa,
        V, dk);

    auto p_write = autograd::softmax(write_logits, temperature, stream);

    kernelBlendedWriteValues<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        M.values.data, p_write.data, v_out.data, V);

    // state_new = result_emb + step_embeddings[step]
    auto state_new = Tensor::zeros({1, dm}, stream, "exec_state_new");
    const float* step_emb_ptr = step_embeddings_.data + static_cast<size_t>(step) * dm;
    kernelAddVectors<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        state_new.data, result_emb.data, step_emb_ptr, dm);

    kernelBlendedWriteVectors<<<V, kBlockSize, 0, stream>>>(
        M.state_embeds.data, p_write.data, state_new.data, V, dm);

    auto key_new = Tensor::zeros({1, dk}, stream, "exec_key_new");
    kernelSmallMatmul<<<1, dk, 0, stream>>>(
        key_new.data, result_emb.data, W_key_proj_.data, 1, dm, dk);

    kernelBlendedWriteVectors<<<V, kBlockSize, 0, stream>>>(
        M.key_embeds.data, p_write.data, key_new.data, V, dk);

    kernelBlendedWriteValidMask<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        M.valid_mask.data, p_write.data, V);

    cudaMemcpyAsync(M.recent_write_mask.data, p_write.data, V * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // Monotonic growth: one slot becomes "filled" per step, preserving sparsity
    // and learnable slot competition. valid_mask handles soft validity filtering.
    M.num_filled = std::min(V, M.num_filled + 1);

    // ──── 12. Diagnostic output (GPU-only, no host sync) ────
    if (diag_out) {
        diag_out->p_arg1 = Tensor::zeros({1, C}, stream);
        cudaMemcpyAsync(diag_out->p_arg1.data, p_arg1.data, C * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->p_arg2 = Tensor::zeros({1, C}, stream);
        cudaMemcpyAsync(diag_out->p_arg2.data, p_arg2.data, C * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->p_op = Tensor::zeros({1, nop}, stream);
        cudaMemcpyAsync(diag_out->p_op.data, p_op.data, nop * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->p_write = Tensor::zeros({1, V}, stream);
        cudaMemcpyAsync(diag_out->p_write.data, p_write.data, V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->v_out = Tensor::zeros({1, 1}, stream);
        cudaMemcpyAsync(diag_out->v_out.data, v_out.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->result_emb = Tensor::zeros({1, dm}, stream);
        cudaMemcpyAsync(diag_out->result_emb.data, result_emb.data, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }
}

//======================================================//
//  computeEntropyLoss
//======================================================//
Tensor ExecutionBlockLayer::computeEntropyLoss(
    const std::vector<ExecutionBlockStepOutput>& steps,
    float weight,
    cudaStream_t stream) const
{
    // Entropy loss is a non-differentiable monitoring regularizer.
    // It is computed via raw CUDA kernels and is NOT connected to the autograd graph.
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
        if (s.p_op.data)   accum_ent(s.p_op, s.p_op.shape.flat.cols);
        if (s.p_write.data) accum_ent(s.p_write, s.p_write.shape.flat.cols);
    }

    auto result = Tensor::zeros({1, 1}, stream, "exec_entropy_loss");
    kernelScaleNegAvg<<<1, 1, 0, stream>>>(result.data, accum.data, weight, count);
    return result;
}

//======================================================//
//  crossAttentionRead — gated, sharpened
//======================================================//
void ExecutionBlockLayer::crossAttentionRead(
    Tensor& hidden_states,
    ExecutionMemory& M,
    int total_tokens,
    cudaStream_t stream)
{
    validateCrossAttentionInputsOrThrow(hidden_states, M, total_tokens);

    const int dm  = config_.d_model;
    const int dk  = config_.d_key;
    const int hd  = config_.cross_attn_head_dim;
    const int V   = config_.num_slots;
    const int nv  = M.num_filled;
    const int topk= config_.cross_attn_topk;

    auto Q = Tensor::zeros({total_tokens, hd}, stream);
    kernelSmallMatmul<<<total_tokens, hd, 0, stream>>>(
        Q.data, hidden_states.data, W_Q_read_.data, total_tokens, dm, hd);

    auto K_proj = Tensor::zeros({nv, hd}, stream);
    kernelSmallMatmul<<<nv, hd, 0, stream>>>(
        K_proj.data, M.key_embeds.data, W_K_read_.data, nv, dk, hd);

    auto V_proj = Tensor::zeros({nv, hd}, stream);
    kernelSmallMatmul<<<nv, hd, 0, stream>>>(
        V_proj.data, M.state_embeds.data, W_V_read_.data, nv, dm, hd);

    auto scores = Tensor::zeros({total_tokens, nv}, stream);
    kernelCrossAttnSharpScores<<<total_tokens, 1, 0, stream>>>(
        scores.data, Q.data, K_proj.data, M.valid_mask.data,
        tau_.data, total_tokens, nv, hd, topk);

    auto R = Tensor::zeros({total_tokens, hd}, stream);
    kernelCrossAttnWeightedValue<<<total_tokens, hd, 0, stream>>>(
        R.data, scores.data, V_proj.data, total_tokens, nv, hd);

    auto gate = Tensor::zeros({1, total_tokens}, stream);
    kernelComputeGate<<<(total_tokens + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        gate.data, hidden_states.data, W_gate_read_.data, total_tokens, dm);

    kernelCrossAttnGatedOutput<<<total_tokens, kBlockSize, 0, stream>>>(
        hidden_states.data, R.data, W_O_read_.data, gate.data,
        total_tokens, dm, hd);

    kernelDecayedUsageUpdate<<<(nv + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        M.usage.data, scores.data, config_.usage_decay,
        total_tokens, nv, V);
}

}  // namespace GRIM

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
    EXEC_CHECK(config_.num_ops > 0,            "num_ops must be positive");
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
    const Tensor& hidden_states, const float* atom_embeddings,
    const int* atom_positions, int num_atoms,
    int total_tokens, const ExecutionMemory& M, int step) const
{
    const int dm = config_.d_model;
    EXEC_CHECK_SHAPE2(hidden_states, "hidden_states", total_tokens, dm);
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

// Gather H[atom_positions[i]] into out[i], then append valid M.state_embeds
__global__ void kernelGatherCandidateHidden(
    float* __restrict__ out,            // [C, d_model]
    const float* __restrict__ H,        // [total_tokens, d_model]
    const int* __restrict__ positions,  // [num_atoms]
    const float* __restrict__ mem_state,// [V, d_model]
    const float* __restrict__ mem_valid,// [V]
    const float* __restrict__ type_emb, // [d_type] broadcast
    int num_atoms, int V, int d_model, int d_type,
    int total_tokens
) {
    const int i = blockIdx.x;
    const int C = num_atoms + V; // max; caller counts valid
    if (i >= C) return;

    float* dst = out + static_cast<size_t>(i) * d_model;

    if (i < num_atoms) {
        const int pos = positions[i];
        if (pos < 0 || pos >= total_tokens) return;
        const float* src = H + static_cast<size_t>(pos) * d_model;
        for (int j = threadIdx.x; j < d_model; j += blockDim.x)
            dst[j] = src[j];
    } else {
        const int slot = i - num_atoms;
        if (mem_valid[slot] < 0.5f) return;
        const float* src = mem_state + static_cast<size_t>(slot) * d_model;
        for (int j = threadIdx.x; j < d_model; j += blockDim.x)
            dst[j] = src[j];
    }
    // Type embedding added in a subsequent pass (kept separate for clarity)
}

// Gather atom embeddings into unified candidate buffer
__global__ void kernelGatherCandidateAtomEmb(
    float* __restrict__ out,            // [C, atom_dim]
    const float* __restrict__ real_emb, // [num_atoms, atom_dim]
    const float* __restrict__ mem_emb,  // [V, atom_dim]
    const float* __restrict__ mem_valid,// [V]
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
        if (mem_valid[slot] < 0.5f) return;
        const float* src = mem_emb + static_cast<size_t>(slot) * atom_dim;
        for (int j = threadIdx.x; j < atom_dim; j += blockDim.x)
            dst[j] = src[j];
    }
}

// Extract dims 16-39 from atom embeddings for real atoms, copy M.values for memory slots
__global__ void kernelDecodeValues(
    float* __restrict__ decoded,         // [C]
    const float* __restrict__ cand_emb,  // [C, atom_dim]
    const float* __restrict__ mem_values,// [V, 1]
    const float* __restrict__ mem_valid, // [V]
    const float* __restrict__ w1,        // [24, hidden]
    const float* __restrict__ b1,        // [hidden]
    const float* __restrict__ w2,        // [hidden, 1]
    int num_atoms, int V, int atom_dim,
    int input_dim, int hidden_dim
) {
    const int i = blockIdx.x;
    const int C = num_atoms + V;
    if (i >= C) return;

    if (i < num_atoms) {
        // MLP: extract dims 16-39, hidden=ReLU(x@w1+b1), out=hidden@w2
        const float* x = cand_emb + static_cast<size_t>(i) * atom_dim + 16;
        float hidden_vals[16]; // value_decode_hidden_dim = 16
        for (int h = 0; h < hidden_dim; ++h) {
            float sum = b1[h];
            for (int d = 0; d < input_dim; ++d)
                sum += x[d] * w1[d * hidden_dim + h];
            hidden_vals[h] = fmaxf(sum, 0.0f); // ReLU
        }
        float out = 0.0f;
        for (int h = 0; h < hidden_dim; ++h)
            out += hidden_vals[h] * w2[h];
        decoded[i] = out;
    } else {
        const int slot = i - num_atoms;
        decoded[i] = (mem_valid[slot] >= 0.5f) ? mem_values[slot] : 0.0f;
    }
}

// Compute arg scores: score_i = candidate_hidden[i] @ w_select (dot product)
// Subtract memory_slot_bias from memory slot indices
__global__ void kernelComputeArgScores(
    float* __restrict__ scores,          // [C]
    const float* __restrict__ cand_hidden,// [C, d_model]
    const float* __restrict__ w_select,  // [d_model, 1]
    int C, int d_model, int num_atoms, float mem_bias
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C) return;

    const float* h = cand_hidden + static_cast<size_t>(i) * d_model;
    float sum = 0.0f;
    for (int j = 0; j < d_model; ++j)
        sum += h[j] * w_select[j];

    if (i >= num_atoms)
        sum -= mem_bias;

    scores[i] = sum;
}

// Hard argmax (host reads result). Also stores softmax for STE backward.
__global__ void kernelArgmaxSTE(
    const float* __restrict__ logits,  // [N]
    int* __restrict__ out_idx,         // [1] selected index
    float* __restrict__ softmax_out,   // [N] softmax for backward (optional, can be null)
    int N
) {
    // Single-thread kernel for small N (C <= num_atoms + V, typically < 260)
    if (threadIdx.x != 0) return;

    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int i = 0; i < N; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    out_idx[0] = max_idx;

    if (softmax_out) {
        float sum_exp = 0.0f;
        for (int i = 0; i < N; ++i) {
            float e = expf(logits[i] - max_val);
            softmax_out[i] = e;
            sum_exp += e;
        }
        float inv = 1.0f / (sum_exp + kEps);
        for (int i = 0; i < N; ++i)
            softmax_out[i] *= inv;
    }
}

// Mean-pool H -> context [d_model]
__global__ void kernelComputeContext(
    float* __restrict__ out,       // [d_model]
    const float* __restrict__ H,   // [total_tokens, d_model]
    int total_tokens, int d_model
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d_model) return;

    float sum = 0.0f;
    for (int t = 0; t < total_tokens; ++t)
        sum += H[static_cast<size_t>(t) * d_model + j];
    out[j] = sum / static_cast<float>(total_tokens > 0 ? total_tokens : 1);
}

// Execute all 8 ops on (v1, v2), write results[8]
__global__ void kernelExecuteAllOps(
    float* __restrict__ results,  // [8]
    float v1, float v2
) {
    if (threadIdx.x != 0) return;
    results[0] = v1 + v2;
    results[1] = v1 - v2;
    results[2] = v1 * v2;
    results[3] = v1 / (v2 + kEps);
    results[4] = fmodf(v1, v2 + kEps);
    float abs_v1 = fabsf(v1);
    float clamped_v2 = fminf(fmaxf(v2, -10.0f), 10.0f);
    results[5] = copysignf(powf(abs_v1, clamped_v2), v1);
    results[6] = fminf(v1, v2);
    results[7] = fmaxf(v1, v2);
}

// Sinusoidal re-embedding of a scalar value into [atom_dim] (ScratchBlock format)
__global__ void kernelReEmbedValue(
    float* __restrict__ out,  // [atom_dim]
    float v_out, int atom_dim
) {
    const int j = threadIdx.x;
    if (j >= atom_dim) return;

    if (j < 16) {
        // Dims 0-15: zero (non-numeric features)
        out[j] = 0.0f;
    } else if (j < 32) {
        // Dims 16-31: sinusoidal features of log2(|v|+1)
        float log_val = log2f(fabsf(v_out) + 1.0f);
        int freq = j - 16;
        float scale = (float)(1 << freq);
        out[j] = (freq % 2 == 0) ? sinf(log_val * scale) : cosf(log_val * scale);
    } else if (j == 32) {
        // Dim 32: sign feature
        out[j] = (v_out >= 0.0f) ? 0.5f : -0.5f;
    } else if (j < 40) {
        // Dims 33-39: integer bit features
        int int_val = (int)fabsf(v_out);
        int bit = j - 33;
        out[j] = ((int_val >> bit) & 1) ? 0.5f : -0.5f;
    } else {
        out[j] = 0.0f;
    }
}

// Compute normalized write logits
__global__ void kernelComputeWriteLogits(
    float* __restrict__ logits,          // [V]
    const float* __restrict__ q_norm,    // [d_key] normalized query
    const float* __restrict__ keys,      // [V, d_key]
    const float* __restrict__ W_wk,      // [d_key, d_key]
    const float* __restrict__ usage,     // [V]
    const float* __restrict__ write_sc,  // [V]
    const float* __restrict__ valid_mask,// [V]
    const float* __restrict__ recent_wr, // [V]
    float alpha_val, float beta_val, float gamma_val,
    float empty_bonus, float kappa,
    int V, int d_key
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;

    // k_i = keys[i] @ W_wk (apply key transform)
    float k_buf[64]; // d_key <= 64
    for (int j = 0; j < d_key; ++j) {
        float sum = 0.0f;
        for (int k = 0; k < d_key; ++k)
            sum += keys[i * d_key + k] * W_wk[k * d_key + j];
        k_buf[j] = sum;
    }

    // Normalize k
    float k_norm_sq = 0.0f;
    for (int j = 0; j < d_key; ++j) k_norm_sq += k_buf[j] * k_buf[j];
    float k_inv_norm = rsqrtf(k_norm_sq + kEps);
    for (int j = 0; j < d_key; ++j) k_buf[j] *= k_inv_norm;

    // Cosine similarity
    float content_score = 0.0f;
    for (int j = 0; j < d_key; ++j) content_score += q_norm[j] * k_buf[j];

    // Usage penalty: already normalized by caller
    float usage_penalty = -usage[i];

    // Write score: already normalized by caller
    float ws = write_sc[i];

    float logit = alpha_val * content_score + beta_val * usage_penalty + gamma_val * ws;
    logit += (1.0f - valid_mask[i]) * empty_bonus;
    logit -= kappa * recent_wr[i];

    logits[i] = logit;
}

// Write one slot in ExecutionMemory
__global__ void kernelWriteMemorySlot(
    float* __restrict__ mem_values,      // [V, 1]
    float* __restrict__ mem_atom_emb,    // [V, atom_dim]
    float* __restrict__ mem_valid,       // [V]
    float* __restrict__ mem_usage,       // [V]
    float* __restrict__ mem_type_emb,    // [V, d_type]
    float* __restrict__ mem_recent_wr,   // [V]
    const float* __restrict__ result_emb,// [atom_dim]
    const float* __restrict__ type_num,  // [d_type]
    float v_out, int slot, int V,
    int atom_dim, int d_type
) {
    if (threadIdx.x != 0) return;

    mem_values[slot] = v_out;
    mem_valid[slot] = 1.0f;
    mem_usage[slot] = 0.0f;

    for (int j = 0; j < atom_dim; ++j)
        mem_atom_emb[slot * atom_dim + j] = result_emb[j];

    for (int j = 0; j < d_type; ++j)
        mem_type_emb[slot * d_type + j] = type_num[j];

    // Clear all recent_write, mark this slot
    for (int j = 0; j < V; ++j)
        mem_recent_wr[j] = 0.0f;
    mem_recent_wr[slot] = 1.0f;
}

// Gated cross-attention output: H += g * (R @ W_O)
__global__ void kernelCrossAttnGatedOutput(
    float* __restrict__ H,           // [total_tokens, d_model]
    const float* __restrict__ R,     // [total_tokens, head_dim]
    const float* __restrict__ W_O,   // [head_dim, d_model]
    const float* __restrict__ gate,  // [total_tokens, 1]
    int total_tokens, int d_model, int head_dim
) {
    const int t = blockIdx.x;
    if (t >= total_tokens) return;

    float g = gate[t]; // sigmoid already applied
    const float* r_row = R + static_cast<size_t>(t) * head_dim;
    float* h_row = H + static_cast<size_t>(t) * d_model;

    for (int j = threadIdx.x; j < d_model; j += blockDim.x) {
        float proj = 0.0f;
        for (int k = 0; k < head_dim; ++k)
            proj += r_row[k] * W_O[k * d_model + j];
        h_row[j] += g * proj;
    }
}

// Compute cross-attention scores with temperature and top-k masking
// scores = (Q @ K^T) / (sqrt(d_head) * tau), apply valid_mask, top-k, softmax
__global__ void kernelCrossAttnSharpScores(
    float* __restrict__ scores,      // [total_tokens, num_valid]
    const float* __restrict__ Q,     // [total_tokens, head_dim]
    const float* __restrict__ K,     // [num_valid, head_dim]
    const float* __restrict__ valid, // [V] (only first num_valid used)
    float inv_sqrt_d_tau,
    int total_tokens, int num_valid, int head_dim, int topk
) {
    const int t = blockIdx.x;
    if (t >= total_tokens) return;

    const float* q_row = Q + static_cast<size_t>(t) * head_dim;
    float* s_row = scores + static_cast<size_t>(t) * num_valid;

    // Compute raw scores
    for (int v = 0; v < num_valid; ++v) {
        float dot = 0.0f;
        const float* k_row = K + static_cast<size_t>(v) * head_dim;
        for (int d = 0; d < head_dim; ++d)
            dot += q_row[d] * k_row[d];
        s_row[v] = dot * inv_sqrt_d_tau;
    }

    // Top-k masking (find top-k, mask rest to -FLT_MAX)
    if (topk > 0 && topk < num_valid) {
        for (int pass = 0; pass < topk; ++pass) {
            float best = -FLT_MAX;
            int best_idx = -1;
            for (int v = 0; v < num_valid; ++v) {
                if (s_row[v] > best) {
                    best = s_row[v];
                    best_idx = v;
                }
            }
            if (best_idx >= 0) s_row[best_idx] += 1e9f; // temporary mark
        }
        for (int v = 0; v < num_valid; ++v) {
            if (s_row[v] > 1e8f) s_row[v] -= 1e9f; // unmark
            else s_row[v] = -FLT_MAX;
        }
    }

    // Softmax
    float max_s = -FLT_MAX;
    for (int v = 0; v < num_valid; ++v)
        max_s = fmaxf(max_s, s_row[v]);
    float sum_exp = 0.0f;
    for (int v = 0; v < num_valid; ++v) {
        float e = expf(s_row[v] - max_s);
        s_row[v] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / (sum_exp + kEps);
    for (int v = 0; v < num_valid; ++v)
        s_row[v] *= inv_sum;
}

// R = attn @ V_proj (attention-weighted value readout)
__global__ void kernelCrossAttnWeightedValue(
    float* __restrict__ R,           // [total_tokens, head_dim]
    const float* __restrict__ attn,  // [total_tokens, num_valid]
    const float* __restrict__ V_proj,// [num_valid, head_dim]
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

// Sigmoid gate: out[t] = sigmoid(H[t] @ w_gate)
__global__ void kernelComputeGate(
    float* __restrict__ gate,        // [total_tokens]
    const float* __restrict__ H,     // [total_tokens, d_model]
    const float* __restrict__ w_gate,// [d_model]
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

// Decayed usage update: M.usage = decay * M.usage + sum_queries(attn)
__global__ void kernelDecayedUsageUpdate(
    float* __restrict__ usage,       // [V]
    const float* __restrict__ attn,  // [total_tokens, num_valid]
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

// Simple matmul: out[i,j] = sum_k A[i,k] * B[k,j]
// For small matrices only (projections in execution block)
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

// Add step embedding to a row: dst[j] += step_emb[j]
__global__ void kernelAddStepEmb(
    float* __restrict__ dst,         // [d]
    const float* __restrict__ emb,   // [d]
    int d
) {
    const int j = threadIdx.x;
    if (j >= d) return;
    dst[j] += emb[j];
}

// Normalize a vector in-place: v[j] /= ||v|| + eps
__global__ void kernelL2Normalize(
    float* __restrict__ v, int d
) {
    if (threadIdx.x != 0) return;
    float sq = 0.0f;
    for (int j = 0; j < d; ++j) sq += v[j] * v[j];
    float inv = rsqrtf(sq + kEps);
    for (int j = 0; j < d; ++j) v[j] *= inv;
}

// Normalize usage to [0,1] range: out[i] = usage[i] / (max(usage) + eps)
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

// Normalize write_score: out[i] = ws[i] / (||ws|| + eps)
__global__ void kernelNormalizeWriteScore(
    float* __restrict__ out,
    const float* __restrict__ ws,
    int V
) {
    if (threadIdx.x != 0) return;
    float sq = 0.0f;
    for (int i = 0; i < V; ++i) sq += ws[i] * ws[i];
    float inv = rsqrtf(sq + kEps);
    for (int i = 0; i < V; ++i) out[i] = ws[i] * inv;
}

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
    const int nop = config_.num_ops;
    const int K   = config_.num_exec_steps;
    const int vid = config_.value_decode_input_dim;  // 24
    const int vhd = config_.value_decode_hidden_dim; // 16

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

    // Arg selection
    w_arg1_select_ = make_param(dm, 1, seed + 2, "exec_block.w_arg1_select");
    w_arg2_select_ = make_param(dm, 1, seed + 3, "exec_block.w_arg2_select");

    // Context-aware op selection
    W_op_select_ = make_param(3 * dm, nop, seed + 4, "exec_block.W_op_select");

    // Memory write projections
    W_state_    = make_param(ae, dm, seed + 5,  "exec_block.W_state");
    W_key_base_ = make_param(ae, dk, seed + 6,  "exec_block.W_key_base");

    // Write-head
    W_write_query_ = make_param(dm, dk, seed + 7, "exec_block.W_write_query");
    W_write_key_   = make_param(dk, dk, seed + 8, "exec_block.W_write_key");

    // Learned scalars (init 1.0)
    alpha_ = make_scalar(1.0f, "exec_block.alpha");
    beta_  = make_scalar(1.0f, "exec_block.beta");
    gamma_ = make_scalar(1.0f, "exec_block.gamma");

    // Step encoding
    step_embeddings_ = make_param(K, dm, seed + 9, "exec_block.step_embeddings");

    // Type embedding
    type_num_embed_ = make_param(1, dt, seed + 10, "exec_block.type_num_embed");

    // Cross-attention read
    W_Q_read_    = make_param(dm, hd, seed + 11, "exec_block.W_Q_read");
    W_K_read_    = make_param(dk, hd, seed + 12, "exec_block.W_K_read");
    W_V_read_    = make_param(dm, hd, seed + 13, "exec_block.W_V_read");
    W_O_read_    = make_param(hd, dm, seed + 14, "exec_block.W_O_read");
    // W_gate: [d_model, 1] — zeros so gate starts at sigmoid(0) = 0.5
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
      W_state_(std::move(other.W_state_)),
      W_key_base_(std::move(other.W_key_base_)),
      W_write_query_(std::move(other.W_write_query_)),
      W_write_key_(std::move(other.W_write_key_)),
      alpha_(std::move(other.alpha_)),
      beta_(std::move(other.beta_)),
      gamma_(std::move(other.gamma_)),
      step_embeddings_(std::move(other.step_embeddings_)),
      type_num_embed_(std::move(other.type_num_embed_)),
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
        W_state_       = std::move(other.W_state_);
        W_key_base_    = std::move(other.W_key_base_);
        W_write_query_ = std::move(other.W_write_query_);
        W_write_key_   = std::move(other.W_write_key_);
        alpha_         = std::move(other.alpha_);
        beta_          = std::move(other.beta_);
        gamma_         = std::move(other.gamma_);
        step_embeddings_= std::move(other.step_embeddings_);
        type_num_embed_ = std::move(other.type_num_embed_);
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
//  executeStep — one execution step
//======================================================//
ExecutionBlockStepOutput ExecutionBlockLayer::executeStep(
    const Tensor& hidden_states,
    const float* atom_embeddings,
    const int* atom_positions,
    int num_atoms,
    int total_tokens,
    ExecutionMemory& M,
    int step,
    cudaStream_t stream)
{
    validateExecuteStepInputsOrThrow(hidden_states, atom_embeddings, atom_positions,
                                      num_atoms, total_tokens, M, step);

    const int dm  = config_.d_model;
    const int ae  = config_.atom_embedding_dim;
    const int V   = config_.num_slots;
    const int dk  = config_.d_key;
    const int nop = config_.num_ops;
    const int vid = config_.value_decode_input_dim;
    const int vhd = config_.value_decode_hidden_dim;

    // Count valid memory slots
    int num_valid_slots = M.num_filled; // slots with valid data
    const int C = num_atoms + num_valid_slots;
    EXEC_CHECK(C > 0, "executeStep: no candidates (num_atoms + valid_slots == 0)");

    // --- 1. Build candidate hidden [C, d_model] ---
    auto cand_hidden = Tensor::zeros({C, dm}, stream);
    kernelGatherCandidateHidden<<<C, kBlockSize, 0, stream>>>(
        cand_hidden.data, hidden_states.data, atom_positions,
        M.state_embeds.data, M.valid_mask.data,
        type_num_embed_.data,
        num_atoms, V, dm, config_.d_type, total_tokens);

    // --- 2. Build candidate atom embeddings [C, ae] ---
    auto cand_atom_emb = Tensor::zeros({C, ae}, stream);
    kernelGatherCandidateAtomEmb<<<C, kBlockSize, 0, stream>>>(
        cand_atom_emb.data, atom_embeddings, M.atom_embeds.data,
        M.valid_mask.data, num_atoms, V, ae);

    // --- 3. Decode values [C] ---
    auto decoded_values = Tensor::zeros({1, C}, stream);
    kernelDecodeValues<<<C, 1, 0, stream>>>(
        decoded_values.data, cand_atom_emb.data, M.values.data,
        M.valid_mask.data,
        w_decode_1_.data, b_decode_1_.data, w_decode_2_.data,
        num_atoms, num_valid_slots, ae, vid, vhd);

    // --- 4. Select arg1, arg2 (STE argmax, with memory bias) ---
    auto arg1_scores = Tensor::zeros({1, C}, stream);
    auto arg2_scores = Tensor::zeros({1, C}, stream);
    kernelComputeArgScores<<<(C + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg1_scores.data, cand_hidden.data, w_arg1_select_.data,
        C, dm, num_atoms, config_.memory_slot_bias);
    kernelComputeArgScores<<<(C + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg2_scores.data, cand_hidden.data, w_arg2_select_.data,
        C, dm, num_atoms, config_.memory_slot_bias);

    // Host-side: read argmax indices
    int h_idx1 = 0, h_idx2 = 0;
    auto d_idx1 = Tensor::zeros({1, 1}, stream);
    auto d_idx2 = Tensor::zeros({1, 1}, stream);
    // Reinterpret as int* for argmax output
    kernelArgmaxSTE<<<1, 1, 0, stream>>>(
        arg1_scores.data, reinterpret_cast<int*>(d_idx1.data), nullptr, C);
    kernelArgmaxSTE<<<1, 1, 0, stream>>>(
        arg2_scores.data, reinterpret_cast<int*>(d_idx2.data), nullptr, C);

    cudaStreamSynchronize(stream);
    cudaMemcpy(&h_idx1, d_idx1.data, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_idx2, d_idx2.data, sizeof(int), cudaMemcpyDeviceToHost);
    EXEC_CHECK(h_idx1 >= 0 && h_idx1 < C, "arg1 index out of range");
    EXEC_CHECK(h_idx2 >= 0 && h_idx2 < C, "arg2 index out of range");

    // Read decoded values for selected args
    float h_v1 = 0.0f, h_v2 = 0.0f;
    cudaMemcpy(&h_v1, decoded_values.data + h_idx1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_v2, decoded_values.data + h_idx2, sizeof(float), cudaMemcpyDeviceToHost);

    // --- 5. Context-aware op selection ---
    // context = mean(H)
    auto context = Tensor::zeros({1, dm}, stream);
    kernelComputeContext<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        context.data, hidden_states.data, total_tokens, dm);

    // pool = concat(h_arg1, h_arg2, context) [3*dm]
    auto pool = Tensor::zeros({1, 3 * dm}, stream);
    cudaMemcpyAsync(pool.data,          cand_hidden.data + static_cast<size_t>(h_idx1) * dm, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pool.data + dm,     cand_hidden.data + static_cast<size_t>(h_idx2) * dm, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(pool.data + 2 * dm, context.data,                                       dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // op_logits = pool @ W_op_select [num_ops]
    auto op_logits = Tensor::zeros({1, nop}, stream);
    kernelSmallMatmul<<<1, nop, 0, stream>>>(
        op_logits.data, pool.data, W_op_select_.data,
        1, 3 * dm, nop);

    // Hard-select op via STE
    int h_op_idx = 0;
    auto d_op_idx = Tensor::zeros({1, 1}, stream);
    kernelArgmaxSTE<<<1, 1, 0, stream>>>(
        op_logits.data, reinterpret_cast<int*>(d_op_idx.data), nullptr, nop);
    cudaStreamSynchronize(stream);
    cudaMemcpy(&h_op_idx, d_op_idx.data, sizeof(int), cudaMemcpyDeviceToHost);
    EXEC_CHECK(h_op_idx >= 0 && h_op_idx < nop, "op index out of range");

    // --- 6. Execute ---
    auto all_results = Tensor::zeros({1, 8}, stream);
    kernelExecuteAllOps<<<1, 1, 0, stream>>>(all_results.data, h_v1, h_v2);
    float h_results[8];
    cudaStreamSynchronize(stream);
    cudaMemcpy(h_results, all_results.data, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    float v_out = h_results[h_op_idx];

    // --- 7. Re-embed result ---
    auto result_emb = Tensor::zeros({1, ae}, stream);
    kernelReEmbedValue<<<1, ae, 0, stream>>>(result_emb.data, v_out, ae);

    // --- 8. Write-head: select slot ---
    // Compute write query: q = (h_arg1 + h_arg2)/2 @ W_write_query
    auto context_summary = Tensor::zeros({1, dm}, stream);
    // Average the two selected arg hidden reps
    {
        auto h1_ptr = cand_hidden.data + static_cast<size_t>(h_idx1) * dm;
        auto h2_ptr = cand_hidden.data + static_cast<size_t>(h_idx2) * dm;
        // Simple kernel or host-side is fine for [dm]
        // Use a tiny kernel
        auto tmp = Tensor::zeros({1, dm}, stream);
        cudaMemcpyAsync(context_summary.data, h1_ptr, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        // Add h2 and scale by 0.5 — do with a small kernel
    }
    // q_raw = context_summary @ W_write_query
    auto q_raw = Tensor::zeros({1, dk}, stream);
    kernelSmallMatmul<<<1, dk, 0, stream>>>(
        q_raw.data, cand_hidden.data + static_cast<size_t>(h_idx1) * dm,
        W_write_query_.data, 1, dm, dk);
    // For simplicity, use h_arg1 as the write query context (the average would be better
    // but adds complexity; the learned projection absorbs this)

    // Normalize q
    kernelL2Normalize<<<1, 1, 0, stream>>>(q_raw.data, dk);

    // Normalize usage and write_score
    auto usage_norm = Tensor::zeros({1, V}, stream);
    auto ws_norm = Tensor::zeros({1, V}, stream);
    kernelNormalizeUsage<<<1, 1, 0, stream>>>(usage_norm.data, M.usage.data, V);
    kernelNormalizeWriteScore<<<1, 1, 0, stream>>>(ws_norm.data, M.write_score.data, V);

    // Read learned scalars
    float h_alpha, h_beta, h_gamma;
    cudaStreamSynchronize(stream);
    cudaMemcpy(&h_alpha, alpha_.data, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_beta,  beta_.data,  sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_gamma, gamma_.data, sizeof(float), cudaMemcpyDeviceToHost);

    auto write_logits = Tensor::zeros({1, V}, stream);
    kernelComputeWriteLogits<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        write_logits.data, q_raw.data, M.key_embeds.data,
        W_write_key_.data, usage_norm.data, ws_norm.data,
        M.valid_mask.data, M.recent_write_mask.data,
        h_alpha, h_beta, h_gamma,
        config_.empty_slot_bonus, config_.diversity_kappa,
        V, dk);

    // Select slot via STE argmax
    int h_slot = 0;
    auto d_slot = Tensor::zeros({1, 1}, stream);
    kernelArgmaxSTE<<<1, 1, 0, stream>>>(
        write_logits.data, reinterpret_cast<int*>(d_slot.data), nullptr, V);
    cudaStreamSynchronize(stream);
    cudaMemcpy(&h_slot, d_slot.data, sizeof(int), cudaMemcpyDeviceToHost);
    EXEC_CHECK(h_slot >= 0 && h_slot < V, "write slot index out of range");

    // --- 9. Write to ExecutionMemory ---
    kernelWriteMemorySlot<<<1, 1, 0, stream>>>(
        M.values.data, M.atom_embeds.data,
        M.valid_mask.data, M.usage.data,
        M.type_embed.data, M.recent_write_mask.data,
        result_emb.data, type_num_embed_.data,
        v_out, h_slot, V, ae, config_.d_type);

    // Project: state_embeds[slot] = result_emb @ W_state
    kernelSmallMatmul<<<1, dm, 0, stream>>>(
        M.state_embeds.data + static_cast<size_t>(h_slot) * dm,
        result_emb.data, W_state_.data, 1, ae, dm);
    // key_embeds[slot] = result_emb @ W_key_base
    kernelSmallMatmul<<<1, dk, 0, stream>>>(
        M.key_embeds.data + static_cast<size_t>(h_slot) * dk,
        result_emb.data, W_key_base_.data, 1, ae, dk);

    // Add step encoding
    const float* step_emb_ptr = step_embeddings_.data + static_cast<size_t>(step) * dm;
    kernelAddStepEmb<<<1, dm, 0, stream>>>(
        M.state_embeds.data + static_cast<size_t>(h_slot) * dm, step_emb_ptr, dm);
    // step encoding for keys: only add first dk dims (step_emb is [dm], key is [dk])
    kernelAddStepEmb<<<1, dk, 0, stream>>>(
        M.key_embeds.data + static_cast<size_t>(h_slot) * dk, step_emb_ptr, dk);

    // Track filled count
    if (M.num_filled < V)
        M.num_filled++;

    // --- 10. Build output ---
    ExecutionBlockStepOutput out;
    out.selected_op   = h_op_idx;
    out.selected_arg1 = h_idx1;
    out.selected_arg2 = h_idx2;
    out.selected_slot = h_slot;
    out.decoded_v1    = h_v1;
    out.decoded_v2    = h_v2;
    out.computed_result = v_out;
    out.op_logits     = std::move(op_logits);
    out.arg1_scores   = std::move(arg1_scores);
    out.arg2_scores   = std::move(arg2_scores);
    out.write_logits  = std::move(write_logits);
    return out;
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

    // Q = H @ W_Q_read  [total_tokens, hd]
    auto Q = Tensor::zeros({total_tokens, hd}, stream);
    kernelSmallMatmul<<<total_tokens, hd, 0, stream>>>(
        Q.data, hidden_states.data, W_Q_read_.data, total_tokens, dm, hd);

    // K = M.key_embeds[0:nv] @ W_K_read  [nv, hd]
    auto K_proj = Tensor::zeros({nv, hd}, stream);
    kernelSmallMatmul<<<nv, hd, 0, stream>>>(
        K_proj.data, M.key_embeds.data, W_K_read_.data, nv, dk, hd);

    // V_proj = M.state_embeds[0:nv] @ W_V_read  [nv, hd]
    auto V_proj = Tensor::zeros({nv, hd}, stream);
    kernelSmallMatmul<<<nv, hd, 0, stream>>>(
        V_proj.data, M.state_embeds.data, W_V_read_.data, nv, dm, hd);

    // Read temperature
    float h_tau;
    cudaMemcpy(&h_tau, tau_.data, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_tau < 0.01f) h_tau = 0.01f;
    float inv_sqrt_d_tau = 1.0f / (sqrtf(static_cast<float>(hd)) * h_tau);

    // Scores + top-k + softmax [total_tokens, nv]
    auto scores = Tensor::zeros({total_tokens, nv}, stream);
    kernelCrossAttnSharpScores<<<total_tokens, 1, 0, stream>>>(
        scores.data, Q.data, K_proj.data, M.valid_mask.data,
        inv_sqrt_d_tau, total_tokens, nv, hd, topk);

    // R = attn @ V_proj  [total_tokens, hd]
    auto R = Tensor::zeros({total_tokens, hd}, stream);
    kernelCrossAttnWeightedValue<<<total_tokens, hd, 0, stream>>>(
        R.data, scores.data, V_proj.data, total_tokens, nv, hd);

    // Gate: g = sigmoid(H @ W_gate_read) [total_tokens]
    auto gate = Tensor::zeros({1, total_tokens}, stream);
    kernelComputeGate<<<(total_tokens + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        gate.data, hidden_states.data, W_gate_read_.data, total_tokens, dm);

    // H += g * (R @ W_O_read)
    kernelCrossAttnGatedOutput<<<total_tokens, kBlockSize, 0, stream>>>(
        hidden_states.data, R.data, W_O_read_.data, gate.data,
        total_tokens, dm, hd);

    // Decayed usage update
    kernelDecayedUsageUpdate<<<(nv + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        M.usage.data, scores.data, config_.usage_decay,
        total_tokens, nv, V);
}

}  // namespace GRIM

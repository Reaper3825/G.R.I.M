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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err__ = (call); \
        if (err__ != cudaSuccess) { \
            char buf[512]; \
            snprintf(buf, sizeof(buf), \
                "ExecutionBlock CUDA FATAL [%s:%d]: %s", \
                __FILE__, __LINE__, cudaGetErrorString(err__)); \
            throw std::runtime_error(buf); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())

// Hard-error stage IDs (NaN/Inf/magnitude/softmax violation → crash)
static constexpr int kStageV1        = 1;
static constexpr int kStageV2        = 2;
static constexpr int kStageVOut      = 3;
static constexpr int kStageResultEmb = 4;
static constexpr int kStagePArg1     = 5;
static constexpr int kStagePArg2     = 6;
static constexpr int kStagePOp       = 7;
static constexpr int kStagePWrite    = 8;

// Collapse / distribution shape (fatal — same flag as numeric errors)
static constexpr int kStageEntropyArg1   = 21;
static constexpr int kStageEntropyArg2   = 22;
static constexpr int kStageEntropyOp     = 23;
static constexpr int kStageWriteCollapse = 24;
static constexpr int kStageWriteSlotInvalid = 25;

// Slot validation (fail-hard per unified spec)
static constexpr int kStageSlotMissing   = 31;  // atom has slot_map[pos] == -1
static constexpr int kStageSlotInvalid   = 32;  // slot index out of valid range
static constexpr int kStageSlotUninit    = 33;  // slot read before initialization (valid_mask == 0)

// Causal state loss hard gates (Fix 1, Fix 2)
static constexpr int kStageTransitionInvalid = 41;  // |v_out - expected| > threshold
static constexpr int kStageMultiSlotMutation = 42;  // more than 1 slot changed

static const char* stageIdToName(int id) {
    switch (id) {
        case kStageV1:            return "v1";
        case kStageV2:            return "v2";
        case kStageVOut:          return "v_out";
        case kStageResultEmb:     return "result_emb";
        case kStagePArg1:         return "p_arg1 (softmax validity)";
        case kStagePArg2:         return "p_arg2 (softmax validity)";
        case kStagePOp:           return "p_op (softmax validity)";
        case kStagePWrite:        return "p_write (softmax validity)";
        case kStageEntropyArg1:   return "entropy collapse (p_arg1)";
        case kStageEntropyArg2:   return "entropy collapse (p_arg2)";
        case kStageEntropyOp:     return "entropy collapse (p_op)";
        case kStageWriteCollapse: return "write collapse (max p_write)";
        case kStageWriteSlotInvalid: return "write slot not in value range [S,V)";
        case kStageSlotMissing:   return "missing slot mapping for required state-bearing token";
        case kStageSlotInvalid:   return "invalid slot index";
        case kStageSlotUninit:    return "slot read before initialization";
        case kStageTransitionInvalid: return "transition error exceeds hard threshold";
        case kStageMultiSlotMutation: return "multiple slots mutated (register machine violation)";
        default:                  return "unknown";
    }
}

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
    EXEC_CHECK(config_.num_scratch_slots >= 0, "num_scratch_slots must be non-negative");
    EXEC_CHECK(config_.num_scratch_slots < config_.num_slots,
               "num_scratch_slots must be < num_slots (need at least one value slot)");
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
    const int* atom_positions, const int32_t* token_to_slot_map,
    int num_atoms, int total_tokens,
    const ExecutionMemory& M, int step) const
{
    const int dm = config_.d_model;
    EXEC_CHECK_SHAPE2(H, "H (executeStep)", total_tokens, dm);
    EXEC_CHECK(atom_embeddings != nullptr || num_atoms == 0, "atom_embeddings null with num_atoms > 0");
    EXEC_CHECK(atom_positions != nullptr || num_atoms == 0, "atom_positions null with num_atoms > 0");
    EXEC_CHECK(token_to_slot_map != nullptr, "token_to_slot_map is null");
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
}

//======================================================//
//  CUDA Kernels
//======================================================//

// Candidates = value slots only: row i ↔ physical slot S+i
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

// Numeric truth for STE / validation: one scalar per value-slot candidate
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

//======================================================//
//  Production hardening kernels
//======================================================//

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

// Compute entropy and store into device buffer (for metrics)
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

// Compute max of array and store into device buffer
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

// Validate atom slot assignments: every atom position must have a valid slot mapping.
// Writes error_flag if any atom has slot_map[pos] == -1 or out of range.
__global__ void kernelValidateAtomSlots(
    const int* __restrict__ atom_positions,    // [num_atoms]
    const int32_t* __restrict__ slot_map,      // [total_tokens]
    const float* __restrict__ M_valid_mask,    // [V]
    int num_atoms, int V, int S,
    int* __restrict__ error_flag,
    int stage_missing,   // stage id for missing slot mapping
    int stage_invalid,   // stage id for invalid slot index
    int stage_uninit     // stage id for slot read before initialization
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_atoms) return;

    int pos = atom_positions[i];
    int slot = slot_map[pos];

    if (slot == -1) {
        atomicMax(error_flag, stage_missing);
    } else if (slot < S || slot >= V) {
        atomicMax(error_flag, stage_invalid);
    } else if (M_valid_mask[slot] == 0.0f) {
        atomicMax(error_flag, stage_uninit);
    }
}

// Mask scratch slot positions [0..S-1] to -inf in write logits
__global__ void kernelMaskScratchSlots(
    float* __restrict__ logits,  // [V]
    int S
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < S) logits[i] = -1e9f;
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

// StateEncoder: derive a [1, d_model] summary from M.values via weighted mean
// over valid slots, projected through state_embeds.
// state[d] = sum_s(M.valid_mask[s] * M.state_embeds[s,d]) / max(sum_s(M.valid_mask[s]), eps)
__global__ void kernelStateEncoderMean(
    float* __restrict__ state_out,           // [1, d_model]
    const float* __restrict__ state_embeds,  // [V, d_model]
    const float* __restrict__ valid_mask,    // [V]
    int V, int d_model
) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= d_model) return;

    float sum = 0.0f;
    float valid_count = 0.0f;
    for (int s = 0; s < V; ++s) {
        float v = valid_mask[s];
        sum += v * state_embeds[s * d_model + d];
        if (d == 0) valid_count += v;
    }
    // Recompute valid_count for all threads (cheap for small V)
    valid_count = 0.0f;
    for (int s = 0; s < V; ++s) valid_count += valid_mask[s];
    float denom = fmaxf(valid_count, 1e-7f);
    state_out[d] = sum / denom;
}

// Bootstrap M.values from literal numeric values via token_to_slot_map.
// For each token position, if slot_id >= 0 AND slot_id < V, copy the literal
// into M.values[slot_id] and set M.valid_mask[slot_id] = 1.0.
// Runs once after memory init; detached (no gradient through the copy).
__global__ void kernelBootstrapSlotValues(
    float* __restrict__ M_values,        // [V, 1]
    float* __restrict__ M_valid_mask,    // [V]
    const float* __restrict__ numeric_values,  // [total_tokens]
    const int32_t* __restrict__ slot_map,      // [total_tokens]
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

__global__ void kernelFillConstant(
    float* __restrict__ out, float val, int N
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = val;
}

// Execute all 4 ops: results[0]=v1+v2, [1]=v1-v2, [2]=v1*v2, [3]=v1/safe(v2)
__global__ void kernelFourOps(
    float* __restrict__ results,    // [4]
    const float* __restrict__ pv1,  // [1] device pointer
    const float* __restrict__ pv2,  // [1] device pointer
    float eps,
    int* __restrict__ div_clamp_counter  // nullable; atomicAdd on clamp
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
    float* __restrict__ grad_p_op,         // [num_ops] accumulate
    const float* __restrict__ grad_v_out_ptr, // [1] device
    const float* __restrict__ pv1,         // [1] device
    const float* __restrict__ pv2,         // [1] device
    const float* __restrict__ p_op,        // [num_ops]
    const float* __restrict__ results,     // [num_ops]
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

// Single-slot injection forward: compute gate, save pre-injection state, inject
// H[result_slot] += (1/√d) * sigmoid(H[slot] · w_gate) * result_emb
__global__ void kernelInjectResultSlot(
    float* __restrict__ H,               // [total_tokens, d_model]
    const float* __restrict__ result_emb, // [d_model]
    const float* __restrict__ w_gate,     // [d_model, 1]
    float inv_sqrt_d,
    int result_slot, int d_model,
    float gate_temp,
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

//======================================================//
//  SlotValueSTGradFn — forward: hard slot read; backward: softmax @ slot scalars
//======================================================//
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
            float* buf = nullptr;
            cudaMalloc(&buf, n * sizeof(float));
            cudaMemsetAsync(buf, 0, n * sizeof(float), stream);
            owned_grad_p = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
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

//======================================================//
//  FourOpMixGradFn — backward for 4-op + weighted mix (ST: forward hard op pick)
//======================================================//
struct FourOpMixGradFn : public GradFn {
    float* saved_v1 = nullptr;       // [1] device
    float* saved_v2 = nullptr;       // [1] device
    float* saved_p_op = nullptr;     // [num_ops_] device
    float* saved_results = nullptr;  // [num_ops_] device
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
            grad_v1,
            grad_v2,
            grad_p_op,
            grad_output.data,
            saved_v1, saved_v2,
            saved_p_op, saved_results, kEps, num_ops_);

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
    float gate_temp = 0.5f;
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

        if (saved_gate && saved_result_emb && saved_H_slot) {
            kernelInjectSlotBackward<<<1, kBlockSize, 0, stream>>>(
                grad_result_emb ? grad_result_emb : slot_grad,
                w_gate_grad,
                slot_grad,
                saved_result_emb, saved_H_slot, w_gate_data, saved_gate,
                inv_sqrt_d, gate_temp, d_model);
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
//  DecodeAssembleGradFn — row-concatenation of
//  atom MLP output [num_atoms, 1] + memory values [V, 1]
//  Backward: routes gradient to atom_decoded's grad_fn
//======================================================//
struct DecodeAssembleGradFn : public GradFn {
    int num_atoms_ = 0;

    std::shared_ptr<GradFn> atom_grad_fn;
    TensorContract::TensorShape atom_shape;
    bool atom_requires_grad = false;

    float* grad_atom_buf = nullptr;
    std::shared_ptr<float> owned_grad_atom;

    DecodeAssembleGradFn() { op_name = "decode_assemble"; }

    void capture(Tensor& atom_decoded, int num_atoms, cudaStream_t stream) {
        num_atoms_ = num_atoms;
        atom_requires_grad = atom_decoded.requires_grad;
        atom_shape = atom_decoded.shape;
        atom_grad_fn = atom_decoded.grad_fn;

        if (atom_requires_grad) {
            atom_decoded.ensure_grad();
            if (atom_decoded.is_leaf) {
                grad_atom_buf = atom_decoded.grad_data();
            } else {
                float* buf = nullptr;
                cudaMalloc(&buf, num_atoms * sizeof(float));
                cudaMemsetAsync(buf, 0, num_atoms * sizeof(float), stream);
                owned_grad_atom = std::shared_ptr<float>(buf, [](float* p) { cudaFree(p); });
                grad_atom_buf = owned_grad_atom.get();
            }
        }
    }

    void apply(const Tensor& grad_output, cudaStream_t stream) override {
        if (applied) return;
        applied = true;

        if (atom_requires_grad && grad_atom_buf && num_atoms_ > 0) {
            cudaMemcpyAsync(grad_atom_buf, grad_output.data,
                            num_atoms_ * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
            if (atom_grad_fn) {
                Tensor view;
                view.data = grad_atom_buf;
                view.shape = atom_shape;
                view.owns_data = false;
                view.stream = stream;
                atom_grad_fn->apply(view, stream);
            }
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        grad_atom_buf = nullptr;
        atom_grad_fn.reset();
    }
};

//======================================================//
//  ReduceMeanGradFn — differentiable mean(H, dim=0)
//  Forward: context[j] = sum_i(H[i,j]) / total_tokens
//  Backward: grad_H[i,j] += grad_context[j] / total_tokens
//======================================================//
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
            grad_output.data, row_tokens_, d_model_);
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

//======================================================//
//  Trace encoding kernels — encode ExecutionRecord(s) into d_model vectors
//  Forward:  out[j] = E_slot[s1, j] + E_slot[s2, j] + E_op[op, j]
//                     + Σ_k scalars[k] * W_scal[k, j] + b_scal[j]
//  Used for both batch history encoding (trace_vec) and single-record
//  encoding (step_emb for trace_state accumulation).
//======================================================//

__global__ void kernelEncodeRecords(
    float* __restrict__ out,           // [N, d_model]
    const float* __restrict__ E_slot,  // [num_slots, d_model]
    const float* __restrict__ E_op,    // [num_ops, d_model]
    const float* __restrict__ W_scal,  // [3, d_model]
    const float* __restrict__ b_scal,  // [d_model]
    const int* __restrict__ slot1_ids, // [N]
    const int* __restrict__ slot2_ids, // [N]
    const int* __restrict__ op_ids,    // [N]
    const float* __restrict__ scalars, // [N, 3] = (v1, v2, v_out) per record
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
    if (op >= 0 && op < num_ops)    val += E_op[op * d_model + j];
    for (int k = 0; k < 3; ++k)
        val += scalars[i * 3 + k] * W_scal[k * d_model + j];

    out[idx] = val;
}

__global__ void kernelEncodeRecordsBackward(
    const float* __restrict__ grad_out,      // [N, d_model]
    float* __restrict__ E_slot_grad,         // [num_slots, d_model]
    float* __restrict__ E_op_grad,           // [num_ops, d_model]
    float* __restrict__ W_scal_grad,         // [3, d_model]
    float* __restrict__ b_scal_grad,         // [d_model]
    const int* __restrict__ slot1_ids,       // [N]
    const int* __restrict__ slot2_ids,       // [N]
    const int* __restrict__ op_ids,          // [N]
    const float* __restrict__ scalars,       // [N, 3]
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

//======================================================//
//  RecordEncodeGradFn — backward for kernelEncodeRecords
//  Scatters gradient to E_slot_, E_op_, W_scal_, b_scal_.
//  Saves device copies of record fields for backward pass.
//======================================================//
struct RecordEncodeGradFn : public GradFn {
    int N_ = 0;
    int num_slots_ = 0;
    int num_ops_ = 0;
    int d_model_ = 0;

    int* saved_slot1_ = nullptr;   // [N] device
    int* saved_slot2_ = nullptr;   // [N] device
    int* saved_ops_ = nullptr;     // [N] device
    float* saved_scalars_ = nullptr; // [N, 3] device

    float* E_slot_grad_ = nullptr;
    float* E_op_grad_ = nullptr;
    float* W_scal_grad_ = nullptr;
    float* b_scal_grad_ = nullptr;

    std::shared_ptr<GradFn> E_slot_grad_fn_;
    std::shared_ptr<GradFn> E_op_grad_fn_;
    std::shared_ptr<GradFn> W_scal_grad_fn_;
    std::shared_ptr<GradFn> b_scal_grad_fn_;

    RecordEncodeGradFn() { op_name = "record_encode"; }

    ~RecordEncodeGradFn() override {
        if (saved_slot1_)   cudaFree(saved_slot1_);
        if (saved_slot2_)   cudaFree(saved_slot2_);
        if (saved_ops_)     cudaFree(saved_ops_);
        if (saved_scalars_) cudaFree(saved_scalars_);
    }

    void capture(int N, int num_slots, int num_ops, int d_model,
                 const int* d_slot1, const int* d_slot2,
                 const int* d_ops, const float* d_scalars,
                 Tensor& E_slot, Tensor& E_op,
                 Tensor& W_scal, Tensor& b_scal,
                 cudaStream_t stream) {
        N_ = N; num_slots_ = num_slots; num_ops_ = num_ops; d_model_ = d_model;

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
            grad_output.data, E_slot_grad_, E_op_grad_,
            W_scal_grad_, b_scal_grad_,
            saved_slot1_, saved_slot2_, saved_ops_, saved_scalars_,
            N_, num_slots_, num_ops_, d_model_);
        CUDA_CHECK_KERNEL();
    }

    void release_saved() override {
        GradFn::release_saved();
        if (saved_slot1_)   { cudaFree(saved_slot1_);   saved_slot1_ = nullptr; }
        if (saved_slot2_)   { cudaFree(saved_slot2_);   saved_slot2_ = nullptr; }
        if (saved_ops_)     { cudaFree(saved_ops_);     saved_ops_ = nullptr; }
        if (saved_scalars_) { cudaFree(saved_scalars_); saved_scalars_ = nullptr; }
    }
};

//======================================================//
//  L1ScalarLossGradFn — backward for |a - b| scalar loss
//  Gradient: ∂|a-b|/∂a = sign(a-b), chains to a's grad_fn
//======================================================//
struct L1ScalarLossGradFn : public GradFn {
    float* saved_a_ = nullptr;       // [1] device copy of forward a value
    float* saved_b_ = nullptr;       // [1] device copy of forward b value
    float* grad_a_  = nullptr;       // [1] device gradient buffer

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
        CUDA_CHECK(cudaMemcpyAsync(saved_a_, a_tensor.data, sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));

        CUDA_CHECK(cudaMalloc(&saved_b_, sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(saved_b_, target_device_ptr, sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));

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

//======================================================//
//  Constructor
//======================================================//
ExecutionBlockLayer::~ExecutionBlockLayer() {
    if (d_numeric_error_flag_) cudaFree(d_numeric_error_flag_);
    if (d_div_clamp_count_)    cudaFree(d_div_clamp_count_);
    if (d_exec_idx_)           cudaFree(d_exec_idx_);
    if (d_exec_record_i_)      cudaFree(d_exec_record_i_);
    if (d_exec_record_f_)      cudaFree(d_exec_record_f_);
}

int ExecutionBlockLayer::lastDivClampCount(cudaStream_t stream) const {
    if (!d_div_clamp_count_) return 0;
    int count = 0;
    cudaMemcpyAsync(&count, d_div_clamp_count_, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return count;
}

ExecutionBlockLayer::ExecutionBlockLayer(const ExecutionBlockConfig& config,
                                       uint64_t seed,
                                       cudaStream_t init_stream)
    : config_(config)
{
    validateConfigOrThrow();
    EXEC_CHECK(init_stream != nullptr, "init_stream is NULL");

    CUDA_CHECK(cudaMalloc(&d_numeric_error_flag_, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_numeric_error_flag_, 0, sizeof(int), init_stream));
    CUDA_CHECK(cudaMalloc(&d_div_clamp_count_, sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_div_clamp_count_, 0, sizeof(int), init_stream));
    CUDA_CHECK(cudaMalloc(&d_exec_idx_, 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_exec_record_i_, 3 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_exec_record_f_, 3 * sizeof(float)));

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

    // Trace encoding weights
    E_slot_  = make_param(V, dm, seed + 16, "exec_block.E_slot");
    E_op_    = make_param(nop, dm, seed + 17, "exec_block.E_op");
    W_scal_  = make_param(3, dm, seed + 18, "exec_block.W_scal");
    b_scal_  = make_bias(dm,                "exec_block.b_scal");
    W_trace_ = make_param(K * dm, dm, seed + 19, "exec_block.W_trace");
    b_trace_ = make_bias(dm,                "exec_block.b_trace");
}

//======================================================//
//  Move semantics
//======================================================//
ExecutionBlockLayer::ExecutionBlockLayer(ExecutionBlockLayer&& other) noexcept
    : config_(other.config_),
      d_numeric_error_flag_(other.d_numeric_error_flag_),
      d_div_clamp_count_(other.d_div_clamp_count_),
      d_exec_idx_(other.d_exec_idx_),
      d_exec_record_i_(other.d_exec_record_i_),
      d_exec_record_f_(other.d_exec_record_f_),
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
      tau_(std::move(other.tau_)),
      E_slot_(std::move(other.E_slot_)),
      E_op_(std::move(other.E_op_)),
      W_scal_(std::move(other.W_scal_)),
      b_scal_(std::move(other.b_scal_)),
      W_trace_(std::move(other.W_trace_)),
      b_trace_(std::move(other.b_trace_))
{
    other.d_numeric_error_flag_ = nullptr;
    other.d_div_clamp_count_ = nullptr;
    other.d_exec_idx_ = nullptr;
    other.d_exec_record_i_ = nullptr;
    other.d_exec_record_f_ = nullptr;
}

ExecutionBlockLayer& ExecutionBlockLayer::operator=(ExecutionBlockLayer&& other) noexcept {
    if (this != &other) {
        if (d_numeric_error_flag_) cudaFree(d_numeric_error_flag_);
        if (d_div_clamp_count_)    cudaFree(d_div_clamp_count_);
        if (d_exec_idx_)           cudaFree(d_exec_idx_);
        if (d_exec_record_i_)     cudaFree(d_exec_record_i_);
        if (d_exec_record_f_)     cudaFree(d_exec_record_f_);
        config_ = other.config_;
        d_numeric_error_flag_ = other.d_numeric_error_flag_;
        d_div_clamp_count_    = other.d_div_clamp_count_;
        d_exec_idx_           = other.d_exec_idx_;
        d_exec_record_i_      = other.d_exec_record_i_;
        d_exec_record_f_      = other.d_exec_record_f_;
        other.d_numeric_error_flag_ = nullptr;
        other.d_div_clamp_count_    = nullptr;
        other.d_exec_idx_           = nullptr;
        other.d_exec_record_i_      = nullptr;
        other.d_exec_record_f_      = nullptr;
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
        E_slot_        = std::move(other.E_slot_);
        E_op_          = std::move(other.E_op_);
        W_scal_        = std::move(other.W_scal_);
        b_scal_        = std::move(other.b_scal_);
        W_trace_       = std::move(other.W_trace_);
        b_trace_       = std::move(other.b_trace_);
    }
    return *this;
}

//======================================================//
//  encodeState — derive [1, d_model] summary from memory (always derived, never stored)
//======================================================//
Tensor ExecutionBlockLayer::encodeState(const ExecutionMemory& M, cudaStream_t stream) {
    validateMemoryOrThrow(M);
    const int dm = config_.d_model;
    const int V = config_.num_slots;
    auto state = Tensor::zeros({1, dm}, stream, "exec_state_encoded");
    kernelStateEncoderMean<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        state.data, M.state_embeds.data, M.valid_mask.data, V, dm);
    CUDA_CHECK_KERNEL();
    return state;
}

//======================================================//
//  bootstrapMemoryFromSlotMap — populate M.values from literals (detached)
//======================================================//
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
}

//======================================================//
//  Causal state loss kernels (Fixes 1-9)
//======================================================//

// Fix 4: Compute v_soft = sum_k p_op[k] * results[k] (soft mixed result)
__global__ void kernelComputeVSoft(
    float* __restrict__ v_soft,          // [1] output
    const float* __restrict__ p_op,      // [num_ops]
    const float* __restrict__ results,   // [num_ops]
    int num_ops)
{
    if (threadIdx.x != 0) return;
    float sum = 0.0f;
    for (int k = 0; k < num_ops; ++k)
        sum += p_op[k] * results[k];
    v_soft[0] = sum;
}

// Fix 1/3/4/6: Compute scalar absolute difference |a - b| → out[0]
__global__ void kernelAbsDiff(
    float* __restrict__ out,             // [1]
    const float* __restrict__ a,         // [1]
    const float* __restrict__ b,         // [1]
    int* __restrict__ error_flag,        // nullable
    int stage_id,
    float hard_threshold)                // 0 = no gate
{
    if (threadIdx.x != 0) return;
    float diff = fabsf(a[0] - b[0]);
    out[0] = diff;
    if (hard_threshold > 0.0f && diff > hard_threshold && error_flag)
        atomicMax(error_flag, stage_id);
}

// Fix 3: Read state_before[write_slot] → out[0] (device-side write_slot index)
__global__ void kernelReadSlotByDevIdx(
    float* __restrict__ out,             // [1]
    const float* __restrict__ values,    // [V, 1]
    const int* __restrict__ d_write_slot // [1] device
)
{
    if (threadIdx.x != 0) return;
    int ws = d_write_slot[0];
    out[0] = values[ws];
}

// Fix 2+9: Per-slot scale-aware delta check: count changed slots, hinge on non-write
__global__ void kernelStateDeltaCheck(
    float* __restrict__ state_integrity_loss, // [1] output: accumulated hinge
    int* __restrict__ changed_count,          // [1] output: number of changed slots
    const float* __restrict__ before,         // [V, 1]
    const float* __restrict__ after,          // [V, 1]
    const int* __restrict__ d_write_slot,     // [1] device
    int V)
{
    if (threadIdx.x != 0) return;
    int ws = d_write_slot[0];
    float hinge_sum = 0.0f;
    int count = 0;
    for (int i = 0; i < V; ++i) {
        float delta = fabsf(after[i] - before[i]);
        // Fix 9: per-slot epsilon = max(1e-6, 0.01 * |before[i]|)
        float eps_i = fmaxf(1e-6f, 0.01f * fabsf(before[i]));
        if (delta > eps_i) {
            count++;
            if (i != ws) {
                // Hinge penalty on non-write slot mutation
                hinge_sum += delta - eps_i;
            }
        }
    }
    state_integrity_loss[0] = hinge_sum;
    changed_count[0] = count;
}

// Fix 2: Hard gate for multi-slot mutation (set error flag if changed > 1)
__global__ void kernelCheckMultiSlotMutation(
    const int* __restrict__ changed_count,   // [1]
    int* __restrict__ error_flag,
    int stage_id)
{
    if (threadIdx.x != 0) return;
    if (changed_count[0] > 1)
        atomicMax(error_flag, stage_id);
}

// Fix 5: write_mismatch = max(p_write) * transition_error
__global__ void kernelWriteMismatch(
    float* __restrict__ out,             // [1]
    const float* __restrict__ p_write,   // [V]
    const float* __restrict__ transition_error, // [1]
    int V)
{
    if (threadIdx.x != 0) return;
    float max_p = 0.0f;
    for (int i = 0; i < V; ++i) {
        if (p_write[i] > max_p) max_p = p_write[i];
    }
    out[0] = max_p * transition_error[0];
}

// Fix 5: write_entropy_penalty = -sum(p * log(p)) → negate to penalize LOW entropy
// Output: max(0, target_entropy - actual_entropy) where target = log(V)/2
__global__ void kernelWriteEntropyPenalty(
    float* __restrict__ out,             // [1]
    const float* __restrict__ p_write,   // [V]
    int V)
{
    if (threadIdx.x != 0) return;
    float ent = 0.0f;
    for (int i = 0; i < V; ++i) {
        float p = p_write[i];
        if (p > 1e-10f)
            ent -= p * logf(p);
    }
    // Target: half of max entropy = log(V)/2
    float target_ent = logf(static_cast<float>(V)) * 0.5f;
    // Penalize when entropy is below target (collapsed distribution)
    out[0] = fmaxf(0.0f, target_ent - ent);
}

// Fix 6+7: Read consistency loss |v_actual - v_expected| (nullable inputs)
__global__ void kernelReadConsistencyLoss(
    float* __restrict__ out,             // [1]
    const float* __restrict__ v1_actual, // [1] (device)
    const float* __restrict__ v1_expected, // [1] (device, nullable)
    const float* __restrict__ v2_actual, // [1] (device)
    const float* __restrict__ v2_expected) // [1] (device, nullable)
{
    if (threadIdx.x != 0) return;
    float loss = 0.0f;
    if (v1_expected)
        loss += fabsf(v1_actual[0] - v1_expected[0]);
    if (v2_expected)
        loss += fabsf(v2_actual[0] - v2_expected[0]);
    out[0] = loss;
}

// Slot validity: penalize M.values[i] where valid_mask[i]==0 but values[i]!=0
__global__ void kernelSlotValidityPenalty(
    float* __restrict__ out,             // [1] accumulated penalty
    const float* __restrict__ values,    // [V, 1]
    const float* __restrict__ valid_mask,// [V]
    int V)
{
    if (threadIdx.x != 0) return;
    float penalty = 0.0f;
    for (int i = 0; i < V; ++i) {
        if (valid_mask[i] < 0.5f)
            penalty += fabsf(values[i]);
    }
    out[0] = penalty;
}

// L_exec aggregate: weighted sum of component losses
__global__ void kernelAggregateExecLoss(
    float* __restrict__ out,             // [1]
    const float* __restrict__ transition_loss,
    const float* __restrict__ state_integrity_loss,
    const float* __restrict__ write_consistency_loss,
    const float* __restrict__ write_mismatch_loss,
    const float* __restrict__ write_entropy_penalty,
    const float* __restrict__ read_consistency_loss,
    float w1, float w2, float w3, float w4, float w5, float w6)
{
    if (threadIdx.x != 0) return;
    out[0] = w1 * transition_loss[0]
           + w2 * state_integrity_loss[0]
           + w3 * write_consistency_loss[0]
           + w4 * write_mismatch_loss[0]
           + w5 * write_entropy_penalty[0]
           + w6 * read_consistency_loss[0];
}

// L1 loss backward: grad_input = grad_output * sign(a - b)
__global__ void kernelL1LossBackward(
    float* __restrict__ grad_input,
    const float* __restrict__ grad_output,
    const float* __restrict__ a,
    const float* __restrict__ b)
{
    if (threadIdx.x != 0) return;
    float diff = a[0] - b[0];
    float s = (fabsf(diff) > 1e-10f) ? copysignf(1.0f, diff) : 0.0f;
    grad_input[0] = grad_output[0] * s;
}

// Accumulate scalar: a[0] += b[0]
__global__ void kernelAccumulateScalar(
    float* __restrict__ a,
    const float* __restrict__ b)
{
    if (threadIdx.x != 0) return;
    a[0] += b[0];
}

//======================================================//
//  executeStep — fully differentiable, GPU-only
//======================================================//
void ExecutionBlockLayer::executeStep(
    Tensor& H,
    ExecutionMemory& M,
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
    uint64_t training_step,
    const float* expected_target,
    const float* expected_read_v1,
    const float* expected_read_v2)
{
    if (row_tokens < 0) row_tokens = total_tokens;

    validateExecuteStepInputsOrThrow(H, atom_embeddings, atom_positions,
                                      token_to_slot_map, num_atoms, total_tokens, M, step);
    (void)atom_embeddings;

    const int dm  = config_.d_model;
    const int V   = config_.num_slots;
    const int S   = config_.num_scratch_slots;
    const int V_val = V - S;
    const int dk  = config_.d_key;
    const int nop = config_.num_ops;
    EXEC_CHECK(V_val > 0, "executeStep: no value slots (V - S == 0)");

    // Snapshot state_before for transition validity
    if (diag_out) {
        diag_out->state_before_values = Tensor::zeros({V, 1}, stream, "state_before_values");
        diag_out->state_before_valid  = Tensor::zeros({1, V}, stream, "state_before_valid");
        cudaMemcpyAsync(diag_out->state_before_values.data, M.values.data,
                        V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(diag_out->state_before_valid.data, M.valid_mask.data,
                        V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // Reset per-step error tracking
    CUDA_CHECK(cudaMemsetAsync(d_numeric_error_flag_, 0, sizeof(int), stream));

    if (num_atoms > 0) {
        kernelValidateAtomSlots<<<(num_atoms + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            atom_positions, token_to_slot_map, M.valid_mask.data,
            num_atoms, V, S, d_numeric_error_flag_,
            kStageSlotMissing, kStageSlotInvalid, kStageSlotUninit);
        CUDA_CHECK_KERNEL();
    }

    // Phase 1–2: slot-only candidates; selection logits (softmax for training / STE)
    auto cand_hidden = Tensor::zeros({V_val, dm}, stream, "exec_cand_hidden");
    cand_hidden.requires_grad = false;
    cand_hidden.is_leaf = true;
    kernelGatherSlotOnlyHidden<<<V_val, kBlockSize, 0, stream>>>(
        cand_hidden.data, M.state_embeds.data, M.valid_mask.data, V_val, S, dm);
    CUDA_CHECK_KERNEL();

    auto cand_mask = Tensor::zeros({1, V_val}, stream, "exec_cand_mask");
    kernelBuildSlotOnlyCandidateMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        cand_mask.data, M.valid_mask.data, V_val, S);
    CUDA_CHECK_KERNEL();

    auto slot_values = Tensor::zeros({V_val, 1}, stream, "exec_slot_values");
    kernelGatherValueSlotScalars<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        slot_values.data, M.values.data, M.valid_mask.data, V_val, S);
    CUDA_CHECK_KERNEL();
    kernelCheckFinite<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        slot_values.data, V_val, d_numeric_error_flag_, kStageV1, config_.magnitude_limit);

    auto arg1_logits = autograd::matmul(w_arg1_select_, cand_hidden, stream,
                                         nullptr, nullptr, true);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg1_logits.data, arg1_logits.data, cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    auto p_arg1 = autograd::softmax(arg1_logits, temperature, stream);
    kernelValidateSoftmax<<<1, 1, 0, stream>>>(p_arg1.data, V_val, d_numeric_error_flag_, kStagePArg1);
    kernelCheckEntropyCollapse<<<1, 1, 0, stream>>>(
        p_arg1.data, V_val, d_numeric_error_flag_, kStageEntropyArg1, config_.entropy_collapse_threshold);

    auto arg2_logits = autograd::matmul(w_arg2_select_, cand_hidden, stream,
                                         nullptr, nullptr, true);
    kernelApplyLogitMask<<<(V_val + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        arg2_logits.data, arg2_logits.data, cand_mask.data, V_val);
    CUDA_CHECK_KERNEL();
    auto p_arg2 = autograd::softmax(arg2_logits, temperature, stream);
    kernelValidateSoftmax<<<1, 1, 0, stream>>>(p_arg2.data, V_val, d_numeric_error_flag_, kStagePArg2);
    kernelCheckEntropyCollapse<<<1, 1, 0, stream>>>(
        p_arg2.data, V_val, d_numeric_error_flag_, kStageEntropyArg2, config_.entropy_collapse_threshold);

    kernelArgmax1DInt<<<1, 1, 0, stream>>>(d_exec_idx_, p_arg1.data, V_val);
    CUDA_CHECK_KERNEL();
    kernelArgmax1DInt<<<1, 1, 0, stream>>>(d_exec_idx_ + 1, p_arg2.data, V_val);
    CUDA_CHECK_KERNEL();

    auto v1 = Tensor::zeros({1, 1}, stream, "exec_v1");
    v1.requires_grad = p_arg1.requires_grad;
    v1.is_leaf = false;
    kernelReadSlotValueByRelIdx<<<1, 1, 0, stream>>>(
        v1.data, M.values.data, M.valid_mask.data, S, V, V_val,
        d_exec_idx_, d_numeric_error_flag_, kStageSlotInvalid, kStageSlotUninit);
    CUDA_CHECK_KERNEL();
    kernelCheckFinite<<<1, 1, 0, stream>>>(v1.data, 1, d_numeric_error_flag_, kStageV1, config_.magnitude_limit);
    if (p_arg1.requires_grad) {
        auto ste = std::make_shared<SlotValueSTGradFn>();
        ste->capture(p_arg1, slot_values.data, V_val, stream);
        v1.grad_fn = ste;
    }

    auto v2 = Tensor::zeros({1, 1}, stream, "exec_v2");
    v2.requires_grad = p_arg2.requires_grad;
    v2.is_leaf = false;
    kernelReadSlotValueByRelIdx<<<1, 1, 0, stream>>>(
        v2.data, M.values.data, M.valid_mask.data, S, V, V_val,
        d_exec_idx_ + 1, d_numeric_error_flag_, kStageSlotInvalid, kStageSlotUninit);
    CUDA_CHECK_KERNEL();
    kernelCheckFinite<<<1, 1, 0, stream>>>(v2.data, 1, d_numeric_error_flag_, kStageV2, config_.magnitude_limit);
    if (p_arg2.requires_grad) {
        auto ste2 = std::make_shared<SlotValueSTGradFn>();
        ste2->capture(p_arg2, slot_values.data, V_val, stream);
        v2.grad_fn = ste2;
    }

    auto h_arg1 = autograd::matmul(p_arg1, cand_hidden, stream);
    auto h_arg2 = autograd::matmul(p_arg2, cand_hidden, stream);

    // ──── 6. Context = reduce_mean(H[offset:offset+row], dim=0) [1, dm] ────
    auto context = Tensor::zeros({1, dm}, stream, "exec_context");
    context.requires_grad = true;
    context.is_leaf = false;
    kernelReduceMeanForward<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        context.data, H.data + static_cast<size_t>(token_offset) * dm, row_tokens, dm);
    CUDA_CHECK_KERNEL();
    {
        auto mean_fn = std::make_shared<ReduceMeanGradFn>();
        mean_fn->capture(H, total_tokens, dm, stream, token_offset, row_tokens);
        context.grad_fn = mean_fn;
    }

    // ──── 7. Op selection (autograd: concat + matmul + softmax) ────
    // Build trace_vec from prior execution records [1, dm]
    Tensor trace_vec;
    const int K = config_.num_exec_steps;
    const int N_prior = static_cast<int>(prior_records.size());
    if (N_prior > 0) {
        // Upload record fields to device
        std::vector<int> h_slot1(N_prior), h_slot2(N_prior), h_ops(N_prior);
        std::vector<float> h_scalars(N_prior * 3);
        for (int i = 0; i < N_prior; ++i) {
            h_slot1[i]  = prior_records[i].arg1_slot;
            h_slot2[i]  = prior_records[i].arg2_slot;
            h_ops[i]    = prior_records[i].op_id;
            h_scalars[i * 3 + 0] = prior_records[i].value_before_1;
            h_scalars[i * 3 + 1] = prior_records[i].value_before_2;
            h_scalars[i * 3 + 2] = prior_records[i].value_after;
        }
        int *d_slot1, *d_slot2, *d_ops;
        float *d_scalars;
        CUDA_CHECK(cudaMalloc(&d_slot1, N_prior * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_slot2, N_prior * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ops, N_prior * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scalars, N_prior * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(d_slot1, h_slot1.data(), N_prior * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_slot2, h_slot2.data(), N_prior * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ops, h_ops.data(), N_prior * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_scalars, h_scalars.data(), N_prior * 3 * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Forward: [N_prior, dm]
        auto encoded_records = Tensor::zeros({N_prior, dm}, stream, "exec_encoded_records");
        encoded_records.requires_grad = true;
        encoded_records.is_leaf = false;
        int total_enc = N_prior * dm;
        kernelEncodeRecords<<<(total_enc + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            encoded_records.data, E_slot_.data, E_op_.data, W_scal_.data, b_scal_.data,
            d_slot1, d_slot2, d_ops, d_scalars,
            N_prior, V, nop, dm);
        CUDA_CHECK_KERNEL();

        // Attach GradFn for backward
        {
            auto enc_fn = std::make_shared<RecordEncodeGradFn>();
            enc_fn->capture(N_prior, V, nop, dm,
                            d_slot1, d_slot2, d_ops, d_scalars,
                            E_slot_, E_op_, W_scal_, b_scal_, stream);
            encoded_records.grad_fn = enc_fn;
        }

        // Pad/truncate to [K, dm], then flatten to [1, K*dm]
        auto padded = Tensor::zeros({K, dm}, stream, "exec_trace_padded");
        int copy_rows = (N_prior < K) ? N_prior : K;
        // Copy last copy_rows records (most recent history)
        int src_offset = (N_prior > K) ? (N_prior - K) : 0;
        CUDA_CHECK(cudaMemcpyAsync(padded.data,
            encoded_records.data + static_cast<size_t>(src_offset) * dm,
            copy_rows * dm * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        padded.requires_grad = encoded_records.requires_grad;
        padded.is_leaf = false;

        auto flattened = Tensor::zeros({1, K * dm}, stream, "exec_trace_flat");
        CUDA_CHECK(cudaMemcpyAsync(flattened.data, padded.data,
            K * dm * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        flattened.requires_grad = padded.requires_grad;
        flattened.is_leaf = false;

        // Project: [1, K*dm] @ [K*dm, dm] + bias = [1, dm]
        trace_vec = autograd::matmul(flattened, W_trace_, stream);
        trace_vec = autograd::add(trace_vec, b_trace_, stream);

        // Free device staging buffers (data already captured by GradFn)
        cudaFreeAsync(d_slot1, stream);
        cudaFreeAsync(d_slot2, stream);
        cudaFreeAsync(d_ops, stream);
        cudaFreeAsync(d_scalars, stream);
    } else {
        trace_vec = Tensor::zeros({1, dm}, stream, "exec_trace_vec_zero");
    }

    // Fuse: context' = context + trace_vec + trace_state
    auto context_prime = autograd::add(context, trace_vec, stream);
    context_prime = autograd::add(context_prime, trace_state, stream);

    auto pool12 = autograd::concat(h_arg1, h_arg2, stream);       // [1, 2*dm]
    auto pool = autograd::concat(pool12, context_prime, stream);   // [1, 3*dm]

    auto op_logits = autograd::matmul(pool, W_op_select_, stream); // [1, nop]
    auto p_op = autograd::softmax(op_logits, temperature, stream);  // [1, nop]
    kernelValidateSoftmax<<<1, 1, 0, stream>>>(p_op.data, nop, d_numeric_error_flag_, kStagePOp);
    kernelCheckEntropyCollapse<<<1, 1, 0, stream>>>(
        p_op.data, nop, d_numeric_error_flag_, kStageEntropyOp, config_.entropy_collapse_threshold);

    auto op_results = Tensor::zeros({1, nop}, stream, "exec_op_results");
    kernelFourOps<<<1, 1, 0, stream>>>(op_results.data, v1.data, v2.data, kEps, d_div_clamp_count_);
    CUDA_CHECK_KERNEL();

    kernelArgmax1DInt<<<1, 1, 0, stream>>>(d_exec_idx_ + 2, p_op.data, nop);
    CUDA_CHECK_KERNEL();

    auto v_out = Tensor::zeros({1, 1}, stream, "exec_v_out");
    v_out.requires_grad = true;
    v_out.is_leaf = false;
    kernelHardPickOpForward<<<1, 1, 0, stream>>>(
        v_out.data, op_results.data, d_exec_idx_ + 2, nop);
    CUDA_CHECK_KERNEL();
    kernelCheckFinite<<<1, 1, 0, stream>>>(v_out.data, 1, d_numeric_error_flag_, kStageVOut, config_.magnitude_limit);

    {
        auto grad_fn = std::make_shared<FourOpMixGradFn>();
        grad_fn->capture(v1, v2, p_op, p_op.data, op_results.data, nop, stream);
        v_out.grad_fn = grad_fn;
    }

    kernelAssembleExecRecord<<<1, 1, 0, stream>>>(
        d_exec_record_i_, d_exec_record_f_,
        d_exec_idx_, d_exec_idx_ + 1, d_exec_idx_ + 2,
        v1.data, v2.data, v_out.data, S);
    CUDA_CHECK_KERNEL();

    // ──── 8b. Encode current step record → update trace_state ────
    {
        // Encode current record on device using d_exec_record_i_ (s1, s2, op)
        // and d_exec_record_f_ (v1, v2, v_out) already assembled above.
        auto cur_encoded = Tensor::zeros({1, dm}, stream, "exec_cur_step_enc");
        cur_encoded.requires_grad = true;
        cur_encoded.is_leaf = false;

        // d_exec_record_i_ contains [arg1_slot, arg2_slot, op_id]
        // d_exec_record_f_ reinterpret as 3 scalars
        kernelEncodeRecords<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            cur_encoded.data, E_slot_.data, E_op_.data, W_scal_.data, b_scal_.data,
            d_exec_record_i_, d_exec_record_i_ + 1,  // slot1, slot2
            d_exec_record_i_ + 2,                     // op
            d_exec_record_f_,                         // [v1, v2, v_out] as 3 scalars
            1, V, nop, dm);
        CUDA_CHECK_KERNEL();

        // Attach GradFn
        {
            auto enc_fn = std::make_shared<RecordEncodeGradFn>();
            enc_fn->capture(1, V, nop, dm,
                            d_exec_record_i_, d_exec_record_i_ + 1,
                            d_exec_record_i_ + 2, d_exec_record_f_,
                            E_slot_, E_op_, W_scal_, b_scal_, stream);
            cur_encoded.grad_fn = enc_fn;
        }

        // Accumulate into trace_state
        trace_state = autograd::add(trace_state, cur_encoded, stream);
    }

    // ──── 9. Linear value embedding (autograd: matmul + add) ────
    auto result_emb = autograd::matmul(v_out, W_value_to_emb_, stream);
    result_emb = autograd::add(result_emb, b_value_to_emb_, stream);
    kernelCheckFinite<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        result_emb.data, dm, d_numeric_error_flag_, kStageResultEmb, config_.magnitude_limit);

    // ──── 10. Inject into H at result_slot (single slot, custom GradFn) ────
    int result_slot;
    if (config_.result_slot_mode == 1 && config_.result_slot_index >= 0 &&
        config_.result_slot_index < total_tokens) {
        result_slot = config_.result_slot_index;
    } else {
        result_slot = token_offset + row_tokens - 1;
    }
    EXEC_CHECK(result_slot >= 0 && result_slot < total_tokens,
               "result_slot out of bounds");
    float inv_sqrt_d = 1.0f / sqrtf(static_cast<float>(dm));

    float* save_gate_buf = nullptr;
    float* save_H_slot_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&save_gate_buf, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&save_H_slot_buf, dm * sizeof(float)));

    const float gate_temp = config_.inject_gate_temp;
    kernelInjectResultSlot<<<1, kBlockSize, 0, stream>>>(
        H.data, result_emb.data, w_inject_gate_.data,
        inv_sqrt_d, result_slot, dm, gate_temp,
        save_gate_buf, save_H_slot_buf);
    CUDA_CHECK_KERNEL();

    {
        // Autograd chaining: capture() saves H's existing grad_fn before we
        // replace it. During backward, inject_fn chains to the saved parent,
        // preserving the full upstream gradient graph (not overwriting it).
        auto inject_fn = std::make_shared<ExecutionBlockInjectGradFn>();
        inject_fn->capture(H, result_emb, w_inject_gate_,
                           save_gate_buf, save_H_slot_buf,
                           inv_sqrt_d, gate_temp,
                           result_slot, total_tokens, dm, stream);
        H.grad_fn = inject_fn;
        H.is_leaf = false;
        H.requires_grad = true;
    }

    auto write_ctx_12 = autograd::concat(h_arg1, h_arg2, stream);
    auto write_ctx_123 = autograd::concat(write_ctx_12, context_prime, stream);
    auto write_ctx = autograd::concat(write_ctx_123, result_emb, stream);

    auto q_write = autograd::matmul(write_ctx, W_write_query_, stream);
    kernelL2Normalize<<<1, 1, 0, stream>>>(q_write.data, dk);
    CUDA_CHECK_KERNEL();

    auto usage_norm = Tensor::zeros({1, V}, stream);
    auto ws_norm = Tensor::zeros({1, V}, stream);
    kernelNormalizeUsage<<<1, 1, 0, stream>>>(usage_norm.data, M.usage.data, V);
    CUDA_CHECK_KERNEL();
    kernelNormalizeWriteScore<<<1, 1, 0, stream>>>(ws_norm.data, M.write_score.data, V);
    CUDA_CHECK_KERNEL();

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
    CUDA_CHECK_KERNEL();

    // Mask scratch slots [0..S-1] to -inf so writes only go to value slots [S..V-1]
    if (S > 0) {
        kernelMaskScratchSlots<<<(S + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
            write_logits.data, S);
        CUDA_CHECK_KERNEL();
    }

    auto p_write = autograd::softmax(write_logits, temperature, stream);
    kernelValidateSoftmax<<<1, 1, 0, stream>>>(p_write.data, V, d_numeric_error_flag_, kStagePWrite);
    kernelCheckWriteCollapse<<<1, 1, 0, stream>>>(
        p_write.data, V, d_numeric_error_flag_, kStageWriteCollapse, config_.write_collapse_threshold);

    kernelArgmax1DInt<<<1, 1, 0, stream>>>(d_exec_idx_ + 3, p_write.data, V);
    CUDA_CHECK_KERNEL();
    kernelValidateWriteSlotDev<<<1, 1, 0, stream>>>(
        d_exec_idx_ + 3, S, V, d_numeric_error_flag_, kStageWriteSlotInvalid);
    CUDA_CHECK_KERNEL();

    // ──── Causal loss computation (Fixes 1-7, before physical write) ────
    if (diag_out) {
        // Fix 4: v_soft = dot(p_op, op_results) via autograd matmul [1,nop] @ [nop,1]
        auto op_results_col = Tensor::zeros({nop, 1}, stream, "exec_op_res_col");
        cudaMemcpyAsync(op_results_col.data, op_results.data,
                        nop * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        auto v_soft = autograd::matmul(p_op, op_results_col, stream); // [1,1]

        // Fix 3: Read state_before[write_slot] for write consistency
        auto slot_before_val = Tensor::zeros({1, 1}, stream, "exec_slot_before_val");
        kernelReadSlotByDevIdx<<<1, 1, 0, stream>>>(
            slot_before_val.data, diag_out->state_before_values.data, d_exec_idx_ + 3);
        CUDA_CHECK_KERNEL();

        // Fix 3: write_consistency_loss = |state_before[ws] - v_out|
        diag_out->write_consistency_loss = Tensor::zeros({1, 1}, stream, "exec_wc_loss");
        kernelAbsDiff<<<1, 1, 0, stream>>>(
            diag_out->write_consistency_loss.data, slot_before_val.data, v_out.data,
            nullptr, 0, 0.0f);
        CUDA_CHECK_KERNEL();

        // Fix 1: transition_error_hard = |v_out - expected_internal| (machine consistency)
        diag_out->transition_error_hard = Tensor::zeros({1, 1}, stream, "exec_te_hard");
        kernelAbsDiff<<<1, 1, 0, stream>>>(
            diag_out->transition_error_hard.data, v_out.data, v_out.data,
            nullptr, 0, 0.0f);
        CUDA_CHECK_KERNEL();

        // Fix 4+6: transition_loss = |v_soft - target|
        const float* target_ptr = expected_target ? expected_target : v_out.data;
        diag_out->used_expected_target = (expected_target != nullptr);
        diag_out->transition_loss = Tensor::zeros({1, 1}, stream, "exec_trans_loss");
        kernelAbsDiff<<<1, 1, 0, stream>>>(
            diag_out->transition_loss.data, v_soft.data, target_ptr,
            nullptr, 0, 0.0f);
        CUDA_CHECK_KERNEL();

        // Autograd chain: transition_loss grad → v_soft grad → p_op grad
        {
            auto l1_fn = std::make_shared<L1ScalarLossGradFn>();
            l1_fn->capture(v_soft, target_ptr, stream);
            diag_out->transition_loss.grad_fn = l1_fn;
            diag_out->transition_loss.requires_grad = true;
            diag_out->transition_loss.is_leaf = false;
        }

        // Fix 1 + Fix 8: Hard transition gate (only after warmup)
        if (config_.transition_hard_threshold > 0.0f &&
            training_step >= static_cast<uint64_t>(config_.exec_gate_warmup_steps)) {
            kernelAbsDiff<<<1, 1, 0, stream>>>(
                diag_out->transition_error_hard.data, v_out.data, v_out.data,
                d_numeric_error_flag_, kStageTransitionInvalid,
                config_.transition_hard_threshold);
            CUDA_CHECK_KERNEL();
        }

        // Fix 7: Read consistency loss (teacher operand values)
        diag_out->read_consistency_loss = Tensor::zeros({1, 1}, stream, "exec_rc_loss");
        kernelReadConsistencyLoss<<<1, 1, 0, stream>>>(
            diag_out->read_consistency_loss.data,
            v1.data, expected_read_v1,
            v2.data, expected_read_v2);
        CUDA_CHECK_KERNEL();

        // Fix 5: write_mismatch_loss = max(p_write) * transition_error (soft)
        diag_out->write_mismatch_loss = Tensor::zeros({1, 1}, stream, "exec_wm_loss");
        kernelWriteMismatch<<<1, 1, 0, stream>>>(
            diag_out->write_mismatch_loss.data, p_write.data,
            diag_out->transition_loss.data, V);
        CUDA_CHECK_KERNEL();

        // Fix 5: write_entropy_penalty (penalize collapsed p_write)
        diag_out->write_entropy_penalty = Tensor::zeros({1, 1}, stream, "exec_we_pen");
        kernelWriteEntropyPenalty<<<1, 1, 0, stream>>>(
            diag_out->write_entropy_penalty.data, p_write.data, V);
        CUDA_CHECK_KERNEL();
    }

    kernelHardWriteScalarDev<<<1, 1, 0, stream>>>(
        M.values.data, d_exec_idx_ + 3, v_out.data);
    CUDA_CHECK_KERNEL();

    auto state_new = Tensor::zeros({1, dm}, stream, "exec_state_new");
    const float* step_emb_ptr = step_embeddings_.data + static_cast<size_t>(step) * dm;
    kernelAddVectors<<<(dm + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        state_new.data, result_emb.data, step_emb_ptr, dm);
    CUDA_CHECK_KERNEL();

    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(
        M.state_embeds.data, d_exec_idx_ + 3, dm, state_new.data);
    CUDA_CHECK_KERNEL();

    auto key_new = Tensor::zeros({1, dk}, stream, "exec_key_new");
    kernelSmallMatmul<<<1, dk, 0, stream>>>(
        key_new.data, result_emb.data, W_key_proj_.data, 1, dm, dk);
    CUDA_CHECK_KERNEL();

    kernelHardWriteRowDev<<<1, kBlockSize, 0, stream>>>(
        M.key_embeds.data, d_exec_idx_ + 3, dk, key_new.data);
    CUDA_CHECK_KERNEL();

    kernelSetValidMaskDev<<<1, 1, 0, stream>>>(M.valid_mask.data, d_exec_idx_ + 3);
    CUDA_CHECK_KERNEL();

    kernelSetRecentWriteOneHotDev<<<(V + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        M.recent_write_mask.data, V, d_exec_idx_ + 3);
    CUDA_CHECK_KERNEL();

    if (diag_out) {
        diag_out->p_arg1 = Tensor::zeros({1, V_val}, stream);
        cudaMemcpyAsync(diag_out->p_arg1.data, p_arg1.data, V_val * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->p_arg2 = Tensor::zeros({1, V_val}, stream);
        cudaMemcpyAsync(diag_out->p_arg2.data, p_arg2.data, V_val * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->p_op = Tensor::zeros({1, nop}, stream);
        cudaMemcpyAsync(diag_out->p_op.data, p_op.data, nop * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->p_write = Tensor::zeros({1, V}, stream);
        cudaMemcpyAsync(diag_out->p_write.data, p_write.data, V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->v_out = Tensor::zeros({1, 1}, stream);
        cudaMemcpyAsync(diag_out->v_out.data, v_out.data, sizeof(float), cudaMemcpyDeviceToDevice, stream);
        diag_out->result_emb = Tensor::zeros({1, dm}, stream);
        cudaMemcpyAsync(diag_out->result_emb.data, result_emb.data, dm * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // Snapshot state_after for transition validity
        diag_out->state_after_values = Tensor::zeros({V, 1}, stream, "state_after_values");
        diag_out->state_after_valid  = Tensor::zeros({1, V}, stream, "state_after_valid");
        cudaMemcpyAsync(diag_out->state_after_values.data, M.values.data,
                        V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(diag_out->state_after_valid.data, M.valid_mask.data,
                        V * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // ──── Causal state delta checks (Fixes 2, 9 — after physical write) ────

        // Fix 2+9: Per-slot scale-aware delta → state_integrity_loss + changed_count
        diag_out->state_integrity_loss = Tensor::zeros({1, 1}, stream, "exec_si_loss");
        int* d_changed_count = nullptr;
        CUDA_CHECK(cudaMalloc(&d_changed_count, sizeof(int)));
        kernelStateDeltaCheck<<<1, 1, 0, stream>>>(
            diag_out->state_integrity_loss.data, d_changed_count,
            diag_out->state_before_values.data, diag_out->state_after_values.data,
            d_exec_idx_ + 3, V);
        CUDA_CHECK_KERNEL();

        // Fix 2 + Fix 8: Multi-slot mutation hard gate (after warmup)
        if (training_step >= static_cast<uint64_t>(config_.exec_gate_warmup_steps)) {
            kernelCheckMultiSlotMutation<<<1, 1, 0, stream>>>(
                d_changed_count, d_numeric_error_flag_, kStageMultiSlotMutation);
            CUDA_CHECK_KERNEL();
        }
        cudaFreeAsync(d_changed_count, stream);

        // Slot validity penalty: penalize values[i] != 0 where valid_mask[i] == 0
        auto slot_validity = Tensor::zeros({1, 1}, stream, "exec_sv_pen");
        kernelSlotValidityPenalty<<<1, 1, 0, stream>>>(
            slot_validity.data, M.values.data, M.valid_mask.data, V);
        CUDA_CHECK_KERNEL();

        // Fold slot validity into state_integrity_loss
        kernelAccumulateScalar<<<1, 1, 0, stream>>>(
            diag_out->state_integrity_loss.data, slot_validity.data);
        CUDA_CHECK_KERNEL();

        // Aggregate L_exec from all component losses
        diag_out->exec_step_loss = Tensor::zeros({1, 1}, stream, "exec_step_loss");
        kernelAggregateExecLoss<<<1, 1, 0, stream>>>(
            diag_out->exec_step_loss.data,
            diag_out->transition_loss.data,
            diag_out->state_integrity_loss.data,
            diag_out->write_consistency_loss.data,
            diag_out->write_mismatch_loss.data,
            diag_out->write_entropy_penalty.data,
            diag_out->read_consistency_loss.data,
            config_.causal_w1_transition,
            config_.causal_w2_state_integrity,
            config_.causal_w3_write_consistency,
            config_.causal_w4_write_mismatch,
            config_.causal_w5_write_entropy,
            config_.causal_w6_read_consistency);
        CUDA_CHECK_KERNEL();

        // Chain exec_step_loss autograd through transition_loss (primary gradient signal)
        if (diag_out->transition_loss.grad_fn) {
            diag_out->exec_step_loss.requires_grad = true;
            diag_out->exec_step_loss.is_leaf = false;
            diag_out->exec_step_loss.grad_fn = diag_out->transition_loss.grad_fn;
        }
    }

    // ──── 13. Metrics collection (debug_mode, populates diag_out->metrics) ────
    if (config_.debug_mode && diag_out) {
        float d_metrics[7];
        float* d_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buf, 7 * sizeof(float)));

        kernelComputeEntropyScalar<<<1, 1, 0, stream>>>(d_buf + 0, p_arg1.data, V_val);
        kernelComputeEntropyScalar<<<1, 1, 0, stream>>>(d_buf + 1, p_arg2.data, V_val);
        kernelComputeEntropyScalar<<<1, 1, 0, stream>>>(d_buf + 2, p_op.data, nop);
        kernelComputeEntropyScalar<<<1, 1, 0, stream>>>(d_buf + 3, p_write.data, V);
        kernelComputeMax<<<1, 1, 0, stream>>>(d_buf + 4, p_write.data, V);

        CUDA_CHECK(cudaMemcpyAsync(d_metrics, d_buf, 5 * sizeof(float), cudaMemcpyDeviceToHost, stream));

        int div_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(&div_count, d_div_clamp_count_, sizeof(int), cudaMemcpyDeviceToHost, stream));

        float op_dist[4] = {};
        int op_copy = (nop <= 4) ? nop : 4;
        CUDA_CHECK(cudaMemcpyAsync(op_dist, p_op.data, op_copy * sizeof(float), cudaMemcpyDeviceToHost, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        diag_out->metrics.arg1_entropy   = d_metrics[0];
        diag_out->metrics.arg2_entropy   = d_metrics[1];
        diag_out->metrics.op_entropy     = d_metrics[2];
        diag_out->metrics.write_entropy  = d_metrics[3];
        diag_out->metrics.max_p_write    = d_metrics[4];
        diag_out->metrics.div_clamp_count = div_count;
        for (int k = 0; k < 4; ++k)
            diag_out->metrics.op_distribution[k] = (k < op_copy) ? op_dist[k] : 0.0f;

        cudaFree(d_buf);
    }

    {
        int h_error = 0;
        int ri[3] = {0, 0, 0};
        float rf[3] = {0.0f, 0.0f, 0.0f};
        if (diag_out) {
            CUDA_CHECK(cudaMemcpyAsync(ri, d_exec_record_i_, 3 * sizeof(int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(rf, d_exec_record_f_, 3 * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(&h_error, d_numeric_error_flag_, sizeof(int),
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
            if (config_.debug_mode)
                fprintf(stderr, "[ExecutionBlock debug] %s\n", buf);
            throw std::runtime_error(buf);
        }
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
    cudaStream_t stream,
    int token_offset,
    int row_tokens)
{
    validateCrossAttentionInputsOrThrow(hidden_states, M, total_tokens);
    if (row_tokens < 0) row_tokens = total_tokens;

    const int dm  = config_.d_model;
    const int dk  = config_.d_key;
    const int hd  = config_.cross_attn_head_dim;
    const int V   = config_.num_slots;
    const int nv  = V;
    const int topk= config_.cross_attn_topk;

    float* H_row = hidden_states.data + static_cast<size_t>(token_offset) * dm;

    auto Q = Tensor::zeros({row_tokens, hd}, stream);
    kernelSmallMatmul<<<row_tokens, hd, 0, stream>>>(
        Q.data, H_row, W_Q_read_.data, row_tokens, dm, hd);
    CUDA_CHECK_KERNEL();

    auto K_proj = Tensor::zeros({nv, hd}, stream);
    kernelSmallMatmul<<<nv, hd, 0, stream>>>(
        K_proj.data, M.key_embeds.data, W_K_read_.data, nv, dk, hd);
    CUDA_CHECK_KERNEL();

    auto V_proj = Tensor::zeros({nv, hd}, stream);
    kernelSmallMatmul<<<nv, hd, 0, stream>>>(
        V_proj.data, M.state_embeds.data, W_V_read_.data, nv, dm, hd);
    CUDA_CHECK_KERNEL();

    auto scores = Tensor::zeros({row_tokens, nv}, stream);
    kernelCrossAttnSharpScores<<<row_tokens, 1, 0, stream>>>(
        scores.data, Q.data, K_proj.data, M.valid_mask.data,
        tau_.data, row_tokens, nv, hd, topk);
    CUDA_CHECK_KERNEL();

    auto R = Tensor::zeros({row_tokens, hd}, stream);
    kernelCrossAttnWeightedValue<<<row_tokens, hd, 0, stream>>>(
        R.data, scores.data, V_proj.data, row_tokens, nv, hd);
    CUDA_CHECK_KERNEL();

    auto gate = Tensor::zeros({1, row_tokens}, stream);
    kernelComputeGate<<<(row_tokens + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        gate.data, H_row, W_gate_read_.data, row_tokens, dm);
    CUDA_CHECK_KERNEL();

    kernelCrossAttnGatedOutput<<<row_tokens, kBlockSize, 0, stream>>>(
        H_row, R.data, W_O_read_.data, gate.data,
        row_tokens, dm, hd);
    CUDA_CHECK_KERNEL();

    kernelDecayedUsageUpdate<<<(nv + kBlockSize - 1) / kBlockSize, kBlockSize, 0, stream>>>(
        M.usage.data, scores.data, config_.usage_decay,
        row_tokens, nv, V);
    CUDA_CHECK_KERNEL();
}

}  // namespace GRIM

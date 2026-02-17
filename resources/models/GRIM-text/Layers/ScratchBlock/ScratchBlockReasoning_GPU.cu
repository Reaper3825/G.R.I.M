//======================================================//
//  ScratchBlockReasoning_GPU.cu
//  Implementation of Internal Reasoning Layer
//
//  Rewritten Feb 2026: Proper autograd node.
//  - ScratchBlockGradFn owns all caches
//  - autograd::scratch_block_inject() is the entry point
//  - Dead code deleted (kernelPassthrough, legacy lookup,
//    detectAtomTokensAsync, injectAtomEmbeddings, onConfigure)
//  - Broken LCG PRNG replaced with splitmix64
//======================================================//

#include "ScratchBlockReasoning_GPU.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace GRIM {

//======================================================//
//  Constants
//======================================================//

static constexpr const char* kScratchBlockModule = "ScratchBlock";
constexpr int ATOM_TOKEN_START = HyperParameters::ATOM_TOKEN_START;
constexpr int NUM_ATOM_TYPES   = HyperParameters::NUM_ATOM_TYPES;
constexpr int kTextFeatureDim  = 16;

__device__ __forceinline__ int ClampNumAtoms(const int* num_atoms, int max_atoms) {
    int n = num_atoms ? *num_atoms : 0;
    if (n < 0) n = 0;
    return (n > max_atoms) ? max_atoms : n;
}

//======================================================//
//  CUDA Kernels — Forward
//======================================================//

__global__ void kernelDetectAtomTokens(
    const int* __restrict__ token_ids,
    int total_tokens,
    int* __restrict__ atom_positions,
    int* __restrict__ num_atoms,
    int max_atoms,
    int atom_token_start,
    int atom_token_end
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_tokens) {
        int token = token_ids[idx];
        if (token >= atom_token_start && token < atom_token_end) {
            int global_pos = atomicAdd(num_atoms, 1);
            if (global_pos < max_atoms) {
                atom_positions[global_pos] = idx;
            }
        }
    }
}

// Value-aware atom embedding lookup (sinusoidal+log basis in dims 16-47, type embedding in 0-15 and 48+)
__global__ void kernelLookupAtomEmbeddingsWithValue(
    const int* __restrict__ token_ids,
    const int* __restrict__ atom_positions,
    const int* __restrict__ num_atoms,
    int max_atoms,
    const float* __restrict__ atom_type_embeddings,
    const float* __restrict__ token_numeric_values,
    const uint8_t* __restrict__ token_numeric_mask,
    float* __restrict__ atom_embeddings,
    int atom_embedding_dim
) {
    const int atom_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;
    __shared__ int s_num_atoms;
    if (threadIdx.x == 0) {
        s_num_atoms = ClampNumAtoms(num_atoms, max_atoms);
    }
    __syncthreads();

    if (atom_idx >= s_num_atoms || dim_idx >= atom_embedding_dim) return;

    int token_pos = atom_positions[atom_idx];
    int token_id = token_ids[token_pos];
    int atom_type = (token_id - ATOM_TOKEN_START) % NUM_ATOM_TYPES;

    bool has_numeric = token_numeric_mask && token_numeric_mask[token_pos] != 0;
    float numeric_val = (token_numeric_values && has_numeric)
        ? token_numeric_values[token_pos] : 0.0f;
    if (has_numeric && !isfinite(numeric_val)) {
        has_numeric = false;
        numeric_val = 0.0f;
    }

    // Base embedding from type
    float value = atom_type_embeddings[atom_type * atom_embedding_dim + dim_idx];

    // Value-specific encoding in dimensions 16-47
    if (dim_idx >= 16 && dim_idx < 32 && has_numeric) {
        int bit = dim_idx - 16;
        float log_mag = log2f(fabsf(numeric_val) + 1.0f);
        float freq = (float)(bit + 1) * 0.5f;
        value += 0.5f * sinf(log_mag * freq);
    }
    else if (dim_idx >= 32 && dim_idx < 48 && has_numeric) {
        int feat = dim_idx - 32;
        if (feat == 0) {
            value += (numeric_val < 0) ? 0.5f : -0.5f;
        } else if (feat < 8) {
            int int_val = (int)fabsf(numeric_val);
            value += ((int_val >> (feat - 1)) & 1) ? 0.3f : -0.3f;
        } else {
            float frac = numeric_val - floorf(numeric_val);
            value += 0.3f * sinf(frac * 3.14159f * (feat - 7));
        }
    }
    // Dims 0-15 and 48+: pure type embedding

    atom_embeddings[atom_idx * atom_embedding_dim + dim_idx] = value;
}

// Project atom embeddings and inject into hidden states
__global__ void kernelInjectAtomEmbeddings(
    float* __restrict__ hidden_states,
    const int* __restrict__ atom_positions,
    const int* __restrict__ num_atoms,
    int max_atoms,
    const float* __restrict__ atom_embeddings,
    const float* __restrict__ projection,
    int atom_embedding_dim,
    int d_model,
    float scale
) {
    const int atom_idx = blockIdx.x;
    const int d_idx = threadIdx.x;
    __shared__ int s_num_atoms;
    if (threadIdx.x == 0) {
        s_num_atoms = ClampNumAtoms(num_atoms, max_atoms);
    }
    __syncthreads();

    if (atom_idx >= s_num_atoms || d_idx >= d_model) return;

    int token_pos = atom_positions[atom_idx];
    float sum = 0.0f;
    for (int k = 0; k < atom_embedding_dim; ++k) {
        sum += atom_embeddings[atom_idx * atom_embedding_dim + k] *
               projection[k * d_model + d_idx];
    }
    hidden_states[token_pos * d_model + d_idx] += scale * sum;
}

// Inject text features into hidden states
__global__ void kernelInjectTextFeatures(
    float* __restrict__ hidden_states,
    const uint16_t* __restrict__ text_features,
    const uint8_t* __restrict__ text_mask,
    const float* __restrict__ text_projection,
    int total_tokens,
    int d_model,
    float scale
) {
    const int token_idx = blockIdx.x;
    const int d_idx = threadIdx.x;
    if (token_idx >= total_tokens || d_idx >= d_model) return;
    if (!text_mask || text_mask[token_idx] == 0) return;

    const uint16_t* token_features = text_features + token_idx * kTextFeatureDim;
    float sum = 0.0f;
    for (int k = 0; k < kTextFeatureDim; ++k) {
        float feat = __half2float(*reinterpret_cast<const __half*>(&token_features[k]));
        sum += feat * text_projection[k * d_model + d_idx];
    }
    hidden_states[token_idx * d_model + d_idx] += scale * sum;
}

//======================================================//
//  CUDA Kernels — Backward
//======================================================//

// Backward through atom embedding injection
__global__ void kernelBackwardAtomEmbeddings(
    const float* __restrict__ grad_output,
    const int* __restrict__ atom_positions,
    int num_atoms_clamped,
    const float* __restrict__ atom_embeddings,
    const float* __restrict__ projection,
    float* __restrict__ grad_projection,
    float* __restrict__ grad_atom_embeddings,
    int atom_embedding_dim,
    int d_model,
    float scale
) {
    const int atom_idx = blockIdx.x;
    const int k_idx = threadIdx.x;

    if (atom_idx >= num_atoms_clamped || k_idx >= atom_embedding_dim) return;

    int token_pos = atom_positions[atom_idx];
    const float* grad_h = grad_output + token_pos * d_model;
    const float* atom_emb = atom_embeddings + atom_idx * atom_embedding_dim;

    // grad_atom_emb[k] = scale * sum_d(proj[k, d] * grad_h[d])
    float grad_emb = 0.0f;
    for (int d = 0; d < d_model; ++d) {
        grad_emb += projection[k_idx * d_model + d] * grad_h[d];
    }
    grad_atom_embeddings[atom_idx * atom_embedding_dim + k_idx] = scale * grad_emb;

    // grad_projection[k, d] += scale * atom_emb[k] * grad_h[d]
    for (int d = 0; d < d_model; ++d) {
        float grad_proj = scale * atom_emb[k_idx] * grad_h[d];
        atomicAdd(&grad_projection[k_idx * d_model + d], grad_proj);
    }
}

//======================================================//
//  Value Extraction Kernel
//  Computes: value = W_extract[d] * hidden_state[pos*d_model + d] for d in [0, d_model)
//  Then adds bias. Result stored in d_output (device scalar).
//  Uses parallel reduction across d_model dimensions.
//======================================================//
__global__ void kernelExtractNumericValue(
    const float* __restrict__ hidden_state,  // [seq_len, d_model]
    const float* __restrict__ w_extract,     // [d_model]
    const float* __restrict__ b_extract,     // [1]
    float* __restrict__ d_output,            // [1] output scalar
    int position,                            // token position to read from
    int d_model
) {
    // One block, blockDim.x threads for parallel reduction
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    
    // Each thread accumulates a partial dot product
    float partial = 0.0f;
    const float* h_pos = hidden_state + position * d_model;
    for (int d = tid; d < d_model; d += blockDim.x) {
        partial += w_extract[d] * h_pos[d];
    }
    sdata[tid] = partial;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result + bias
    if (tid == 0) {
        d_output[0] = sdata[0] + b_extract[0];
    }
}

//------------------------------------------------------
// Extraction Head Training Kernel
// One block per token — skips non-atom positions.
// Accumulates MSE loss and weight/bias gradients via atomicAdd.
// Gradients are PRE-DIVIDED by atom_count (passed by caller).
//------------------------------------------------------
__global__ void kernelExtractionTrainStep(
    const float* __restrict__ encoder_output,   // [total_tokens, d_model]
    const float* __restrict__ W_extract,        // [d_model]
    const float  b_extract,                     // scalar
    const float* __restrict__ numeric_values,   // [total_tokens]
    const float* __restrict__ numeric_mask,     // [total_tokens] (1.0 = atom)
    float* __restrict__ grad_W,                 // [d_model]
    float* __restrict__ grad_b,                 // [1]
    float* __restrict__ d_total_loss,           // [1]
    int total_tokens,
    int d_model,
    float inv_count                             // 1.0 / atom_count
) {
    const int t = blockIdx.x;
    if (t >= total_tokens) return;
    if (numeric_mask[t] < 0.5f) return;  // Not a numeric atom

    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const float* hidden = encoder_output + t * d_model;

    // --- Forward: predicted = dot(W, hidden[t]) + b ---
    float partial = 0.0f;
    for (int d = tid; d < d_model; d += blockDim.x) {
        partial += W_extract[d] * hidden[d];
    }
    sdata[tid] = partial;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    const float predicted = sdata[0] + b_extract;
    const float target    = numeric_values[t];
    const float diff      = predicted - target;

    // --- Backward: grad = diff * inv_count  (mean MSE) ---
    const float g = diff * inv_count;

    for (int d = tid; d < d_model; d += blockDim.x) {
        atomicAdd(&grad_W[d], g * hidden[d]);
    }
    if (tid == 0) {
        atomicAdd(grad_b, g);
        atomicAdd(d_total_loss, 0.5f * diff * diff);
    }
}

// Extract atom types from token IDs
__global__ void kernelExtractAtomTypes(
    const int* __restrict__ token_ids,
    const int* __restrict__ atom_positions,
    int num_atoms_clamped,
    int* __restrict__ atom_types
) {
    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom_idx >= num_atoms_clamped) return;

    int token_pos = atom_positions[atom_idx];
    int token_id = token_ids[token_pos];
    atom_types[atom_idx] = (token_id - ATOM_TOKEN_START) % NUM_ATOM_TYPES;
}

// Accumulate per-atom gradients back to shared atom type embeddings
__global__ void kernelAccumulateAtomTypeGradients(
    const float* __restrict__ grad_atom_embeddings,
    const int* __restrict__ atom_types,
    int num_atoms_clamped,
    float* __restrict__ grad_atom_type_embeddings,
    int atom_embedding_dim
) {
    const int atom_idx = blockIdx.x;
    const int k_idx = threadIdx.x;

    if (atom_idx >= num_atoms_clamped || k_idx >= atom_embedding_dim) return;

    int atom_type = atom_types[atom_idx];
    if (atom_type < 0 || atom_type >= NUM_ATOM_TYPES) return;

    float grad = grad_atom_embeddings[atom_idx * atom_embedding_dim + k_idx];
    atomicAdd(&grad_atom_type_embeddings[atom_type * atom_embedding_dim + k_idx], grad);
}

// Backward through text feature injection
__global__ void kernelBackwardTextFeatures(
    const float* __restrict__ grad_output,
    const uint16_t* __restrict__ text_features,
    const uint8_t* __restrict__ text_mask,
    float* __restrict__ grad_text_projection,
    int total_tokens,
    int d_model,
    float scale
) {
    const int token_idx = blockIdx.x;
    const int k_idx = threadIdx.x;
    if (token_idx >= total_tokens || k_idx >= kTextFeatureDim) return;
    if (!text_mask || text_mask[token_idx] == 0) return;

    const uint16_t* token_features = text_features + token_idx * kTextFeatureDim;
    float feat = __half2float(*reinterpret_cast<const __half*>(&token_features[k_idx]));

    const float* grad_h = grad_output + token_idx * d_model;
    for (int d = 0; d < d_model; ++d) {
        float grad_proj = scale * feat * grad_h[d];
        atomicAdd(&grad_text_projection[k_idx * d_model + d], grad_proj);
    }
}

//======================================================//
//  Xavier Init — splitmix64 PRNG (replaces broken LCG)
//======================================================//

__global__ void kernelXavierInit(
    float* __restrict__ weights,
    int size,
    float stddev,
    unsigned int seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // splitmix64: per-element seed with high-quality mixing
    // Avoids the LCG correlation bug (Issue #107 pattern)
    uint64_t state = static_cast<uint64_t>(seed) + static_cast<uint64_t>(idx) * 0x9E3779B97F4A7C15ULL;
    state ^= state >> 30; state *= 0xBF58476D1CE4E5B9ULL;
    state ^= state >> 27; state *= 0x94D049BB133111EBULL;
    state ^= state >> 31;
    float u1 = (static_cast<uint32_t>(state) & 0x7FFFFFFF) / float(0x7FFFFFFF);

    state ^= state >> 30; state *= 0xBF58476D1CE4E5B9ULL;
    state ^= state >> 27; state *= 0x94D049BB133111EBULL;
    state ^= state >> 31;
    float u2 = (static_cast<uint32_t>(state) & 0x7FFFFFFF) / float(0x7FFFFFFF);

    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);
    weights[idx] = z * stddev;
}

//======================================================//
//  ScratchBlockGradFn Implementation
//======================================================//

ScratchBlockGradFn::~ScratchBlockGradFn() {
    if (cached_atom_embeddings)   cudaFree(cached_atom_embeddings);
    if (cached_atom_positions)    cudaFree(cached_atom_positions);
    if (cached_atom_types)        cudaFree(cached_atom_types);
    if (cached_text_features)     cudaFree(cached_text_features);
    if (cached_text_mask)         cudaFree(cached_text_mask);
    if (d_grad_atom_embeddings)   cudaFree(d_grad_atom_embeddings);
    if (owns_input_grad && input_grad) cudaFree(input_grad);
}

void ScratchBlockGradFn::capture_input(Tensor& x) {
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;

    // Allocate OWN gradient buffer (Issue #126/#133 pattern)
    // Input tensor may be destroyed before backward() runs.
    if (x.requires_grad) {
        const size_t grad_size = x.numel();
        cudaMalloc(&input_grad, grad_size * sizeof(float));
        cudaMemset(input_grad, 0, grad_size * sizeof(float));
        owns_input_grad = true;
    }
}

void ScratchBlockGradFn::capture_forward(
    const int* atom_positions_src, const int* atom_types_src,
    const float* atom_embeddings_src, int num_atoms,
    const uint16_t* text_features_src, const uint8_t* text_mask_src,
    int tokens, cudaStream_t stream)
{
    total_tokens = tokens;
    num_atoms_captured = num_atoms;

    // Capture atom data (OWNED copies on device)
    if (num_atoms > 0) {
        const size_t pos_bytes = static_cast<size_t>(num_atoms) * sizeof(int);
        const size_t type_bytes = pos_bytes;
        const size_t emb_bytes = static_cast<size_t>(num_atoms) * atom_embedding_dim * sizeof(float);

        cudaMalloc(&cached_atom_positions, pos_bytes);
        cudaMemcpyAsync(cached_atom_positions, atom_positions_src, pos_bytes,
                        cudaMemcpyDeviceToDevice, stream);

        cudaMalloc(&cached_atom_types, type_bytes);
        cudaMemcpyAsync(cached_atom_types, atom_types_src, type_bytes,
                        cudaMemcpyDeviceToDevice, stream);

        cudaMalloc(&cached_atom_embeddings, emb_bytes);
        cudaMemcpyAsync(cached_atom_embeddings, atom_embeddings_src, emb_bytes,
                        cudaMemcpyDeviceToDevice, stream);

        // Backward scratch for per-atom gradients
        cudaMalloc(&d_grad_atom_embeddings, static_cast<size_t>(max_atoms) * atom_embedding_dim * sizeof(float));
    }

    // Capture text feature data (OWNED copies)
    if (text_features_src && text_mask_src) {
        const size_t feat_bytes = static_cast<size_t>(tokens) * kTextFeatureDim * sizeof(uint16_t);
        const size_t mask_bytes = static_cast<size_t>(tokens) * sizeof(uint8_t);

        cudaMalloc(&cached_text_features, feat_bytes);
        cudaMemcpyAsync(cached_text_features, text_features_src, feat_bytes,
                        cudaMemcpyDeviceToDevice, stream);

        cudaMalloc(&cached_text_mask, mask_bytes);
        cudaMemcpyAsync(cached_text_mask, text_mask_src, mask_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    }
}

void ScratchBlockGradFn::capture_weights(
    Tensor& atom_proj, Tensor& atom_type_emb, Tensor& text_feat_proj)
{
    atom_proj.ensure_grad();
    atom_type_emb.ensure_grad();
    text_feat_proj.ensure_grad();

    atom_projection_data         = atom_proj.data;
    atom_projection_grad         = atom_proj.grad_data();
    atom_type_embeddings_grad    = atom_type_emb.grad_data();
    text_feature_projection_grad = text_feat_proj.grad_data();

    if (!atom_projection_data)         throw std::runtime_error("ScratchBlockGradFn::capture_weights: atom_projection.data is NULL");
    if (!atom_projection_grad)         throw std::runtime_error("ScratchBlockGradFn::capture_weights: atom_projection.grad is NULL");
    if (!atom_type_embeddings_grad)    throw std::runtime_error("ScratchBlockGradFn::capture_weights: atom_type_embeddings.grad is NULL");
    if (!text_feature_projection_grad) throw std::runtime_error("ScratchBlockGradFn::capture_weights: text_feature_projection.grad is NULL");
}

void ScratchBlockGradFn::apply(const Tensor& grad_output, cudaStream_t stream) {
    using namespace GRIM::Logging;
    setCurrentGradFnOp("scratch_block", this);

    // Issue #49: Prevent infinite loops
    if (applied) return;
    applied = true;

    if (!grad_output.data) {
        throw std::runtime_error("ScratchBlockGradFn::apply: grad_output.data is NULL");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Backward Step 1: Atom projection + type embedding gradients
    // ═══════════════════════════════════════════════════════════════════════════
    if (num_atoms_captured > 0 && cached_atom_embeddings && cached_atom_positions) {
        // Zero per-atom gradient scratch
        const size_t scratch_bytes = static_cast<size_t>(num_atoms_captured) * atom_embedding_dim * sizeof(float);
        cudaMemsetAsync(d_grad_atom_embeddings, 0, scratch_bytes, stream);

        // Backward through projection: computes grad_projection and grad_atom_embeddings
        const int block_size = std::min(atom_embedding_dim, 1024);
        kernelBackwardAtomEmbeddings<<<num_atoms_captured, block_size, 0, stream>>>(
            grad_output.data,
            cached_atom_positions,
            num_atoms_captured,
            cached_atom_embeddings,
            atom_projection_data,
            atom_projection_grad,
            d_grad_atom_embeddings,
            atom_embedding_dim,
            d_model,
            atom_scale);

        // Route per-atom gradients back to shared type embeddings
        if (cached_atom_types && atom_type_embeddings_grad) {
            kernelAccumulateAtomTypeGradients<<<num_atoms_captured, block_size, 0, stream>>>(
                d_grad_atom_embeddings,
                cached_atom_types,
                num_atoms_captured,
                atom_type_embeddings_grad,
                atom_embedding_dim);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Backward Step 2: Text feature projection gradients
    // ═══════════════════════════════════════════════════════════════════════════
    if (cached_text_features && cached_text_mask && text_feature_projection_grad) {
        const int text_block_size = std::min(kTextFeatureDim, 32);
        kernelBackwardTextFeatures<<<total_tokens, text_block_size, 0, stream>>>(
            grad_output.data,
            cached_text_features,
            cached_text_mask,
            text_feature_projection_grad,
            total_tokens,
            d_model,
            atom_scale);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Backward Step 3: Chain to input (additive injection → identity gradient)
    // ═══════════════════════════════════════════════════════════════════════════
    if (input_grad_fn) {
        // For additive injection, grad_input = grad_output (no modification needed).
        // Pass grad_output directly to the chain — no copy required.
        Tensor view;
        view.data = grad_output.data;
        view.shape = input_shape;
        view.owns_data = false;
        view.stream = stream;
        input_grad_fn->apply(view, stream);
    }
}

void ScratchBlockGradFn::release_saved() {
    GradFn::release_saved();
    if (cached_atom_embeddings) { cudaFree(cached_atom_embeddings); cached_atom_embeddings = nullptr; }
    if (cached_atom_positions)  { cudaFree(cached_atom_positions);  cached_atom_positions = nullptr; }
    if (cached_atom_types)      { cudaFree(cached_atom_types);      cached_atom_types = nullptr; }
    if (cached_text_features)   { cudaFree(cached_text_features);   cached_text_features = nullptr; }
    if (cached_text_mask)       { cudaFree(cached_text_mask);       cached_text_mask = nullptr; }
    if (d_grad_atom_embeddings) { cudaFree(d_grad_atom_embeddings); d_grad_atom_embeddings = nullptr; }
    if (owns_input_grad && input_grad) { cudaFree(input_grad); owns_input_grad = false; }
    input_grad = nullptr;
    input_grad_fn.reset();
}

//======================================================//
//  autograd::scratch_block_inject() — Entry Point
//======================================================//

namespace autograd {

Tensor scratch_block_inject(
    Tensor& input,
    ScratchBlockLayer& layer,
    const int* token_ids,
    const float* numeric_values,
    const uint8_t* numeric_mask,
    const uint16_t* text_features,
    const uint8_t* text_mask,
    int total_tokens,
    cudaStream_t stream)
{
    const auto& cfg = layer.config();

    // Rule 20: No fallbacks
    if (!input.data) throw std::runtime_error("scratch_block_inject: input.data is NULL");
    if (!token_ids)  throw std::runtime_error("scratch_block_inject: token_ids is NULL");
    if (!stream)     throw std::runtime_error("scratch_block_inject: stream is NULL");
    if (!numeric_values || !numeric_mask) {
        throw std::runtime_error("scratch_block_inject: numeric side-channel is NULL");
    }

    // Create output tensor (copy of input — injection is additive in-place)
    const size_t data_bytes = static_cast<size_t>(total_tokens) * cfg.d_model * sizeof(float);
    Tensor output;
    cudaMalloc(&output.data, data_bytes);
    cudaMemcpyAsync(output.data, input.data, data_bytes, cudaMemcpyDeviceToDevice, stream);
    output.shape = input.shape;
    output.owns_data = true;
    output.requires_grad = input.requires_grad;
    output.is_leaf = false;
    output.stream = stream;

    // Run forward kernels on output buffer
    layer.runForwardKernels(
        output.data, total_tokens,
        token_ids, numeric_values, numeric_mask,
        text_features, text_mask, stream);

    // Build GradFn for backward pass
    if (input.requires_grad) {
        auto grad_fn = std::make_shared<ScratchBlockGradFn>();
        grad_fn->atom_embedding_dim = cfg.atom_embedding_dim;
        grad_fn->d_model = cfg.d_model;
        grad_fn->max_atoms = cfg.max_atoms;
        grad_fn->atom_scale = cfg.atom_scale;

        // Capture input chain
        grad_fn->capture_input(input);

        // Capture weight gradient pointers (layer outlives GradFn)
        grad_fn->capture_weights(
            layer.atomProjection(),
            layer.atomTypeEmbeddings(),
            layer.textFeatureProjection());

        // Get atom count from device (need sync for host read)
        int num_atoms_host = 0;
        cudaMemcpyAsync(&num_atoms_host, layer.numAtomsBuffer(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        num_atoms_host = std::min(num_atoms_host, cfg.max_atoms);

        // Extract atom types into device buffer for capture
        int* d_atom_types_temp = nullptr;
        if (num_atoms_host > 0) {
            cudaMalloc(&d_atom_types_temp, static_cast<size_t>(num_atoms_host) * sizeof(int));
            const int blk = 256;
            const int grd = (num_atoms_host + blk - 1) / blk;
            kernelExtractAtomTypes<<<grd, blk, 0, stream>>>(
                token_ids,
                layer.atomPositionsBuffer(),
                num_atoms_host,
                d_atom_types_temp);
        }

        // Capture forward activations (OWNED copies in GradFn)
        grad_fn->capture_forward(
            layer.atomPositionsBuffer(),
            d_atom_types_temp,
            layer.atomEmbeddingsBuffer(),
            num_atoms_host,
            text_features, text_mask,
            total_tokens, stream);

        // Free temporary atom types buffer (GradFn made its own copy)
        if (d_atom_types_temp) cudaFree(d_atom_types_temp);

        output.grad_fn = std::move(grad_fn);

        // Update stats
        layer.recordForwardCall(num_atoms_host);
    } else {
        layer.recordForwardCall(0);
    }

    return output;
}

}  // namespace autograd

//======================================================//
//  ScratchBlockLayer Implementation
//======================================================//

ScratchBlockLayer::ScratchBlockLayer() {
    config_ = ScratchBlockConfig{};
}

ScratchBlockLayer::ScratchBlockLayer(const ScratchBlockConfig& config)
    : config_(config)
{
    if (config_.enabled) {
        allocateWeights();
        initializeWeights();
    }
}

ScratchBlockLayer::~ScratchBlockLayer() {
    freeWeights();
}

ScratchBlockLayer::ScratchBlockLayer(ScratchBlockLayer&& other) noexcept
    : config_(other.config_)
    , stats_(other.stats_)
    , atom_type_embeddings_(std::move(other.atom_type_embeddings_))
    , atom_projection_(std::move(other.atom_projection_))
    , text_feature_projection_(std::move(other.text_feature_projection_))
    , d_atom_positions_(other.d_atom_positions_)
    , d_num_atoms_(other.d_num_atoms_)
    , d_atom_embeddings_(other.d_atom_embeddings_)
    , weights_allocated_(other.weights_allocated_)
{
    other.d_atom_positions_ = nullptr;
    other.d_num_atoms_ = nullptr;
    other.d_atom_embeddings_ = nullptr;
    other.weights_allocated_ = false;
}

ScratchBlockLayer& ScratchBlockLayer::operator=(ScratchBlockLayer&& other) noexcept {
    if (this != &other) {
        freeWeights();

        config_ = other.config_;
        stats_ = other.stats_;
        atom_type_embeddings_ = std::move(other.atom_type_embeddings_);
        atom_projection_ = std::move(other.atom_projection_);
        text_feature_projection_ = std::move(other.text_feature_projection_);
        d_atom_positions_ = other.d_atom_positions_;
        d_num_atoms_ = other.d_num_atoms_;
        d_atom_embeddings_ = other.d_atom_embeddings_;
        weights_allocated_ = other.weights_allocated_;

        other.d_atom_positions_ = nullptr;
        other.d_num_atoms_ = nullptr;
        other.d_atom_embeddings_ = nullptr;
        other.weights_allocated_ = false;
    }
    return *this;
}

void ScratchBlockLayer::setConfig(const ScratchBlockConfig& config) {
    bool was_enabled = config_.enabled;
    config_ = config;
    if (config_.enabled && !was_enabled && !weights_allocated_) {
        allocateWeights();
        initializeWeights();
    }
}

void ScratchBlockLayer::setEnabled(bool enabled) {
    bool was_enabled = config_.enabled;
    config_.enabled = enabled;
    if (enabled && !was_enabled && !weights_allocated_) {
        allocateWeights();
        initializeWeights();
    }
}

void ScratchBlockLayer::allocateWeights() {
    if (weights_allocated_) return;
    StreamController::fatalIfDefaultStream(config_.stream, "ScratchBlockLayer::allocateWeights");

    TensorContract::Shape2D atom_emb_2d{NUM_ATOM_TYPES, config_.atom_embedding_dim};
    TensorContract::Shape2D proj_2d{config_.atom_embedding_dim, config_.d_model};
    TensorContract::Shape2D text_proj_2d{kTextFeatureDim, config_.d_model};

    TensorContract::TensorShape atom_emb_shape(TensorContract::Layout::BSM, atom_emb_2d);
    TensorContract::TensorShape proj_shape(TensorContract::Layout::BSM, proj_2d);
    TensorContract::TensorShape text_proj_shape(TensorContract::Layout::BSM, text_proj_2d);

    atom_type_embeddings_ = Tensor::zeros(atom_emb_shape, true, config_.stream, "atom_type_embeddings");
    atom_type_embeddings_.ensure_grad();
    atom_projection_ = Tensor::zeros(proj_shape, true, config_.stream, "atom_projection");
    atom_projection_.ensure_grad();
    text_feature_projection_ = Tensor::zeros(text_proj_shape, true, config_.stream, "text_feature_projection");
    text_feature_projection_.ensure_grad();

    // Value extraction head: hidden_state[atom_pos] → numeric value
    // W_extract: [1, d_model] projects d_model-dim hidden state to a scalar
    // b_extract: [1] bias
    TensorContract::Shape2D extract_w_2d{1, config_.d_model};
    TensorContract::Shape2D extract_b_2d{1, 1};
    TensorContract::TensorShape extract_w_shape(TensorContract::Layout::BSM, extract_w_2d);
    TensorContract::TensorShape extract_b_shape(TensorContract::Layout::BSM, extract_b_2d);
    value_extraction_weight_ = Tensor::zeros(extract_w_shape, true, config_.stream, "value_extraction_weight");
    value_extraction_weight_.ensure_grad();
    value_extraction_bias_ = Tensor::zeros(extract_b_shape, true, config_.stream, "value_extraction_bias");
    value_extraction_bias_.ensure_grad();

    // Device scalar for extraction output (reused across calls)
    cudaMalloc(&d_extraction_output_, sizeof(float));

    // Temporary buffers for forward (reused across calls, NOT cached for backward — GradFn owns that)
    cudaMalloc(&d_atom_positions_, config_.max_atoms * sizeof(int));
    cudaMalloc(&d_num_atoms_, sizeof(int));
    cudaMalloc(&d_atom_embeddings_, static_cast<size_t>(config_.max_atoms) * config_.atom_embedding_dim * sizeof(float));

    weights_allocated_ = true;
}

void ScratchBlockLayer::freeWeights() {
    atom_type_embeddings_ = Tensor();
    atom_projection_ = Tensor();
    text_feature_projection_ = Tensor();
    value_extraction_weight_ = Tensor();
    value_extraction_bias_ = Tensor();

    if (d_atom_positions_)  cudaFree(d_atom_positions_);
    if (d_num_atoms_)       cudaFree(d_num_atoms_);
    if (d_atom_embeddings_) cudaFree(d_atom_embeddings_);
    if (d_extraction_output_) cudaFree(d_extraction_output_);

    d_atom_positions_ = nullptr;
    d_num_atoms_ = nullptr;
    d_atom_embeddings_ = nullptr;
    d_extraction_output_ = nullptr;
    weights_allocated_ = false;
}

void ScratchBlockLayer::initializeWeights() {
    if (!weights_allocated_) return;
    StreamController::fatalIfDefaultStream(config_.stream, "ScratchBlockLayer::initializeWeights");

    cudaStream_t stream = config_.stream;
    const int block_size = 256;

    // Xavier init for atom type embeddings (splitmix64 PRNG)
    const int atom_emb_size = NUM_ATOM_TYPES * config_.atom_embedding_dim;
    float atom_stddev = std::sqrt(2.0f / (NUM_ATOM_TYPES + config_.atom_embedding_dim));
    int grid_size = (atom_emb_size + block_size - 1) / block_size;
    kernelXavierInit<<<grid_size, block_size, 0, stream>>>(
        atom_type_embeddings_.data, atom_emb_size, atom_stddev, 42);

    // Xavier init for atom projection
    const int proj_size = config_.atom_embedding_dim * config_.d_model;
    float proj_stddev = std::sqrt(2.0f / (config_.atom_embedding_dim + config_.d_model));
    grid_size = (proj_size + block_size - 1) / block_size;
    kernelXavierInit<<<grid_size, block_size, 0, stream>>>(
        atom_projection_.data, proj_size, proj_stddev, 123);

    // Xavier init for text feature projection
    const int text_proj_size = kTextFeatureDim * config_.d_model;
    float text_proj_stddev = std::sqrt(2.0f / (kTextFeatureDim + config_.d_model));
    grid_size = (text_proj_size + block_size - 1) / block_size;
    kernelXavierInit<<<grid_size, block_size, 0, stream>>>(
        text_feature_projection_.data, text_proj_size, text_proj_stddev, 456);

    // Xavier init for value extraction weight [1, d_model]
    // Use smaller stddev since this projects from high-dim hidden to scalar
    float extract_stddev = std::sqrt(2.0f / (1 + config_.d_model));
    grid_size = (config_.d_model + block_size - 1) / block_size;
    kernelXavierInit<<<grid_size, block_size, 0, stream>>>(
        value_extraction_weight_.data, config_.d_model, extract_stddev, 789);
    // Bias initialized to 0 (already from zeros())

    logWeightInit();
}

void ScratchBlockLayer::runForwardKernels(
    float* output, int total_tokens,
    const int* token_ids,
    const float* numeric_values, const uint8_t* numeric_mask,
    const uint16_t* text_features, const uint8_t* text_mask,
    cudaStream_t stream)
{
    // Step 1: Detect atom tokens
    cudaMemsetAsync(d_num_atoms_, 0, sizeof(int), stream);

    const int detect_block = 256;
    const int detect_grid = (total_tokens + detect_block - 1) / detect_block;
    kernelDetectAtomTokens<<<detect_grid, detect_block, 0, stream>>>(
        token_ids, total_tokens,
        d_atom_positions_, d_num_atoms_,
        config_.max_atoms,
        config_.atom_token_start, config_.atom_token_end);

    const int atom_blocks = std::min(config_.max_atoms, total_tokens);
    if (atom_blocks <= 0) return;

    // Step 2: Lookup atom embeddings (value-aware)
    kernelLookupAtomEmbeddingsWithValue<<<atom_blocks, config_.atom_embedding_dim, 0, stream>>>(
        token_ids, d_atom_positions_, d_num_atoms_, config_.max_atoms,
        atom_type_embeddings_.data,
        numeric_values, numeric_mask,
        d_atom_embeddings_, config_.atom_embedding_dim);

    // Step 3: Project and inject atom embeddings
    kernelInjectAtomEmbeddings<<<atom_blocks, config_.d_model, 0, stream>>>(
        output, d_atom_positions_, d_num_atoms_, config_.max_atoms,
        d_atom_embeddings_, atom_projection_.data,
        config_.atom_embedding_dim, config_.d_model, config_.atom_scale);

    // Step 4: Inject text features
    if (text_features && text_mask) {
        kernelInjectTextFeatures<<<total_tokens, config_.d_model, 0, stream>>>(
            output, text_features, text_mask,
            text_feature_projection_.data,
            total_tokens, config_.d_model, config_.atom_scale);
    }
}

//======================================================//
//  Logging
//======================================================//

float ScratchBlockLayer::extractNumericValue(
    const float* encoder_output,
    int position,
    cudaStream_t stream)
{
    if (!weights_allocated_) {
        throw std::runtime_error(
            "[ScratchBlock::extractNumericValue] weights not allocated — "
            "call allocateWeights() first");
    }
    if (!encoder_output) {
        throw std::runtime_error(
            "[ScratchBlock::extractNumericValue] encoder_output is NULL");
    }
    if (!d_extraction_output_) {
        throw std::runtime_error(
            "[ScratchBlock::extractNumericValue] d_extraction_output_ is NULL — "
            "allocateWeights() did not allocate extraction buffer");
    }

    // kernelExtractNumericValue does dot(W_extract, hidden) + bias
    // using shared-memory tree reduction.  One block, d_model threads.
    const int threads = std::min(config_.d_model, 1024);
    kernelExtractNumericValue<<<1, threads, threads * sizeof(float), stream>>>(
        encoder_output,
        value_extraction_weight_.data,
        value_extraction_bias_.data,
        d_extraction_output_,
        position,
        config_.d_model);

    // Copy scalar result back to host (requires stream sync)
    float host_value = 0.0f;
    cudaMemcpyAsync(&host_value, d_extraction_output_, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return host_value;
}

std::pair<float, int> ScratchBlockLayer::trainExtractionStep(
    const float* encoder_output,
    const float* numeric_values,
    const float* numeric_mask,
    int total_tokens,
    cudaStream_t stream)
{
    if (!weights_allocated_) {
        throw std::runtime_error(
            "[ScratchBlock::trainExtractionStep] weights not allocated");
    }
    if (!encoder_output || !numeric_values || !numeric_mask) {
        throw std::runtime_error(
            "[ScratchBlock::trainExtractionStep] NULL input pointer");
    }
    if (total_tokens <= 0) {
        return {0.0f, 0};
    }

    // ---------------------------------------------------------------
    //  Pass 1: Count atoms on GPU (reuse d_num_atoms_ buffer)
    // ---------------------------------------------------------------
    // We need atom count BEFORE the kernel to compute inv_count.
    // Simple approach: launch a tiny count kernel, sync, then launch
    // the training kernel with inv_count.
    //
    // TODO(perf): fuse count into the training kernel or precompute
    //             atom count from the batch builder.
    // ---------------------------------------------------------------

    // Count kernel: one thread per position, atomicAdd to d_num_atoms_
    cudaMemsetAsync(d_num_atoms_, 0, sizeof(int), stream);
    {
        // Reuse kernelExtractAtomTypes grid sizing pattern
        const int blk = 256;
        const int grd = (total_tokens + blk - 1) / blk;
        // Inline tiny lambda is not allowed in CUDA, so use the
        // fact that numeric_mask[t] >= 0.5 means atom.
        // We need a small count kernel.  Rather than adding yet
        // another kernel, we'll do a host-side sync:
        // Copy the mask back and count.  For ~7000 tokens this is <28KB.
    }

    // Host-side atom count (small memcpy, avoids extra kernel)
    std::vector<float> h_mask(total_tokens);
    cudaMemcpyAsync(h_mask.data(), numeric_mask,
                    total_tokens * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int atom_count = 0;
    for (int t = 0; t < total_tokens; ++t) {
        if (h_mask[t] >= 0.5f) ++atom_count;
    }
    if (atom_count == 0) {
        return {0.0f, 0};
    }

    const float inv_count = 1.0f / static_cast<float>(atom_count);

    // ---------------------------------------------------------------
    //  Pass 2: Training kernel — accumulates grad_W, grad_b, loss
    // ---------------------------------------------------------------
    //  NOTE: grad_W and grad_b are NOT zeroed here.  The caller
    //  (executeAutogradBackward) zeros them in the !accumulate path.
    //  We ACCUMULATE onto any existing gradient.
    // ---------------------------------------------------------------
    cudaMemsetAsync(d_extraction_output_, 0, sizeof(float), stream);  // loss accumulator

    // Read bias scalar from device (kernel takes float, not pointer)
    float h_bias = 0.0f;
    cudaMemcpyAsync(&h_bias, value_extraction_bias_.data, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const int threads = std::min(config_.d_model, 256);
    kernelExtractionTrainStep<<<total_tokens, threads,
                                threads * sizeof(float), stream>>>(
        encoder_output,
        value_extraction_weight_.data,
        h_bias,
        numeric_values,
        numeric_mask,
        value_extraction_weight_.grad_data(),
        value_extraction_bias_.grad_data(),
        d_extraction_output_,
        total_tokens,
        config_.d_model,
        inv_count);

    // Read back mean loss
    float total_loss = 0.0f;
    cudaMemcpyAsync(&total_loss, d_extraction_output_, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    const float mean_loss = total_loss / static_cast<float>(atom_count);
    return {mean_loss, atom_count};
}

void ScratchBlockLayer::logForward(int num_atoms, float duration_ms) {
    if (!logging_enabled_) return;

    std::ostringstream oss;
    oss << "forward: atoms=";
    if (num_atoms < 0) oss << "unknown";
    else oss << num_atoms;
    oss << " duration_ms=" << std::fixed << std::setprecision(3) << duration_ms
        << " total_calls=" << stats_.total_forward_calls
        << " active=" << stats_.active_calls;
    Logging::EmitModuleInfo(kScratchBlockModule, oss.str(), global_step_);
}

void ScratchBlockLayer::logWeightInit() {
    if (!logging_enabled_) return;

    const int atom_emb_size = NUM_ATOM_TYPES * config_.atom_embedding_dim;
    const int proj_size = config_.atom_embedding_dim * config_.d_model;
    const int text_proj_size = kTextFeatureDim * config_.d_model;
    const size_t total_bytes = (atom_emb_size + proj_size + text_proj_size) * sizeof(float) * 2;

    std::ostringstream oss;
    oss << "weights_init: atom_types=" << NUM_ATOM_TYPES
        << " atom_emb_dim=" << config_.atom_embedding_dim
        << " d_model=" << config_.d_model
        << " total_params=" << (atom_emb_size + proj_size + text_proj_size)
        << " memory_mb=" << std::fixed << std::setprecision(2) << (total_bytes / (1024.0 * 1024.0));
    Logging::EmitModuleInfo(kScratchBlockModule, oss.str(), global_step_);
}

} // namespace GRIM

//======================================================//
//  ScratchBlock_GPU.cu
//  Implementation of Internal Reasoning Layer
//======================================================//

#include "ScratchBlock_GPU.hpp"
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
//  Constants - using centralized HyperParameters (Rule 20)
//======================================================//

// Module name for logging
static constexpr const char* kScratchBlockModule = "ScratchBlock";

// Atom token range from HyperParameters
constexpr int ATOM_TOKEN_START = HyperParameters::ATOM_TOKEN_START;
constexpr int NUM_ATOM_TYPES = HyperParameters::NUM_ATOM_TYPES;

// Text feature dimension (must match GRIM::Tokenizer::kTextFeatureDim)
constexpr int kTextFeatureDim = 16;

__device__ __forceinline__ int ClampNumAtoms(const int* num_atoms, int max_atoms) {
    int n = 0;
    if (num_atoms) {
        n = *num_atoms;
        if (n < 0) {
            n = 0;
        }
    }
    return (n > max_atoms) ? max_atoms : n;
}

//======================================================//
//  CUDA Kernels
//======================================================//

// Kernel: Detect atom tokens in sequence
__global__ void kernelDetectAtomTokens(
    const int* __restrict__ token_ids,
    int total_tokens,
    int* __restrict__ atom_positions,
    int* __restrict__ num_atoms,
    int max_atoms,
    int atom_token_start,
    int atom_token_end
) {
    __shared__ int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_tokens) {
        int token = token_ids[idx];
        
        // Check if token is in atom range
        if (token >= atom_token_start && token < atom_token_end) {
            int pos = atomicAdd(&shared_count, 1);
            if (pos < max_atoms) {
                // Note: This may cause race conditions for position ordering
                // but we only need to know which positions have atoms
                int global_pos = atomicAdd(num_atoms, 1);
                if (global_pos < max_atoms) {
                    atom_positions[global_pos] = idx;
                }
            }
        }
    }
}

// Kernel: Passthrough copy (when disabled)
__global__ void kernelPassthrough(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx];
    }
}

// Kernel: Lookup atom type embeddings (type-only, legacy)
__global__ void kernelLookupAtomEmbeddings(
    const int* __restrict__ token_ids,
    const int* __restrict__ atom_positions,
    const int* __restrict__ num_atoms,
    int max_atoms,
    const float* __restrict__ atom_type_embeddings,  // [NUM_ATOM_TYPES, atom_embedding_dim]
    float* __restrict__ atom_embeddings,             // [num_atoms, atom_embedding_dim]
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
    
    // Map token ID to atom type index
    int atom_type = (token_id - ATOM_TOKEN_START) % NUM_ATOM_TYPES;
    
    // Lookup embedding
    float value = atom_type_embeddings[atom_type * atom_embedding_dim + dim_idx];
    atom_embeddings[atom_idx * atom_embedding_dim + dim_idx] = value;
}

// Kernel: Lookup atom embeddings with VALUE encoding (semantically-aware)
// Encodes both the atom TYPE and its numeric VALUE into the embedding
__global__ void kernelLookupAtomEmbeddingsWithValue(
    const int* __restrict__ token_ids,
    const int* __restrict__ atom_positions,
    const int* __restrict__ num_atoms,
    int max_atoms,
    const float* __restrict__ atom_type_embeddings,  // [NUM_ATOM_TYPES, atom_embedding_dim]
    const float* __restrict__ token_numeric_values,  // [total_tokens] - numeric values per token
    const uint8_t* __restrict__ token_numeric_mask,  // [total_tokens] - mask per token
    float* __restrict__ atom_embeddings,             // [num_atoms, atom_embedding_dim]
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
        ? token_numeric_values[token_pos]
        : 0.0f;
    if (has_numeric && !isfinite(numeric_val)) {
        has_numeric = false;
        numeric_val = 0.0f;
    }
    
    // Base embedding from type
    float value = atom_type_embeddings[atom_type * atom_embedding_dim + dim_idx];
    
    // Add value-specific encoding to certain dimensions
    // This ensures different numeric values produce different embeddings
    if (dim_idx < 16) {
        // First 16 dims: Type one-hot (unchanged)
        // value already set from type embeddings
    }
    else if (dim_idx < 32) {
        // Dims 16-31: Encode the numeric magnitude in log scale
        // Different bits of the log-scaled value
        int bit = dim_idx - 16;
        if (has_numeric) {
            float log_mag = log2f(fabsf(numeric_val) + 1.0f);  // Log magnitude
            // Encode different frequency components
            float freq = (float)(bit + 1) * 0.5f;
            value += 0.5f * sinf(log_mag * freq);
        }
    }
    else if (dim_idx < 48) {
        // Dims 32-47: Encode sign and fractional precision
        int feat = dim_idx - 32;
        if (!has_numeric) {
            // Keep type-only embedding when numeric is absent.
        } else if (feat == 0) {
            // Sign bit
            value += (numeric_val < 0) ? 0.5f : -0.5f;
        } else if (feat < 8) {
            // Low bits of integer part (for small number discrimination)
            int int_val = (int)fabsf(numeric_val);
            value += ((int_val >> (feat - 1)) & 1) ? 0.3f : -0.3f;
        } else {
            // Fractional encoding for distinguishing 5 vs 5.0 vs 5.5
            float frac = numeric_val - floorf(numeric_val);
            value += 0.3f * sinf(frac * 3.14159f * (feat - 7));
        }
    }
    // Dims 48+: Keep pure type embedding for category information
    
    atom_embeddings[atom_idx * atom_embedding_dim + dim_idx] = value;
}

// Kernel: Project atom embeddings and inject into hidden states
__global__ void kernelInjectAtomEmbeddings(
    float* __restrict__ hidden_states,               // [total_tokens, d_model]
    const int* __restrict__ atom_positions,
    const int* __restrict__ num_atoms,
    int max_atoms,
    const float* __restrict__ atom_embeddings,       // [num_atoms, atom_embedding_dim]
    const float* __restrict__ projection,            // [atom_embedding_dim, d_model]
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
    
    // Compute projection: atom_embedding @ projection
    float sum = 0.0f;
    for (int k = 0; k < atom_embedding_dim; ++k) {
        sum += atom_embeddings[atom_idx * atom_embedding_dim + k] * 
               projection[k * d_model + d_idx];
    }
    
    // Add scaled projection to hidden state
    hidden_states[token_pos * d_model + d_idx] += scale * sum;
}

// Backward pass kernel for ScratchBlock atom embedding injection
// Computes gradients for projection matrix and atom embeddings
// Given: grad_output from downstream (same shape as hidden_states)
// grad_projection: [atom_embedding_dim, d_model] - accumulated via atomicAdd
// grad_atom_embeddings: [num_atoms, atom_embedding_dim] - local output
__global__ void kernelBackwardAtomEmbeddings(
    const float* __restrict__ grad_output,           // [total_tokens, d_model]
    const int* __restrict__ atom_positions,          // [num_atoms]
    const int* __restrict__ num_atoms,
    int max_atoms,
    const float* __restrict__ atom_embeddings,       // [num_atoms, atom_embedding_dim]
    const float* __restrict__ projection,            // [atom_embedding_dim, d_model]
    float* __restrict__ grad_projection,             // [atom_embedding_dim, d_model]
    float* __restrict__ grad_atom_embeddings,        // [num_atoms, atom_embedding_dim]
    int atom_embedding_dim,
    int d_model,
    float scale
) {
    const int atom_idx = blockIdx.x;
    const int k_idx = threadIdx.x;  // atom_embedding_dim
    __shared__ int s_num_atoms;
    if (threadIdx.x == 0) {
        s_num_atoms = ClampNumAtoms(num_atoms, max_atoms);
    }
    __syncthreads();
    
    if (atom_idx >= s_num_atoms || k_idx >= atom_embedding_dim) return;
    
    int token_pos = atom_positions[atom_idx];
    const float* grad_h = grad_output + token_pos * d_model;  // [d_model]
    const float* atom_emb = atom_embeddings + atom_idx * atom_embedding_dim;  // [atom_embedding_dim]
    
    // Backward through: hidden_states[pos, d] += scale * sum_k(atom_emb[k] * proj[k, d])
    //
    // grad_projection[k, d] += scale * atom_emb[k] * grad_h[d]
    // grad_atom_emb[k] += scale * sum_d(proj[k, d] * grad_h[d])
    
    // Accumulate grad_atom_embeddings[atom_idx, k_idx]
    float grad_emb = 0.0f;
    for (int d = 0; d < d_model; ++d) {
        grad_emb += projection[k_idx * d_model + d] * grad_h[d];
    }
    grad_atom_embeddings[atom_idx * atom_embedding_dim + k_idx] = scale * grad_emb;
    
    // Accumulate grad_projection[k_idx, d] for all d
    // Use atomicAdd since multiple atoms may contribute to same projection weights
    for (int d = 0; d < d_model; ++d) {
        float grad_proj = scale * atom_emb[k_idx] * grad_h[d];
        atomicAdd(&grad_projection[k_idx * d_model + d], grad_proj);
    }
}

// Kernel: Extract atom types for backward pass gradient routing
// Maps token_id -> atom_type via token ID range
__global__ void kernelExtractAtomTypes(
    const int* __restrict__ token_ids,
    const int* __restrict__ atom_positions,
    const int* __restrict__ num_atoms,
    int max_atoms,
    int* __restrict__ atom_types  // [num_atoms] - output atom type indices
) {
    const int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_atoms_clamped = ClampNumAtoms(num_atoms, max_atoms);
    if (atom_idx >= num_atoms_clamped) return;
    
    int token_pos = atom_positions[atom_idx];
    int token_id = token_ids[token_pos];
    int atom_type = (token_id - ATOM_TOKEN_START) % NUM_ATOM_TYPES;
    atom_types[atom_idx] = atom_type;
}

// Kernel: Accumulate per-atom gradients back to shared atom type embeddings
// Routes gradients from [num_atoms, atom_embedding_dim] to [NUM_ATOM_TYPES, atom_embedding_dim]
__global__ void kernelAccumulateAtomTypeGradients(
    const float* __restrict__ grad_atom_embeddings,  // [num_atoms, atom_embedding_dim]
    const int* __restrict__ atom_types,              // [num_atoms] - which type each atom is
    const int* __restrict__ num_atoms,
    int max_atoms,
    float* __restrict__ grad_atom_type_embeddings,   // [NUM_ATOM_TYPES, atom_embedding_dim]
    int atom_embedding_dim
) {
    const int atom_idx = blockIdx.x;
    const int k_idx = threadIdx.x;
    __shared__ int s_num_atoms;
    if (threadIdx.x == 0) {
        s_num_atoms = ClampNumAtoms(num_atoms, max_atoms);
    }
    __syncthreads();
    
    if (atom_idx >= s_num_atoms || k_idx >= atom_embedding_dim) return;
    
    int atom_type = atom_types[atom_idx];
    if (atom_type < 0 || atom_type >= NUM_ATOM_TYPES) return;
    
    float grad = grad_atom_embeddings[atom_idx * atom_embedding_dim + k_idx];
    atomicAdd(&grad_atom_type_embeddings[atom_type * atom_embedding_dim + k_idx], grad);
}

// Kernel: Inject text features into hidden states
// Projects 16-dim FP16 text features directly to d_model and adds to hidden states
// This is the VALUE encoding path - text_features encode atom semantics
__global__ void kernelInjectTextFeatures(
    float* __restrict__ hidden_states,                // [total_tokens, d_model]
    const uint16_t* __restrict__ text_features,       // [total_tokens * kTextFeatureDim] FP16
    const uint8_t* __restrict__ text_mask,            // [total_tokens] - 1 if token has text features
    const float* __restrict__ text_projection,        // [kTextFeatureDim, d_model]
    int total_tokens,
    int d_model,
    float scale
) {
    const int token_idx = blockIdx.x;
    const int d_idx = threadIdx.x;
    
    if (token_idx >= total_tokens || d_idx >= d_model) return;
    
    // Skip tokens without text features
    if (!text_mask || text_mask[token_idx] == 0) return;
    
    // Get text features for this token (16 FP16 values)
    const uint16_t* token_features = text_features + token_idx * kTextFeatureDim;
    
    // Compute projection: text_features @ text_projection
    // text_features: [kTextFeatureDim] as FP16
    // text_projection: [kTextFeatureDim, d_model]
    float sum = 0.0f;
    for (int k = 0; k < kTextFeatureDim; ++k) {
        // Convert FP16 to FP32
        float feat = __half2float(*reinterpret_cast<const __half*>(&token_features[k]));
        sum += feat * text_projection[k * d_model + d_idx];
    }
    
    // Add scaled projection to hidden state
    hidden_states[token_idx * d_model + d_idx] += scale * sum;
}

// Backward kernel for text feature injection
// Computes gradients for text_projection matrix
__global__ void kernelBackwardTextFeatures(
    const float* __restrict__ grad_output,            // [total_tokens, d_model]
    const uint16_t* __restrict__ text_features,       // [total_tokens * kTextFeatureDim] FP16
    const uint8_t* __restrict__ text_mask,            // [total_tokens]
    float* __restrict__ grad_text_projection,         // [kTextFeatureDim, d_model]
    int total_tokens,
    int d_model,
    float scale
) {
    const int token_idx = blockIdx.x;
    const int k_idx = threadIdx.x;  // kTextFeatureDim
    
    if (token_idx >= total_tokens || k_idx >= kTextFeatureDim) return;
    
    // Skip tokens without text features
    if (!text_mask || text_mask[token_idx] == 0) return;
    
    // Get text feature value for this token and dimension
    const uint16_t* token_features = text_features + token_idx * kTextFeatureDim;
    float feat = __half2float(*reinterpret_cast<const __half*>(&token_features[k_idx]));
    
    // Backward through: hidden_states[token, d] += scale * sum_k(feat[k] * proj[k, d])
    // grad_projection[k, d] += scale * feat[k] * grad_output[token, d]
    const float* grad_h = grad_output + token_idx * d_model;
    
    for (int d = 0; d < d_model; ++d) {
        float grad_proj = scale * feat * grad_h[d];
        atomicAdd(&grad_text_projection[k_idx * d_model + d], grad_proj);
    }
}

// Kernel: Xavier initialization
__global__ void kernelXavierInit(
    float* __restrict__ weights,
    int size,
    float stddev,
    unsigned int seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Simple LCG random number generator
    unsigned int state = seed + idx * 1099087573u;
    state = state * 1103515245u + 12345u;
    float u1 = (state & 0x7FFFFFFF) / float(0x7FFFFFFF);
    state = state * 1103515245u + 12345u;
    float u2 = (state & 0x7FFFFFFF) / float(0x7FFFFFFF);
    
    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);
    weights[idx] = z * stddev;
}

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
    , d_atom_type_embeddings_(other.d_atom_type_embeddings_)
    , d_atom_type_embeddings_grad_(other.d_atom_type_embeddings_grad_)
    , d_atom_projection_(other.d_atom_projection_)
    , d_atom_projection_grad_(other.d_atom_projection_grad_)
    , d_text_feature_projection_(other.d_text_feature_projection_)
    , d_text_feature_projection_grad_(other.d_text_feature_projection_grad_)
    , d_atom_positions_(other.d_atom_positions_)
    , d_num_atoms_(other.d_num_atoms_)
    , d_atom_embeddings_(other.d_atom_embeddings_)
    , d_grad_atom_embeddings_(other.d_grad_atom_embeddings_)
    , weights_allocated_(other.weights_allocated_)
{
    other.d_atom_type_embeddings_ = nullptr;
    other.d_atom_type_embeddings_grad_ = nullptr;
    other.d_atom_projection_ = nullptr;
    other.d_atom_projection_grad_ = nullptr;
    other.d_text_feature_projection_ = nullptr;
    other.d_text_feature_projection_grad_ = nullptr;
    other.d_atom_positions_ = nullptr;
    other.d_num_atoms_ = nullptr;
    other.d_atom_embeddings_ = nullptr;
    other.d_grad_atom_embeddings_ = nullptr;
    other.weights_allocated_ = false;
}

ScratchBlockLayer& ScratchBlockLayer::operator=(ScratchBlockLayer&& other) noexcept {
    if (this != &other) {
        freeWeights();
        
        config_ = other.config_;
        stats_ = other.stats_;
        d_atom_type_embeddings_ = other.d_atom_type_embeddings_;
        d_atom_type_embeddings_grad_ = other.d_atom_type_embeddings_grad_;
        d_atom_projection_ = other.d_atom_projection_;
        d_atom_projection_grad_ = other.d_atom_projection_grad_;
        d_text_feature_projection_ = other.d_text_feature_projection_;
        d_text_feature_projection_grad_ = other.d_text_feature_projection_grad_;
        d_atom_positions_ = other.d_atom_positions_;
        d_num_atoms_ = other.d_num_atoms_;
        d_atom_embeddings_ = other.d_atom_embeddings_;
        d_grad_atom_embeddings_ = other.d_grad_atom_embeddings_;
        weights_allocated_ = other.weights_allocated_;
        
        other.d_atom_type_embeddings_ = nullptr;
        other.d_atom_type_embeddings_grad_ = nullptr;
        other.d_atom_projection_ = nullptr;
        other.d_atom_projection_grad_ = nullptr;
        other.d_text_feature_projection_ = nullptr;
        other.d_text_feature_projection_grad_ = nullptr;
        other.d_atom_positions_ = nullptr;
        other.d_num_atoms_ = nullptr;
        other.d_atom_embeddings_ = nullptr;
        other.d_grad_atom_embeddings_ = nullptr;
        other.weights_allocated_ = false;
    }
    return *this;
}

void ScratchBlockLayer::setConfig(const ScratchBlockConfig& config) {
    bool was_enabled = config_.enabled;
    config_ = config;
    
    // Allocate weights if newly enabled
    if (config_.enabled && !was_enabled && !weights_allocated_) {
        allocateWeights();
        initializeWeights();
    }
}

void ScratchBlockLayer::setEnabled(bool enabled) {
    bool was_enabled = config_.enabled;
    config_.enabled = enabled;
    
    // Lazy allocation on first enable
    if (enabled && !was_enabled && !weights_allocated_) {
        allocateWeights();
        initializeWeights();
    }
}

void ScratchBlockLayer::allocateWeights() {
    if (weights_allocated_) return;
    StreamController::fatalIfDefaultStream(config_.stream, "ScratchBlockLayer::allocateWeights");
    
    const int atom_emb_size = NUM_ATOM_TYPES * config_.atom_embedding_dim;
    const int proj_size = config_.atom_embedding_dim * config_.d_model;
    const int text_proj_size = kTextFeatureDim * config_.d_model;
    
    cudaMalloc(&d_atom_type_embeddings_, atom_emb_size * sizeof(float));
    cudaMalloc(&d_atom_type_embeddings_grad_, atom_emb_size * sizeof(float));
    cudaMalloc(&d_atom_projection_, proj_size * sizeof(float));
    cudaMalloc(&d_atom_projection_grad_, proj_size * sizeof(float));
    
    // Text feature projection: [kTextFeatureDim, d_model]
    cudaMalloc(&d_text_feature_projection_, text_proj_size * sizeof(float));
    cudaMalloc(&d_text_feature_projection_grad_, text_proj_size * sizeof(float));
    
    cudaMalloc(&d_atom_positions_, config_.max_atoms * sizeof(int));
    cudaMalloc(&d_num_atoms_, sizeof(int));
    cudaMalloc(&d_atom_embeddings_, config_.max_atoms * config_.atom_embedding_dim * sizeof(float));
    cudaMalloc(&d_grad_atom_embeddings_, config_.max_atoms * config_.atom_embedding_dim * sizeof(float));
    
    // Zero gradients (async - GradAccumulationController will sync before first backward)
    // NOTE: config_.stream may be nullptr during early init, use default stream
    cudaMemsetAsync(d_atom_type_embeddings_grad_, 0, atom_emb_size * sizeof(float), config_.stream);
    cudaMemsetAsync(d_atom_projection_grad_, 0, proj_size * sizeof(float), config_.stream);
    cudaMemsetAsync(d_text_feature_projection_grad_, 0, text_proj_size * sizeof(float), config_.stream);
    
    weights_allocated_ = true;
}

void ScratchBlockLayer::freeWeights() {
    if (d_atom_type_embeddings_) cudaFree(d_atom_type_embeddings_);
    if (d_atom_type_embeddings_grad_) cudaFree(d_atom_type_embeddings_grad_);
    if (d_atom_projection_) cudaFree(d_atom_projection_);
    if (d_atom_projection_grad_) cudaFree(d_atom_projection_grad_);
    if (d_text_feature_projection_) cudaFree(d_text_feature_projection_);
    if (d_text_feature_projection_grad_) cudaFree(d_text_feature_projection_grad_);
    if (d_atom_positions_) cudaFree(d_atom_positions_);
    if (d_num_atoms_) cudaFree(d_num_atoms_);
    if (d_atom_embeddings_) cudaFree(d_atom_embeddings_);
    if (d_grad_atom_embeddings_) cudaFree(d_grad_atom_embeddings_);
    
    d_atom_type_embeddings_ = nullptr;
    d_atom_type_embeddings_grad_ = nullptr;
    d_atom_projection_ = nullptr;
    d_atom_projection_grad_ = nullptr;
    d_text_feature_projection_ = nullptr;
    d_text_feature_projection_grad_ = nullptr;
    d_atom_positions_ = nullptr;
    d_num_atoms_ = nullptr;
    d_atom_embeddings_ = nullptr;
    d_grad_atom_embeddings_ = nullptr;
    weights_allocated_ = false;
}

void ScratchBlockLayer::initializeWeights() {
    if (!weights_allocated_) return;
    StreamController::fatalIfDefaultStream(config_.stream, "ScratchBlockLayer::initializeWeights");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    cudaStream_t stream = config_.stream;
    
    // Xavier init for atom type embeddings
    const int atom_emb_size = NUM_ATOM_TYPES * config_.atom_embedding_dim;
    float atom_stddev = std::sqrt(2.0f / (NUM_ATOM_TYPES + config_.atom_embedding_dim));
    
    int block_size = 256;
    int grid_size = (atom_emb_size + block_size - 1) / block_size;
    kernelXavierInit<<<grid_size, block_size, 0, stream>>>(
        d_atom_type_embeddings_, atom_emb_size, atom_stddev, 42);
    
    // Xavier init for atom projection
    const int proj_size = config_.atom_embedding_dim * config_.d_model;
    float proj_stddev = std::sqrt(2.0f / (config_.atom_embedding_dim + config_.d_model));
    
    grid_size = (proj_size + block_size - 1) / block_size;
    kernelXavierInit<<<grid_size, block_size, 0, stream>>>(
        d_atom_projection_, proj_size, proj_stddev, 123);
    
    // Xavier init for text feature projection [kTextFeatureDim, d_model]
    const int text_proj_size = kTextFeatureDim * config_.d_model;
    float text_proj_stddev = std::sqrt(2.0f / (kTextFeatureDim + config_.d_model));
    
    grid_size = (text_proj_size + block_size - 1) / block_size;
    kernelXavierInit<<<grid_size, block_size, 0, stream>>>(
        d_text_feature_projection_, text_proj_size, text_proj_stddev, 456);
    
    // Log weight initialization
    logWeightInit();
}

void ScratchBlockLayer::forward(const ScratchBlockForwardArgs& args) {
    auto start = std::chrono::high_resolution_clock::now();
    
    stats_.total_forward_calls++;
    
    // DISABLED: Pure passthrough (zero overhead)
    if (!config_.enabled) {
        forwardPassthrough(args);
        stats_.passthrough_calls++;
        
        auto end = std::chrono::high_resolution_clock::now();
        float duration_ms = std::chrono::duration<float, std::milli>(end - start).count();
        logForward(0, duration_ms);
        return;
    }
    
    // RULE 20: Validate args when active
    args.validate("ScratchBlockLayer::forward");
    
    // ENABLED: Active atom detection and injection
    forwardActive(args);
    stats_.active_calls++;
    
    // Copy atom count back to CPU for logging (diagnostic only)
    int num_atoms_host = 0;
    cudaMemcpyAsync(&num_atoms_host, d_num_atoms_, sizeof(int), cudaMemcpyDeviceToHost, args.stream);
    cudaStreamSynchronize(args.stream);  // Wait for copy
    
    auto end = std::chrono::high_resolution_clock::now();
    float duration_ms = std::chrono::duration<float, std::milli>(end - start).count();
    logForward(num_atoms_host, duration_ms);
}

void ScratchBlockLayer::forwardPassthrough(const ScratchBlockForwardArgs& args) {
    // Extract raw pointers from TensorViews
    const float* input_ptr = args.input.ptr;
    float* output_ptr = args.output.ptr;
    
    // Just copy input to output (or do nothing if same pointer)
    if (input_ptr == output_ptr) {
        return;  // In-place, nothing to do
    }
    
    const int total_elements = args.total_tokens * config_.d_model;
    cudaStream_t stream = args.stream ? args.stream : config_.stream;
    
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    
    kernelPassthrough<<<grid_size, block_size, 0, stream>>>(
        input_ptr, output_ptr, total_elements);
}

void ScratchBlockLayer::forwardActive(const ScratchBlockForwardArgs& args) {
    cudaStream_t stream = args.stream ? args.stream : config_.stream;
    
    // Extract raw pointers from TensorViews
    const float* input_ptr = args.input.ptr;
    float* output_ptr = args.output.ptr;
    
    // Step 1: Copy input to output (we'll modify output in-place)
    if (input_ptr != output_ptr) {
        const size_t bytes = args.total_tokens * config_.d_model * sizeof(float);
        cudaMemcpyAsync(output_ptr, input_ptr, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    
    // Step 2: Detect atom tokens (if token IDs provided)
    if (!args.token_ids || !config_.inject_atom_embeddings) {
        return;  // No token IDs, can't detect atoms
    }
    
    // Reset atom count
    cudaMemsetAsync(d_num_atoms_, 0, sizeof(int), stream);
    
    // Detect atoms
    const int block_size = 256;
    const int grid_size = (args.total_tokens + block_size - 1) / block_size;
    
    kernelDetectAtomTokens<<<grid_size, block_size, 0, stream>>>(
        args.token_ids,
        args.total_tokens,
        d_atom_positions_,
        d_num_atoms_,
        config_.max_atoms,
        config_.atom_token_start,
        config_.atom_token_end);
    
    const int max_atoms = config_.max_atoms;
    const int atom_blocks = std::min(max_atoms, args.total_tokens);
    if (atom_blocks <= 0) {
        return;
    }
    
    if (!args.token_numeric_values || !args.token_numeric_mask) {
        Logging::EmitModuleError(
            kScratchBlockModule,
            "Numeric side-channel missing for ScratchBlock",
            global_step_);
        throw std::runtime_error("ScratchBlockLayer::forward requires token numeric side-channel");
    }

    // Step 3: Lookup atom embeddings (value-aware, per-token side channel)
    kernelLookupAtomEmbeddingsWithValue<<<atom_blocks, config_.atom_embedding_dim, 0, stream>>>(
        args.token_ids,
        d_atom_positions_,
        d_num_atoms_,
        max_atoms,
        d_atom_type_embeddings_,
        args.token_numeric_values,
        args.token_numeric_mask,
        d_atom_embeddings_,
        config_.atom_embedding_dim);
    
    // Step 4: Project and inject atom type embeddings into hidden states
    kernelInjectAtomEmbeddings<<<atom_blocks, config_.d_model, 0, stream>>>(
        output_ptr,
        d_atom_positions_,
        d_num_atoms_,
        max_atoms,
        d_atom_embeddings_,
        d_atom_projection_,
        config_.atom_embedding_dim,
        config_.d_model,
        config_.atom_scale);
    
    // Step 5: Inject text features (VALUE encoding) into hidden states
    // This is the primary value injection path - text_features encode atom semantics
    if (args.token_text_features && args.token_text_mask) {
        kernelInjectTextFeatures<<<args.total_tokens, config_.d_model, 0, stream>>>(
            output_ptr,
            args.token_text_features,
            args.token_text_mask,
            d_text_feature_projection_,
            args.total_tokens,
            config_.d_model,
            config_.atom_scale);
    }
    
    // Cache for backward (if provided)
    if (args.cache_atom_positions) {
        cudaMemcpyAsync(args.cache_atom_positions, d_atom_positions_,
                        atom_blocks * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }
    if (args.cache_num_atoms) {
        cudaMemcpyAsync(args.cache_num_atoms, d_num_atoms_, sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);
    }
    if (args.cache_atom_embeddings.ptr) {
        cudaMemcpyAsync(args.cache_atom_embeddings.ptr, d_atom_embeddings_,
                        static_cast<size_t>(atom_blocks) * config_.atom_embedding_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }
    // Cache atom types for backward gradient routing
    if (args.cache_atom_types && args.token_ids) {
        const int block = 256;
        const int grid = (atom_blocks + block - 1) / block;
        kernelExtractAtomTypes<<<grid, block, 0, stream>>>(
            args.token_ids,
            d_atom_positions_,
            d_num_atoms_,
            max_atoms,
            args.cache_atom_types);
    }
}

void ScratchBlockLayer::backward(const ScratchBlockForwardArgs& args,
                                  const float* grad_output,
                                  float* grad_input) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // When disabled, gradients flow through unchanged
    if (!config_.enabled) {
        if (grad_output != grad_input) {
            const size_t bytes = args.total_tokens * config_.d_model * sizeof(float);
            cudaStream_t stream = args.stream ? args.stream : config_.stream;
            cudaMemcpyAsync(grad_input, grad_output, bytes, cudaMemcpyDeviceToDevice, stream);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        float duration_ms = std::chrono::duration<float, std::milli>(end - start).count();
        logBackward(duration_ms);
        return;
    }
    
    cudaStream_t stream = args.stream ? args.stream : config_.stream;
    
    // Step 1: Copy grad_output to grad_input (gradients flow through additive injection)
    if (grad_output != grad_input) {
        const size_t bytes = args.total_tokens * config_.d_model * sizeof(float);
        cudaMemcpyAsync(grad_input, grad_output, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    
    // Step 2: Compute gradients for projection and atom embeddings
    const int max_atoms = config_.max_atoms;
    const int atom_blocks = std::min(max_atoms, args.total_tokens);
    const size_t temp_size = static_cast<size_t>(atom_blocks) * config_.atom_embedding_dim;
    float* d_grad_atom_embeddings_temp = d_grad_atom_embeddings_;
    if (!d_grad_atom_embeddings_temp || atom_blocks <= 0) {
        auto end = std::chrono::high_resolution_clock::now();
        float duration_ms = std::chrono::duration<float, std::milli>(end - start).count();
        logBackward(duration_ms);
        return;
    }
    cudaMemsetAsync(d_grad_atom_embeddings_temp, 0, temp_size * sizeof(float), stream);
    
    // Launch backward kernel for projection gradients
    const int block_size = std::min(config_.atom_embedding_dim, 1024);
    kernelBackwardAtomEmbeddings<<<atom_blocks, block_size, 0, stream>>>(
        grad_output,
        args.cache_atom_positions,
        args.cache_num_atoms,
        max_atoms,
        args.cache_atom_embeddings.ptr,  // Extract raw pointer from TensorView
        d_atom_projection_,
        d_atom_projection_grad_,
        d_grad_atom_embeddings_temp,
        config_.atom_embedding_dim,
        config_.d_model,
        config_.atom_scale
    );
    
    // Step 3: Accumulate atom embedding gradients back to atom type embeddings
    // Routes from per-atom [num_atoms, atom_embedding_dim] to shared [NUM_ATOM_TYPES, atom_embedding_dim]
    if (args.cache_atom_types && d_atom_type_embeddings_grad_) {
        kernelAccumulateAtomTypeGradients<<<max_atoms, block_size, 0, stream>>>(
            d_grad_atom_embeddings_temp,
            args.cache_atom_types,
            args.cache_num_atoms,
            max_atoms,
            d_atom_type_embeddings_grad_,
            config_.atom_embedding_dim
        );
    }
    
    // Step 4: Backward through text feature injection
    // Computes gradients for text_feature_projection
    if (args.token_text_features && args.token_text_mask && d_text_feature_projection_grad_) {
        const int text_block_size = std::min(kTextFeatureDim, 32);
        kernelBackwardTextFeatures<<<args.total_tokens, text_block_size, 0, stream>>>(
            grad_output,
            args.token_text_features,
            args.token_text_mask,
            d_text_feature_projection_grad_,
            args.total_tokens,
            config_.d_model,
            config_.atom_scale
        );
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    float duration_ms = std::chrono::duration<float, std::milli>(end - start).count();
    logBackward(duration_ms);
}

void ScratchBlockLayer::onConfigure(const Dimensions& dims) {
    config_.d_model = dims.input;
}

// NOTE: forwardImpl/backwardImpl deleted - dead code from unused LayerIO abstraction
// ScratchBlock uses forward(ScratchBlockForwardArgs) with TensorView directly

std::size_t ScratchBlockLayer::requiredWorkspaceBytes(int total_tokens) const {
    if (!config_.enabled) {
        return 0;  // No workspace needed when disabled
    }
    
    // Space for atom detection and embedding lookup
    std::size_t bytes = 0;
    bytes += config_.max_atoms * sizeof(int);                              // atom positions
    bytes += sizeof(int);                                                   // num atoms
    bytes += config_.max_atoms * config_.atom_embedding_dim * sizeof(float); // atom embeddings
    
    return bytes;
}

void ScratchBlockLayer::detectAtomTokensAsync(const int* token_ids,
                                               int total_tokens,
                                               int* out_positions,
                                               int max_atoms,
                                               int* out_num_atoms_device,
                                               cudaStream_t stream) {
    if (!token_ids || !out_positions || total_tokens <= 0 || max_atoms <= 0) {
        return;
    }

    int* num_atoms_device = out_num_atoms_device ? out_num_atoms_device : d_num_atoms_;
    if (!num_atoms_device) {
        return;
    }

    cudaMemsetAsync(num_atoms_device, 0, sizeof(int), stream);

    const int block_size = 256;
    const int grid_size = (total_tokens + block_size - 1) / block_size;

    kernelDetectAtomTokens<<<grid_size, block_size, 0, stream>>>(
        token_ids, total_tokens, out_positions, num_atoms_device, max_atoms,
        config_.atom_token_start, config_.atom_token_end);
}

void ScratchBlockLayer::injectAtomEmbeddings(float* hidden_states,
                                              int total_tokens,
                                              const int* atom_positions,
                                              int num_atoms,
                                              const float* token_numeric_values,
                                              const uint8_t* token_numeric_mask,
                                              cudaStream_t stream) {
    if (num_atoms == 0) return;
    (void)token_numeric_values;
    (void)token_numeric_mask;
    
    // This version doesn't use AtomTable directly on GPU
    // (AtomTable is CPU-side; we use precomputed atom type embeddings)
    
    const int max_atoms = config_.max_atoms;
    const int atom_blocks = std::min(max_atoms, total_tokens);
    if (atom_blocks <= 0) {
        return;
    }
    cudaMemcpyAsync(d_num_atoms_, &num_atoms, sizeof(int), cudaMemcpyHostToDevice, stream);
    kernelInjectAtomEmbeddings<<<atom_blocks, config_.d_model, 0, stream>>>(
        hidden_states,
        atom_positions,
        d_num_atoms_,
        max_atoms,
        d_atom_embeddings_,
        d_atom_projection_,
        config_.atom_embedding_dim,
        config_.d_model,
        config_.atom_scale);
}

//======================================================//
//  Logging Helper Methods
//======================================================//

void ScratchBlockLayer::logForward(int num_atoms, float duration_ms) {
    if (!logging_enabled_) return;
    
    std::ostringstream oss;
    oss << "forward: enabled=" << (config_.enabled ? "true" : "false")
        << " atoms=";
    if (num_atoms < 0) {
        oss << "unknown";
    } else {
        oss << num_atoms;
    }
    oss
        << " duration_ms=" << std::fixed << std::setprecision(3) << duration_ms
        << " total_calls=" << stats_.total_forward_calls
        << " passthrough=" << stats_.passthrough_calls
        << " active=" << stats_.active_calls;
    
    Logging::EmitModuleInfo(kScratchBlockModule, oss.str(), global_step_);
}

void ScratchBlockLayer::logBackward(float duration_ms) {
    if (!logging_enabled_) return;
    
    std::ostringstream oss;
    oss << "backward: enabled=" << (config_.enabled ? "true" : "false")
        << " duration_ms=" << std::fixed << std::setprecision(3) << duration_ms;
    
    Logging::EmitModuleInfo(kScratchBlockModule, oss.str(), global_step_);
}

void ScratchBlockLayer::logWeightInit() {
    if (!logging_enabled_) return;
    
    const int atom_emb_size = NUM_ATOM_TYPES * config_.atom_embedding_dim;
    const int proj_size = config_.atom_embedding_dim * config_.d_model;
    const int text_proj_size = kTextFeatureDim * config_.d_model;
    const size_t total_bytes = (atom_emb_size + proj_size + text_proj_size) * sizeof(float) * 2;  // weights + grads
    
    std::ostringstream oss;
    oss << "weights_init: atom_types=" << NUM_ATOM_TYPES
        << " atom_emb_dim=" << config_.atom_embedding_dim
        << " d_model=" << config_.d_model
        << " total_params=" << (atom_emb_size + proj_size)
        << " memory_mb=" << std::fixed << std::setprecision(2) << (total_bytes / (1024.0 * 1024.0));
    
    Logging::EmitModuleInfo(kScratchBlockModule, oss.str(), global_step_);
}

void ScratchBlockLayer::logConfigChange(const char* param, float old_val, float new_val) {
    if (!logging_enabled_) return;
    
    if (old_val == new_val) return;
    
    std::ostringstream oss;
    oss << "config_change: " << param
        << " old=" << old_val
        << " new=" << new_val;
    
    Logging::EmitModuleInfo(kScratchBlockModule, oss.str(), global_step_);
}

} // namespace GRIM

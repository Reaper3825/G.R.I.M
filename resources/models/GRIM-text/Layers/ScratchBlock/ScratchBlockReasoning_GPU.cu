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

// ATOM_NUM=1 is the only atom type
__device__ __forceinline__ bool isNumericAtomType(int atom_type) {
    return atom_type == 1;
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
// UNIFIED: Uses numeric_values + atom_flags + text_features for ALL atom types.
// numeric_values carries AtomTable packed values; atom_flags carries type-specific metadata;
// text_features carries 16-dim FP16 raw-text surface analysis (length, char ratios, separators).
// All three signals are combined into a single atom embedding vector, eliminating the
// separate text feature injection path (Path B).
__global__ void kernelLookupAtomEmbeddingsWithValue(
    const int* __restrict__ token_ids,
    const int* __restrict__ atom_positions,
    const int* __restrict__ num_atoms,
    int max_atoms,
    const float* __restrict__ atom_type_embeddings,
    const float* __restrict__ token_numeric_values,
    const uint32_t* __restrict__ token_atom_flags,
    const uint16_t* __restrict__ text_features,
    const uint8_t* __restrict__ atom_mask,
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

    float numeric_val = token_numeric_values ? token_numeric_values[token_pos] : 0.0f;
    uint32_t flags = token_atom_flags ? token_atom_flags[token_pos] : 0u;
    bool has_value = (atom_mask[token_pos] != 0) && isfinite(numeric_val);
    bool has_flags = (flags != 0u);

    // Base embedding from type
    float value = atom_type_embeddings[atom_type * atom_embedding_dim + dim_idx];

    // Dims 16-31: Sinusoidal value encoding — for ANY atom with meaningful packed value
    if (dim_idx >= 16 && dim_idx < 32 && has_value) {
        int bit = dim_idx - 16;
        float log_mag = log2f(fabsf(numeric_val) + 1.0f);
        float freq = (float)(bit + 1) * 0.5f;
        value += 0.5f * sinf(log_mag * freq);
    }
    // Dims 32-39: Numeric structure OR flag bits
    else if (dim_idx >= 32 && dim_idx < 40) {
        int feat = dim_idx - 32;
        if (feat == 0 && has_value) {
            // Sign feature (works for all numeric types)
            value += (numeric_val < 0) ? 0.5f : -0.5f;
        } else if (feat < 8 && has_value) {
            // Integer bit features of packed value
            int int_val = (int)fabsf(numeric_val);
            value += ((int_val >> (feat - 1)) & 1) ? 0.3f : -0.3f;
        }
    }
    // Dims 40-47: Flag-based features — encode AtomTable metadata bits
    else if (dim_idx >= 40 && dim_idx < 48 && has_flags) {
        int bit = dim_idx - 40;
        value += ((flags >> bit) & 1u) ? 0.3f : -0.3f;
    }
    // Dims 48-63: Text features — raw-text surface analysis (absorbed from former Path B)
    // These carry instance-specific signals: char composition, separator presence.
    // EXCLUDED: dims 8-9 (text length) — representation artifact, not semantic signal.
    else if (dim_idx >= 48 && dim_idx < 64 && text_features && atom_mask) {
        if (atom_mask[token_pos] != 0) {
            int feat_idx = dim_idx - 48;
            if (feat_idx != 8 && feat_idx != 9) {  // Skip length dims
                float feat = __half2float(*reinterpret_cast<const __half*>(
                    &text_features[token_pos * kTextFeatureDim + feat_idx]));
                value += feat;  // Additive — learned type embedding in these dims acts as bias
            }
        }
    }
    // Dims 0-15: pure learned type embedding

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
    // NOTE: Text features are now merged INTO atom embeddings (dims 48-63)
    // during forward. cached_atom_embeddings already contains the merged signal.
    // No separate text feature capture needed.
}

void ScratchBlockGradFn::capture_weights(
    Tensor& atom_proj, Tensor& atom_type_emb)
{
    atom_proj.ensure_grad();
    atom_type_emb.ensure_grad();

    atom_projection_data         = atom_proj.data;
    atom_projection_grad         = atom_proj.grad_data();
    atom_type_embeddings_grad    = atom_type_emb.grad_data();

    if (!atom_projection_data)         throw std::runtime_error("ScratchBlockGradFn::capture_weights: atom_projection.data is NULL");
    if (!atom_projection_grad)         throw std::runtime_error("ScratchBlockGradFn::capture_weights: atom_projection.grad is NULL");
    if (!atom_type_embeddings_grad)    throw std::runtime_error("ScratchBlockGradFn::capture_weights: atom_type_embeddings.grad is NULL");
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
    //  Backward Step 2: Chain to input (additive injection → identity gradient)
    // ═══════════════════════════════════════════════════════════════════════════
    // NOTE: Text feature backward (former Step 2) ELIMINATED — text features are
    // now absorbed into atom embeddings (dims 48-63). Their gradients flow through
    // kernelBackwardAtomEmbeddings → atom_projection + atom_type_embeddings.
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
    const uint16_t* text_features,
    const uint8_t* atom_mask,
    const uint32_t* atom_flags,
    int total_tokens,
    cudaStream_t stream)
{
    const auto& cfg = layer.config();

    // Rule 20: No fallbacks
    if (!input.data) throw std::runtime_error("scratch_block_inject: input.data is NULL");
    if (!token_ids)  throw std::runtime_error("scratch_block_inject: token_ids is NULL");
    if (!stream)     throw std::runtime_error("scratch_block_inject: stream is NULL");
    if (!numeric_values) {
        throw std::runtime_error("scratch_block_inject: numeric_values is NULL");
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
        token_ids, numeric_values,
        text_features, atom_mask, atom_flags, stream);

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
            layer.atomTypeEmbeddings());

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

    TensorContract::TensorShape atom_emb_shape(TensorContract::Layout::BSM, atom_emb_2d);
    TensorContract::TensorShape proj_shape(TensorContract::Layout::BSM, proj_2d);

    atom_type_embeddings_ = Tensor::zeros(atom_emb_shape, true, config_.stream, "atom_type_embeddings");
    atom_type_embeddings_.ensure_grad();
    atom_projection_ = Tensor::zeros(proj_shape, true, config_.stream, "atom_projection");
    atom_projection_.ensure_grad();

    // Temporary buffers for forward (reused across calls, NOT cached for backward — GradFn owns that)
    cudaMalloc(&d_atom_positions_, config_.max_atoms * sizeof(int));
    cudaMalloc(&d_num_atoms_, sizeof(int));
    cudaMalloc(&d_atom_embeddings_, static_cast<size_t>(config_.max_atoms) * config_.atom_embedding_dim * sizeof(float));

    weights_allocated_ = true;
}

void ScratchBlockLayer::freeWeights() {
    atom_type_embeddings_ = Tensor();
    atom_projection_ = Tensor();

    if (d_atom_positions_)  cudaFree(d_atom_positions_);
    if (d_num_atoms_)       cudaFree(d_num_atoms_);
    if (d_atom_embeddings_) cudaFree(d_atom_embeddings_);

    d_atom_positions_ = nullptr;
    d_num_atoms_ = nullptr;
    d_atom_embeddings_ = nullptr;
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

    logWeightInit();
}

void ScratchBlockLayer::runForwardKernels(
    float* output, int total_tokens,
    const int* token_ids,
    const float* numeric_values,
    const uint16_t* text_features, const uint8_t* atom_mask,
    const uint32_t* atom_flags,
    cudaStream_t stream)
{
    // Step 1: Detect atom tokens
    cudaMemsetAsync(d_num_atoms_, 0, sizeof(int), stream);

    const int detect_block = 256;
    const int detect_grid = total_tokens > 0 ? (total_tokens + detect_block - 1) / detect_block : 0;
    if (detect_grid > 0) {
        kernelDetectAtomTokens<<<detect_grid, detect_block, 0, stream>>>(
            token_ids, total_tokens,
            d_atom_positions_, d_num_atoms_,
            config_.max_atoms,
            config_.atom_token_start, config_.atom_token_end);
    }

    const int atom_blocks = std::min(config_.max_atoms, total_tokens);
    if (atom_blocks <= 0) return;

    // Step 2: Lookup atom embeddings (unified: numeric_values + atom_flags + text_features)
    kernelLookupAtomEmbeddingsWithValue<<<atom_blocks, config_.atom_embedding_dim, 0, stream>>>(
        token_ids, d_atom_positions_, d_num_atoms_, config_.max_atoms,
        atom_type_embeddings_.data,
        numeric_values,
        atom_flags,
        text_features, atom_mask,
        d_atom_embeddings_, config_.atom_embedding_dim);

    // Step 3: Project and inject atom embeddings (single unified projection)
    kernelInjectAtomEmbeddings<<<atom_blocks, config_.d_model, 0, stream>>>(
        output, d_atom_positions_, d_num_atoms_, config_.max_atoms,
        d_atom_embeddings_, atom_projection_.data,
        config_.atom_embedding_dim, config_.d_model, config_.atom_scale);
    // NOTE: Text features are now absorbed into dims 48-63 of atom embeddings above.
    // The separate text feature injection path (Path B) has been eliminated.
}

//======================================================//
//  Logging
//======================================================//

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
    const size_t total_bytes = (atom_emb_size + proj_size) * sizeof(float) * 2;

    std::ostringstream oss;
    oss << "weights_init: atom_types=" << NUM_ATOM_TYPES
        << " atom_emb_dim=" << config_.atom_embedding_dim
        << " d_model=" << config_.d_model
        << " total_params=" << (atom_emb_size + proj_size)
        << " memory_mb=" << std::fixed << std::setprecision(2) << (total_bytes / (1024.0 * 1024.0));
    Logging::EmitModuleInfo(kScratchBlockModule, oss.str(), global_step_);
}

} // namespace GRIM

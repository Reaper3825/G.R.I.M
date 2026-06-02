//======================================================//
//  ScratchBlockReasoning_GPU.cu
//  Implementation of Internal Reasoning Layer
//
//  Rewritten Feb 2026: Proper autograd node.
//  - ScratchBlockGradFn owns all tape-local caches
//  - autograd::scratch_block_inject() remains the additive entry point
//  - autograd::scratch_block_project_all_tokens() is a legacy removal target
//  - Semantic atom data is authored before Phase2 on BatchPayload/Bindings
//  - ScratchBlockLayer device buffers are transient staging only, not the
//    semantic owner of atom position/type/value facts
//  - Registry-owned parameter initialization owned by ParameterGroupRegistration
//======================================================//

#include "ScratchBlockReasoning_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <string>

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM {

//======================================================//
//  Constants
//======================================================//

static constexpr const char* kScratchBlockModule = "ScratchBlock";
constexpr int ATOM_TOKEN_START = HyperParameters::ATOM_TOKEN_START;
constexpr int NUM_ATOM_TYPES   = GRIM::Tokenizer::kAtomTypeCount;

namespace {

void validateScratchBlockBatchInputs(
    const char* caller,
    const HyperParameters::ScratchBlockConstructionHP& hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream)
{
    if (!hp.enabled) {
        throw std::runtime_error(std::string(caller) + ": ScratchBlockConstructionHP.enabled=false");
    }
    payload.validate(caller);
    if (!bindings.d_numeric_values) {
        throw std::runtime_error(std::string(caller) + ": BatchDeviceBindings.d_numeric_values is NULL");
    }
    if (!bindings.d_atom_flags) {
        throw std::runtime_error(std::string(caller) + ": BatchDeviceBindings.d_atom_flags is NULL");
    }
    if (!bindings.d_token_to_slot_map) {
        throw std::runtime_error(std::string(caller) + ": BatchDeviceBindings.d_token_to_slot_map is NULL");
    }
    if (!bindings.d_atom_positions) {
        throw std::runtime_error(std::string(caller) + ": BatchDeviceBindings.d_atom_positions is NULL");
    }
    if (!bindings.d_atom_types) {
        throw std::runtime_error(std::string(caller) + ": BatchDeviceBindings.d_atom_types is NULL");
    }
    if (!stream) {
        throw std::runtime_error(std::string(caller) + ": stream is NULL");
    }
    if (payload.total_tokens <= 0) {
        throw std::runtime_error(std::string(caller) + ": payload.total_tokens must be positive");
    }
    if (payload.authoredAtomCount() > hp.max_atoms) {
        throw std::runtime_error(
            std::string(caller) + ": payload.authoredAtomCount()=" +
            std::to_string(payload.authoredAtomCount()) +
            " exceeds ScratchBlockConstructionHP.max_atoms=" +
            std::to_string(hp.max_atoms));
    }
}

void validateScratchBlockParameterTensors(
    const GRIM::ScratchBlockParameterTensors& scratch_parameters,
    const char* caller)
{
    if (!scratch_parameters.atom_type_embeddings.data) {
        throw std::runtime_error(std::string(caller) + ": atom_type_embeddings.data is NULL");
    }
    if (!scratch_parameters.atom_projection.data) {
        throw std::runtime_error(std::string(caller) + ": atom_projection.data is NULL");
    }
    if (!scratch_parameters.structured_gate_weight.data) {
        throw std::runtime_error(std::string(caller) + ": structured_gate_weight.data is NULL");
    }
}

} // namespace

//======================================================//
//  CUDA Kernels — Forward
//======================================================//

// Value-aware atom embedding lookup (sinusoidal+log basis in dims 16-47, learned type embedding elsewhere).
// UNIFIED: Uses numeric_values + atom_flags for atom metadata.
// numeric_values carries AtomTable packed values; atom_flags carries type-specific metadata;
// these signals are combined into a single atom embedding vector.
__global__ void kernelLookupAtomEmbeddingsWithValue(
    const int* __restrict__ atom_positions,
    int num_atoms,
    const int* __restrict__ atom_types,
    const float* __restrict__ atom_type_embeddings,
    const float* __restrict__ token_numeric_values,
    const uint32_t* __restrict__ token_atom_flags,
    const int32_t* __restrict__ token_to_slot_map,
    float* __restrict__ atom_embeddings,
    int atom_embedding_dim,
    int execution_first_type_only
) {
    const int atom_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;
    if (atom_idx >= num_atoms || dim_idx >= atom_embedding_dim) return;

    int token_pos = atom_positions[atom_idx];
    int atom_type = atom_types[atom_idx];
    if (atom_type < 0 || atom_type >= NUM_ATOM_TYPES) return;

    // Execution-first training: atom placeholders must not leak value OR type
    // into the residual stream. Emit an exact zero structured vector here.
    if (execution_first_type_only) {
        atom_embeddings[atom_idx * atom_embedding_dim + dim_idx] = 0.0f;
        return;
    }

    // Slot-bound tokens: numeric truth lives in M.values, not in literal embeddings.
    // Suppress literal numeric injection (has_value = false) for slot-bound positions.
    int slot_id = (token_to_slot_map != nullptr) ? token_to_slot_map[token_pos] : -1;
    bool is_slot_bound = (slot_id >= 0);

    float numeric_val = token_numeric_values ? token_numeric_values[token_pos] : 0.0f;
    uint32_t flags = token_atom_flags ? token_atom_flags[token_pos] : 0u;
    bool has_value = !is_slot_bound && isfinite(numeric_val);
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
    // Dims outside explicit numeric/flag bands remain pure learned type embedding.

    atom_embeddings[atom_idx * atom_embedding_dim + dim_idx] = value;
}

// Project atom embeddings and inject into hidden states
__global__ void kernelInjectAtomEmbeddings(
    float* __restrict__ hidden_states,
    const int* __restrict__ atom_positions,
    int num_atoms,
    const float* __restrict__ atom_embeddings,
    const float* __restrict__ projection,
    int atom_embedding_dim,
    int d_model,
    float scale
) {
    const int atom_idx = blockIdx.x;
    const int d_idx = threadIdx.x;
    if (atom_idx >= num_atoms || d_idx >= d_model) return;

    int token_pos = atom_positions[atom_idx];
    float sum = 0.0f;
    for (int k = 0; k < atom_embedding_dim; ++k) {
        sum += atom_embeddings[atom_idx * atom_embedding_dim + k] *
               projection[k * d_model + d_idx];
    }
    hidden_states[token_pos * d_model + d_idx] += scale * sum;
}

// Project atom embeddings into a full-token z tensor. z is pre-zeroed; only
// structured/atom rows are written. This makes all-token gating shape-stable
// while preserving exact no-op behavior for ordinary tokens.
__global__ void kernelProjectAtomEmbeddingsAllTokens(
    float* __restrict__ structured_state,
    const int* __restrict__ atom_positions,
    int num_atoms,
    const float* __restrict__ atom_embeddings,
    const float* __restrict__ projection,
    int atom_embedding_dim,
    int d_model,
    float scale
) {
    const int atom_idx = blockIdx.x;
    const int d_idx = threadIdx.x;
    if (atom_idx >= num_atoms || d_idx >= d_model) return;

    const int token_pos = atom_positions[atom_idx];
    float sum = 0.0f;
    for (int k = 0; k < atom_embedding_dim; ++k) {
        sum += atom_embeddings[atom_idx * atom_embedding_dim + k] *
               projection[k * d_model + d_idx];
    }
    structured_state[token_pos * d_model + d_idx] = scale * sum;
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
//  ScratchBlockGradFn Implementation
//======================================================//

ScratchBlockGradFn::~ScratchBlockGradFn() {
    if (cached_atom_embeddings)   cudaFree(cached_atom_embeddings);
    if (cached_atom_positions)    cudaFree(cached_atom_positions);
    if (cached_atom_types)        cudaFree(cached_atom_types);
    if (d_grad_atom_embeddings)   cudaFree(d_grad_atom_embeddings);
}

struct ScratchBlockProjectionGradFn : public GradFn {
    float* cached_atom_embeddings = nullptr;
    int* cached_atom_positions = nullptr;
    int* cached_atom_types = nullptr;
    int num_atoms_captured = 0;
    int atom_embedding_dim = 0;
    int d_model = 0;
    int max_atoms = 0;
    float atom_scale = 1.0f;

    float* atom_projection_data = nullptr;
    float* atom_projection_grad = nullptr;
    float* atom_type_embeddings_grad = nullptr;
    float* d_grad_atom_embeddings = nullptr;
    bool accumulate_atom_type_gradients = true;

    ScratchBlockProjectionGradFn() { op_name = "scratch_block_project_all_tokens"; }
    ~ScratchBlockProjectionGradFn() override { release_saved(); }

    void capture_weights(Tensor& atom_proj, Tensor& atom_type_emb) {
        atom_proj.ensure_grad();
        atom_type_emb.ensure_grad();
        atom_projection_data = atom_proj.data;
        atom_projection_grad = atom_proj.grad_data();
        atom_type_embeddings_grad = atom_type_emb.grad_data();
        if (!atom_projection_data) throw std::runtime_error("ScratchBlockProjectionGradFn::capture_weights: atom_projection.data is NULL");
        if (!atom_projection_grad) throw std::runtime_error("ScratchBlockProjectionGradFn::capture_weights: atom_projection.grad is NULL");
        if (!atom_type_embeddings_grad) throw std::runtime_error("ScratchBlockProjectionGradFn::capture_weights: atom_type_embeddings.grad is NULL");
    }

    void capture_forward(const int* atom_positions_src,
                         const int* atom_types_src,
                         const float* atom_embeddings_src,
                         int num_atoms,
                         cudaStream_t stream) {
        num_atoms_captured = num_atoms;
        if (num_atoms <= 0) return;

        const size_t pos_bytes = static_cast<size_t>(num_atoms) * sizeof(int);
        const size_t emb_bytes = static_cast<size_t>(num_atoms) * atom_embedding_dim * sizeof(float);

        cudaMallocOrThrow(reinterpret_cast<void**>(&cached_atom_positions), pos_bytes, "ScratchBlockProjectionGradFn_atom_positions");
        cudaMemcpyAsync(cached_atom_positions, atom_positions_src, pos_bytes, cudaMemcpyDeviceToDevice, stream);

        cudaMallocOrThrow(reinterpret_cast<void**>(&cached_atom_types), pos_bytes, "ScratchBlockProjectionGradFn_atom_types");
        cudaMemcpyAsync(cached_atom_types, atom_types_src, pos_bytes, cudaMemcpyDeviceToDevice, stream);

        cudaMallocOrThrow(reinterpret_cast<void**>(&cached_atom_embeddings), emb_bytes, "ScratchBlockProjectionGradFn_atom_embeddings");
        cudaMemcpyAsync(cached_atom_embeddings, atom_embeddings_src, emb_bytes, cudaMemcpyDeviceToDevice, stream);

        cudaMallocOrThrow(reinterpret_cast<void**>(&d_grad_atom_embeddings),
                          static_cast<size_t>(max_atoms) * atom_embedding_dim * sizeof(float),
                          "ScratchBlockProjectionGradFn_grad_atom_embeddings");
    }

    void apply_impl(const Tensor& grad_output, cudaStream_t stream) override {
        setCurrentGradFnOp("scratch_block_project_all_tokens", this);
        if (applied) return;
        applied = true;
        if (!grad_output.data) {
            throw std::runtime_error("ScratchBlockProjectionGradFn::apply: grad_output.data is NULL");
        }
        if (num_atoms_captured <= 0) return;
        if (!cached_atom_embeddings || !cached_atom_positions || !cached_atom_types) {
            throw std::runtime_error("ScratchBlockProjectionGradFn::apply: cached atom buffers are NULL");
        }

        const size_t scratch_bytes = static_cast<size_t>(num_atoms_captured) * atom_embedding_dim * sizeof(float);
        cudaMemsetAsync(d_grad_atom_embeddings, 0, scratch_bytes, stream);

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

        if (accumulate_atom_type_gradients) {
            kernelAccumulateAtomTypeGradients<<<num_atoms_captured, block_size, 0, stream>>>(
                d_grad_atom_embeddings,
                cached_atom_types,
                num_atoms_captured,
                atom_type_embeddings_grad,
                atom_embedding_dim);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        if (cached_atom_embeddings) { cudaFree(cached_atom_embeddings); cached_atom_embeddings = nullptr; }
        if (cached_atom_positions) { cudaFree(cached_atom_positions); cached_atom_positions = nullptr; }
        if (cached_atom_types) { cudaFree(cached_atom_types); cached_atom_types = nullptr; }
        if (d_grad_atom_embeddings) { cudaFree(d_grad_atom_embeddings); d_grad_atom_embeddings = nullptr; }
        atom_projection_data = nullptr;
        atom_projection_grad = nullptr;
        atom_type_embeddings_grad = nullptr;
    }
};

void ScratchBlockGradFn::capture_input(Tensor& x) {
    input_shape = x.shape;
    input_grad_fn = x.grad_fn;
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

        cudaMallocOrThrow(reinterpret_cast<void**>(&cached_atom_positions), pos_bytes, "ScratchBlockGradFn_atom_positions");
        cudaMemcpyAsync(cached_atom_positions, atom_positions_src, pos_bytes,
                        cudaMemcpyDeviceToDevice, stream);

        cudaMallocOrThrow(reinterpret_cast<void**>(&cached_atom_types), type_bytes, "ScratchBlockGradFn_atom_types");
        cudaMemcpyAsync(cached_atom_types, atom_types_src, type_bytes,
                        cudaMemcpyDeviceToDevice, stream);

        cudaMallocOrThrow(reinterpret_cast<void**>(&cached_atom_embeddings), emb_bytes, "ScratchBlockGradFn_atom_embeddings");
        cudaMemcpyAsync(cached_atom_embeddings, atom_embeddings_src, emb_bytes,
                        cudaMemcpyDeviceToDevice, stream);

        // Backward scratch for per-atom gradients
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_grad_atom_embeddings), static_cast<size_t>(max_atoms) * atom_embedding_dim * sizeof(float), "ScratchBlockGradFn_grad_atom_emb");
    }
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

void ScratchBlockGradFn::apply_impl(const Tensor& grad_output, cudaStream_t stream) {
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
        if (accumulate_atom_type_gradients && cached_atom_types && atom_type_embeddings_grad) {
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
    input_grad_fn.reset();
}

//======================================================//
//  autograd::scratch_block_inject() — Entry Point
//======================================================//

namespace autograd {

Tensor scratch_block_inject(
    Tensor& input,
    ScratchBlockLayer& layer,
    GRIM::ScratchBlockParameterTensors& scratch_parameters,
    const HyperParameters::ScratchBlockConstructionHP& hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream,
    bool execution_first_type_only)
{
    validateScratchBlockBatchInputs(
        "scratch_block_inject",
        hp,
        payload,
        bindings,
        stream);
    validateScratchBlockParameterTensors(scratch_parameters, "scratch_block_inject");

    if (!input.data) throw std::runtime_error("scratch_block_inject: input.data is NULL");

    // Create output tensor (copy of input — injection is additive in-place)
    const size_t data_bytes = static_cast<size_t>(payload.total_tokens) * hp.d_model * sizeof(float);
    Tensor output;
    cudaMallocOrThrow(reinterpret_cast<void**>(&output.data), data_bytes, "scratch_block_inject_output");
    cudaMemcpyAsync(output.data, input.data, data_bytes, cudaMemcpyDeviceToDevice, stream);
    output.shape = input.shape;
    output.owns_data = true;
    output.requires_grad = input.requires_grad;
    output.is_leaf = false;
    output.stream = stream;

    // Run forward kernels on output buffer
    layer.runForwardKernels(
        output.data,
        scratch_parameters,
        hp,
        payload,
        bindings,
        stream,
        execution_first_type_only);

    // Build GradFn for backward pass
    if (input.requires_grad) {
        auto grad_fn = std::make_shared<ScratchBlockGradFn>();
        grad_fn->atom_embedding_dim = hp.atom_embedding_dim;
        grad_fn->d_model = hp.d_model;
        grad_fn->max_atoms = hp.max_atoms;
        grad_fn->atom_scale = hp.atom_scale;
        grad_fn->accumulate_atom_type_gradients = !execution_first_type_only;

        // Capture input chain
        grad_fn->capture_input(input);

        // Capture registry-owned parameter gradient pointers
        grad_fn->capture_weights(
            scratch_parameters.atom_projection,
            scratch_parameters.atom_type_embeddings);

        const int num_atoms_host = payload.authoredAtomCount();

        // Capture forward activations (OWNED copies in GradFn)
        grad_fn->capture_forward(
            bindings.d_atom_positions,
            bindings.d_atom_types,
            layer.atomEmbeddingScratchBuffer(),
            num_atoms_host,
            payload.total_tokens, stream);

        output.grad_fn = std::move(grad_fn);
    }

    return output;
}

// Legacy removal target. Keep compiled only until the experiment boundary is
// fully deleted; shared forward must not add new callers.
Tensor scratch_block_project_all_tokens(
    ScratchBlockLayer& layer,
    GRIM::ScratchBlockParameterTensors& scratch_parameters,
    const HyperParameters::ScratchBlockConstructionHP& hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream,
    bool execution_first_type_only,
    bool connect_parameter_graph,
    const ScratchBlockProjectionParameterViews* parameter_views)
{
    validateScratchBlockBatchInputs(
        "scratch_block_project_all_tokens",
        hp,
        payload,
        bindings,
        stream);
    validateScratchBlockParameterTensors(scratch_parameters, "scratch_block_project_all_tokens");
    if (connect_parameter_graph && parameter_views) {
        throw std::runtime_error(
            "scratch_block_project_all_tokens: connect_parameter_graph=true cannot use explicit read-only parameter views");
    }

    const Tensor& atom_type_embeddings =
        (parameter_views && parameter_views->atom_type_embeddings)
            ? *parameter_views->atom_type_embeddings
            : scratch_parameters.atom_type_embeddings;
    const Tensor& atom_projection =
        (parameter_views && parameter_views->atom_projection)
            ? *parameter_views->atom_projection
            : scratch_parameters.atom_projection;

    if (!atom_projection.data || !atom_type_embeddings.data) {
        throw std::runtime_error("scratch_block_project_all_tokens: ScratchBlock weights are not allocated");
    }

    Tensor structured_state = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(payload.total_tokens, hp.d_model),
        connect_parameter_graph,
        stream,
        "scratch_structured_state_z");

    const int atom_blocks = payload.authoredAtomCount();
    if (atom_blocks > 0) {
        const int type_only_flag = execution_first_type_only ? 1 : 0;
        kernelLookupAtomEmbeddingsWithValue<<<atom_blocks, hp.atom_embedding_dim, 0, stream>>>(
            bindings.d_atom_positions,
            atom_blocks,
            bindings.d_atom_types,
            atom_type_embeddings.data,
            bindings.d_numeric_values,
            bindings.d_atom_flags,
            bindings.d_token_to_slot_map,
            layer.atomEmbeddingScratchBuffer(),
            hp.atom_embedding_dim,
            type_only_flag);

        kernelProjectAtomEmbeddingsAllTokens<<<atom_blocks, hp.d_model, 0, stream>>>(
            structured_state.data,
            bindings.d_atom_positions,
            atom_blocks,
            layer.atomEmbeddingScratchBuffer(),
            atom_projection.data,
            hp.atom_embedding_dim,
            hp.d_model,
            hp.atom_scale);
    }

    if (connect_parameter_graph && (atom_projection.requires_grad || atom_type_embeddings.requires_grad)) {
        auto grad_fn = std::make_shared<ScratchBlockProjectionGradFn>();
        grad_fn->atom_embedding_dim = hp.atom_embedding_dim;
        grad_fn->d_model = hp.d_model;
        grad_fn->max_atoms = hp.max_atoms;
        grad_fn->atom_scale = hp.atom_scale;
        grad_fn->accumulate_atom_type_gradients = !execution_first_type_only;
        grad_fn->capture_weights(scratch_parameters.atom_projection, scratch_parameters.atom_type_embeddings);

        const int num_atoms_host = payload.authoredAtomCount();

        grad_fn->capture_forward(
            bindings.d_atom_positions,
            bindings.d_atom_types,
            layer.atomEmbeddingScratchBuffer(),
            num_atoms_host,
            stream);
        structured_state.is_leaf = false;
        structured_state.grad_fn = std::move(grad_fn);
    }

    return structured_state;
}

}  // namespace autograd

//======================================================//
//  ScratchBlockLayer Implementation
//======================================================//

ScratchBlockLayer::ScratchBlockLayer(
    const HyperParameters::ScratchBlockConstructionHP& hp,
    cudaStream_t init_stream)
{
    if (hp.enabled) {
        if (!init_stream) {
            throw std::runtime_error("ScratchBlockLayer: init_stream is NULL — startup model assembly must provide a construction stream");
        }
        allocateRuntimeBuffers(hp, init_stream);
    }
}

ScratchBlockLayer::~ScratchBlockLayer() {
    releaseRuntimeBuffers();
}

ScratchBlockLayer::ScratchBlockLayer(ScratchBlockLayer&& other) noexcept
    : d_atom_embeddings_(other.d_atom_embeddings_)
{
    other.d_atom_embeddings_ = nullptr;
}

ScratchBlockLayer& ScratchBlockLayer::operator=(ScratchBlockLayer&& other) noexcept {
    if (this != &other) {
        releaseRuntimeBuffers();

        d_atom_embeddings_ = other.d_atom_embeddings_;

        other.d_atom_embeddings_ = nullptr;
    }
    return *this;
}

void ScratchBlockLayer::allocateRuntimeBuffers(
    const HyperParameters::ScratchBlockConstructionHP& hp,
    cudaStream_t init_stream) {
    if (d_atom_embeddings_) {
        throw std::runtime_error("ScratchBlockLayer::allocateRuntimeBuffers: runtime buffers already allocated");
    }
    StreamController::fatalIfDefaultStream(init_stream, "ScratchBlockLayer::allocateRuntimeBuffers");

    // Temporary embedding scratch for forward (reused across calls, NOT cached
    // for backward — GradFn owns the tape-local snapshots it needs).
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_atom_embeddings_), static_cast<size_t>(hp.max_atoms) * hp.atom_embedding_dim * sizeof(float), "ScratchBlockLayer_atom_embeddings");
}

void ScratchBlockLayer::releaseRuntimeBuffers() {
    if (d_atom_embeddings_) cudaFree(d_atom_embeddings_);

    d_atom_embeddings_ = nullptr;
}

void ScratchBlockLayer::runForwardKernels(
    float* output,
    const GRIM::ScratchBlockParameterTensors& scratch_parameters,
    const HyperParameters::ScratchBlockConstructionHP& hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream,
    bool execution_first_type_only)
{
    validateScratchBlockBatchInputs(
        "ScratchBlockLayer::runForwardKernels",
        hp,
        payload,
        bindings,
        stream);
    validateScratchBlockParameterTensors(scratch_parameters, "ScratchBlockLayer::runForwardKernels");
    if (!output) {
        throw std::runtime_error("ScratchBlockLayer::runForwardKernels: output is NULL");
    }

    const int atom_blocks = payload.authoredAtomCount();
    if (atom_blocks > hp.max_atoms) {
        throw std::runtime_error(
            "ScratchBlockLayer::runForwardKernels: payload authoredAtomCount=" +
            std::to_string(atom_blocks) + " exceeds hp.max_atoms=" +
            std::to_string(hp.max_atoms));
    }
    if (atom_blocks <= 0) return;

    // Step 1: Lookup atom embeddings from authored compact atom data.
    const int type_only_flag = execution_first_type_only ? 1 : 0;
    kernelLookupAtomEmbeddingsWithValue<<<atom_blocks, hp.atom_embedding_dim, 0, stream>>>(
        bindings.d_atom_positions,
        atom_blocks,
        bindings.d_atom_types,
        scratch_parameters.atom_type_embeddings.data,
        bindings.d_numeric_values,
        bindings.d_atom_flags,
        bindings.d_token_to_slot_map,
        d_atom_embeddings_, hp.atom_embedding_dim, type_only_flag);

    // Step 2: Project and inject authored atom embeddings.
    kernelInjectAtomEmbeddings<<<atom_blocks, hp.d_model, 0, stream>>>(
        output, bindings.d_atom_positions, atom_blocks,
        d_atom_embeddings_, scratch_parameters.atom_projection.data,
        hp.atom_embedding_dim, hp.d_model, hp.atom_scale);
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
    oss << " duration_ms=" << std::fixed << std::setprecision(3) << duration_ms;
    Logging::EmitModuleInfo(kScratchBlockModule, oss.str(), global_step_);
}

} // namespace GRIM

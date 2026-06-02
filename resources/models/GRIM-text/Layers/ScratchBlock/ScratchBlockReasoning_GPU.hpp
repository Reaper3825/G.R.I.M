//======================================================//
//  ScratchBlockReasoning_GPU.hpp
//  Internal Reasoning Layer for GRIM-text
//
//  Proper autograd node: forward returns Tensor with
//  ScratchBlockGradFn attached. GradFn owns all tape-local
//  caches internally. Semantic atom data is authored on the
//  prepared BatchPayload/BatchDeviceBindings boundary, not on
//  ScratchBlockLayer.
//
//  Forward:  output = input + scale * project(atom_emb)
//  Backward: grad_input = grad_output (additive identity)
//            + parameter gradients for projection/embeddings when the
//              forward path actually depends on them
//
//  Author: GRIM Team
//  Date: December 2025, Rewritten February 2026
//======================================================//

#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "../grim_layer_gpu.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

namespace GRIM {

struct ScratchBlockProjectionParameterViews {
    const Tensor* atom_type_embeddings = nullptr;
    const Tensor* atom_projection = nullptr;
};

//======================================================//
//  ScratchBlockGradFn — Proper autograd backward node
//
//  Owns all cached activation data needed for backward.
//  Follows BiasAddGradFn/DropoutGradFn ownership pattern:
//  - Allocates own buffers in capture()
//  - Destructor frees all owned buffers
//  - Chains to input_grad_fn
//======================================================//
struct ScratchBlockGradFn : public GradFn {
    //--- Cached forward activations (OWNED — freed in destructor) ---
    float* cached_atom_embeddings = nullptr;  // [num_atoms_captured, atom_embedding_dim]
    int*   cached_atom_positions  = nullptr;  // [num_atoms_captured]
    int*   cached_atom_types      = nullptr;  // [num_atoms_captured]
    int    num_atoms_captured     = 0;

    //--- Geometry ---
    int total_tokens     = 0;
    int atom_embedding_dim = 0;
    int d_model          = 0;
    int max_atoms        = 0;
    float atom_scale     = 0.1f;

    //--- References to registry-owned trainable tensors (NOT owned — registry outlives GradFn) ---
    float* atom_projection_data         = nullptr;  // [atom_embedding_dim, d_model]
    float* atom_projection_grad         = nullptr;
    float* atom_type_embeddings_grad    = nullptr;  // [NUM_ATOM_TYPES, atom_embedding_dim]
    bool accumulate_atom_type_gradients = true;

    //--- Temporary backward scratch (OWNED) ---
    float* d_grad_atom_embeddings = nullptr;  // [max_atoms, atom_embedding_dim]

    //--- Input gradient chain ---
    TensorContract::TensorShape input_shape;
    std::shared_ptr<GradFn> input_grad_fn;

    ScratchBlockGradFn() { op_name = "scratch_block"; }

    ~ScratchBlockGradFn() override;

    /// Capture input tensor state for backward chain (Issue #48: stable data, not Tensor*)
    void capture_input(Tensor& input);

    /// Capture forward activations: atom positions, types, embeddings
    void capture_forward(
        const int* atom_positions, const int* atom_types,
        const float* atom_embeddings, int num_atoms,
        int total_tokens, cudaStream_t stream);

    /// Capture references to registry-owned parameter gradient buffers
    void capture_weights(Tensor& atom_proj, Tensor& atom_type_emb);

    void apply_impl(const Tensor& grad_output, cudaStream_t stream) override;

    void release_saved() override;
};

//======================================================//
//  ScratchBlock Layer
//======================================================//
class ScratchBlockLayer final : public Layer<ScratchBlockLayer, float> {
public:
    static constexpr LayerType layer_type = LayerType::kUnknown;

    ScratchBlockLayer(const HyperParameters::ScratchBlockConstructionHP& hp,
                      cudaStream_t init_stream);
    ~ScratchBlockLayer();

    // Disable copy
    ScratchBlockLayer(const ScratchBlockLayer&) = delete;
    ScratchBlockLayer& operator=(const ScratchBlockLayer&) = delete;

    // Move support
    ScratchBlockLayer(ScratchBlockLayer&&) noexcept;
    ScratchBlockLayer& operator=(ScratchBlockLayer&&) noexcept;

    // Logging control
    void setLoggingEnabled(bool enabled) { logging_enabled_ = enabled; }
    bool isLoggingEnabled() const noexcept { return logging_enabled_; }
    void setGlobalStep(std::uint64_t step) { global_step_ = step; }
    std::uint64_t globalStep() const noexcept { return global_step_; }

    //--------------------------------------------------//
    // Internal scratch access (for autograd::scratch_block_inject)
    //
    // This is transient forward staging only. Semantic atom position/type/value
    // data is authored before Phase2 on BatchPayload/BatchDeviceBindings.
    //--------------------------------------------------//

    float* atomEmbeddingScratchBuffer() { return d_atom_embeddings_; }

    /// Run forward CUDA kernels (embed lookup, projection+inject) using the
    /// explicit grouped ScratchBlock HP and active batch bindings.
    /// Modifies output in-place while consuming authored atom positions/types
    /// from the payload/bindings boundary.
    void runForwardKernels(
        float* output,
        const GRIM::ScratchBlockParameterTensors& scratch_parameters,
        const HyperParameters::ScratchBlockConstructionHP& hp,
        const Batching::BatchPayload& payload,
        const Batching::BatchDeviceBindings& bindings,
        cudaStream_t stream,
        bool execution_first_type_only = false);

private:
    void allocateRuntimeBuffers(const HyperParameters::ScratchBlockConstructionHP& hp,
                                cudaStream_t init_stream);
    void releaseRuntimeBuffers();

    // Temporary forward scratch (reused across calls). Authored compact atom
    // positions/types live on BatchPayload/BatchDeviceBindings, not here.
    float* d_atom_embeddings_ = nullptr;  // [max_atoms, atom_embedding_dim]

    bool logging_enabled_   = false;
    std::uint64_t global_step_ = 0;

    // Logging helpers
    void logForward(int num_atoms, float duration_ms);
};

//======================================================//
//  Autograd Entry Point
//======================================================//
namespace autograd {

/// Inject ScratchBlock atom/text embeddings into token representations.
/// Returns NEW Tensor with ScratchBlockGradFn attached to autograd graph.
///
/// Forward:  output[t] = input[t] + scale * project(atom_emb[t])
///   atom_emb includes learned type features plus numeric value and atom flags,
///   except execution-first placeholder mode where atom_emb is exactly zero.
/// Backward: grad_input = grad_output (additive identity), plus parameter gradients
///
/// @param input          Embedding tensor [total_tokens, d_model] with grad_fn chain
/// @param layer          ScratchBlock runtime shell (owns reusable runtime buffers)
/// @param hp             Explicit grouped ScratchBlock construction/view payload
/// @param payload        Caller-owned active batch payload
/// @param bindings       Caller-owned device bindings for payload
/// @param stream         CUDA stream
Tensor scratch_block_inject(
    Tensor& input,
    ScratchBlockLayer& layer,
    GRIM::ScratchBlockParameterTensors& scratch_parameters,
    const HyperParameters::ScratchBlockConstructionHP& hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream,
    bool execution_first_type_only = false);

/// Legacy removal target. Build a full-token structured state tensor
/// z [total_tokens, d_model]. Shared forward must not add new callers while
/// experiments keep ScratchBlock invocation disabled.
Tensor scratch_block_project_all_tokens(
    ScratchBlockLayer& layer,
    GRIM::ScratchBlockParameterTensors& scratch_parameters,
    const HyperParameters::ScratchBlockConstructionHP& hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    cudaStream_t stream,
    bool execution_first_type_only = false,
    bool connect_parameter_graph = true,
    const ScratchBlockProjectionParameterViews* parameter_views = nullptr);

}  // namespace autograd

} // namespace GRIM

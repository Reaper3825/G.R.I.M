//======================================================//
//  LM Head Layer - GPU (Pattern B: Layer Ownership)
//  Linear projection from hidden states to vocabulary logits
//
//  Owns: weights [vocab_size, d_model], bias [vocab_size] (optional),
//        final_rms_gamma [d_model] (pre-LM-head normalization).
//
//  Architecture: logits = projected(centered(RMSNorm(encoder_output))) @ W^T + bias
//  Where W is either tied to embedding weights or independently allocated.
//
//  Backward is handled automatically by the autograd tape system:
//    grad_W = centered^T @ grad_logits
//    grad_input = grad_logits @ W  (flows back through centering + RMSNorm ops)
//    grad_bias = sum(grad_logits, dim=0)
//    grad_gamma via RMSNormGradFn
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <cstdint>

#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {

//======================================================//
//  LMHeadLayer - Self-Allocating (Pattern B: Layer Ownership)
//
//  Owns its own weights (allocated in constructor) OR aliases
//  embedding weights when tie_embeddings=true (Issue #60).
//  Also owns final_rms_gamma for pre-LM-head normalization.
//
//  forward() applies: RMSNorm → optional centering → optional PC1 projection → matmul → bias
//  backward() handled by autograd chain (RMSNormGradFn → MatMulGradFn etc.)
//======================================================//

class LMHeadLayer {
public:
    // Rule 20: Default constructor deleted
    LMHeadLayer() = delete;

    /// Self-allocating constructor (Pattern B - Layer Ownership)
    ///
    /// When tied_embedding_weights is non-null, weights alias embedding (Issue #60 weight tying).
    /// When null, weights are independently allocated with Xavier init.
    /// final_rms_gamma is ALWAYS self-allocated (initialized to 1.0).
    ///
    /// @param hp                   Grouped construction hyperparameters
    /// @param seed                  Xavier init seed for independent weights
    /// @param init_stream           CUDA stream for allocation
    /// @param tied_embedding_weights If non-null, weights alias this tensor (from_ptr + share_grad)
    explicit LMHeadLayer(const HyperParameters::LMHeadLayerConstructionHP& hp,
                         uint64_t seed,
                         cudaStream_t init_stream,
                         Tensor* tied_embedding_weights = nullptr);

    ~LMHeadLayer() = default;

    // Non-copyable (GPU resource ownership)
    LMHeadLayer(const LMHeadLayer&) = delete;
    LMHeadLayer& operator=(const LMHeadLayer&) = delete;

    // Allow move
    LMHeadLayer(LMHeadLayer&& other) noexcept;
    LMHeadLayer& operator=(LMHeadLayer&& other) noexcept;

    //--------------------------------------------------
    // Grouped HP snapshot
    //--------------------------------------------------
    const HyperParameters::LMHeadLayerConstructionHP& hp() const noexcept { return hp_; }

    //--------------------------------------------------
    // Weight Accessors (for training/serialization)
    //--------------------------------------------------
    Tensor& weights() { return weights_; }
    Tensor& bias() { return bias_; }

    // ---- final_rms_gamma access ----
    // The READ accessor is always safe (telemetry, save, MTP forward, warn-checks).
    const Tensor& finalRmsGamma() const { return final_rms_gamma_frozen_or_trained_; }

    // The WRITE accessor THROWS if the gamma is configured frozen.
    // Only four legitimate writers exist:
    //   1. LMHeadLayer ctor (uses the private member directly)
    //   2. Startup/Model/ParameterGroupRegistration (register final_rms_gamma)
    //   3. AutogradTraining (zero_grad before backward)
    //   4. grim_model_serialization::load (assignWrite to .data)
    // Any other path that obtains a mutable reference will trip the throw and
    // produce a stack trace pinpointing the leak.
    Tensor& finalRmsGammaMutable_UnfrozenOnly(const char* caller) {
        if (hp_.freeze_final_rms_gamma) {
            throw std::runtime_error(
                std::string("[FROZEN-GAMMA-LEAK] mutable access to final_rms_gamma while "
                            "freeze_final_rms_gamma=true. caller=") +
                (caller ? caller : "<unknown>"));
        }
        return final_rms_gamma_frozen_or_trained_;
    }

    const Tensor& weights() const { return weights_; }
    const Tensor& bias() const { return bias_; }

    /// Whether this layer allocated its own weights (false = tied to embedding)
    bool ownsWeights() const { return owns_weights_; }

    /// Whether weights are initialized and ready for forward pass
    bool weightsReady() const { return weights_.data != nullptr; }

    //--------------------------------------------------
    // Forward Pass - Autograd
    //--------------------------------------------------
    /// LM head forward with autograd tracking:
    ///   0. Optional: RMSNorm(input, final_rms_gamma_frozen_or_trained_) — pre-LM-head normalization
    ///   1. Optional: center_columns_by_sequence_lengths on normalized input (Issue #125/#132)
    ///   2. Optional: project_out_pc1 on the current LM input (composes after centering when both are enabled)
    ///   3. logits = input @ weights^T  (autograd::matmul, transpose_b=true)
    ///   4. Optional: center_rows on logits (numerical stability)
    ///   5. Optional: logits += bias  (autograd::broadcast_add)
    ///
    /// Builds compute graph for automatic backward().
    ///
    /// @param input                    [total_tokens, d_model] - encoder output (MUST have grad_fn if training)
    /// @param out_centered_hidden      Output: centered hidden states (valid only if centering enabled, for diagnostics)
    /// @param d_sequence_lengths       Device [batch_size] real lengths for padding-aware hidden centering
    /// @param batch_size               Number of flattened sequences/samples
    /// @param rows_per_sequence        Padded contiguous rows per sequence/sample in the flattened input
    /// @param stream                   CUDA stream from the caller's forward payload/request
    /// @param cublas_handle            cuBLAS handle from the caller's forward payload/request
    /// @return logits [total_tokens, vocab_size] with grad_fn attached
    Tensor forward(const Tensor& input, Tensor& out_centered_hidden,
                   const int* d_sequence_lengths, int batch_size, int rows_per_sequence,
                   cudaStream_t stream, cublasHandle_t cublas_handle);



private:
    // Immutable grouped read view from HyperparameterGroupings.hpp. This is not
    // a second authored config owner; it is the layer's durable construction HP
    // snapshot needed after startup-local grouping objects go out of scope.
    HyperParameters::LMHeadLayerConstructionHP hp_{};

    // Weight Tensors with autograd (requires_grad=true)
    Tensor weights_;          // [vocab_size, d_model] — owned or aliased from embedding
    Tensor bias_;             // [vocab_size] — optional, always owned
    // Renamed (April 2026) so any stray reference to the old name `final_rms_gamma_`
    // outside this class fails to compile. Combined with the split const/mutable
    // accessors above, this constrains writes to the four declared paths.
    Tensor final_rms_gamma_frozen_or_trained_;  // [d_model] — pre-LM-head RMSNorm gamma, always owned

    // Workspace for the effective LM-head weight matrix. It is always hard
    // token-type gated. When hidden-state centering is enabled, active dims are
    // also row-centered inside the token-type subspace. Held as a member so its
    // data buffer outlives forward() — the matmul GradFn captures W_eff via this
    // tensor and backward must dereference its .data and grad chain after
    // forward() has returned.
    Tensor centered_weights_;

    bool owns_weights_ = true;  // false when tied to embedding weights
};

} // namespace GRIM

#endif // USE_CUDA

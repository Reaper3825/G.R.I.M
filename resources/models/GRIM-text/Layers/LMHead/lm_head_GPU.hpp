//======================================================//
//  LM Head Layer - GPU (registry-owned parameter tensors)
//  Linear projection from hidden states to vocabulary logits
//
//  Borrows: weights [vocab_size, d_model], bias [vocab_size] (optional),
//           final_rms_gamma [d_model] (pre-LM-head normalization).
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
#include "../../Shared/Forward/ModelForwardOutputs.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"

namespace GRIM {

struct LMHeadParameterViews {
    const Tensor* weights = nullptr;
    const Tensor* bias = nullptr;
    const Tensor* final_rms_gamma = nullptr;
};

struct LMHeadParameterTensors {
    Tensor weights;
    Tensor bias;
    Tensor final_rms_gamma;
    bool owns_weights = true;
};

// Local experiment toggle only. Keep this LM-head-local until we decide
// whether token-layout gating should become an authored config field.
//
// IMPORTANT: setting this false disables the LM-head-side hard token-type
// gate without changing embedding lookup, so tied embeddings no longer have
// strict embedding/LM symmetry. That asymmetry is intentional for local
// experiments.
inline constexpr bool kEnableLmHeadTokenTypeGateExperiment = false;

//======================================================//
//  LMHeadLayer - Borrowed parameter owner
//
//  Startup-owned LMHeadParameterTensors hold the durable weights, optional
//  bias, and final_rms_gamma. This layer validates and uses those tensors for
//  forward/backward, but it is not the durable owner.
//
//  forward() applies: RMSNorm → optional centering → optional PC1 projection → matmul → bias
//  backward() handled by autograd chain (RMSNormGradFn → MatMulGradFn etc.)
//======================================================//

class LMHeadLayer {
public:
    // Rule 20: Default constructor deleted
    LMHeadLayer() = delete;

    /// Constructor that populates a startup-owned LM-head tensor owner.
    ///
    /// When tied_embedding_weights is non-null, the registry-owned LM-head
    /// weight tensor aliases embedding (Issue #60 weight tying). When null,
    /// the registry-owned LM-head weight tensor is independently allocated with
    /// Xavier init. final_rms_gamma is always allocated into the supplied
    /// owner and initialized to 1.0.
    ///
    /// @param hp                   Grouped construction hyperparameters
    /// @param parameter_tensors     Durable startup-owned LM-head tensor owner
    /// @param seed                  Xavier init seed for independent weights
    /// @param init_stream           CUDA stream for allocation
    /// @param tied_embedding_weights If non-null, weights alias this tensor (from_ptr + share_grad)
    explicit LMHeadLayer(const HyperParameters::LMHeadLayerConstructionHP& hp,
                         GRIM::LMHeadParameterTensors& parameter_tensors,
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
    Tensor& weights() { return requireParameters("LMHeadLayer::weights").weights; }
    Tensor& bias() { return requireParameters("LMHeadLayer::bias").bias; }

    // ---- final_rms_gamma access ----
    // The READ accessor is always safe (telemetry, save, MTP forward, warn-checks).
    const Tensor& finalRmsGamma() const { return requireParameters("LMHeadLayer::finalRmsGamma").final_rms_gamma; }

    // The WRITE accessor THROWS if the gamma is configured frozen.
    // Only four legitimate writers exist:
    //   1. LMHeadLayer ctor (uses the private member directly)
    //   2. Startup/Model/ParameterGroupRegistration (top-level startup registration)
    //   3. AutogradTraining (zero_grad before backward)
    //   4. grim_model_serialization::load (assignWrite to .data)
    // Any other path that obtains a mutable reference will trip the throw and
    // produce a stack trace pinpointing the leak.
    Tensor& finalRmsGammaMutable_UnfrozenOnly(const char* caller) {
        if (hp_.freeze_learned_rms_gammas) {
            throw std::runtime_error(
                std::string("[FROZEN-GAMMA-LEAK] mutable access to final_rms_gamma while "
                            "freeze_learned_rms_gammas=true. caller=") +
                (caller ? caller : "<unknown>"));
        }
        return requireParameters("LMHeadLayer::finalRmsGammaMutable_UnfrozenOnly").final_rms_gamma;
    }

    const Tensor& weights() const { return requireParameters("LMHeadLayer::weights const").weights; }
    const Tensor& bias() const { return requireParameters("LMHeadLayer::bias const").bias; }

    /// Whether this layer allocated its own weights (false = tied to embedding)
    bool ownsWeights() const { return requireParameters("LMHeadLayer::ownsWeights").owns_weights; }

    /// Whether weights are initialized and ready for forward pass
    bool weightsReady() const { return requireParameters("LMHeadLayer::weightsReady").weights.data != nullptr; }

    //--------------------------------------------------
    // Forward Pass - Autograd
    //--------------------------------------------------
    /// LM head forward with autograd tracking:
    ///   0. Optional: RMSNorm(input, final_rms_gamma_frozen_or_trained_) — pre-LM-head normalization
    ///   1. Optional: center_columns_by_causal_prefix_lengths on normalized input (Issue #125/#132)
    ///   2. Optional: project_out_pc1 on the current LM input (composes after centering when both are enabled)
    ///   3. logits = input @ weights^T  (autograd::matmul, transpose_b=true)
    ///   4. Optional: center_rows on logits (numerical stability)
    ///   5. Optional: logits += bias  (autograd::broadcast_add)
    ///
    /// Builds compute graph for automatic backward().
    ///
    /// @param input                    [total_tokens, d_model] - encoder output (MUST have grad_fn if training)
    /// @param payload                  Host-side batch contents and real lengths; inference geometry comes from payload,
    ///                                 while fixed-shape training geometry is validated against config-authored hp_
    /// @param stream                   CUDA stream from the caller's forward payload/request
    /// @param cublas_handle            cuBLAS handle from the caller's forward payload/request
    /// @param forward_outputs          Canonical per-call forward sink. This function
    ///                                 writes `forward_outputs.lm_head_input_tensor`
    ///                                 when it materializes a transformed LM input
    ///                                 and always writes `forward_outputs.logits_tensor`.
    void forward(
        const Tensor& input,
        const Batching::BatchPayload& payload,
        cudaStream_t stream,
        cublasHandle_t cublas_handle,
        Forward::ModelForwardOutputs& forward_outputs,
        const LMHeadParameterViews* parameter_views = nullptr);



private:
    LMHeadParameterTensors& requireParameters(const char* caller) {
        if (!parameters_) {
            throw std::runtime_error(std::string(caller) + ": LMHeadLayer parameter owner is not initialized");
        }
        return *parameters_;
    }

    const LMHeadParameterTensors& requireParameters(const char* caller) const {
        if (!parameters_) {
            throw std::runtime_error(std::string(caller) + ": LMHeadLayer parameter owner is not initialized");
        }
        return *parameters_;
    }

    // Immutable grouped read view from HyperparameterGroupings.hpp. This is not
    // a second authored config owner; it is the layer's durable construction HP
    // snapshot needed after startup-local grouping objects go out of scope.
    HyperParameters::LMHeadLayerConstructionHP hp_{};

    LMHeadParameterTensors* parameters_ = nullptr;
};

} // namespace GRIM

#endif // USE_CUDA

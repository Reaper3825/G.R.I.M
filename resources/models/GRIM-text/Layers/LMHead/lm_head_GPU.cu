//======================================================//
//  lm_head_GPU.cu
//  GPU-accelerated LM Head layer using autograd
//  Pattern B: Layer Ownership — self-allocates weights
//
//  Owns: weights [vocab_size, d_model] (or aliased from embedding),
//        bias [vocab_size] (optional), final_rms_gamma [d_model].
//
//  Forward: RMSNorm → optional centering → optional PC1 projection → logits = input @ W^T → bias
//
//  ISSUE #56 pattern: The LM head returns any materialized LM-input tensor so
//  the caller can keep tape-local state alive without the layer mutating a
//  caller-owned runtime sink.
//
//  PyTorch equivalent:
//    class LMHead(nn.Module):
//        def __init__(self, d_model, vocab_size, bias=True):
//            self.norm = nn.RMSNorm(d_model)
//            self.proj = nn.Linear(d_model, vocab_size, bias=bias)
//        def forward(self, x):
//            return self.proj(self.norm(x))
//======================================================//

#include "lm_head_GPU.hpp"
#include "../../Shared/TensorContract/LMHeadGemmDiagnostics.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

#include <stdexcept>
#include <cstdio>
#include <string>
#include <vector>

namespace GRIM {

//======================================================//
//  Self-Allocating Constructor (Pattern B)
//======================================================//

LMHeadLayer::LMHeadLayer(const HyperParameters::LMHeadLayerConstructionHP& hp,
                         uint64_t seed,
                         cudaStream_t init_stream,
                         Tensor* tied_embedding_weights)
    : hp_(hp), owns_weights_(!tied_embedding_weights)
{
    // Rule 20: Fail loud on invalid configuration
    if (hp_.d_model <= 0) {
        throw std::runtime_error("LMHeadLayer: d_model must be positive, got " + std::to_string(hp_.d_model));
    }
    if (hp_.vocab_size <= 0) {
        throw std::runtime_error("LMHeadLayer: vocab_size must be positive, got " + std::to_string(hp_.vocab_size));
    }
    if (!init_stream) {
        throw std::runtime_error("LMHeadLayer: init_stream is NULL — CUDA stream required for allocation");
    }
    if (hp_.tie_embeddings != (tied_embedding_weights != nullptr)) {
        throw std::runtime_error(
            "LMHeadLayer: tie_embeddings grouping/runtime ownership mismatch. "
            "HyperparameterGroupings requested tie_embeddings=" +
            std::string(hp_.tie_embeddings ? "true" : "false") +
            " but tied_embedding_weights is " +
            std::string(tied_embedding_weights ? "non-null" : "NULL"));
    }

    // ══════════════════════════════════════════════════════════════
    //  WEIGHTS: Either tied from embedding or independently allocated
    // ══════════════════════════════════════════════════════════════
    if (tied_embedding_weights) {
        // WEIGHT TYING (Issue #60): LM head shares BOTH data AND grad with embedding
        if (!tied_embedding_weights->data) {
            throw std::runtime_error("LMHeadLayer: tied_embedding_weights->data is NULL — embedding MUST be initialized before LM head");
        }
        // Validate shape
        const auto& ews = tied_embedding_weights->shape.as_2d();
        if (ews.rows != hp_.vocab_size || ews.cols != hp_.d_model) {
            throw std::runtime_error(
                "LMHeadLayer: tied embedding shape mismatch. Expected [" +
                std::to_string(hp_.vocab_size) + "," + std::to_string(hp_.d_model) +
                "], got [" + std::to_string(ews.rows) + "," + std::to_string(ews.cols) + "]");
        }

        weights_ = Tensor::from_ptr(
            tied_embedding_weights->data,
            tied_embedding_weights->shape,
            false, true,
            "lm_head.weights_tied"
        );
        // CRITICAL: Share the grad (Issue #59 shared_ptr semantics)
        weights_.share_grad(*tied_embedding_weights);
        weights_.owns_data = false;  // embedding_weights owns the data
        weights_.requires_grad = true;

        fprintf(stdout, "[LMHeadLayer] Weights TIED to embedding: data=%p grad=%p\n",
                (void*)weights_.data, (void*)weights_.grad_data());
    } else {
        // Independent weights: Xavier init
        weights_ = Tensor::zeros({hp_.vocab_size, hp_.d_model}, init_stream, "lm_head.weights");
        weights_.requires_grad_();
        weights_.ensure_grad();
        Tensor::xavier_uniform_(weights_, seed, init_stream);

        fprintf(stdout, "[LMHeadLayer] Weights INDEPENDENT: [%d, %d] Xavier seed=%llu\n",
        hp_.vocab_size, hp_.d_model, static_cast<unsigned long long>(seed));
    }

    // ══════════════════════════════════════════════════════════════
    //  BIAS: Optional, always independently allocated
    // ══════════════════════════════════════════════════════════════
    if (hp_.use_bias) {
        bias_ = Tensor::zeros({hp_.vocab_size}, init_stream, "lm_head.bias");
        bias_.requires_grad_();
        bias_.ensure_grad();
        fprintf(stdout, "[LMHeadLayer] Bias: [%d] initialized to zeros\n", hp_.vocab_size);
    }

    // ══════════════════════════════════════════════════════════════
    //  FINAL RMSNORM GAMMA: Always self-allocated, initialized to 1.0
    // ══════════════════════════════════════════════════════════════
    {
        final_rms_gamma_frozen_or_trained_ = Tensor::zeros({hp_.d_model}, init_stream, "final_rms_gamma");

        // freeze_learned_rms_gammas=true: γ stays at 1.0 forever — do NOT mark as a leaf
        // parameter. autograd will skip producing its gradient and buildParameterGroups
        // will skip registering it (gated on has_grad()).
        if (!hp_.freeze_learned_rms_gammas) {
            final_rms_gamma_frozen_or_trained_.requires_grad_();
            final_rms_gamma_frozen_or_trained_.ensure_grad();
        }

        // Initialize gamma to 1.0 (identity normalization at start)
        std::vector<float> ones(hp_.d_model, 1.0f);
        cudaMemcpyAsync(final_rms_gamma_frozen_or_trained_.data, ones.data(),
            hp_.d_model * sizeof(float), cudaMemcpyHostToDevice, init_stream);

        fprintf(stdout, "[LMHeadLayer] Final RMSNorm gamma: [%d] initialized to 1.0 (eps=%.1e) frozen=%s\n",
            hp_.d_model, hp_.rms_epsilon,
            hp_.freeze_learned_rms_gammas ? "true" : "false");
    }

    fprintf(stdout, "[LMHeadLayer] Initialized: owns_weights=%s, has_bias=%s, has_rms_norm=true\n",
            owns_weights_ ? "true" : "false",
            hp_.use_bias ? "true" : "false");
}

LMHeadLayer::LMHeadLayer(LMHeadLayer&& other) noexcept
    : hp_(other.hp_)
    , weights_(std::move(other.weights_))
    , bias_(std::move(other.bias_))
    , final_rms_gamma_frozen_or_trained_(std::move(other.final_rms_gamma_frozen_or_trained_))
    , owns_weights_(other.owns_weights_) {
    other.owns_weights_ = false;
}

LMHeadLayer& LMHeadLayer::operator=(LMHeadLayer&& other) noexcept {
    if (this != &other) {
        hp_ = other.hp_;
        weights_ = std::move(other.weights_);
        bias_ = std::move(other.bias_);
        final_rms_gamma_frozen_or_trained_ = std::move(other.final_rms_gamma_frozen_or_trained_);
        owns_weights_ = other.owns_weights_;
        other.owns_weights_ = false;
    }
    return *this;
}

//======================================================//
//  Forward Pass
//======================================================//

LMHeadForwardResult LMHeadLayer::forward(const Tensor& input,
                                         const Batching::BatchPayload& payload,
                                         cudaStream_t stream, cublasHandle_t cublas_handle,
                                         const LMHeadParameterViews* parameter_views) {
    LMHeadForwardResult result;
    const Tensor& lm_weights =
        (parameter_views && parameter_views->weights) ? *parameter_views->weights : weights_;
    const Tensor& lm_bias =
        (parameter_views && parameter_views->bias) ? *parameter_views->bias : bias_;
    const Tensor& lm_final_rms_gamma =
        (parameter_views && parameter_views->final_rms_gamma)
            ? *parameter_views->final_rms_gamma
            : final_rms_gamma_frozen_or_trained_;

    // Rule 20: Crash on invalid state
    if (!lm_weights.data) {
        throw std::runtime_error("LMHeadLayer::forward: selected weights view is not initialized");
    }
    if (!stream) {
        throw std::runtime_error("LMHeadLayer::forward: stream is NULL");
    }
    if (!cublas_handle) {
        throw std::runtime_error("LMHeadLayer::forward: cublas_handle is NULL");
    }

    autograd::set_autograd_cublas_handle(cublas_handle);

    const int d_model = hp_.d_model;
    int batch_size = 0;
    int rows_per_sequence = 0;
    int total_tokens = 0;

    if (payload.isTraining()) {
        batch_size = hp_.training_batch_size;
        rows_per_sequence = hp_.training_rows_per_sequence;
        if (batch_size <= 0 || rows_per_sequence <= 0) {
            throw std::runtime_error(
                "LMHeadLayer::forward: training payload requires config-authored fixed shape, got training_batch_size=" +
                std::to_string(batch_size) + " training_rows_per_sequence=" +
                std::to_string(rows_per_sequence));
        }
        total_tokens = batch_size * rows_per_sequence;
    } else {
        batch_size = payload.batch_size;
        rows_per_sequence = payload.max_seq_len;
        total_tokens = payload.total_tokens;
        const int input_rows = input.shape.as_2d().rows;
        if (input_rows != total_tokens) {
            throw std::runtime_error("LMHeadLayer::forward: input rows (" + std::to_string(input_rows) +
                                     ") != payload total_tokens (" + std::to_string(total_tokens) + ")");
        }
    }

    if (rows_per_sequence <= 0) {
        throw std::runtime_error("LMHeadLayer::forward: rows_per_sequence must be > 0, got " +
                                 std::to_string(rows_per_sequence));
    }
    if (batch_size <= 0) {
        throw std::runtime_error("LMHeadLayer::forward: batch_size must be > 0, got " +
                                 std::to_string(batch_size));
    }
    if (static_cast<int>(payload.seq_lengths.size()) != batch_size) {
        throw std::runtime_error("LMHeadLayer::forward: payload.seq_lengths size (" +
                                 std::to_string(payload.seq_lengths.size()) +
                                 ") != batch_size (" + std::to_string(batch_size) + ")");
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 0: Optional Final RMSNorm (pre-LM-head normalization)
    //
    // Normalizes encoder output before projection: y = RMSNorm(x, gamma)
    // Autograd graph: input → RMSNormGradFn → normalized
    // ════════════════════════════════════════════════════════════════════

    const Tensor* current_input = &input;
    Tensor normalized;

    if (lm_final_rms_gamma.data) {
        normalized = autograd::rms_norm(input, lm_final_rms_gamma, hp_.rms_epsilon, stream);
        current_input = &normalized;
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 1: Optional hidden-state geometry projection chain
    //
    //   Column centering h: Σ_t h[t,d] = 0 for each feature d   (Issue #125)
    //     Removes shared direction across positions → reduces cos(h_i, h_j).
    //
    //   Row centering of WEIGHT matrix instead of h (April 2026 reformulation,
    //   applied below at STEP 2 — see commentary there):
    //     Constrains Σ_d W[v,d] = 0 for each vocab v. Mathematically equivalent
    //     invariance to the original Issue #132 row-centering-of-h, with two
    //     advantages:
    //       (a) preserves per-position energy in h — no rms bifurcation when h
    //           drifts toward the all-ones direction (the failure mode that
    //           caused rms_max/rms_min to climb 1.02x → 4.0x and produced
    //           spurious high ρ from tiny denominators);
    //       (b) STRONGER guarantee — also makes back-propagated grad_h satisfy
    //           Σ_d grad_h[t,d] = 0 automatically (Issue #132's original
    //           row-centering-of-h only enforced the property in forward).
    // ════════════════════════════════════════════════════════════════════

    const Tensor* matmul_input = current_input;
    Tensor centered_hidden_for_pc1;

    if (hp_.center_hidden_states) {
        if (rows_per_sequence <= 1) {
            throw std::runtime_error("LMHeadLayer::forward: center_hidden_states requires rows_per_sequence > 1; single-token decode cannot column-center hidden states without erasing the signal");
        }
        // Column-center h with a strict-past causal prefix mean inside each
        // sequence: removes the running shared direction across valid positions
        // without coupling samples inside the batch, including PAD activations
        // in the mean, erasing token 0, or leaking future tokens into the
        // current LM position.
        // Row-centering moved to W at STEP 2 (April 2026 reformulation).
        centered_hidden_for_pc1 = autograd::center_columns_by_causal_prefix_lengths(
            *current_input, payload.seq_lengths, batch_size, rows_per_sequence, stream);
        matmul_input = &centered_hidden_for_pc1;
    }

    if (hp_.project_out_pc1) {
        // Issue #149: project out dominant PC1 direction via power iteration.
        // g is RMS-normalized (g·g = D), so the projection coefficient is (h·g)/D:
        //   h̃[t] = h[t] - (h[t]·g / D) * g     where g = PC1(H), stop-gradient
        // Backward: grad_h += (I - gg^T/D) * grad_h̃  (accumulates into input grad)
        result.lm_input_tensor = autograd::project_out_pc1(*matmul_input, hp_.pc1_power_iters, stream);
        matmul_input = &result.lm_input_tensor;
    } else if (hp_.center_hidden_states) {
        // Materialize the causal-prefix centered tensor so it survives this
        // scope (Issue #127) and callers can explicitly keep the live LM-input
        // handle inside the forward boundary.
        result.lm_input_tensor = std::move(centered_hidden_for_pc1);
        matmul_input = &result.lm_input_tensor;
    } else {
        if (current_input == &normalized) {
            // RMSNorm was applied but no centering — preserve the normalized
            // tensor in the returned LM-input view so it does not dangle when
            // this function returns (normalized is a local).
            result.lm_input_tensor = std::move(normalized);
            matmul_input = &result.lm_input_tensor;
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 2: Linear projection  logits = lm_input @ weights^T
    //
    // autograd::matmul builds the computation graph:
    //   MatMulGradFn::apply() computes:
    //     grad_input  = grad_output @ weights      (for backward to encoder)
    //     grad_weights = lm_input^T @ grad_output  (for weight update)
    // ════════════════════════════════════════════════════════════════════
    if (!lm_weights.shape.is_2d_layout()) {
        throw std::runtime_error("LMHeadLayer::forward: weights must be 2D [vocab_size, d_model]");
    }
    const auto weights_shape = lm_weights.shape.as_2d();
    if (weights_shape.rows != hp_.vocab_size || weights_shape.cols != hp_.d_model) {
        throw std::runtime_error(
            "LMHeadLayer::forward: weights shape mismatch. expected=[" +
            std::to_string(hp_.vocab_size) + "," + std::to_string(hp_.d_model) +
            "] got=[" + std::to_string(weights_shape.rows) + "," +
            std::to_string(weights_shape.cols) + "]");
    }

    // Default path: hard type gate W_eff so the LM row indexed by token ID only
    // sees the TokenLayout class subspace assigned to that token type.
    // Local experiment path: bypass the hard token-type gate inside the LM head
    // only, without plumbing a new authored config field yet.
    const bool use_token_type_gate = GRIM::kEnableLmHeadTokenTypeGateExperiment;
    const bool use_centered_weights = hp_.center_hidden_states;
    Tensor effective_weights_storage;
    const Tensor* effective_weights = &lm_weights;
    if (use_centered_weights && use_token_type_gate) {
        effective_weights_storage = autograd::center_rows_by_token_type_gate(lm_weights, stream);
        effective_weights_storage.name = "lm_head.centered_token_type_gated_weights";
        effective_weights = &effective_weights_storage;
    } else if (use_centered_weights) {
        effective_weights_storage = autograd::center_rows(lm_weights, stream);
        effective_weights_storage.name = "lm_head.centered_weights";
        effective_weights = &effective_weights_storage;
    } else if (use_token_type_gate) {
        effective_weights_storage = autograd::type_gate_rows_by_token_type(lm_weights, stream);
        effective_weights_storage.name = "lm_head.token_type_gated_weights";
        effective_weights = &effective_weights_storage;
    }

    if (!matmul_input->data) {
        throw std::runtime_error("LMHeadLayer::forward: matmul input has null data - cannot compute weight gradient. "
            "Check encoder output and centering/PC1 buffers.");
    }
    const float* a_cache = matmul_input->data;  // Explicit cache for grad_B = lm_input^T @ grad_output

    result.logits = autograd::matmul(
        *matmul_input,
        *effective_weights,
        stream,
        a_cache,
        nullptr,  // matmul copies the effective B cache it needs for backward
        true  // transpose_b=true: logits = input @ W^T
    );

    autograd::logLmHeadGemmForwardEquation(
        *matmul_input,
        *effective_weights,
        result.logits,
        hp_.center_hidden_states,
        hp_.project_out_pc1,
        use_centered_weights,
        use_token_type_gate,
        total_tokens,
        d_model,
        hp_.vocab_size,
        stream);

    // Validate output shape
    const auto expected_shape = TensorContract::TensorShape::make_LOGITS(total_tokens, hp_.vocab_size);
    const size_t logits_elements = result.logits.shape.total_elements();
    const size_t expected_elements = expected_shape.total_elements();
    if (logits_elements != expected_elements) {
        throw std::runtime_error(
            "LMHeadLayer::forward: logits shape validation FAILED\n"
            "  Got: " + std::to_string(logits_elements) + " elements\n"
            "  Expected: " + std::to_string(expected_elements) + " elements (" +
                std::to_string(total_tokens) + "x" + std::to_string(hp_.vocab_size) + ")");
    }
    result.logits.shape = expected_shape;

    // ════════════════════════════════════════════════════════════════════
    // STEP 3: Optional logit centering (numerical stability)
    //
    // Softmax is shift-invariant: softmax(x - c) = softmax(x)
    // So centering doesn't change predictions but keeps logits near zero.
    // ════════════════════════════════════════════════════════════════════
    if (hp_.center_logits) {
        result.logits = autograd::center_rows(result.logits, stream);
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 4: Optional bias addition
    //
    // autograd::broadcast_add builds BiasAddGradFn:
    //   grad_logits passes through to input
    //   grad_bias = sum(grad_logits, dim=0)
    // ════════════════════════════════════════════════════════════════════
    if (hp_.use_bias && lm_bias.data) {
        result.logits = autograd::broadcast_add(result.logits, lm_bias, stream);
        result.logits.shape = expected_shape;  // Preserve LOGITS layout after broadcast_add
    }

    // CRITICAL (Issue #56): Return the output Tensor plus any materialized LM
    // input Tensor. The caller owns where those Category 1 tensors live; the
    // layer must not smuggle them into runtime sinks through output refs.
    return result;
}

} // namespace GRIM

//======================================================//
//  lm_head_GPU.cu
//  GPU-accelerated LM Head layer using autograd
//  Pattern B: Layer Ownership — self-allocates weights
//
//  Owns: weights [vocab_size, d_model] (or aliased from embedding),
//        bias [vocab_size] (optional), final_rms_gamma [d_model].
//
//  Forward: RMSNorm → centering → logits = input @ W^T → bias
//
//  ISSUE #56 pattern: Intermediate tensors kept alive via caller-owned
//  storage so autograd graph survives until backward().
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
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

#include <stdexcept>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

namespace GRIM {

//======================================================//
//  Self-Allocating Constructor (Pattern B)
//======================================================//

LMHeadLayer::LMHeadLayer(const LMHeadLayerConfig& config,
                         uint64_t seed,
                         cudaStream_t init_stream,
                         Tensor* tied_embedding_weights)
    : config_(config), owns_weights_(!tied_embedding_weights)
{
    // Rule 20: Fail loud on invalid configuration
    if (config_.d_model <= 0) {
        throw std::runtime_error("LMHeadLayer: d_model must be positive, got " + std::to_string(config_.d_model));
    }
    if (config_.vocab_size <= 0) {
        throw std::runtime_error("LMHeadLayer: vocab_size must be positive, got " + std::to_string(config_.vocab_size));
    }
    if (!init_stream) {
        throw std::runtime_error("LMHeadLayer: init_stream is NULL — CUDA stream required for allocation");
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
        if (ews.rows != config_.vocab_size || ews.cols != config_.d_model) {
            throw std::runtime_error(
                "LMHeadLayer: tied embedding shape mismatch. Expected [" +
                std::to_string(config_.vocab_size) + "," + std::to_string(config_.d_model) +
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
        weights_ = Tensor::zeros({config_.vocab_size, config_.d_model}, init_stream, "lm_head.weights");
        weights_.requires_grad_();
        weights_.ensure_grad();
        Tensor::xavier_uniform_(weights_, seed, init_stream);

        fprintf(stdout, "[LMHeadLayer] Weights INDEPENDENT: [%d, %d] Xavier seed=%llu\n",
                config_.vocab_size, config_.d_model, static_cast<unsigned long long>(seed));
    }

    // ══════════════════════════════════════════════════════════════
    //  BIAS: Optional, always independently allocated
    // ══════════════════════════════════════════════════════════════
    if (config_.use_bias) {
        bias_ = Tensor::zeros({config_.vocab_size}, init_stream, "lm_head.bias");
        bias_.requires_grad_();
        bias_.ensure_grad();
        fprintf(stdout, "[LMHeadLayer] Bias: [%d] initialized to zeros\n", config_.vocab_size);
    }

    // ══════════════════════════════════════════════════════════════
    //  FINAL RMSNORM GAMMA: Always self-allocated, initialized to 1.0
    // ══════════════════════════════════════════════════════════════
    if (config_.has_final_rms_norm) {
        final_rms_gamma_frozen_or_trained_ = Tensor::zeros({config_.d_model}, init_stream, "final_rms_gamma");

        // freeze_final_rms_gamma=true: γ stays at 1.0 forever — do NOT mark as a leaf
        // parameter. autograd will skip producing its gradient and buildParameterGroups
        // will skip registering it (gated on has_grad()).
        if (!config_.freeze_final_rms_gamma) {
            final_rms_gamma_frozen_or_trained_.requires_grad_();
            final_rms_gamma_frozen_or_trained_.ensure_grad();
        }

        // Initialize gamma to 1.0 (identity normalization at start)
        std::vector<float> ones(config_.d_model, 1.0f);
        cudaMemcpyAsync(final_rms_gamma_frozen_or_trained_.data, ones.data(),
                        config_.d_model * sizeof(float), cudaMemcpyHostToDevice, init_stream);

        fprintf(stdout, "[LMHeadLayer] Final RMSNorm gamma: [%d] initialized to 1.0 (eps=%.1e) frozen=%s\n",
                config_.d_model, config_.rms_epsilon,
                config_.freeze_final_rms_gamma ? "true" : "false");
    }

    // Set cuBLAS handle for autograd (if available at init time)
    if (config_.cublas_handle) {
        autograd::set_autograd_cublas_handle(config_.cublas_handle);
    }

    fprintf(stdout, "[LMHeadLayer] Initialized: owns_weights=%s, has_bias=%s, has_rms_norm=%s\n",
            owns_weights_ ? "true" : "false",
            config_.use_bias ? "true" : "false",
            config_.has_final_rms_norm ? "true" : "false");
}

LMHeadLayer::LMHeadLayer(LMHeadLayer&& other) noexcept
    : config_(other.config_)
    , weights_(std::move(other.weights_))
    , bias_(std::move(other.bias_))
    , final_rms_gamma_frozen_or_trained_(std::move(other.final_rms_gamma_frozen_or_trained_))
    , owns_weights_(other.owns_weights_) {
    other.config_.cublas_handle = nullptr;
    other.owns_weights_ = false;
}

LMHeadLayer& LMHeadLayer::operator=(LMHeadLayer&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
        weights_ = std::move(other.weights_);
        bias_ = std::move(other.bias_);
        final_rms_gamma_frozen_or_trained_ = std::move(other.final_rms_gamma_frozen_or_trained_);
        owns_weights_ = other.owns_weights_;
        other.config_.cublas_handle = nullptr;
        other.owns_weights_ = false;
    }
    return *this;
}

//======================================================//
//  Forward Pass
//======================================================//

Tensor LMHeadLayer::forward(const Tensor& input, Tensor& out_centered_hidden) {
    // Rule 20: Crash on invalid state
    if (!weights_.data) {
        throw std::runtime_error("LMHeadLayer::forward: weights not initialized");
    }
    if (!config_.stream) {
        throw std::runtime_error("LMHeadLayer::forward: stream is NULL — call setStream() before forward");
    }
    if (!config_.cublas_handle) {
        throw std::runtime_error("LMHeadLayer::forward: cublas_handle is NULL — call setCublasHandle() before forward");
    }

    const cudaStream_t stream = config_.stream;
    const int total_tokens = input.shape.as_2d().rows;
    const int d_model = input.shape.as_2d().cols;

    if (d_model != config_.d_model) {
        throw std::runtime_error("LMHeadLayer::forward: input d_model mismatch (" +
                                 std::to_string(d_model) + " vs config " +
                                 std::to_string(config_.d_model) + ")");
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 0: Optional Final RMSNorm (pre-LM-head normalization)
    //
    // Normalizes encoder output before projection: y = RMSNorm(x, gamma)
    // Autograd graph: input → RMSNormGradFn → normalized
    // ════════════════════════════════════════════════════════════════════

    const Tensor* current_input = &input;
    Tensor normalized;

    if (config_.has_final_rms_norm && final_rms_gamma_frozen_or_trained_.data) {
        // Only flip requires_grad when γ is trainable. When frozen the rms_norm
        // GradFn will skip the gamma path entirely (no grad accumulation).
        if (!config_.freeze_final_rms_gamma) {
            final_rms_gamma_frozen_or_trained_.requires_grad = true;
        }
        normalized = autograd::rms_norm(input, final_rms_gamma_frozen_or_trained_, config_.rms_epsilon, stream);
        current_input = &normalized;
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 1: Optional hidden state centering (Issue #125 / reformulated #132, April 2026)
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

    if (config_.center_hidden_states) {
        // Column-center h: removes common direction across positions (Issue #125).
        // Row-centering moved to W at STEP 2 (April 2026 reformulation).
        Tensor col_centered = autograd::center_columns(*current_input, stream);
        // Store in out_centered_hidden so it survives this scope (Issue #127)
        // and so cached_encoder_output reflects the actual matmul input for diagnostics.
        out_centered_hidden = std::move(col_centered);
        matmul_input = &out_centered_hidden;
    } else if (config_.project_out_pc1) {
        // Issue #149: project out dominant PC1 direction via power iteration.
        // g is RMS-normalized (g·g = D), so the projection coefficient is (h·g)/D:
        //   h̃[t] = h[t] - (h[t]·g / D) * g     where g = PC1(H), stop-gradient
        // Backward: grad_h += (I - gg^T/D) * grad_h̃  (accumulates into input grad)
        // Returns owning tensor with separate output buffer — input is not mutated.
        out_centered_hidden = autograd::project_out_pc1(*current_input, config_.pc1_power_iters, stream);
        matmul_input = &out_centered_hidden;
    } else {
        if (current_input == &normalized) {
            // RMSNorm was applied but no centering — preserve the normalized
            // tensor in out_centered_hidden so cached_a doesn't dangle when
            // this function returns (normalized is a local).
            out_centered_hidden = std::move(normalized);
            matmul_input = &out_centered_hidden;
        } else {
            out_centered_hidden = Tensor();
        }
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 2: Linear projection  logits = lm_input @ weights^T
    //
    // autograd::matmul builds the computation graph:
    //   MatMulGradFn::apply() computes:
    //     grad_input  = grad_output @ weights      (for backward to encoder)
    //     grad_weights = lm_input^T @ grad_output  (for weight update)
    //
    // Pass explicit a_cache (like FFN) so grad_B computation has valid source.
    // Avoids MatMulGradFn::set_cache_copy "a_cache is NULL" when tensor.data
    // is null (moved-from, zero-size, or lifecycle edge case).
    // ════════════════════════════════════════════════════════════════════
    weights_.requires_grad = true;
    weights_.shape = TensorContract::TensorShape::make_BSM(config_.vocab_size, config_.d_model);

    // April 2026: When hidden-state centering is enabled, project through a
    // row-centered VIEW of W (Σ_d W[v,d]=0). Forward and backward both pick up
    // the invariance described above. centered_weights_ is held as a member so
    // its data buffer and CenterRowsGradFn chain survive past this function and
    // remain valid for backward(). When centering is disabled we project through
    // raw weights_ as before.
    const Tensor* effective_weights = &weights_;
    if (config_.center_hidden_states) {
        centered_weights_ = autograd::center_rows(weights_, stream);
        effective_weights = &centered_weights_;
    } else {
        // Drop any stale buffer from a previous centered run — keeps memory honest.
        centered_weights_ = Tensor();
    }

    if (!matmul_input->data) {
        throw std::runtime_error("LMHeadLayer::forward: matmul input has null data - cannot compute weight gradient. "
            "Check encoder output and centering/PC1 buffers.");
    }
    const float* a_cache = matmul_input->data;  // Explicit cache for grad_B = lm_input^T @ grad_output

    Tensor logits = autograd::matmul(
        *matmul_input,
        *effective_weights,
        stream,
        a_cache,
        nullptr,  // weights persist across calls (raw or centered held by member)
        true  // transpose_b=true: logits = input @ W^T
    );
    // Validate output shape
    const auto expected_shape = TensorContract::TensorShape::make_LOGITS(total_tokens, config_.vocab_size);
    const size_t logits_elements = logits.shape.total_elements();
    const size_t expected_elements = expected_shape.total_elements();
    if (logits_elements != expected_elements) {
        throw std::runtime_error(
            "LMHeadLayer::forward: logits shape validation FAILED\n"
            "  Got: " + std::to_string(logits_elements) + " elements\n"
            "  Expected: " + std::to_string(expected_elements) + " elements (" +
            std::to_string(total_tokens) + "x" + std::to_string(config_.vocab_size) + ")");
    }
    logits.shape = expected_shape;

    // ════════════════════════════════════════════════════════════════════
    // STEP 3: Optional logit centering (numerical stability)
    //
    // Softmax is shift-invariant: softmax(x - c) = softmax(x)
    // So centering doesn't change predictions but keeps logits near zero.
    // ════════════════════════════════════════════════════════════════════
    if (config_.center_logits) {
        logits = autograd::center_rows(logits, stream);
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 4: Optional bias addition
    //
    // autograd::broadcast_add builds BiasAddGradFn:
    //   grad_logits passes through to input
    //   grad_bias = sum(grad_logits, dim=0)
    // ════════════════════════════════════════════════════════════════════
    if (config_.use_bias && bias_.data) {
        bias_.requires_grad = true;
        logits = autograd::broadcast_add(logits, bias_, stream);
        logits.shape = expected_shape;  // Preserve LOGITS layout after broadcast_add
    }

    // CRITICAL (Issue #56): Return the output Tensor.
    // The returned Tensor owns the grad_fn chain. If it were destroyed here,
    // the entire autograd graph would be deleted during forward pass.
    return logits;
}

} // namespace GRIM

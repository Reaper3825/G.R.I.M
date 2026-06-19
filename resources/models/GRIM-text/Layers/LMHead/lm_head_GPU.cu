//======================================================//
//  lm_head_GPU.cu
//  GPU-accelerated LM head forward using autograd
//  LM head tensors live on a startup-owned registry bundle.
//
//  Borrows: weights [vocab_size, d_model] (or aliased from embedding),
//           bias [vocab_size] (optional), final_rms_gamma [d_model].
//
//  Forward: RMSNorm → optional centering → optional PC1 projection → logits = input @ W^T → bias
//
//  ISSUE #56 pattern: The LM head writes any materialized LM-input tensor plus
//  logits into the canonical shared-forward sink owned by the active caller.
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
//  Forward Pass
//======================================================//

void forwardLmHead(
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameter_tensors,
    const Tensor& input,
    const Batching::BatchPayload& payload,
    cudaStream_t stream,
    cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs) {
    forward_outputs.lm_head_input_tensor = Tensor();
    forward_outputs.logits_tensor = Tensor();
    const Tensor& lm_weights = parameter_tensors.weights;
    const Tensor& lm_bias = parameter_tensors.bias;
    const Tensor& lm_final_rms_gamma = parameter_tensors.final_rms_gamma;

    // Rule 20: Crash on invalid state
    if (!lm_weights.data) {
        throw std::runtime_error("forwardLmHead: weights tensor is not initialized");
    }
    if (!stream) {
        throw std::runtime_error("forwardLmHead: stream is NULL");
    }
    if (!cublas_handle) {
        throw std::runtime_error("forwardLmHead: cublas_handle is NULL");
    }

    autograd::set_autograd_cublas_handle(cublas_handle);

    const int d_model = hp.d_model;
    int batch_size = 0;
    int rows_per_sequence = 0;
    int total_tokens = 0;

    if (payload.isTraining()) {
        batch_size = hp.training_batch_size;
        rows_per_sequence = hp.training_rows_per_sequence;
        if (batch_size <= 0 || rows_per_sequence <= 0) {
            throw std::runtime_error(
                "forwardLmHead: training payload requires config-authored fixed shape, got training_batch_size=" +
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
            throw std::runtime_error("forwardLmHead: input rows (" + std::to_string(input_rows) +
                                     ") != payload total_tokens (" + std::to_string(total_tokens) + ")");
        }
    }

    if (rows_per_sequence <= 0) {
        throw std::runtime_error("forwardLmHead: rows_per_sequence must be > 0, got " +
                                 std::to_string(rows_per_sequence));
    }
    if (batch_size <= 0) {
        throw std::runtime_error("forwardLmHead: batch_size must be > 0, got " +
                                 std::to_string(batch_size));
    }
    if (static_cast<int>(payload.seq_lengths.size()) != batch_size) {
        throw std::runtime_error("forwardLmHead: payload.seq_lengths size (" +
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
        normalized = autograd::rms_norm(input, lm_final_rms_gamma, hp.rms_epsilon, stream);
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

    if (hp.center_hidden_states) {
        if (rows_per_sequence <= 1) {
            throw std::runtime_error("forwardLmHead: center_hidden_states requires rows_per_sequence > 1; single-token decode cannot column-center hidden states without erasing the signal");
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

    if (hp.project_out_pc1) {
        // Issue #149: project out dominant PC1 direction via power iteration.
        // g is RMS-normalized (g·g = D), so the projection coefficient is (h·g)/D:
        //   h̃[t] = h[t] - (h[t]·g / D) * g     where g = PC1(H), stop-gradient
        // Backward: grad_h += (I - gg^T/D) * grad_h̃  (accumulates into input grad)
        forward_outputs.lm_head_input_tensor = autograd::project_out_pc1(*matmul_input, hp.pc1_power_iters, stream);
        matmul_input = &forward_outputs.lm_head_input_tensor;
    } else if (hp.center_hidden_states) {
        // Materialize the causal-prefix centered tensor so it survives this
        // scope (Issue #127) and callers can explicitly keep the live LM-input
        // handle inside the forward boundary.
        forward_outputs.lm_head_input_tensor = std::move(centered_hidden_for_pc1);
        matmul_input = &forward_outputs.lm_head_input_tensor;
    } else {
        if (current_input == &normalized) {
            // RMSNorm was applied but no centering — preserve the normalized
            // tensor in the forward sink LM-input view so it does not dangle when
            // this function returns (normalized is a local).
            forward_outputs.lm_head_input_tensor = std::move(normalized);
            matmul_input = &forward_outputs.lm_head_input_tensor;
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
        throw std::runtime_error("forwardLmHead: weights must be 2D [vocab_size, d_model]");
    }
    const auto weights_shape = lm_weights.shape.as_2d();
    if (weights_shape.rows != hp.vocab_size || weights_shape.cols != hp.d_model) {
        throw std::runtime_error(
            "forwardLmHead: weights shape mismatch. expected=[" +
            std::to_string(hp.vocab_size) + "," + std::to_string(hp.d_model) +
            "] got=[" + std::to_string(weights_shape.rows) + "," +
            std::to_string(weights_shape.cols) + "]");
    }

    // Default path: hard type gate W_eff so the LM row indexed by token ID only
    // sees the TokenLayout class subspace assigned to that token type.
    // Local experiment path: bypass the hard token-type gate inside the LM head
    // only, without plumbing a new authored config field yet.
    const bool use_token_type_gate = GRIM::kEnableLmHeadTokenTypeGateExperiment;
    const bool use_centered_weights = hp.center_hidden_states;
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
        throw std::runtime_error("forwardLmHead: matmul input has null data - cannot compute weight gradient. "
            "Check encoder output and centering/PC1 buffers.");
    }
    forward_outputs.logits_tensor = autograd::matmul(
        *matmul_input,
        *effective_weights,
        stream,
        true  // transpose_b=true: logits = input @ W^T
    );

    autograd::logLmHeadGemmForwardEquation(
        *matmul_input,
        *effective_weights,
        forward_outputs.logits_tensor,
        hp.center_hidden_states,
        hp.project_out_pc1,
        use_centered_weights,
        use_token_type_gate,
        total_tokens,
        d_model,
        payload.vocab_size,
        stream);

    // Validate output geometry without restamping layout metadata here.
    const size_t logits_elements = forward_outputs.logits_tensor.shape.total_elements();
    const size_t expected_elements = static_cast<size_t>(total_tokens) *
                                     static_cast<size_t>(payload.vocab_size);
    if (logits_elements != expected_elements) {
        throw std::runtime_error(
            "forwardLmHead: logits shape validation FAILED\n"
            "  Got: " + std::to_string(logits_elements) + " elements\n"
            "  Expected: " + std::to_string(expected_elements) + " elements (" +
                std::to_string(total_tokens) + "x" + std::to_string(payload.vocab_size) + ")");
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 3: Optional logit centering (numerical stability)
    //
    // Softmax is shift-invariant: softmax(x - c) = softmax(x)
    // So centering doesn't change predictions but keeps logits near zero.
    // ════════════════════════════════════════════════════════════════════
    if (hp.center_logits) {
        forward_outputs.logits_tensor = autograd::center_rows(forward_outputs.logits_tensor, stream);
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 4: Optional bias addition
    //
    // autograd::broadcast_add builds BiasAddGradFn:
    //   grad_logits passes through to input
    //   grad_bias = sum(grad_logits, dim=0)
    // ════════════════════════════════════════════════════════════════════
    // Apply the bias whenever the bias tensor exists. The tensor is allocated by
    // initializeLmHeadParameterTensors when EITHER use_bias OR the dedicated
    // unigram bias is enabled, so gating on data presence (rather than use_bias)
    // lets the log p(v) unigram bias take effect with use_bias=false.
    if (lm_bias.data) {
        forward_outputs.logits_tensor = autograd::broadcast_add(forward_outputs.logits_tensor, lm_bias, stream);
    }
}

} // namespace GRIM

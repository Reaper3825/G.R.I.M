//======================================================//
//  lm_head_GPU.cu
//  GPU-accelerated LM head forward using autograd
//  LM head tensors live on a startup-owned registry bundle.
//
//  Borrows: weights [vocab_size, d_model] (or aliased from embedding),
//           bias [vocab_size] (optional), final_rms_gamma [d_model],
//           mlp_W_gate/mlp_W_up/mlp_W_down (optional residual SwiGLU adapter).
//
//  Forward: RMSNorm → optional residual SwiGLU adapter → logits = input @ W_eff^T → bias
//
//  ISSUE #56 pattern: The LM head writes any materialized LM-input tensor plus
//  logits into the canonical shared-forward sink owned by the active caller.
//
//  PyTorch equivalent:
//    class LMHead(nn.Module):
//        def __init__(self, d_model, vocab_size, mlp_d_ff, alpha, bias=True):
//            self.norm = nn.RMSNorm(d_model)
//            self.gate = nn.Linear(d_model, mlp_d_ff, bias=False)
//            self.up   = nn.Linear(d_model, mlp_d_ff, bias=False)
//            self.down = nn.Linear(mlp_d_ff, d_model, bias=False)
//            self.alpha = alpha
//            self.proj = nn.Linear(d_model, vocab_size, bias=bias)
//        def forward(self, x):
//            z = self.norm(x)
//            u = z + self.alpha * self.down(F.silu(self.gate(z)) * self.up(z))
//            return self.proj(u)
//======================================================//

#include "lm_head_GPU.hpp"
#include "../../Shared/Goal/MeanPool_GPU.hpp"
#include "../../Shared/TensorContract/LMHeadGemmDiagnostics.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <vector>

namespace GRIM {

//======================================================//
//  Forward Pass
//======================================================//

namespace {
const Tensor& normalizeReadoutRows(
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameter_tensors,
    const Tensor& input,
    cudaStream_t stream, cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs) {
    forward_outputs.final_normalized_hidden_states = Tensor();
    forward_outputs.mean_pool = Tensor();
    forward_outputs.lm_head_mlp_gate_out = Tensor();
    forward_outputs.lm_head_mlp_silu_out = Tensor();
    forward_outputs.lm_head_mlp_up_out = Tensor();
    forward_outputs.lm_head_mlp_swiglu_out = Tensor();
    forward_outputs.lm_head_mlp_residual_out = Tensor();
    forward_outputs.logits_tensor = Tensor();
    const Tensor& lm_weights = parameter_tensors.weights;
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
    if (!input.data || !input.shape.is_2d_layout() ||
        input.shape.as_2d().rows <= 0 ||
        input.shape.as_2d().cols != d_model || hp.vocab_size <= 0)
        throw std::runtime_error("LM-head readout: invalid row geometry");
    // ════════════════════════════════════════════════════════════════════
    // STEP 0: Optional Final RMSNorm (pre-LM-head normalization)
    //
    // Normalizes encoder output before projection: y = RMSNorm(x, gamma)
    // Autograd graph: input → RMSNormGradFn → normalized
    // ════════════════════════════════════════════════════════════════════

    const Tensor* current_input = &input;

    if (lm_final_rms_gamma.data) {
        if (lm_final_rms_gamma.shape.total_elements() != static_cast<size_t>(d_model) ||
            !std::isfinite(hp.rms_epsilon) || hp.rms_epsilon <= 0)
            throw std::runtime_error("LM-head readout: invalid RMSNorm parameters");
        forward_outputs.final_normalized_hidden_states =
            autograd::rms_norm(input, lm_final_rms_gamma, hp.rms_epsilon, stream);
        current_input = &forward_outputs.final_normalized_hidden_states;
    }

    return *current_input;
}

void projectReadoutRows(
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameter_tensors,
    const Tensor& normalized_input,
    const Batching::BatchPayload& payload,
    const LMHeadReadoutSelection* selection,
    cudaStream_t stream,
    Forward::ModelForwardOutputs& forward_outputs,
    const Tensor* prepared_effective_weights) {
    const int d_model = hp.d_model;
    const int total_tokens = selection
        ? selection->position_end - selection->position_begin
        : (hp.atom_insertion_enabled ? payload.atomInsertionGapRowCount() : payload.total_tokens);
    if (!normalized_input.shape.is_2d_layout() ||
        normalized_input.shape.as_2d().rows != total_tokens)
        throw std::runtime_error("LM-head readout: input rows differ from payload selection");
    const Tensor& lm_weights = parameter_tensors.weights;
    const Tensor& lm_bias = parameter_tensors.bias;
    const Tensor* current_input = &normalized_input;

    // ════════════════════════════════════════════════════════════════════
    // STEP 0.5: Optional head-side residual SwiGLU adapter (capacity expansion)
    //
    //   z    = current_input (RMSNorm output when gamma exists)
    //   gate = SiLU(z @ mlp_W_gate)          [total_tokens, mlp_d_ff]
    //   up   = z @ mlp_W_up                  [total_tokens, mlp_d_ff]
    //   mlp  = (gate ⊙ up) @ mlp_W_down      [total_tokens, d_model]
    //   u    = z + mlp_alpha * mlp           [total_tokens, d_model]
    //
    // The gate/silu/up/swiglu intermediates are retained on the forward sink
    // because SiluGradFn and ElementwiseMulGradFn hold non-owning pointers into
    // their input buffers (same contract as the encoder FFN retained tensors).
    // The projection consumes the adapter-enriched state.
    // ════════════════════════════════════════════════════════════════════

    if (hp.mlp_enabled) {
        const Tensor& mlp_W_gate = parameter_tensors.mlp_W_gate;
        const Tensor& mlp_W_up = parameter_tensors.mlp_W_up;
        const Tensor& mlp_W_down = parameter_tensors.mlp_W_down;
        if (!mlp_W_gate.data || !mlp_W_up.data || !mlp_W_down.data) {
            throw std::runtime_error("forwardLmHead: lm_head_mlp_enabled=true but adapter tensors are not initialized (mlp_W_gate/mlp_W_up/mlp_W_down)");
        }
        if (!mlp_W_gate.shape.is_2d_layout() || !mlp_W_up.shape.is_2d_layout() || !mlp_W_down.shape.is_2d_layout()) {
            throw std::runtime_error("forwardLmHead: LM-head adapter weights must be 2D");
        }
        const auto gate_shape = mlp_W_gate.shape.as_2d();
        const auto up_shape = mlp_W_up.shape.as_2d();
        const auto down_shape = mlp_W_down.shape.as_2d();
        if (hp.mlp_d_ff <= 0 || gate_shape.cols != hp.mlp_d_ff ||
            gate_shape.rows != d_model || up_shape.rows != d_model ||
            gate_shape.cols != down_shape.rows || up_shape.cols != down_shape.rows ||
            down_shape.cols != d_model) {
            throw std::runtime_error(
                "forwardLmHead: LM-head adapter shape mismatch. W_gate=[" +
                std::to_string(gate_shape.rows) + "," + std::to_string(gate_shape.cols) +
                "] W_up=[" + std::to_string(up_shape.rows) + "," + std::to_string(up_shape.cols) +
                "] W_down=[" + std::to_string(down_shape.rows) + "," + std::to_string(down_shape.cols) +
                "] expected [d_model,mlp_d_ff]/[d_model,mlp_d_ff]/[mlp_d_ff,d_model] with d_model=" +
                std::to_string(d_model));
        }
        if (!std::isfinite(hp.mlp_alpha) || hp.mlp_alpha <= 0.0f) {
            throw std::runtime_error("forwardLmHead: lm_head_mlp_alpha must be positive finite, got " +
                                     std::to_string(hp.mlp_alpha));
        }

        forward_outputs.lm_head_mlp_gate_out = autograd::matmul(*current_input, mlp_W_gate, stream);
        forward_outputs.lm_head_mlp_silu_out = autograd::silu(
            forward_outputs.lm_head_mlp_gate_out, stream,
            forward_outputs.lm_head_mlp_gate_out.data);
        forward_outputs.lm_head_mlp_up_out = autograd::matmul(*current_input, mlp_W_up, stream);
        forward_outputs.lm_head_mlp_swiglu_out = autograd::elementwise_mul(
            forward_outputs.lm_head_mlp_silu_out, forward_outputs.lm_head_mlp_up_out, stream);

        Tensor mlp_down = autograd::matmul(forward_outputs.lm_head_mlp_swiglu_out, mlp_W_down, stream);
        Tensor mlp_scaled = autograd::mul_scalar(mlp_down, hp.mlp_alpha, stream);
        forward_outputs.lm_head_mlp_residual_out = autograd::add(*current_input, mlp_scaled, stream);
        current_input = &forward_outputs.lm_head_mlp_residual_out;
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 1: Linear projection  logits = lm_input @ weights^T
    //
    // autograd::matmul builds the computation graph:
    //   MatMulGradFn::apply() computes:
    //     grad_input  = grad_output @ weights      (for backward to encoder)
    //     grad_weights = lm_input^T @ grad_output  (for weight update)
    // ════════════════════════════════════════════════════════════════════
    const Tensor* matmul_input = current_input;

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
    Tensor& effective_weights_storage = forward_outputs.lm_head_effective_weights;
    const Tensor* effective_weights = prepared_effective_weights ? prepared_effective_weights : &lm_weights;
    if (use_token_type_gate && !prepared_effective_weights) {
        effective_weights_storage = autograd::type_gate_rows_by_token_type(lm_weights, stream);
        effective_weights_storage.name = "lm_head.token_type_gated_weights";
        effective_weights = &effective_weights_storage;
    }

    if (!effective_weights->data || !effective_weights->shape.is_2d_layout() ||
        effective_weights->shape.as_2d().rows != hp.vocab_size ||
        effective_weights->shape.as_2d().cols != d_model)
        throw std::runtime_error("LM-head readout: invalid effective weights");

    if (!matmul_input->data) {
        throw std::runtime_error("forwardLmHead: matmul input has null data - cannot compute weight gradient. "
            "Check encoder output and LM-head adapter buffers.");
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
        use_token_type_gate,
        total_tokens,
        d_model,
        hp.vocab_size,
        stream);

    // Validate output geometry without restamping layout metadata here.
    const size_t logits_elements = forward_outputs.logits_tensor.shape.total_elements();
    const size_t expected_elements = static_cast<size_t>(total_tokens) *
                                     static_cast<size_t>(hp.vocab_size);
    if (logits_elements != expected_elements) {
        throw std::runtime_error(
            "forwardLmHead: logits shape validation FAILED\n"
            "  Got: " + std::to_string(logits_elements) + " elements\n"
            "  Expected: " + std::to_string(expected_elements) + " elements (" +
                std::to_string(total_tokens) + "x" + std::to_string(hp.vocab_size) + ")");
    }

    // ════════════════════════════════════════════════════════════════════
    // STEP 2: Optional bias addition
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
        if (lm_bias.shape.total_elements() != static_cast<size_t>(hp.vocab_size))
            throw std::runtime_error("LM-head readout: invalid bias geometry");
        forward_outputs.logits_tensor = autograd::broadcast_add(forward_outputs.logits_tensor, lm_bias, stream);
    }
}
} // namespace

void forwardLmHead(
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameter_tensors,
    const Tensor& input,
    const Batching::BatchPayload& payload,
    cudaStream_t stream, cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs) {
    const int batch_size = payload.batch_size;
    const int rows_per_sequence = hp.atom_insertion_enabled
        ? payload.atom_insertion_gap_rows_per_sequence : payload.max_seq_len;
    const int total_tokens = hp.atom_insertion_enabled
        ? payload.atomInsertionGapRowCount() : payload.total_tokens;

    if (hp.atom_insertion_enabled && rows_per_sequence != payload.max_seq_len - 1)
        throw std::runtime_error("forwardLmHead: gap geometry does not match payload sequence length");
    if (!hp.atom_insertion_enabled && payload.isTraining() &&
        (hp.training_batch_size <= 0 || hp.training_rows_per_sequence <= 0 ||
         batch_size != hp.training_batch_size || rows_per_sequence != hp.training_rows_per_sequence))
        throw std::runtime_error("forwardLmHead: training payload differs from config-authored fixed shape");
    if (batch_size <= 0 || rows_per_sequence <= 0 ||
        static_cast<std::int64_t>(total_tokens) !=
            static_cast<std::int64_t>(batch_size) * rows_per_sequence)
        throw std::runtime_error("forwardLmHead: invalid payload rectangle");

    if (!input.shape.is_2d_layout()) {
        throw std::runtime_error("forwardLmHead: input must be a 2D tensor");
    }
    const int input_rows = input.shape.as_2d().rows;
    if (input_rows != total_tokens) {
        throw std::runtime_error(
            "forwardLmHead: input rows (" + std::to_string(input_rows) +
            ") != selected row geometry (" + std::to_string(total_tokens) + ")");
    }

    if (static_cast<int>(payload.seq_lengths.size()) != batch_size) {
        throw std::runtime_error("forwardLmHead: payload.seq_lengths size (" +
                                 std::to_string(payload.seq_lengths.size()) +
                                 ") != batch_size (" + std::to_string(batch_size) + ")");
    }

    if (payload.vocab_size != hp.vocab_size)
        throw std::runtime_error("forwardLmHead: payload vocabulary differs from head");
    const Tensor& normalized_input = normalizeReadoutRows(
        hp, parameter_tensors, input, stream, cublas_handle, forward_outputs);

    // Each training batch row is one sequence containing prompt and response
    // tokens. Select only its post-prompt token span here;
    // meanPoolHiddenStates itself remains a generic operation over
    // caller-authored token spans. A sequence
    // without a prompt span is pooled over all of its real (non-PAD) rows.
    if (!payload.prompt_lengths.empty() || !payload.prompt_end_positions.empty()) {
        if (static_cast<int>(payload.prompt_lengths.size()) != batch_size ||
            static_cast<int>(payload.prompt_end_positions.size()) != batch_size) {
            throw std::runtime_error(
                "forwardLmHead: prompt-boundary array size mismatch");
        }

        std::vector<MeanPoolSequenceSpan> mean_pool_spans;
        mean_pool_spans.reserve(static_cast<std::size_t>(batch_size));
        for (int batch_row = 0; batch_row < batch_size; ++batch_row) {
            const std::size_t row = static_cast<std::size_t>(batch_row);
            const int sequence_length = payload.seq_lengths[row];
            const int prompt_length = payload.prompt_lengths[row];
            const int prompt_end = payload.prompt_end_positions[row];

            int first_response_token = 0;
            if (prompt_length == 0) {
                if (prompt_end != -1) {
                    throw std::runtime_error(
                        "forwardLmHead: empty prompt requires end=-1 at batch row " +
                        std::to_string(batch_row));
                }
            } else {
                const int prompt_start = prompt_end - prompt_length + 1;
                if (prompt_length < 0 || prompt_start < 0 ||
                    prompt_end < 0 || prompt_end >= sequence_length) {
                    throw std::runtime_error(
                        "forwardLmHead: invalid prompt span at batch row " +
                        std::to_string(batch_row));
                }
                first_response_token = prompt_end + 1;
            }

            const int response_token_count =
                sequence_length - first_response_token;
            if (response_token_count <= 0) {
                throw std::runtime_error(
                    "forwardLmHead: prompt leaves no response tokens to mean-pool at batch row " +
                    std::to_string(batch_row));
            }
            mean_pool_spans.push_back(MeanPoolSequenceSpan{
                batch_row,
                first_response_token,
                sequence_length});
        }
        forward_outputs.mean_pool = meanPoolHiddenStates(
            normalized_input,
            rows_per_sequence,
            mean_pool_spans,
            stream);
    }

    projectReadoutRows(hp, parameter_tensors, normalized_input, payload, nullptr, stream,
                       forward_outputs, nullptr);
}

void forwardLmHeadReadoutChunk(
    const HyperParameters::LMHeadLayerConstructionHP& hp,
    const LMHeadParameterTensors& parameter_tensors,
    const Tensor& input,
    const Batching::BatchPayload& payload,
    const LMHeadReadoutSelection& selection,
    const Tensor* prepared_effective_weights,
    cudaStream_t stream, cublasHandle_t cublas_handle,
    Forward::ModelForwardOutputs& forward_outputs) {
    if (hp.atom_insertion_enabled || payload.EnableAtomIdentification ||
        payload.batch_size <= 0 || payload.max_seq_len <= 0 ||
        static_cast<std::int64_t>(payload.total_tokens) !=
            static_cast<std::int64_t>(payload.batch_size) * payload.max_seq_len ||
        payload.seq_lengths.size() != static_cast<std::size_t>(payload.batch_size) ||
        payload.vocab_size != hp.vocab_size)
        throw std::runtime_error("LM-head chunk: invalid token payload geometry");
    if (!input.data || !input.shape.is_2d_layout() ||
        input.shape.as_2d().rows != payload.total_tokens ||
        input.shape.as_2d().cols != hp.d_model)
        throw std::runtime_error("LM-head chunk: hidden rectangle differs from payload");
    if (selection.batch_row < 0 || selection.batch_row >= payload.batch_size ||
        selection.position_begin < 0 || selection.position_end <= selection.position_begin ||
        selection.position_end > payload.seq_lengths.at(selection.batch_row) ||
        selection.position_end > payload.max_seq_len)
        throw std::runtime_error("LM-head chunk: selection is outside real payload positions");
    const auto first_row = static_cast<std::size_t>(selection.batch_row) * payload.max_seq_len +
                           selection.position_begin;
    auto chunk = Tensor::from_ptr(input.data + first_row * hp.d_model,
        TensorContract::TensorShape::make_BSM(
            selection.position_end - selection.position_begin, hp.d_model),
        false, false, "lens_readout_chunk");
    chunk.stream = stream;
    const Tensor& normalized_input = normalizeReadoutRows(
        hp, parameter_tensors, chunk, stream, cublas_handle, forward_outputs);
    projectReadoutRows(hp, parameter_tensors, normalized_input, payload, &selection, stream,
                       forward_outputs, prepared_effective_weights);
}

} // namespace GRIM

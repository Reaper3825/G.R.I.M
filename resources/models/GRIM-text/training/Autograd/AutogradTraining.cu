//======================================================//
//  AutogradTraining.cu
//  Implementation of autograd-based training flow
//======================================================//

#include "AutogradTraining.hpp"
#include "AutogradSelectorSupervisionLoss.hpp"

// MUST include full definition of GPUGrimEncoder for method calls
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/grim_layer_gpu.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/MTP/MTP_GPU.hpp"
#include "../../Shared/LogRecorder/BatchLogTape.hpp"
#include "../../Shared/UnigramByte/Unigram.hpp"
#include "../../Shared/Execution/ExecutionPayloadValidation.hpp"
#include "../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../../Shared/Execution/DecodeTimeNumPolicy.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>  // std::clamp, std::min, std::max (+ std::min_element/max_element in diagnostics)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "../../Shared/VerboseLogging.hpp"

// Issue #142: DELETED kernelMaskNonContentLogits / launchMaskNonContentLogits.
// Setting special token logits to -inf is NON-STANDARD and was a workaround for
// mode collapse (tok1=PAD at 98% argmax). The standard approach is loss masking
// via target=-1 which is already implemented in:
//   - AutogradLoss.cu: forward skips target==-1, backward zeros grad for target==-1
//   - BatchPayload.cu: defense-masks non-content tokens with target=-1
//   - DataLoader.cu: masks non-content tokens during data loading
// The -inf masking was poisoning every diagnostic that read cached_logits_tensor
// (logit_min=-inf → logit_range=inf → logit_mean=-inf → logit_std=NaN).
// Inference-time masking in grim_language_model_gpu.cu is SEPARATE and stays.

// Logging macros - guarded by VerboseLogging flags for production
#define AG_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[AutogradTraining] INFO: " << msg << std::endl; \
    } \
} while(0)
#define AG_ERROR(msg) do { std::cerr << "[AutogradTraining] ERROR: " << msg << std::endl; } while(0)
#define AG_WARN(msg) do { std::cerr << "[AutogradTraining] WARN: " << msg << std::endl; } while(0)

namespace GRIM {
namespace Autograd {

// Finding 1 (Rule 26): countValidTokensKernel/countValidTokens DELETED — zero callers
// Finding 2 (Rule 26): sumSquaredKernel/computeSumSquared DELETED — only caller was
//   computeGradientNorm() which is redundant with Phase2's computeGradNorm()

// NOTE: linkEncoderWeightsToTrainingState was removed.
// Encoder owns its weights internally; optimizer accesses gradients via
// Tensor& accessors (enc->attnWqkv().grad_data() etc.).
// See buildParameterGroups() in LanguageModel_Training.cu.

// Context initialization lives in AutogradContext.cu so this file can focus on
// the autograd math path: forward, loss, backward, and the training-step bridge.

//======================================================================
// Autograd Forward Pass
// PRODUCTION-READY: Runs entire model with autograd graph intact
//======================================================================

// Rule 20 explicit tape sealing: skip equation-tape D2H/fprintf on non-initial
// accumulation slots. Scope MUST cover the full autograd step (forward + loss +
// backward) — sealing only forward leaves loss/backward to log identical output
// per slot, defeating the optimization and producing duplicate logs.
namespace {
struct TapeSkipScope {
    GRIM::Logging::BatchLogTape* tape;
    bool prev;
    explicit TapeSkipScope(bool skip)
        : tape(GRIM::Logging::getGlobalTape()),
          prev(tape ? tape->skipThisPass() : false) {
        if (tape) tape->setSkipThisPass(skip);
    }
    ~TapeSkipScope() { if (tape) tape->setSkipThisPass(prev); }
    TapeSkipScope(const TapeSkipScope&) = delete;
    TapeSkipScope& operator=(const TapeSkipScope&) = delete;
};

struct GradientSignalProbe {
    bool allocated = false;
    bool finite = true;
    bool nonzero = false;
    float rms = 0.0f;
    size_t sampled = 0;
};

GradientSignalProbe probeGradientSignal(Tensor& tensor, cudaStream_t stream) {
    GradientSignalProbe probe{};
    if (!tensor.data || !tensor.has_grad() || !tensor.grad_data()) {
        return probe;
    }
    probe.allocated = true;
    const size_t count = static_cast<size_t>(tensor.numel());
    probe.sampled = std::min<size_t>(count, 4096);
    if (probe.sampled == 0) {
        probe.finite = false;
        return probe;
    }

    std::vector<float> h_grad(probe.sampled);
    CUDA_CHECK(cudaMemcpyAsync(h_grad.data(), tensor.grad_data(),
                               probe.sampled * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double sum_sq = 0.0;
    for (float v : h_grad) {
        if (!std::isfinite(v)) {
            probe.finite = false;
        }
        if (v != 0.0f) {
            probe.nonzero = true;
        }
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    probe.rms = static_cast<float>(std::sqrt(sum_sq / static_cast<double>(probe.sampled)));
    if (!std::isfinite(probe.rms)) {
        probe.finite = false;
    }
    return probe;
}
}  // namespace

ForwardResult executeAutogradForward(AutogradContext& ctx) {
    ForwardResult result{};
    result.success = false;

    // Skip QKV_EQUATION D2H + fprintf on non-initial accumulation slots.
    // Rule 20 ownership taxonomy: tape skip is now scoped at autogradTrainingStep
    // so forward + loss + backward all observe the same skip flag. The local
    // scope here was a Rule 20 violation — forward sealed early, leaving loss
    // and backward to log on non-initial accumulation slots anyway.
    (void)0;

    // Rule 20: Fail loud
    ctx.validate("executeAutogradForward");
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    auto& intermediates = ts->autograd_intermediates;  // All intermediate tensors go HERE
    const auto* bindings = ctx.device_bindings;
    if (!bindings) {
        throw std::runtime_error(
            "executeAutogradForward: BatchDeviceBindings is NULL — caller must provide an explicit device view");
    }
    
    const auto& payload = *ctx.payload;
    const int total_tokens = payload.total_tokens;
    result.total_tokens = total_tokens;
    result.vocab_size = payload.vocab_size;

    AG_INFO("Autograd Forward: batch=" << ctx.batch_size << " seq=" << ctx.seq_len 
            << " tokens=" << total_tokens << " vocab=" << payload.vocab_size);
    
    // Set autograd cuBLAS handle for all matmul operations
    autograd::set_autograd_cublas_handle(ctx.cublas_handle);
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 1: Embedding Lookup
    //  Input: token_ids [total_tokens]
    //  Output: embeddings [total_tokens, d_model]
    //  
    //  Uses autograd::embedding() for gradient tracking
    // ═══════════════════════════════════════════════════════════════════════════
    
    // RULE 20: Fail loud - validate required buffers. All paths use the explicit
    // BatchDeviceBindings for this step; forward does not read TrainingState
    // cache fields as an alternate source.
    int* token_ids = bindings->d_input_ids;
    if (!token_ids) {
        throw std::runtime_error("AutogradForward: input token device pointer is NULL");
    }
    
    // Embedding weights tensor (owned by EmbeddingLayer — Pattern B)
    Tensor& emb_weights = ctx.embedding_layer->tokenWeights();
    if (!emb_weights.data) {
        throw std::runtime_error("AutogradForward: embedding token_weights.data is NULL");
    }
    emb_weights.requires_grad = ctx.is_training;
    
    // Rule 20: Fail loud on invalid shape - EmbeddingLayer constructor MUST initialize correctly
    if (!emb_weights.shape.is_valid()) {
        throw std::runtime_error("[AutogradTraining] embedding token_weights.shape is INVALID - EmbeddingLayer MUST initialize with correct shape [vocab_size=" 
                                + std::to_string(cfg->vocab_size) + ", d_model=" + std::to_string(cfg->d_model) + "]");
    }
    
    // Use autograd::embedding for proper gradient tracking
    // This performs: output[i] = weight[token_ids[i]] * scale with gradient scatter-add backward
    //
    // Issue #140: REMOVED sqrt(d_model) embedding scaling.
    // AIAYN's sqrt(d_model) was designed for sinusoidal position encodings in the residual stream.
    // GRIM-text uses ALiBi/RoPE (position info INSIDE attention, NOT residual stream).
    // With tied weights, the 27.7x scaling creates gradient asymmetry:
    //   - Embedding backward: grad_W[tok] += grad_encoder[t] * 27.7  (amplified)
    //   - LM head backward:   grad_W = centered^T @ grad_logits      (raw, no scaling)
    // This asymmetry caused non-deterministic embedding gradient spikes (0.5 → 5.2)
    // because atomicAdd scatter order varies per run, and the 27.7x amplification
    // makes small ordering differences into large gradient magnitude differences.
    // Modern LLMs with tied weights (GPT-2, LLaMA, Mistral, Gemma) do NOT scale.
    const float embedding_scale = 1.0f;
    Tensor emb_output = autograd::embedding(
        emb_weights,
        token_ids,  // Use local variable from Tensor cast
        total_tokens,
        ctx.stream,
        embedding_scale  // Issue #140: No scaling (1.0f)
    );
    
    AG_INFO("Step 1b: No position embeddings (using "
            << HyperParameters::positionalEncodingTypeToString(cfg->positional_encoding)
            << " inside attention)");

    // Store in intermediates for backward
    intermediates.embedding_tensor = std::move(emb_output);
    if (cfg->dropout_rate > 0.0f) {
        // Vary seed per step so each batch sees a DIFFERENT dropout mask.
        const uint64_t emb_dropout_seed = ctx.step * 2654435761ULL + 500;
        intermediates.embedding_tensor = autograd::dropout(intermediates.embedding_tensor, cfg->dropout_rate, 
                                                  emb_dropout_seed, ctx.is_training, ctx.stream);
        AG_INFO("Step 1c: Embedding dropout " << (ctx.is_training ? "applied" : "skipped (eval mode)")
                << " (p=" << cfg->dropout_rate << ", step=" << ctx.step << ")");
    }
    
    AG_INFO("Step 1: Embedding complete, shape=[" << total_tokens << ", " << cfg->d_model << "]");
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 1.5: ScratchBlock (numeric/code processing)
    //  NOTE: ScratchBlock operates IN-PLACE on the embedding buffer.
    //  The backward pass uses cached atom positions and types.
    // ═══════════════════════════════════════════════════════════════════════════
    
    if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
        AG_INFO("Step 1.5: Running ScratchBlock injection...");
        // Drain stale error from embedding/dropout so we only report ScratchBlock failures
        (void)cudaGetLastError();

        if (!bindings->d_numeric_values) {
            throw std::runtime_error("executeAutogradForward: ScratchBlock requires BatchDeviceBindings.d_numeric_values");
        }
        if (!bindings->d_token_to_slot_map) {
            throw std::runtime_error("executeAutogradForward: ScratchBlock requires BatchDeviceBindings.d_token_to_slot_map");
        }

        const bool exec_first_type_only =
            cfg->execution_block_enabled && cfg->scratch_block_execution_first_type_only;
        if (cfg->execution_block_enabled && !exec_first_type_only) {
            throw std::runtime_error(
                "executeAutogradForward: execution_block_enabled requires "
                "scratch_block_execution_first_type_only=true to prevent value leakage "
                "into hidden states on arithmetic batches");
        }
        intermediates.embedding_tensor = autograd::scratch_block_inject(
            intermediates.embedding_tensor,
            *ctx.scratch_block,
            token_ids,
            bindings->d_numeric_values,
            bindings->d_text_features,
            bindings->d_atom_mask,
            bindings->d_atom_flags,
            bindings->d_token_to_slot_map,
            total_tokens,
            ctx.stream,
            exec_first_type_only);

        cudaError_t cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            throw std::runtime_error("AutogradForward: ScratchBlock CUDA error: " +
                                     std::string(cudaGetErrorString(cuda_err)));
        }

        AG_INFO("Step 1.5: ScratchBlock complete");
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 2: Encoder Layers (transformer blocks)
    //  Encoder layer outputs stored in intermediates to keep autograd graph alive.
    // ═══════════════════════════════════════════════════════════════════════════
    
    if (!ctx.gpu_encoder) {
        throw std::runtime_error("AutogradForward: gpu_encoder is NULL - pass encoder in context");
    }
    
    const int num_layers = ctx.gpu_encoder->getNumLayers();
    intermediates.encoder_layer_outputs.clear();
    intermediates.layer_intermediates.layers.clear();
    intermediates.embedding_tensor.is_leaf = false;
    
    float* encoder_output = nullptr;
    
    if (!ctx.is_training) {
        // ═══════════════════════════════════════════════════════════════════════
        //  NO-GRAD (validation): Do not store full autograd graph.
        //  Reuse a single layer's intermediates and one running tensor so we use
        //  O(1) memory instead of O(num_layers) — avoids training-level memory.
        // ═══════════════════════════════════════════════════════════════════════
        ForwardIntermediates no_grad_layer_storage;
        Tensor running;
        AG_INFO("Step 2: Running " << num_layers << " encoder layers (no_grad, validation)...");
        
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = ctx.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("AutogradForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }
            // Sync before clear so we never free Q/K buffers while previous layer's
            // kernels (RoPE, etc.) are still in flight — avoids illegal memory access.
            if (layer_idx > 0) {
                cudaError_t sync_err = cudaStreamSynchronize(ctx.stream);
                if (sync_err != cudaSuccess) {
                    throw std::runtime_error("AutogradForward(no_grad): cudaStreamSynchronize failed after layer " +
                        std::to_string(layer_idx - 1) + ": " + cudaGetErrorString(sync_err));
                }
            }
            no_grad_layer_storage.clear();
            Tensor& layer_input = (layer_idx == 0) ? intermediates.embedding_tensor : running;
            running = enc_layer->forward(layer_input, *ctx.payload, ctx.stream, no_grad_layer_storage, ctx.step, layer_idx);
            // Encoder returns a non-owning view of no_grad_layer_storage.output. That storage
            // is cleared next iteration (or destroyed when the loop exits). So running would
            // become a dangling pointer for the next layer OR for post-encoder use (final RMS,
            // LM head) — root cause of inference illegal memory access. Always make running own
            // a copy so we never hold a pointer into no_grad_layer_storage.
            {
                Tensor owned = Tensor::empty(running.shape, false, ctx.stream, "no_grad_layer_output");
                const size_t bytes = static_cast<size_t>(running.shape.total_elements()) * sizeof(float);
                cudaError_t cp_err = cudaMemcpyAsync(owned.data, running.data, bytes, cudaMemcpyDeviceToDevice, ctx.stream);
                if (cp_err != cudaSuccess) {
                    throw std::runtime_error("AutogradForward(no_grad): copy layer output failed: " +
                        std::string(cudaGetErrorString(cp_err)));
                }
                // Ensure copy completes before next iteration's clear() (or storage destructor) frees the source.
                cudaError_t sync_err = cudaStreamSynchronize(ctx.stream);
                if (sync_err != cudaSuccess) {
                    throw std::runtime_error("AutogradForward(no_grad): sync after layer output copy failed: " +
                        std::string(cudaGetErrorString(sync_err)));
                }
                running = std::move(owned);
            }
        }
        
        // Surface any device error from the encoder loop before we use encoder_output.
        {
            cudaError_t enc_sync = cudaStreamSynchronize(ctx.stream);
            if (enc_sync != cudaSuccess) {
                throw std::runtime_error("AutogradForward(no_grad): CUDA error after encoder layers: " +
                    std::string(cudaGetErrorString(enc_sync)) + " (illegal access usually means a kernel wrote/read out of bounds)");
            }
        }
        
        encoder_output = running.data;
        intermediates.encoder_output_tensor = std::move(running);
        intermediates.encoder_output_tensor.requires_grad = false;
        intermediates.encoder_output_tensor.grad_fn.reset();
        intermediates.encoder_output_tensor.stream = ctx.stream;
        AG_INFO("Step 2: All " << num_layers << " encoder layers complete (no_grad)");
    } else {
        // ═══════════════════════════════════════════════════════════════════════
        //  Issue #56: Keep all intermediate tensors alive until backward completes.
        // ═══════════════════════════════════════════════════════════════════════
        intermediates.encoder_layer_outputs.reserve(num_layers);
        intermediates.layer_intermediates.layers.reserve(num_layers);
        
        AG_INFO("Step 2: Running " << num_layers << " encoder layers with autograd...");
        AG_INFO("  embedding_tensor.grad_fn=" << (void*)intermediates.embedding_tensor.grad_fn.get()
                << " requires_grad=" << intermediates.embedding_tensor.requires_grad);
        
        // Resolve execution block layer index
        int exec_layer = -1;
        int exec_K = 0;
        if (ctx.execution_block && cfg->execution_block_enabled && ctx.scratch_block && ctx.scratch_block->isEnabled()) {
            exec_layer = cfg->execution_block_layer;
            if (exec_layer < 0) exec_layer = num_layers - 2;
            if (exec_layer < 0) exec_layer = 0;
            if (exec_layer >= num_layers) exec_layer = num_layers - 1;
            exec_K = cfg->execution_block_num_steps;
        }

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = ctx.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("AutogradForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }
            
            intermediates.layer_intermediates.layers.emplace_back();
            ForwardIntermediates& layer_storage = intermediates.layer_intermediates.layers.back();
            
            Tensor& layer_input = (layer_idx == 0)
                ? intermediates.embedding_tensor
                : intermediates.encoder_layer_outputs.back();
            
            Tensor layer_output = enc_layer->forward(layer_input, *ctx.payload, ctx.stream, layer_storage, ctx.step, layer_idx);

            // ExecutionBlock: run K execution steps at the configured layer
            // Per-row isolation: each batch row gets its own ExecutionMemory
            if (layer_idx == exec_layer && ctx.execution_block) {
                const int ae = cfg->scratch_block_atom_embedding_dim;
                const int V = cfg->execution_block_num_slots;
                const int nop = cfg->execution_block_num_ops;
                const int dk = cfg->execution_block_d_key;
                const int dt = cfg->execution_block_d_type;
                const int B  = ctx.batch_size;
                const int sl = ctx.seq_len;

                intermediates.exec_memories.resize(B);
                intermediates.exec_outputs_per_row.resize(B);

                float T = cfg->execution_block_temp_start;

                // Initialize persistent execution trace per row
                // Vectors sized B for index stability; only active rows get allocated tensors.
                ts->execution_trace_by_row.resize(B);
                ts->trace_state_by_row.resize(B);
                for (int b = 0; b < B; ++b) {
                    ts->execution_trace_by_row[b].clear();
                    const bool row_active = !ctx.payload->execution_active.empty()
                        && ctx.payload->execution_active[b];
                    if (row_active) {
                        ts->trace_state_by_row[b] = Tensor::zeros({1, cfg->d_model}, ctx.stream, "trace_state_row");
                        ts->trace_state_by_row[b].requires_grad_();
                        ts->trace_state_by_row[b].ensure_grad();
                    }
                }

                // Teacher target buffer for causal loss
                const bool have_exec_teacher = (ctx.payload && !ctx.payload->teacher_steps.empty());
                const int B_teacher = have_exec_teacher
                    ? static_cast<int>(ctx.payload->teacher_steps.size()) : 0;
                intermediates.exec_expected_target_tensors.clear();
                intermediates.exec_expected_target_tensors.reserve(
                    static_cast<size_t>(B) * static_cast<size_t>(std::max(0, exec_K)));

                for (int b = 0; b < B; ++b) {
                    // WS7: Only allocate and execute for rows with active compiled payload.
                    const bool row_exec_active = !ctx.payload->execution_active.empty()
                        && ctx.payload->execution_active[b];

                    intermediates.exec_outputs_per_row[b].steps.clear();

                    if (!row_exec_active) continue;

                    auto& M_b = intermediates.exec_memories[b];
                    M_b.allocate(V, ae, cfg->d_model, dk, dt, ctx.stream);
                    M_b.clear(ctx.stream);

                    const int tok_off = b * sl;

                    auto row_atom_view = ctx.scratch_block->extractRowLocalAtomView(
                        tok_off, sl, ctx.stream);

                    // WS7: Execution-active rows MUST have bootstrap data.
                    // Throw instead of silently entering executeStep with empty memory.
                    if (!ctx.device_bindings || !ctx.device_bindings->d_token_to_slot_map
                        || !ctx.device_bindings->d_numeric_values) {
                        throw std::runtime_error(
                            "AutogradTraining: execution-active row " + std::to_string(b)
                            + " has no slot map or numeric values for bootstrap — "
                            "compiled payload marks row active but bootstrap data is missing");
                    }
                    ctx.execution_block->bootstrapMemoryFromSlotMap(
                        M_b,
                        ctx.device_bindings->d_numeric_values + tok_off,
                        ctx.device_bindings->d_token_to_slot_map + tok_off,
                        sl, ctx.stream);

                    for (int step = 0; step < exec_K; ++step) {
                        ExecutionBlockStepOutput step_diag;

                        // Upload teacher expected_value for this step.
                        // teacher_steps is supervision data, not the activation signal.
                        const float* d_expected_target = nullptr;
                        TeacherSelectionTargets selection_targets;  // valid=false by default

                        if (have_exec_teacher && b < B_teacher) {
                            const auto& teacher_row = ctx.payload->teacher_steps[b];
                            if (step < static_cast<int>(teacher_row.size())) {
                                const auto& ts_k = teacher_row[step];
                                float h_val = ts_k.expected_value;
                                Tensor expected_target_tensor = Tensor::empty(
                                    TensorContract::TensorShape::make_BSM(1, 1),
                                    false,
                                    ctx.stream,
                                    "exec_expected_target_owned");
                                cudaMemcpyAsync(expected_target_tensor.data, &h_val,
                                                sizeof(float), cudaMemcpyHostToDevice, ctx.stream);
                                d_expected_target = expected_target_tensor.data;
                                intermediates.exec_expected_target_tensors.push_back(std::move(expected_target_tensor));

                                // Build selection targets for autograd CE if enabled.
                                // Teacher emits absolute slot indices with base_slot=0.
                                // arg softmax spans [0, V-S), write softmax spans [0, V).
                                // S=0 currently, so arg targets = arg_slot directly.
                                if (cfg->structured_ce_enabled) {
                                    const int S_scratch = 0;  // scratch slots not yet used
                                    const int V_val = V - S_scratch;

                                    // Bounds-check every target index before marking valid.
                                    const int arg1_idx = ts_k.arg1_slot - S_scratch;
                                    const int arg2_idx = ts_k.arg2_slot - S_scratch;
                                    const int write_idx = ts_k.write_slot;

                                    if (ts_k.op_id < 0 || ts_k.op_id >= nop)
                                        throw std::runtime_error(
                                            "AutogradTraining: teacher op_id=" + std::to_string(ts_k.op_id)
                                            + " out of range [0," + std::to_string(nop) + ") at row="
                                            + std::to_string(b) + " step=" + std::to_string(step));
                                    if (arg1_idx < 0 || arg1_idx >= V_val)
                                        throw std::runtime_error(
                                            "AutogradTraining: teacher arg1_slot=" + std::to_string(ts_k.arg1_slot)
                                            + " maps to index " + std::to_string(arg1_idx)
                                            + " out of range [0," + std::to_string(V_val) + ") at row="
                                            + std::to_string(b) + " step=" + std::to_string(step));
                                    if (arg2_idx < 0 || arg2_idx >= V_val)
                                        throw std::runtime_error(
                                            "AutogradTraining: teacher arg2_slot=" + std::to_string(ts_k.arg2_slot)
                                            + " maps to index " + std::to_string(arg2_idx)
                                            + " out of range [0," + std::to_string(V_val) + ") at row="
                                            + std::to_string(b) + " step=" + std::to_string(step));
                                    if (write_idx < 0 || write_idx >= V)
                                        throw std::runtime_error(
                                            "AutogradTraining: teacher write_slot=" + std::to_string(ts_k.write_slot)
                                            + " out of range [0," + std::to_string(V) + ") at row="
                                            + std::to_string(b) + " step=" + std::to_string(step));

                                    selection_targets.op_target    = ts_k.op_id;
                                    selection_targets.arg1_target  = arg1_idx;
                                    selection_targets.arg2_target  = arg2_idx;
                                    selection_targets.write_target = write_idx;
                                    selection_targets.valid = true;

                                    // Step mask check: if step is masked, don't supervise
                                    const bool have_step_mask_here =
                                        (ctx.payload && !ctx.payload->teacher_step_mask.empty()
                                         && b < static_cast<int>(ctx.payload->teacher_step_mask.size())
                                         && step < static_cast<int>(ctx.payload->teacher_step_mask[b].size()));
                                    if (have_step_mask_here && ctx.payload->teacher_step_mask[b][step] == 0) {
                                        selection_targets.valid = false;
                                    }
                                }
                            }
                        }

                        ctx.execution_block->executeStep(
                            layer_output, M_b,
                            reinterpret_cast<const int*>(row_atom_view.atom_positions.data),
                            row_atom_view.num_atoms, *ctx.payload, *ctx.device_bindings, b,
                            step, T, ctx.stream,
                            &step_diag,
                            ts->trace_state_by_row[b],
                            ts->execution_trace_by_row[b],
                            d_expected_target,
                            cfg->structured_ce_enabled ? &selection_targets : nullptr);
                        ts->execution_trace_by_row[b].push_back(step_diag.record);
                        intermediates.exec_outputs_per_row[b].steps.push_back(std::move(step_diag));
                    }
                }
            }

            // ExecutionBlock: gated cross-attention read at every layer >= exec_layer (per-row)
            // WS7: Only execution-active rows participate in cross-attention read.
            // Each row returns a [sl, dm] delta — zero_pad places it at the correct
            // offset in [total_tokens, dm], then autograd::add merges into layer_output.
            if (exec_layer >= 0 && layer_idx >= exec_layer
                && ctx.execution_block
                && !intermediates.exec_memories.empty()) {
                const int B  = ctx.batch_size;
                const int sl = ctx.seq_len;
                for (int b = 0; b < B; ++b) {
                    const bool row_exec_active = !ctx.payload->execution_active.empty()
                        && ctx.payload->execution_active[b];
                    if (!row_exec_active) continue;
                    Tensor row_delta = ctx.execution_block->crossAttentionRead(
                        layer_output, intermediates.exec_memories[b],
                        total_tokens, ctx.stream,
                        b * sl, sl,
                        ts->read_gate_accum_tensor.data);
                    Tensor padded = autograd::zero_pad(row_delta, b * sl, total_tokens, ctx.stream);
                    layer_output = autograd::add(layer_output, padded, ctx.stream);
                }
            }

            // Issue #155: Post-layer centering — single centering point AFTER all
            // modifications to layer_output (encoder forward + dropout + crossAttentionRead).
            // The encoder layer's post-FFN centering was removed (moved here) so that
            // crossAttentionRead's in-place injection doesn't bypass it.
            // Post-attention centering remains inside the encoder (between sublayers).
            if (cfg->center_encoder_residuals) {
                layer_output = autograd::center_columns(layer_output, ctx.stream);
            }
            
            intermediates.encoder_layer_outputs.push_back(std::move(layer_output));
        }
        
        AG_INFO("Step 2: All " << num_layers << " encoder layers complete");
        encoder_output = intermediates.encoder_layer_outputs.back().data;
    }
    
    // Final encoder output pointer for LM head and diagnostics
    result.encoder_output = encoder_output;
    
    // Copy raw encoder output to cached buffer for inference slot selector
    // (prefill reads this BEFORE LM head centering is applied).
    // After LM head forward below, we overwrite with the actual LM head input
    // (centered if centering is on, raw otherwise) — single source of truth.
    if (ts->cached_encoder_output.data) {
        cudaMemcpyAsync(ts->cached_encoder_output.data, encoder_output,
                        static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                        cudaMemcpyDeviceToDevice, ctx.stream);
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP 3+4: LM Head → Logits (via PERSISTENT LMHeadLayer)
    //
    //  Pattern B: LMHeadLayer is constructed ONCE in initGPU() and stored on
    //  LanguageModel. The layer self-allocates weights (or aliases embedding
    //  weights for tied config) and owns final_rms_gamma.
    //
    //  LMHeadLayer::forward() encapsulates the FULL pipeline:
    //    Step 0: RMSNorm (final_rms_gamma)
    //    Step 1: Optional column+row centering (Issues #125/#132)
    //    Step 2: Linear projection: logits = centered_encoder @ W^T
    //    Step 3: Optional logit centering (numerical stability)
    //    Step 4: Optional bias addition
    //
    //  Autograd graph built inside forward() - backward handled automatically.
    // ═══════════════════════════════════════════════════════════════════════════
    
    // When training: create encoder output tensor from last layer (preserves grad_fn chain).
    // When validation (no_grad): intermediates.encoder_output_tensor already set in no_grad path above.
    if (ctx.is_training) {
        const bool lmhead_track_grad = true;
        Tensor encoder_output_tensor = Tensor::from_ptr(
            encoder_output,
            TensorContract::TensorShape::make_BSM(total_tokens, cfg->d_model),
            false,
            lmhead_track_grad,
            "encoder_output_for_lmhead"
        );
        encoder_output_tensor.is_leaf = false;
        encoder_output_tensor.stream = ctx.stream;
        encoder_output_tensor.grad_fn = intermediates.encoder_layer_outputs.back().grad_fn;
        intermediates.encoder_output_tensor = std::move(encoder_output_tensor);
    }
    // else: no_grad path already set intermediates.encoder_output_tensor and cleared grad_fn

    // Update persistent LM head with current stream/cublas
    ctx.lm_head->setStream(ctx.stream);
    ctx.lm_head->setCublasHandle(ctx.cublas_handle);

    // Inference path needs cached_logits for return value; training also writes
    // a Cat-3 snapshot to cached_logits_tensor (Rule 20 step-output buffer) so
    // post-backward diagnostics never read intermediates.logits_tensor (Cat 1).
    float* logits_output = ts->cached_logits_tensor.data;
    if (!logits_output) {
        throw std::runtime_error("AutogradTraining: cached_logits_tensor buffer is NULL - TrainingState MUST allocate logits buffer for inference and training");
    }
    
    // Forward pass: builds autograd graph through RMSNorm → centering → matmul → bias
    Tensor logits_tensor = ctx.lm_head->forward(intermediates.encoder_output_tensor, intermediates.centered_encoder_output); // Pass centered output tensor for diagnostics, even if centering is disabled (will be empty tensor in that case, LM head ignores it).
    
    // Overwrite cached_encoder_output with what the LM head actually used.
    // When centering is on, this is the centered output; otherwise the raw encoder output
    // (already written above, so this is a no-op in that case).
    // Single source of truth: every diagnostic reads cached_encoder_output, period.
    if (cfg->lm_head_center_hidden_states && ts->cached_encoder_output.data && intermediates.centered_encoder_output.data) {
        cudaMemcpyAsync(ts->cached_encoder_output.data, intermediates.centered_encoder_output.data,
                       static_cast<size_t>(total_tokens) * cfg->d_model * sizeof(float),
                       cudaMemcpyDeviceToDevice, ctx.stream);
    }
    
    // Pointer for inline diagnostic reads — same data as cached_encoder_output
    float* lm_input_ptr = cfg->lm_head_center_hidden_states 
        ? intermediates.centered_encoder_output.data 
        : intermediates.encoder_output_tensor.data;
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  RULE 21 DIAGNOSTIC: Why is token 277 (or any token) the argmax?
    //  (GUARDED: set ENABLE_EXPENSIVE_DIAGNOSTICS=true in VerboseLogging.hpp to enable)
    // ═══════════════════════════════════════════════════════════════════════════
    if constexpr (GRIM::VerboseLogging::ENABLE_EXPENSIVE_DIAGNOSTICS) {
        //  Equation: logit[v] = Σ_d encoder[pos, d] × W[v, d]
        //  
        //  Per-position argmax analysis: WHY did position choose its predicted token?
        //  The argmax wins because: Σ_d h[d] × W[argmax, d] > Σ_d h[d] × W[v, d] for all v
        //  
        //  This diagnostic shows:
        //  1. Hidden state (encoder output) statistics at sample position
        //  2. Weight row statistics for the predicted argmax token at that position  
        //  3. Dot product decomposition showing WHY argmax wins
        constexpr int kSamplePositions = 1024;  // Sample first 1024 positions
        const int d_model = cfg->d_model;
        const int vocab_size_local = cfg->vocab_size;
        
        // Copy sample data to host
        const int sample_size = std::min(kSamplePositions, total_tokens);
        std::vector<float> h_encoder(sample_size * d_model);
        std::vector<float> h_logits(sample_size * vocab_size_local);
        
        cudaMemcpyAsync(h_encoder.data(), lm_input_ptr,
                        sample_size * d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaMemcpyAsync(h_logits.data(), logits_tensor.data,
                        sample_size * vocab_size_local * sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        
        for (int pos = 0; pos < sample_size; ++pos) {
            const float* h = h_encoder.data() + pos * d_model;
            const float* logits_row = h_logits.data() + pos * vocab_size_local;
            
            // Find argmax and its logit FIRST (since we need to fetch W[argmax])
            int argmax_token = 0;
            float max_logit_val = logits_row[0];
            for (int v = 1; v < vocab_size_local; ++v) {
                if (logits_row[v] > max_logit_val) {
                    max_logit_val = logits_row[v];
                    argmax_token = v;
                }
            }
            
            // Fetch W[argmax] row from device
            std::vector<float> h_weights_argmax(d_model);
            cudaMemcpyAsync(h_weights_argmax.data(), 
                            ctx.lm_head->weights().data + static_cast<size_t>(argmax_token) * d_model,
                            d_model * sizeof(float),
                            cudaMemcpyDeviceToHost, ctx.stream);
            cudaStreamSynchronize(ctx.stream);
            
            // Compute hidden state statistics
            float h_sum = 0.0f, h_sum_sq = 0.0f, h_min = h[0], h_max = h[0];
            for (int d = 0; d < d_model; ++d) {
                h_sum += h[d];
                h_sum_sq += h[d] * h[d];
                h_min = std::min(h_min, h[d]);
                h_max = std::max(h_max, h[d]);
            }
            float h_mean = h_sum / d_model;
            float h_rms = std::sqrt(h_sum_sq / d_model);
            
            // Compute W[argmax] statistics
            float w_sum = 0.0f, w_sum_sq = 0.0f, w_min = h_weights_argmax[0], w_max = h_weights_argmax[0];
            for (int d = 0; d < d_model; ++d) {
                w_sum += h_weights_argmax[d];
                w_sum_sq += h_weights_argmax[d] * h_weights_argmax[d];
                w_min = std::min(w_min, h_weights_argmax[d]);
                w_max = std::max(w_max, h_weights_argmax[d]);
            }
            float w_mean = w_sum / d_model;
            float w_rms = std::sqrt(w_sum_sq / d_model);
            
            // Compute dot product decomposition for argmax token
            float dot_product_argmax = 0.0f;
            float positive_contrib = 0.0f, negative_contrib = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                float contrib = h[d] * h_weights_argmax[d];
                dot_product_argmax += contrib;
                if (contrib > 0) positive_contrib += contrib;
                else negative_contrib += contrib;
            }
            
            // Compute cosine similarity between h and W[argmax]
            // cos = dot / (rms_h * rms_w * d_model)  [equivalent to dot / (||h|| * ||w||)]
            float h_rms_val = std::sqrt(h_sum_sq / d_model);
            float w_rms_val = std::sqrt(w_sum_sq / d_model);
            float cosine_sim = (h_rms_val > 1e-8f && w_rms_val > 1e-8f) 
                               ? (dot_product_argmax / (h_rms_val * w_rms_val * d_model)) : 0.0f;
            
            fprintf(stderr, "═══════════════════════════════════════════════════════════════════════════\n");
            fprintf(stderr, "[LOGIT_ANALYSIS] Position %d: Why does logit[v] = Σ_d h[d] × W[v,d] choose token %d?\n", pos, argmax_token);
            fprintf(stderr, "═══════════════════════════════════════════════════════════════════════════\n");
            
            // Analyze hidden state properties
            fprintf(stderr, "HIDDEN STATE h[pos=%d]:\n", pos);
            fprintf(stderr, "  Statistics: mean=%.6f (offset) rms=%.6f (magnitude) range=[%.6f, %.6f]\n",
                    h_mean, h_rms, h_min, h_max);
            fprintf(stderr, "  ├─ mean≠0 → systematic bias in dot products (mean × W_sum term)\n");
            fprintf(stderr, "  ├─ rms=magnitude → scales all dot products proportionally\n");
            fprintf(stderr, "  └─ range→variation across dimensions affects different weight rows differently\n");
            
            // Analyze weight row for the argmax token (what was actually predicted)
            fprintf(stderr, "WEIGHT ROW W[%d] (predicted token):\n", argmax_token);
            fprintf(stderr, "  Statistics: mean=%.6f rms=%.6f range=[%.6f, %.6f]\n",
                    w_mean, w_rms, w_min, w_max);
            fprintf(stderr, "  ├─ Initialized to mean≈0 (ideal), rms controls sensitivity\n");
            fprintf(stderr, "  └─ Alignment with h determines dot product magnitude\n");
            
            // Decompose the dot product
            fprintf(stderr, "DOT_PRODUCT ANALYSIS Σ_d h[d]×W[%d,d]:\n", argmax_token);
            fprintf(stderr, "  Raw computation: %.6f\n", dot_product_argmax);
            fprintf(stderr, "  ├─ Positive contributions (h×W>0): %.6f (%.1f%%)\n", 
                    positive_contrib, 100.0f * positive_contrib / (std::abs(dot_product_argmax) + 1e-8f));
            fprintf(stderr, "  ├─ Negative contributions (h×W<0): %.6f (%.1f%%)\n",
                    negative_contrib, 100.0f * std::abs(negative_contrib) / (std::abs(dot_product_argmax) + 1e-8f));
            fprintf(stderr, "  ├─ Cosine alignment: %.6f (1.0=perfect alignment, 0=orthogonal, -1=opposite)\n", cosine_sim);
            fprintf(stderr, "  └─ Interpretation: h and W[%d] are %.1f%% aligned\n", argmax_token, 100.0f * cosine_sim);
            
            // Result summary
            fprintf(stderr, "RESULT:\n");
            fprintf(stderr, "  logit[%d]=%.6f (PREDICTED argmax token)\n", argmax_token, max_logit_val);
            fprintf(stderr, "  ├─ Token %d wins because: cos(h, W[%d])=%.3f (alignment) × magnitude\n", 
                    argmax_token, argmax_token, cosine_sim);
            fprintf(stderr, "  └─ Hidden state h[%d] points in direction captured by W[%d]\n", pos, argmax_token);
            
            // Reflection on what this means  
            fprintf(stderr, "REFLECTION:\n");
            if (cosine_sim > 0.5f) {
                fprintf(stderr, "  ⚠️  High alignment (cos=%.3f > 0.5) detected:\n", cosine_sim);
                fprintf(stderr, "     - Hidden state strongly aligned with predicted token weight\n");
                fprintf(stderr, "     - This is expected behavior for confident predictions\n");
                fprintf(stderr, "     - Mode collapse: check if SAME token wins across many positions\n");
            } else if (cosine_sim < 0.1f) {
                fprintf(stderr, "  ⚠️  Low alignment (cos=%.3f < 0.1) detected:\n", cosine_sim);
                fprintf(stderr, "     - Prediction driven more by magnitude than direction\n");
                fprintf(stderr, "     - May indicate weak position differentiation\n");
            } else {
                fprintf(stderr, "  ✓ Moderate alignment (cos=%.3f) is healthy\n", cosine_sim);
            }
            fprintf(stderr, "\n");
        }
    }  // end ENABLE_EXPENSIVE_DIAGNOSTICS guard
    
    // Shape validation, centering, bias addition all handled by LMHeadLayer::forward()

    // Rule 20 ownership taxonomy: snapshot Cat-1 logits.data into the Cat-3
    // workspace `cached_logits_tensor` BEFORE the autograd boundary so
    // post-backward diagnostics (Phase2_TrainingLoop, GuessCacheTraining,
    // ComputeLossBatch) read from TrainingState rather than from
    // intermediates.logits_tensor (which AutogradStepScope clears).
    cudaMemcpyAsync(logits_output, logits_tensor.data,
                    logits_tensor.shape.total_elements() * sizeof(float),
                    cudaMemcpyDeviceToDevice, ctx.stream);
    
    // Move the autograd tensor to intermediates (preserves grad_fn chain)
    intermediates.logits_tensor = std::move(logits_tensor);
    
    // Reasoning head: gated OFF when execution_block_enabled (single execution decision source)
    if (ctx.reasoning_head && ctx.scratch_block && ctx.scratch_block->isEnabled()
        && !cfg->execution_block_enabled) {
        // D2H num_atoms — numAtomsBuffer() is device memory
        int num_atoms = 0;
        cudaMemcpyAsync(&num_atoms, ctx.scratch_block->numAtomsBuffer(),
                        sizeof(int), cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);

        if (num_atoms > 0) {
            const int atom_dim = cfg->scratch_block_atom_embedding_dim;

            // Create canonical copy-first atom embeddings tensor (AutogradIntermediates owns it)
            intermediates.scratch_atom_embeddings = Tensor::empty(
                TensorContract::TensorShape::make_BSM(num_atoms, atom_dim),
                ctx.is_training, ctx.stream, "scratch_atom_embeddings");
            cudaMemcpyAsync(
                intermediates.scratch_atom_embeddings.data,
                ctx.scratch_block->atomEmbeddingsBuffer(),
                static_cast<size_t>(num_atoms) * atom_dim * sizeof(float),
                cudaMemcpyDeviceToDevice, ctx.stream);

            ctx.reasoning_head->setStream(ctx.stream);
            ctx.reasoning_head->setCublasHandle(ctx.cublas_handle);

            ReasoningHeadOutput rh_out = ctx.reasoning_head->forward(
                intermediates.encoder_output_tensor,
                intermediates.scratch_atom_embeddings,
                ctx.scratch_block->atomPositionsBuffer(),
                num_atoms,
                total_tokens,
                ctx.stream);
            intermediates.reasoning_output = std::move(rh_out);
        }
    }
    
    AG_INFO("Forward complete: logits shape=[" << total_tokens << ", " << cfg->vocab_size << "]");
    
    result.success = true;
    return result;
}

//======================================================================
// Loss Config Builder (single conversion point)
//======================================================================

autograd::LossConfig buildLossConfig(const LossContext::LossOptions& opts, const float* d_class_weights) {
    autograd::LossConfig lc{};
    lc.focal_enabled       = opts.focal_enabled;
    lc.focal_alpha         = opts.focal_enabled ? opts.focal_alpha : 1.0f;
    lc.focal_gamma         = opts.focal_enabled ? opts.focal_gamma : 0.0f;
    lc.smoothing_enabled   = opts.label_smoothing_enabled;
    lc.smoothing_epsilon   = opts.label_smoothing_enabled ? opts.label_smoothing_epsilon : 0.0f;
    lc.entropy_reg_enabled = opts.entropy_reg_enabled;
    lc.entropy_reg_lambda  = opts.entropy_reg_enabled ? opts.entropy_reg_lambda : 0.0f;
    lc.class_balanced_enabled = opts.class_balanced_enabled;
    lc.d_class_weights     = opts.class_balanced_enabled ? d_class_weights : nullptr;
    return lc;
}

//======================================================================
// Autograd Loss Computation
//======================================================================

LossResult computeAutogradLoss(
    AutogradContext& ctx
) {
    LossResult result{};
    result.success = false;
    
    // RULE 20: Fail loud
    ctx.validate("computeAutogradLoss");
    if (!ctx.payload) {
        throw std::runtime_error("computeAutogradLoss: ctx.payload is NULL — training path MUST set payload via initAutogradContext(const BatchPayload&, ...)");
    }
    const auto& payload = *ctx.payload;
    payload.validate("computeAutogradLoss");
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    
    // RULE 20: Fail loud - validate logits tensor was populated by forward pass
    auto& intermediates = ts->autograd_intermediates;
    if (!intermediates.logits_tensor.data) {
        throw std::runtime_error("computeAutogradLoss: Logits tensor not initialized - call executeAutogradForward() first");
    }
    
    // BatchPayload owns target semantics; BatchDeviceBindings is the explicit
    // per-step device view of the uploaded target mirror.
    if (!ctx.device_bindings || !ctx.device_bindings->d_target_ids) {
        throw std::runtime_error(
            "computeAutogradLoss: BatchDeviceBindings target pointer is NULL - "
            "caller must upload the Phase1-authored BatchPayload before loss");
    }
    
    const int total_tokens = payload.total_tokens;
    const int vocab_size = payload.vocab_size;
    const int lm_valid_tokens = payload.lm_valid_tokens;
    
    AG_INFO("Computing loss: tokens=" << total_tokens << " vocab=" << vocab_size
            << " lm_valid=" << lm_valid_tokens);
    
    // ═══════════════════════════════════════════════════════════════════════════
    // 1. TEXT CROSS-ENTROPY LOSS (autograd::unified_loss)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Compute text CE - returns scalar Tensor with NLLLossGradFn → LogSoftmaxGradFn chain
    Tensor loss_tensor = autograd::unified_loss(
        intermediates.logits_tensor,
        payload,
        *ctx.device_bindings,
        ctx.loss_config,
        ctx.stream
    );
    
    // Move loss tensor to intermediates (TrainingState owns it during backward)
    intermediates.loss_tensor = std::move(loss_tensor);

    float text_ce_loss = 0.0f;
    cudaMemcpyAsync(&text_ce_loss, intermediates.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    if (!std::isfinite(text_ce_loss)) {
        throw std::runtime_error("computeAutogradLoss: pure text CE is non-finite (" +
                                 std::to_string(text_ce_loss) + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. MTP (multi-token prediction) auxiliary losses: L_total += α/K * Σ_k L_k
    // Delegated to MTP_GPU module (see Shared/MTP/MTP_GPU.cu)
    // ═══════════════════════════════════════════════════════════════════════════
    GRIM::MTP::computeMTPAuxiliaryLosses(ctx, intermediates, result.mtp_diagnostics);

    float mtp_loss = 0.0f;
    if (result.mtp_diagnostics.valid) {
        for (float head_contribution : result.mtp_diagnostics.head_loss) {
            mtp_loss += head_contribution;
        }
        result.mtp_diagnostics.L_total = text_ce_loss + mtp_loss;
    }

    float text_plus_mtp_loss = 0.0f;
    cudaMemcpyAsync(&text_plus_mtp_loss, intermediates.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    if (!std::isfinite(text_plus_mtp_loss)) {
        std::string msg = "computeAutogradLoss: text+MTP loss is non-finite (" + std::to_string(text_plus_mtp_loss) + ")";
        if (result.mtp_diagnostics.valid) {
            msg += " L0_main=" + std::to_string(result.mtp_diagnostics.L0_main);
            for (size_t i = 0; i < result.mtp_diagnostics.head_loss.size(); ++i)
                msg += " head_loss[" + std::to_string(i) + "]=" + std::to_string(result.mtp_diagnostics.head_loss[i]);
        }
        throw std::runtime_error(msg);
    }

    result.text_loss = text_ce_loss;
    result.mtp_loss = mtp_loss;
    result.valid_tokens = lm_valid_tokens;

    // ═══════════════════════════════════════════════════════════════════════════
    // EXECUTION BLOCK LOSS — autograd-connected CE + transition loss
    //
    // Two gradient-connected paths:
    //   1. transition_loss — L1 on soft-computed value vs teacher expected_value
    //   2. selection CE — logits-space CE on op/arg1/arg2/write selections
    //      (computed inside executeStep via autograd::cross_entropy_logits)
    //
    // Monitoring-only (host scalars, no gradients):
    //   - exec_entropy_monitor — distribution collapse detection
    //   - exec_structured_ce — scalar readback of device CE for logging parity
    // ═══════════════════════════════════════════════════════════════════════════
    float exec_structured_ce = 0.0f;
    float exec_entropy_monitor = 0.0f;

    if (cfg->execution_block_enabled && ctx.execution_block
        && !intermediates.exec_outputs_per_row.empty()) {

        const bool have_step_mask = (ctx.payload && !ctx.payload->teacher_step_mask.empty());
        const float ce_weight = cfg->structured_ce_weight;

        // Accumulate autograd losses: transition_loss + selection CE tensors
        int ce_tensor_count = 0;
        float ce_scalar_sum = 0.0f;

        for (int b = 0; b < ctx.batch_size; ++b) {
            if (!ctx.payload->execution_active.empty()
                && !ctx.payload->execution_active[b])
                continue;

            const auto& row_steps = intermediates.exec_outputs_per_row[b].steps;
            for (int k = 0; k < static_cast<int>(row_steps.size()); ++k) {
                // Skip padded steps — no gradient contribution
                if (have_step_mask
                    && b < static_cast<int>(ctx.payload->teacher_step_mask.size())
                    && k < static_cast<int>(ctx.payload->teacher_step_mask[b].size())
                    && ctx.payload->teacher_step_mask[b][k] == 0)
                    continue;

                const auto& sout = row_steps[k];

                // transition_loss → autograd graph (L1 value supervision)
                if (sout.transition_loss.data && sout.transition_loss.grad_fn) {
                    auto scaled = autograd::scale_scalar(
                        sout.transition_loss,
                        cfg->execution_block_causal_w1_transition,
                        ctx.stream);
                    intermediates.loss_tensor = autograd::add(
                        intermediates.loss_tensor, scaled, ctx.stream);
                }

                // Fix #6: div_invalid_penalty → autograd graph
                // Penalizes p_op[3] when division was clamped (|v2| < eps).
                // Gradient flows: penalty → p_op → softmax → op_logits → W_op_select.
                if (sout.div_invalid_penalty.data && sout.div_invalid_penalty.grad_fn) {
                    intermediates.loss_tensor = autograd::add(
                        intermediates.loss_tensor, sout.div_invalid_penalty, ctx.stream);
                }

                // Fix #8: div_magnitude_penalty → autograd graph
                // Penalizes large |v_out| after clamped division.
                // Gradient flows: penalty → v_out → FourOpMixGradFn → v1, v2.
                if (sout.div_magnitude_penalty.data && sout.div_magnitude_penalty.grad_fn) {
                    intermediates.loss_tensor = autograd::add(
                        intermediates.loss_tensor, sout.div_magnitude_penalty, ctx.stream);
                }

                // Fix #7: arg_reinforce_loss → autograd graph
                // REINFORCE: λ * detached(|v_out-target|) * (-log p_arg[k])
                // Gradient flows ONLY to arg logits. No soft weighting of values.
                if (sout.arg_reinforce_loss.data && sout.arg_reinforce_loss.grad_fn) {
                    intermediates.loss_tensor = autograd::add(
                        intermediates.loss_tensor, sout.arg_reinforce_loss, ctx.stream);
                }

                // selection CE → autograd graph (direct per-decision supervision)
                if (cfg->structured_ce_enabled && sout.has_selection_ce) {
                    auto accumulate_ce = [&](const Tensor& ce_tensor, const char* name) {
                        if (!ce_tensor.data || !ce_tensor.grad_fn)
                            throw std::runtime_error(
                                std::string("AutogradTraining: selection CE tensor '")
                                + name + "' has no data/grad_fn at row=" + std::to_string(b)
                                + " step=" + std::to_string(k)
                                + " — CrossEntropyLogitsGradFn was not attached");
                        auto scaled = autograd::scale_scalar(ce_tensor, ce_weight, ctx.stream);
                        intermediates.loss_tensor = autograd::add(
                            intermediates.loss_tensor, scaled, ctx.stream);
                    };

                    accumulate_ce(sout.selection_ce_op,    "selection_ce_op");
                    accumulate_ce(sout.selection_ce_arg1,  "selection_ce_arg1");
                    accumulate_ce(sout.selection_ce_arg2,  "selection_ce_arg2");
                    accumulate_ce(sout.selection_ce_write, "selection_ce_write");

                    // Scalar readback for logging parity (one sync per step is acceptable
                    // since we're already doing cudaMemcpy for expected_value upload)
                    float h_ce[4];
                    cudaMemcpyAsync(&h_ce[0], sout.selection_ce_op.data,    sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                    cudaMemcpyAsync(&h_ce[1], sout.selection_ce_arg1.data,  sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                    cudaMemcpyAsync(&h_ce[2], sout.selection_ce_arg2.data,  sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                    cudaMemcpyAsync(&h_ce[3], sout.selection_ce_write.data, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
                    cudaStreamSynchronize(ctx.stream);
                    ce_scalar_sum += h_ce[0] + h_ce[1] + h_ce[2] + h_ce[3];
                    ce_tensor_count++;
                }
            }
        }

        if (ce_tensor_count > 0) {
            exec_structured_ce = ce_scalar_sum / static_cast<float>(ce_tensor_count);
        }

        // Optional entropy monitor (non-differentiable; not added to loss_tensor)
        if (cfg->entropy_aux_weight > 0.0f) {
            for (int b = 0; b < ctx.batch_size; ++b) {
                const auto& all_steps = intermediates.exec_outputs_per_row[b].steps;
                std::vector<const ExecutionBlockStepOutput*> real_steps;
                if (have_step_mask
                    && b < static_cast<int>(ctx.payload->teacher_step_mask.size())
                    && !ctx.payload->teacher_step_mask[b].empty()) {
                    for (int k = 0; k < static_cast<int>(all_steps.size()); ++k) {
                        if (k < static_cast<int>(ctx.payload->teacher_step_mask[b].size())
                            && ctx.payload->teacher_step_mask[b][k] == 0)
                            continue;
                        real_steps.push_back(&all_steps[k]);
                    }
                } else {
                    for (const auto& s : all_steps)
                        real_steps.push_back(&s);
                }
                Tensor ent = ctx.execution_block->computeEntropyLoss(
                    real_steps,
                    cfg->entropy_aux_weight,
                    ctx.stream);
                float h_ent = 0.0f;
                cudaStreamSynchronize(ctx.stream);
                cudaMemcpy(&h_ent, ent.data, sizeof(float), cudaMemcpyDeviceToHost);
                exec_entropy_monitor += h_ent;
            }
            if (ctx.batch_size > 0)
                exec_entropy_monitor /= static_cast<float>(ctx.batch_size);
        }
    }
    result.entropy_monitor = exec_entropy_monitor;

    // ═══════════════════════════════════════════════════════════════════════════
    // SELECTOR SUPERVISION LOSS — autograd CE through TensorContract
    // Selector supervision is FINAL-STATE selector-only supervision: each row may
    // provide at most one non-Ignore target, and it MUST be attached to the last
    // decode position because the available ExecutionMemory is the row's final
    // post-execution state. Per-token non-Ignore targets would require a
    // timestep-aligned ExecutionMemory snapshot and are rejected loudly.
    // The hidden input is copied into an owned detached Tensor. Gradients train
    // DecodeTimeSlotSelector parameters only; they do not slice back into the
    // encoder hidden tensor because TensorContract has no parent slice/view op.
    // Weighted by cfg->selector_supervision_weight (0 = disabled).
    // ═══════════════════════════════════════════════════════════════════════════
    float selector_supervision_loss = addSelectorSupervisionLoss(ctx, intermediates);
    result.selector_loss = selector_supervision_loss;

    // ═══════════════════════════════════════════════════════════════════════════
    // GROUND-TRUTH LOSS: Read the ACTUAL tensor that backward will differentiate.
    // This is the single source of truth — no manual reconstruction from stale
    // host-side scalars. text_loss was snapshot before exec/selector additions.
    // ═══════════════════════════════════════════════════════════════════════════
    float actual_loss = 0.0f;
    cudaMemcpyAsync(&actual_loss, intermediates.loss_tensor.data, sizeof(float),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    result.loss_value = actual_loss;
    result.numeric_loss = actual_loss - text_ce_loss - mtp_loss - selector_supervision_loss;
    result.aux_loss = actual_loss - text_ce_loss;  // MTP + execution/numeric + selector
    result.weight_text = 1.0f;
    
    if (!std::isfinite(result.loss_value)) {
        throw std::runtime_error("computeAutogradLoss: combined loss is non-finite (actual_tensor=" 
            + std::to_string(actual_loss) + " text_ce=" + std::to_string(text_ce_loss)
            + " mtp=" + std::to_string(mtp_loss)
            + " exec_ce=" + std::to_string(exec_structured_ce)
            + " selector=" + std::to_string(selector_supervision_loss) + ")");
    }
    
    AG_INFO("Loss computed: total=" << actual_loss << " text_ce=" << text_ce_loss
            << " mtp=" << mtp_loss
            << " numeric_exec=" << result.numeric_loss
            << " exec_ce=" << exec_structured_ce
            << " exec_entropy_monitor=" << exec_entropy_monitor
            << " selector=" << selector_supervision_loss
            << " lm_valid=" << lm_valid_tokens);
    
    result.success = true;
    return result;
}

//======================================================================
// Autograd Backward Pass
//======================================================================

BackwardResult executeAutogradBackward(
    AutogradContext& ctx,
    bool accumulate
) {
    BackwardResult result{};
    result.success = false;
    result.grad_rms = 0.0f;
    
    ctx.validate("executeAutogradBackward");
    
    auto* ts = ctx.training_state;
    const auto* cfg = ctx.config;
    auto& intermediates = ts->autograd_intermediates;
    if (!intermediates.loss_tensor.data) {
        throw std::runtime_error("executeAutogradBackward: Loss tensor not initialized - call computeAutogradLoss() first");
    }
    
    if (!intermediates.loss_tensor.grad_fn) {
        throw std::runtime_error("executeAutogradBackward: Loss tensor has no grad_fn - autograd chain broken");
    }
    
    AG_INFO("Executing backward pass (accumulate=" << accumulate << ", scale=" << ctx.grad_scale << ")");

    // Zero gradients if not accumulating
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (!accumulate) {
        // Top-level parameters (Pattern B: owned by EmbeddingLayer)
        ctx.embedding_layer->tokenWeights().zero_grad(ctx.stream);

        // LM Head parameters (Pattern B: owned by persistent LMHeadLayer)
        ctx.lm_head->weights().zero_grad(ctx.stream);
        ctx.lm_head->bias().zero_grad(ctx.stream);
        // γ_final may be frozen (lm_head_freeze_final_rms_gamma=true) — in that case
        // it has no grad tensor and zero_grad would no-op; gate explicitly for clarity.
        if (ctx.lm_head->finalRmsGamma().requires_grad) {
            ctx.lm_head->finalRmsGammaMutable_UnfrozenOnly("AutogradTraining::zero_grad")
                .zero_grad(ctx.stream);
        }

        // Encoder parameters
        if (ctx.gpu_encoder) {
            const int num_layers = ctx.gpu_encoder->getNumLayers();
            for (int layer = 0; layer < num_layers; ++layer) {
                auto* enc = ctx.gpu_encoder->getLayer(layer);
                if (!enc) continue;
                enc->rms1Gamma().zero_grad(ctx.stream);
                enc->rms2Gamma().zero_grad(ctx.stream);
                // Issue #148: Sandwich norm gammas REMOVED
                enc->attnWqkv().zero_grad(ctx.stream);
                enc->attnBqkv().zero_grad(ctx.stream);
                enc->attnWo().zero_grad(ctx.stream);
                enc->attnBo().zero_grad(ctx.stream);
                enc->ffnWGate().zero_grad(ctx.stream);
                enc->ffnW1().zero_grad(ctx.stream);
                enc->ffnW2().zero_grad(ctx.stream);
                enc->ffnB2().zero_grad(ctx.stream);
                enc->layerScale1().zero_grad(ctx.stream);
                enc->layerScale2().zero_grad(ctx.stream);
            }
        }

        // ScratchBlock parameters
        if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
            ctx.scratch_block->atomTypeEmbeddings().zero_grad(ctx.stream);
            ctx.scratch_block->atomProjection().zero_grad(ctx.stream);
        }

        // Numeric head parameters
        // MTP head parameters
        if (ctx.model) {
            GRIM::MTP::zeroMTPGradients(*ctx.model, ctx.stream);
        }

        // ExecutionBlock parameters
        if (ctx.execution_block && cfg->execution_block_enabled) {
            auto& eb = *ctx.execution_block;
            eb.w_decode_1().zero_grad(ctx.stream);
            eb.b_decode_1().zero_grad(ctx.stream);
            eb.w_decode_2().zero_grad(ctx.stream);
            eb.w_arg1_select().zero_grad(ctx.stream);
            eb.w_arg2_select().zero_grad(ctx.stream);
            eb.W_op_select().zero_grad(ctx.stream);
            eb.W_key_proj().zero_grad(ctx.stream);
            eb.W_write_query().zero_grad(ctx.stream);
            eb.W_write_key().zero_grad(ctx.stream);
            eb.alpha().zero_grad(ctx.stream);
            eb.beta().zero_grad(ctx.stream);
            eb.step_embeddings().zero_grad(ctx.stream);
            eb.type_num_embed().zero_grad(ctx.stream);
            eb.W_value_to_emb().zero_grad(ctx.stream);
            eb.b_value_to_emb().zero_grad(ctx.stream);
            eb.w_inject_gate().zero_grad(ctx.stream);
            eb.W_Q_read().zero_grad(ctx.stream);
            eb.W_K_read().zero_grad(ctx.stream);
            eb.W_V_read().zero_grad(ctx.stream);
            eb.W_O_read().zero_grad(ctx.stream);
            eb.W_gate_read().zero_grad(ctx.stream);
            eb.tau().zero_grad(ctx.stream);
            eb.E_slot().zero_grad(ctx.stream);
            eb.E_op().zero_grad(ctx.stream);
            eb.W_scal().zero_grad(ctx.stream);
            eb.b_scal().zero_grad(ctx.stream);
            eb.W_trace().zero_grad(ctx.stream);
            eb.b_trace().zero_grad(ctx.stream);
            eb.W_reason_gate().zero_grad(ctx.stream);
            eb.W_trace_gate().zero_grad(ctx.stream);
        }

        // DecodeTimeSlotSelector parameters
        if (ctx.model && cfg->selector_enabled) {
            auto* selector = ctx.model->getDecodeTimeSlotSelectorLayer();
            if (selector) {
                selector->W_q_select().zero_grad(ctx.stream);
                selector->W_k_select().zero_grad(ctx.stream);
                selector->null_key_select().zero_grad(ctx.stream);
                selector->null_logit_bias().zero_grad(ctx.stream);
            }
        }

        // ReasoningHead parameters
        if (ctx.reasoning_head) {
            ctx.reasoning_head->W_op().zero_grad(ctx.stream);
            ctx.reasoning_head->b_op().zero_grad(ctx.stream);
            ctx.reasoning_head->w_arg1().zero_grad(ctx.stream);
            ctx.reasoning_head->w_arg2().zero_grad(ctx.stream);
        }
    }
    
    // Call backward on the text loss (single loss path)
    // Starting with ctx.grad_scale (usually 1/accumulation_steps)
    AG_INFO("Calling loss_tensor.backward(nullptr, " << ctx.grad_scale << ")...");
    intermediates.loss_tensor.backward(nullptr, ctx.grad_scale);
    AG_INFO("loss_tensor.backward() returned successfully");

    // ════════════════════════════════════════════════════════════════════
    // DIAGNOSTIC: Sample gradient values immediately after backward to
    // determine if backward itself produces zeros or if something later
    // corrupts the buffers.  Issue: "zero gradients every other batch"
    // ════════════════════════════════════════════════════════════════════
    {
        cudaStreamSynchronize(ctx.stream);

        // Sample LM head weight gradient (first element)
        float lm_sample = 0.0f;
        float* lm_grads = ctx.lm_head->weights().grad_data();
        if (lm_grads) {
            cudaMemcpy(&lm_sample, lm_grads, sizeof(float), cudaMemcpyDeviceToHost);
        }

        // Sample first encoder layer attnWqkv gradient
        float enc_sample = 0.0f;
        if (ctx.gpu_encoder && ctx.gpu_encoder->getNumLayers() > 0) {
            auto* enc0 = ctx.gpu_encoder->getLayer(0);
            if (enc0) {
                float* wqkv_grads = enc0->attnWqkv().grad_data();
                if (wqkv_grads) {
                    cudaMemcpy(&enc_sample, wqkv_grads, sizeof(float), cudaMemcpyDeviceToHost);
                }
            }
        }

        // Sample RMSNorm gamma gradient (layer 0)
        float rms_sample = 0.0f;
        if (ctx.gpu_encoder && ctx.gpu_encoder->getNumLayers() > 0) {
            auto* enc0 = ctx.gpu_encoder->getLayer(0);
            if (enc0) {
                float* rms_grads = enc0->rms1Gamma().grad_data();
                if (rms_grads) {
                    cudaMemcpy(&rms_sample, rms_grads, sizeof(float), cudaMemcpyDeviceToHost);
                }
            }
        }

        fprintf(stderr,
            "[GRAD_DIAG] POST-BACKWARD accumulate=%d grad_scale=%.4f "
            "lm_grad[0]=%.10e enc_wqkv_grad[0]=%.10e rms_gamma_grad[0]=%.10e "
            "lm_ptr=%p\n",
            static_cast<int>(accumulate), ctx.grad_scale,
            lm_sample, enc_sample, rms_sample,
            static_cast<void*>(lm_grads));
    }

    // ScratchBlock backward is now automatic via ScratchBlockGradFn in the autograd chain.
    // loss_tensor.backward() → ... → ScratchBlockGradFn::apply() handles parameter gradients.
    
    // Verify gradients are properly connected before optimizer runs
    AG_INFO("Verifying gradients are connected to optimizer...");
    if (!verifyGradientsAreConnected(ctx)) {
        result.error_message = "Gradient connectivity verification failed";
        AG_ERROR("executeAutogradBackward: " << result.error_message);
        return result;
    }
    AG_INFO("Gradient connectivity verified");
    
    // ISSUE #149: Manual parameter gradient scaling REMOVED.
    // We now scale the root gradient (the loss) at the start of backward()
    // which propagates the scale through the entire computation graph.
    // This is mathematically equivalent, more efficient (fewer kernels),
    // and safer against omission bugs when adding new layers.
    
    AG_INFO("Backward complete");
    
    result.success = true;
    return result;
}

//======================================================================
// Helper Functions
//======================================================================

bool verifyGradientsAreConnected(AutogradContext& ctx) {
    bool ok = true;
    const bool text_loss_active = ctx.payload && ctx.payload->lm_valid_tokens > 0;
    const bool selector_loss_active = ctx.training_state
        && !ctx.training_state->autograd_intermediates.selector_fwd_results.empty();
    bool exec_selection_loss_active = false;
    bool exec_write_selection_ce_active = false;
    const bool reasoning_loss_active = false;  // ReasoningHead forward is diagnostic-only until a real reasoning loss path is assembled.
    if (ctx.training_state) {
        const auto& rows = ctx.training_state->autograd_intermediates.exec_outputs_per_row;
        for (const auto& row : rows) {
            for (const auto& step : row.steps) {
                if (step.has_selection_ce) {
                    exec_selection_loss_active = true;
                    exec_write_selection_ce_active = true;
                }
                if ((step.transition_loss.data && step.transition_loss.grad_fn)
                    || (step.arg_reinforce_loss.data && step.arg_reinforce_loss.grad_fn)) {
                    exec_selection_loss_active = true;
                }
            }
            if (exec_selection_loss_active && exec_write_selection_ce_active) break;
        }
    }

    auto requireAllocatedFinite = [&](Tensor& t, const std::string& label) {
        if (!t.data) return;
        if (!t.has_grad()) {
            AG_WARN(label << ".grad is NULL - allocated parameter did not expose optimizer gradient storage");
            ok = false;
            return;
        }
        GradientSignalProbe probe = probeGradientSignal(t, ctx.stream);
        if (!probe.allocated) {
            AG_WARN(label << ".grad_data is NULL despite has_grad=true");
            ok = false;
            return;
        }
        if (!probe.finite) {
            AG_WARN(label << ".grad contains non-finite values in sampled window (sampled="
                    << probe.sampled << ", rms=" << probe.rms << ")");
            ok = false;
        } else {
            AG_INFO(label << ".grad allocated; sampled=" << probe.sampled
                    << " rms=" << probe.rms
                    << " received_nonzero=" << (probe.nonzero ? "yes" : "no"));
        }
    };

    auto requireReceivedGradient = [&](Tensor& t, const std::string& label) {
        requireAllocatedFinite(t, label);
        if (!t.data || !t.has_grad() || !t.grad_data()) return;
        GradientSignalProbe probe = probeGradientSignal(t, ctx.stream);
        if (!probe.nonzero || probe.rms == 0.0f) {
            AG_WARN(label << ".grad is allocated but sampled RMS is zero after this backward "
                    << "(sampled=" << probe.sampled << ") — gradient path did not deliver signal");
            ok = false;
        }
    };
    
    // The autograd system stores gradients in Tensor.grad_ fields (shared_ptr<Tensor>)
    // The optimizer accesses them directly via Tensor.grad_data() pointers.
    // 
    // This function verifies that gradients are properly allocated and accessible.
    // It does NOT copy — gradients are already wired up during initialization.
    // This is a diagnostic check to catch pointer setup bugs before optimizer runs.
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Embedding gradients (may be tied with LM head) — Pattern B: owned by EmbeddingLayer
    // ═══════════════════════════════════════════════════════════════════════════
    // ISSUE #59: Use has_grad() accessor
    if (ctx.embedding_layer->tokenWeights().data) {
        requireAllocatedFinite(ctx.embedding_layer->tokenWeights(), "embedding token_weights");
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LM head gradients (Pattern B: owned by persistent LMHeadLayer)
    // May be tied to embedding (same underlying grad Tensor)
    // ═══════════════════════════════════════════════════════════════════════════
    if (ctx.lm_head->weights().data) {
        if (!ctx.lm_head->weights().has_grad()) {
            AG_WARN("lm_head weights.grad is NULL - gradients NOT flowing to optimizer!");
            ok = false;
        } else {
            // Check if tied to embedding (same underlying grad Tensor)
            if (ctx.lm_head->weights().grad_data() == ctx.embedding_layer->tokenWeights().grad_data()) {
                AG_INFO("LM head gradients TIED to embedding: " << ctx.lm_head->weights().numel() << " elements");
            } else {
                AG_INFO("LM head gradients SEPARATE: " << ctx.lm_head->weights().numel() << " elements at " << ctx.lm_head->weights().grad_data());
            }
            if (text_loss_active) {
                requireReceivedGradient(ctx.lm_head->weights(), "lm_head weights");
            } else {
                requireAllocatedFinite(ctx.lm_head->weights(), "lm_head weights");
            }
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Logits gradients are TensorContract-owned.
    // ═══════════════════════════════════════════════════════════════════════════
    // LMHead returns non-leaf logits, so LogSoftmaxGradFn allocates its own
    // non-leaf input_grad buffer and passes that view directly to the upstream
    // logits grad_fn. TrainingState must not mirror or copy logits gradients.
    
    // NOTE: Encoder gradients are in encoder's internal Tensors, not TrainingState.
    // The optimizer accesses them via Tensor& accessors (enc->attnWqkv() etc.).
    
    // ═══════════════════════════════════════════════════════════════════════════
    // Final RMSNorm gamma (Pattern B: owned by LMHeadLayer)
    // When lm_head_freeze_final_rms_gamma=true, requires_grad=false and we skip the check
    // (no grad is correct, not a bug).
    // ═══════════════════════════════════════════════════════════════════════════
    if (ctx.lm_head->finalRmsGamma().data
        && ctx.lm_head->finalRmsGamma().requires_grad
        && !ctx.lm_head->finalRmsGamma().has_grad()) {
        AG_WARN("final_rms_gamma.grad is NULL - gradients NOT flowing!");
        ok = false;
    } else if (ctx.lm_head->finalRmsGamma().data && ctx.lm_head->finalRmsGamma().requires_grad) {
        requireAllocatedFinite(ctx.lm_head->finalRmsGammaMutable_UnfrozenOnly("verifyGradientsAreConnected"),
                               "final_rms_gamma");
    }

    if (ctx.lm_head->bias().data && !ctx.lm_head->bias().has_grad()) {
        AG_WARN("lm_head_bias.grad is NULL - gradients NOT flowing!");
        ok = false;
    } else if (ctx.lm_head->bias().data) {
        requireAllocatedFinite(ctx.lm_head->bias(), "lm_head_bias");
    }

    if (ctx.gpu_encoder) {
        const int num_layers = ctx.gpu_encoder->getNumLayers();
        for (int layer = 0; layer < num_layers; ++layer) {
            auto* enc = ctx.gpu_encoder->getLayer(layer);
            if (!enc) {
                AG_WARN("encoder layer " << layer << " is NULL during gradient verification");
                ok = false;
                continue;
            }
            auto check = [&](Tensor& t, const char* name) {
                if (t.data) requireAllocatedFinite(t, "layer " + std::to_string(layer) + " " + std::string(name));
            };
            check(enc->rms1Gamma(), "rms1Gamma");
            check(enc->rms2Gamma(), "rms2Gamma");
            // Issue #148: Sandwich norm gammas REMOVED
            check(enc->attnWqkv(), "attnWqkv");
            check(enc->attnBqkv(), "attnBqkv");
            check(enc->attnWo(), "attnWo");
            check(enc->attnBo(), "attnBo");
            check(enc->ffnWGate(), "ffnWGate");
            check(enc->ffnW1(), "ffnW1");
            check(enc->ffnW2(), "ffnW2");
            check(enc->ffnB2(), "ffnB2");
            check(enc->layerScale1(), "layerScale1");
            check(enc->layerScale2(), "layerScale2");
        }
        if (text_loss_active && num_layers > 0) {
            auto* enc0 = ctx.gpu_encoder->getLayer(0);
            if (enc0) {
                requireReceivedGradient(enc0->attnWqkv(), "layer 0 attnWqkv");
            }
        }
    }

    if (ctx.scratch_block && ctx.scratch_block->isEnabled()) {
        auto checkScratch = [&](Tensor& t, const char* name) {
            if (t.data) requireAllocatedFinite(t, "scratch block " + std::string(name));
        };
        checkScratch(ctx.scratch_block->atomTypeEmbeddings(), "atomTypeEmbeddings");
        checkScratch(ctx.scratch_block->atomProjection(), "atomProjection");
    }

    if (ctx.model && !GRIM::MTP::verifyMTPGradients(*ctx.model)) {
        ok = false;
    }

    // ExecutionBlock parameters
    if (ctx.execution_block && ctx.config->execution_block_enabled) {
        auto checkEB = [&](Tensor& t, const char* name) {
            if (t.data) requireAllocatedFinite(t, "exec block " + std::string(name));
        };
        auto& eb = *ctx.execution_block;
        checkEB(eb.w_decode_1(), "w_decode_1");
        checkEB(eb.b_decode_1(), "b_decode_1");
        checkEB(eb.w_decode_2(), "w_decode_2");
        checkEB(eb.w_arg1_select(), "w_arg1_select");
        checkEB(eb.w_arg2_select(), "w_arg2_select");
        checkEB(eb.W_op_select(), "W_op_select");
        checkEB(eb.W_key_proj(), "W_key_proj");
        checkEB(eb.W_write_query(), "W_write_query");
        checkEB(eb.W_write_key(), "W_write_key");
        checkEB(eb.alpha(), "alpha");
        checkEB(eb.beta(), "beta");
        checkEB(eb.step_embeddings(), "step_embeddings");
        checkEB(eb.type_num_embed(), "type_num_embed");
        checkEB(eb.W_value_to_emb(), "W_value_to_emb");
        checkEB(eb.b_value_to_emb(), "b_value_to_emb");
        checkEB(eb.w_inject_gate(), "w_inject_gate");
        checkEB(eb.W_Q_read(), "W_Q_read");
        checkEB(eb.W_K_read(), "W_K_read");
        checkEB(eb.W_V_read(), "W_V_read");
        checkEB(eb.W_O_read(), "W_O_read");
        checkEB(eb.W_gate_read(), "W_gate_read");
        checkEB(eb.tau(), "tau");
        checkEB(eb.E_slot(), "E_slot");
        checkEB(eb.E_op(), "E_op");
        checkEB(eb.W_scal(), "W_scal");
        checkEB(eb.b_scal(), "b_scal");
        checkEB(eb.W_trace(), "W_trace");
        checkEB(eb.b_trace(), "b_trace");
        checkEB(eb.W_reason_gate(), "W_reason_gate");
        checkEB(eb.W_trace_gate(), "W_trace_gate");
        if (exec_selection_loss_active) {
            requireReceivedGradient(eb.W_op_select(), "exec block W_op_select");
            requireReceivedGradient(eb.w_arg1_select(), "exec block w_arg1_select");
        }
        if (exec_write_selection_ce_active) {
            requireReceivedGradient(eb.W_write_query(), "exec block W_write_query");
        }
    }

    // DecodeTimeSlotSelector parameters
    if (ctx.model && ctx.config->selector_enabled) {
        auto* selector = ctx.model->getDecodeTimeSlotSelectorLayer();
        if (selector) {
            auto checkSel = [&](Tensor& t, const char* name) {
                if (t.data) requireAllocatedFinite(t, "selector " + std::string(name));
            };
            checkSel(selector->W_q_select(), "W_q_select");
            checkSel(selector->W_k_select(), "W_k_select");
            checkSel(selector->null_key_select(), "null_key_select");
            checkSel(selector->null_logit_bias(), "null_logit_bias");
            if (selector_loss_active) {
                requireReceivedGradient(selector->W_q_select(), "selector W_q_select");
                requireReceivedGradient(selector->W_k_select(), "selector W_k_select");
                requireReceivedGradient(selector->null_logit_bias(), "selector null_logit_bias");
            }
        }
    }

    // ReasoningHead parameters: executeAutogradForward currently invokes the
    // head for structured diagnostics only. No reasoning loss is assembled into
    // intermediates.loss_tensor, so verifying these params for received gradient
    // would falsely claim training connectivity. Re-enable only alongside a real
    // reasoning loss path.
    if (ctx.reasoning_head && reasoning_loss_active) {
        auto checkRH = [&](Tensor& t, const char* name) {
            if (t.data) requireAllocatedFinite(t, "reasoning head " + std::string(name));
        };
        checkRH(ctx.reasoning_head->W_op(), "W_op");
        checkRH(ctx.reasoning_head->b_op(), "b_op");
        checkRH(ctx.reasoning_head->w_arg1(), "w_arg1");
        checkRH(ctx.reasoning_head->w_arg2(), "w_arg2");
    } else if (ctx.reasoning_head) {
        AG_INFO("ReasoningHead gradient verification skipped: no reasoning loss path is connected to loss_tensor");
    }
    
    return ok;
}

// Finding 2 (Rule 26): computeGradientNorm() DELETED — redundant with
// Phase2's ctx.model->computeGradNorm(true) which produces per-component breakdown.
// The old function duplicated a full L2 norm scan + cudaStreamSynchronize per batch
// whose result was only logged and never consumed by Phase2.

//======================================================================
// Main Entry Point
//======================================================================

LossResult autogradTrainingStep(
    LanguageModel& model,
    TrainingState& training_state,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    bool accumulate,
    float grad_scale,
    uint64_t step
) {
    payload.validate("autogradTrainingStep");

    const auto& cfg = model.getConfig();

    // Execution payload validation (WS4: single shared validator)
    GRIM::Execution::validateExecutionPayload(
        payload, "autogradTrainingStep",
        cfg.execution_block_num_slots, cfg.execution_block_num_ops, cfg.execution_block_num_steps);

    // When execution_block is disabled, teacher_steps are ignored — batch trains with plain cross-entropy.
    if (!payload.teacher_steps.empty() && !cfg.execution_block_enabled) {
        AG_WARN("batch has teacher_steps (arithmetic) but execution_block_enabled=false; "
                "training with plain cross-entropy over text tokens (teacher supervision skipped)");
    }

    // WS8: Structural layer availability — crash loud if config says enabled but layers are missing
    // (mirrors computeLossBatch; ensures both paths throw on same condition)
    if (cfg.execution_block_enabled) {
        if (!model.getExecutionBlockLayer()) {
            throw std::runtime_error(
                "autogradTrainingStep: execution_block_enabled but ExecutionBlock layer is null");
        }
        ScratchBlockLayer* sb_check = model.getScratchBlockLayer();
        if (!sb_check || !sb_check->isEnabled()) {
            throw std::runtime_error(
                "autogradTrainingStep: execution_block_enabled requires ScratchBlock enabled");
        }
    }

    const int total_tokens = payload.total_tokens;
    
    // Get encoder for autograd forward
    GPUGrimEncoder& gpu_encoder = model.getGpuEncoder();
    EmbeddingLayer* embedding_layer = model.getEmbeddingLayer();
    LMHeadLayer* lm_head = model.getLmHeadLayer();
    ScratchBlockLayer* scratch_block = model.getScratchBlockLayer();
    ReasoningHeadLayer* reasoning_head = model.getReasoningHeadLayer();
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();

    // ═══════════════════════════════════════════════════════════════════════════
    // GPU COPIES: handled upstream by LanguageModel::uploadBatchToDevice(payload).
    // initAutogradContext is the single sync-boundary validator for the returned
    // BatchDeviceBindings; this step never authors payload geometry or H2D copies.
    // ═══════════════════════════════════════════════════════════════════════════

    // Logit buffer capacity is still enforced here so a mis-sized payload trips
    // immediately, before any forward kernel is launched.
    const auto& logits_shape = training_state.cached_logits_tensor.shape.require("autogradTrainingStep cached_logits_tensor");
    if (!logits_shape.is_2d_layout()) {
        throw std::runtime_error("autogradTrainingStep: cached_logits_tensor must be a 2D LOGITS buffer");
    }
    const auto logits_dims = logits_shape.as_2d();
    const size_t logit_limit = static_cast<size_t>(logits_dims.rows);
    if (static_cast<size_t>(total_tokens) > logit_limit) {
        throw std::runtime_error(
            "autogradTrainingStep: total_tokens=" + std::to_string(total_tokens) +
            " exceeds logit buffer capacity=" + std::to_string(logit_limit));
    }
    if (payload.vocab_size != cfg.vocab_size) {
        throw std::runtime_error(
            "autogradTrainingStep: payload.vocab_size=" + std::to_string(payload.vocab_size) +
            " != cfg.vocab_size=" + std::to_string(cfg.vocab_size) +
            " — logits buffer width cannot be validated against conflicting vocabularies");
    }
    if (logits_dims.cols != payload.vocab_size) {
        throw std::runtime_error(
            "autogradTrainingStep: cached_logits_tensor cols=" + std::to_string(logits_dims.cols) +
            " != payload/cfg vocab_size=" + std::to_string(payload.vocab_size) +
            " (rows=" + std::to_string(logits_dims.rows) +
            ", total_tokens=" + std::to_string(total_tokens) + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AUTOGRAD CONTEXT
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Write authoritative training step to TrainingState BEFORE building context.
    // This is the ONLY mutation site — eval paths (computeLossBatch) read but never write.
    training_state.autograd_step = step;

    // Rule 20 explicit tape sealing: skip equation-tape D2H/fprintf on non-initial
    // accumulation slots across the ENTIRE step (forward + loss + backward). Sealing
    // only forward (the previous behavior) was a Rule 20 violation: loss and
    // backward kept logging duplicated tape entries on every slot.
    TapeSkipScope tape_skip_scope(accumulate);

    ExecutionBlockLayer* execution_block = model.getExecutionBlockLayer();

    AutogradContext ctx = initAutogradContext(
        &cfg,
        &training_state,
        &gpu_encoder,
        embedding_layer,
        lm_head,
        scratch_block,
        reasoning_head,
        execution_block,
        training_state.cublas_handle,
        stream,
        payload,
        bindings,
        grad_scale,
        step,
        true
    );
    ctx.loss_config = buildLossConfig(model.getLossOptions(), training_state.class_weights_tensor.data);
    ctx.skip_equation_logging = accumulate;  // Skip D2H + fprintf on non-initial accumulation slots
    ctx.model = &model;  // For MTP head access in computeAutogradLoss

    // ═══════════════════════════════════════════════════════════════════════════
    // FORWARD → LOSS → BACKWARD
    // ═══════════════════════════════════════════════════════════════════════════

    // Allocate read-gate accumulator once (Category 3 workspace on TrainingState)
    if (!training_state.read_gate_accum_tensor.data && cfg.execution_block_enabled) {
        training_state.read_gate_accum_tensor = Tensor::zeros({2}, stream, "read_gate_accum");
    }
    // Zero the accumulator before forward (sum=0, count=0)
    if (training_state.read_gate_accum_tensor.data) {
        CUDA_CHECK(cudaMemsetAsync(training_state.read_gate_accum_tensor.data, 0, 2 * sizeof(float), stream));
    }
    
    ForwardResult fwd_result = executeAutogradForward(ctx);
    if (!fwd_result.success) {
        throw std::runtime_error("autogradTrainingStep: Forward failed - " + fwd_result.error_message);
    }

    // Read back the cross-attention read gate accumulator (sum/count on device)
    // Snapshot Category 3 workspace into Category 2 telemetry scalar BEFORE the
    // autograd boundary (Rule 20).
    if (training_state.read_gate_accum_tensor.data) {
        float h_accum[2] = {0.0f, 0.0f};
        CUDA_CHECK(cudaMemcpyAsync(h_accum, training_state.read_gate_accum_tensor.data,
                                   2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        training_state.h_read_gate_mean = (h_accum[1] > 0.0f)
            ? (h_accum[0] / h_accum[1])
            : 0.0f;
    }
    
    LossResult loss_result = computeAutogradLoss(ctx);
    if (!loss_result.success) {
        loss_result.error_message = "autogradTrainingStep: Loss failed - " + loss_result.error_message;
        // Rule 20 single-owner clear: caller's AutogradStepScope handles intermediates.
        return loss_result;
    }
    
    // Rule 20: Non-finite loss means forward produced garbage.
    // Skip backward entirely — don't propagate NaN/Inf gradients.
    if (!std::isfinite(loss_result.loss_value)) {
        loss_result.success = false;
        loss_result.error_message = "Non-finite loss: " + std::to_string(loss_result.loss_value);
        // Rule 20 single-owner clear: caller's AutogradStepScope handles intermediates.
        return loss_result;
    }
    
    BackwardResult bwd_result = executeAutogradBackward(ctx, accumulate);
    if (!bwd_result.success) {
        loss_result.success = false;
        loss_result.error_message = "autogradTrainingStep: Backward failed - " + bwd_result.error_message;
        // Rule 20 single-owner clear: caller's AutogradStepScope handles intermediates.
        return loss_result;
    }
    
    // Post-backward cleanup (matches LanguageModel::backward() behavior)
    training_state.sequence_weight_count = 0;
    
    // Rule 20 ownership taxonomy: AutogradIntermediates::clear() is owned by
    // the caller's AutogradStepScope RAII guard. Do NOT clear here. Post-step
    // diagnostics (Phase2_TrainingLoop, GuessCache, ComputeLossBatch) read
    // from TrainingState::cached_logits_tensor (Cat 3 step-output snapshot),
    // never from intermediates.logits_tensor (Cat 1, transient).
    
    AG_INFO("Training step complete: loss=" << loss_result.loss_value);
    
    return loss_result;
}
}  // namespace Autograd
}  // namespace GRIM

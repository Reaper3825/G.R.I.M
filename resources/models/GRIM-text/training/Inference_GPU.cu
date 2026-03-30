//======================================================//
//  Inference_GPU.cu
//  Inference using autograd forward pass
//  
//  Single inference entry point: executeInferenceForward_()
//  All public inference methods copy data to cached_* tensors
//  then call this private method.
//
//  Rule 20: No backwards compatibility - uses autograd only
//  Rule 26: One inference path, not two
//======================================================//

#include <vector>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "Autograd/AutogradTraining.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Shared/TensorConversion/TensorConversion.hpp"
#include "../Layers/ExecutionBlock/execution_block_GPU.hpp"

namespace GRIM {

namespace {

constexpr int kScratchTextFeatureDim = 16;

void copyTokenSlotMapH2D(TrainingState& ts, cudaStream_t stream, int seq_len,
                         const std::vector<int32_t>& prompt_map) {
    if (!ts.cached_token_to_slot_map.data || seq_len <= 0)
        return;
    auto* dst = reinterpret_cast<int32_t*>(ts.cached_token_to_slot_map.data);
    if (prompt_map.empty()) {
        std::vector<int32_t> neg(static_cast<size_t>(seq_len), -1);
        cudaError_t err = cudaMemcpyAsync(dst, neg.data(),
            static_cast<size_t>(seq_len) * sizeof(int32_t),
            cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("copyTokenSlotMapH2D (sentinel): ") +
                                     cudaGetErrorString(err));
        }
        return;
    }
    if (static_cast<int>(prompt_map.size()) != seq_len) {
        throw std::runtime_error(
            "token_to_slot_map size must match sequence length (or pass empty for all -1)");
    }
    cudaError_t err = cudaMemcpyAsync(dst, prompt_map.data(),
        static_cast<size_t>(seq_len) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("copyTokenSlotMapH2D: ") + cudaGetErrorString(err));
    }
}

ScratchBlockLayer::RowLocalAtomView buildDecodeExecutionAtomView(
    ScratchBlockLayer& scratch_block,
    const LanguageModelConfig& cfg,
    TrainingState& ts,
    cudaStream_t stream,
    int token_pos)
{
    auto throwCuda = [](const char* what, cudaError_t err) {
        throw std::runtime_error(std::string("buildDecodeExecutionAtomView: ") +
                                 what + ": " + cudaGetErrorString(err));
    };

    if (!ts.cached_token_ids_tensor.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_ids_tensor is NULL");
    }
    if (!ts.cached_token_numeric_values.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_numeric_values is NULL");
    }
    if (!ts.cached_token_text_features.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_text_features is NULL");
    }
    if (!ts.cached_token_atom_mask.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_atom_mask is NULL");
    }
    if (!ts.cached_token_atom_flags.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_atom_flags is NULL");
    }
    if (!ts.cached_token_to_slot_map.data) {
        throw std::runtime_error("buildDecodeExecutionAtomView: cached_token_to_slot_map is NULL during decode-time execution");
    }

    auto dummy_hidden = Tensor::zeros({1, cfg.d_model}, stream, "decode_exec_atom_dummy_hidden");
    const bool exec_first_type_only =
        cfg.execution_block_enabled && cfg.scratch_block_execution_first_type_only;

    scratch_block.runForwardKernels(
        dummy_hidden.data,
        1,
        reinterpret_cast<const int*>(ts.cached_token_ids_tensor.data) + token_pos,
        ts.cached_token_numeric_values.data + token_pos,
        reinterpret_cast<const uint16_t*>(ts.cached_token_text_features.data) +
            static_cast<size_t>(token_pos) * kScratchTextFeatureDim,
        reinterpret_cast<const uint8_t*>(ts.cached_token_atom_mask.data) + token_pos,
        reinterpret_cast<const uint32_t*>(ts.cached_token_atom_flags.data) + token_pos,
        reinterpret_cast<const int32_t*>(ts.cached_token_to_slot_map.data) + token_pos,
        stream,
        exec_first_type_only);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) throwCuda("ScratchBlock decode atom path", err);

    auto view = scratch_block.extractRowLocalAtomView(0, 1, stream);

    int32_t slot_id = -1;
    err = cudaMemcpyAsync(&slot_id,
                          reinterpret_cast<const int32_t*>(ts.cached_token_to_slot_map.data) + token_pos,
                          sizeof(int32_t),
                          cudaMemcpyDeviceToHost,
                          stream);
    if (err != cudaSuccess) throwCuda("cudaMemcpyAsync(slot_id)", err);
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) throwCuda("cudaStreamSynchronize(slot_id)", err);

    if (slot_id >= 0 && view.num_atoms == 0) {
        throw std::runtime_error(
            "buildDecodeExecutionAtomView: decode-time execution token is slot-bound but ScratchBlock produced no row-local atom view");
    }

    return view;
}

}  // namespace

//======================================================//
//  executeInferenceForward_ - THE single inference forward path
//  Assumes all data already in cached_* tensors.
//  Creates autograd context, runs forward, returns last-token logits.
//  When populate_kv_cache=true, extracts per-layer K,V from autograd
//  intermediates and converts to BF16 BSHD format in KV cache buffers.
//======================================================//
Vector LanguageModel::executeInferenceForward_(int seq_len, bool populate_kv_cache) {
    if (!training_state_.initialized) {
        throw std::runtime_error("executeInferenceForward_: training state not initialized");
    }
    if (seq_len <= 0 || seq_len > config_.max_seq_len) {
        throw std::runtime_error("executeInferenceForward_: seq_len=" + std::to_string(seq_len) +
                                 " out of range [1, " + std::to_string(config_.max_seq_len) + "]");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

    // Store dimensions for downstream consumers
    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = seq_len;

    // Initialize autograd context (is_training=false disables dropout)
    Autograd::AutogradContext ctx = Autograd::initAutogradContext(
        &config_,
        &training_state_,
        &getGpuEncoder(),
        getEmbeddingLayer(),
        getLmHeadLayer(),
        getScratchBlockLayer(),
        getReasoningHeadLayer(),
        getExecutionBlockLayer(),
        training_state_.cublas_handle,
        stream,
        1,          // batch_size = 1 for inference
        seq_len,
        1.0f,       // grad_scale (unused for inference)
        0,          // step
        false       // is_training (disable dropout)
    );

    // Run autograd forward
    Autograd::ForwardResult result = Autograd::executeAutogradForward(ctx);
    if (!result.success) {
        throw std::runtime_error("executeInferenceForward_: forward failed - " + result.error_message);
    }

    // Extract last token logits
    if (!training_state_.cached_logits_tensor.data) {
        throw std::runtime_error("executeInferenceForward_: cached_logits_tensor not initialized");
    }
    Vector logits(config_.vocab_size);
    const size_t last_token_offset = static_cast<size_t>(seq_len - 1) * config_.vocab_size;
    cudaMemcpyAsync(logits.data.data(),
                    training_state_.cached_logits_tensor.data + last_token_offset,
                    config_.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Persist ExecutionMemory so decode-time execution state survives across
    // autoregressive forwardStep calls.
    if (!training_state_.autograd_intermediates.exec_memories.empty()) {
        training_state_.inference_exec_memory =
            std::move(training_state_.autograd_intermediates.exec_memories[0]);
        training_state_.has_inference_exec_memory = true;
    }

    // Populate KV cache from autograd intermediates (prefill mode).
    // Must happen BEFORE clear() which destroys the intermediate tensors.
    if (populate_kv_cache && !training_state_.kv_cache_k.empty()) {
        const int num_kv_heads = training_state_.num_kv_heads;
        const int head_dim = config_.d_model / config_.num_heads;
        const auto& layers = training_state_.autograd_intermediates.layer_intermediates.layers;
        const int num_layers = static_cast<int>(layers.size());

        for (int i = 0; i < num_layers; ++i) {
            const auto& layer_ints = layers[i];
            if (!layer_ints.K_bhsd.data || !layer_ints.V_bhsd.data) {
                throw std::runtime_error("executeInferenceForward_: layer " + std::to_string(i) +
                                         " K_bhsd/V_bhsd is NULL during KV cache population");
            }
            // K_bhsd: [1, nkv, seq_len, hd] BHSD FP32 (post-RoPE)
            // V_bhsd: [1, nkv, seq_len, hd] BHSD FP32
            // Convert to BSHD BF16 → [1, seq_len, nkv, hd] in cache
            TensorConversion::convert_BHSD_to_BSHD_bf16(
                layer_ints.K_bhsd.data,
                static_cast<__nv_bfloat16*>(training_state_.kv_cache_k[i]),
                1, num_kv_heads, seq_len, head_dim, stream);
            TensorConversion::convert_BHSD_to_BSHD_bf16(
                layer_ints.V_bhsd.data,
                static_cast<__nv_bfloat16*>(training_state_.kv_cache_v[i]),
                1, num_kv_heads, seq_len, head_dim, stream);
        }
    }

    // Free all autograd intermediates — inference never runs backward.
    // Without this, grad_fn chains and cached tensors leak across generation steps.
    training_state_.autograd_intermediates.clear();

    return logits;
}

//======================================================//
//  forwardInit - Prefill phase: copy prompt to device, run forward
//======================================================//
Vector LanguageModel::forwardInit(const std::vector<int>& prompt_tokens,
                                  const std::vector<float>& prompt_numeric_values,
                                  const std::vector<uint8_t>& prompt_atom_mask,
                                  const std::vector<int32_t>& prompt_token_to_slot_map) {
    const int seq_len = static_cast<int>(prompt_tokens.size());
    
    if (!training_state_.initialized) {
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
    }

    if (seq_len <= 0) {
        throw std::runtime_error("forwardInit: seq_len <= 0");
    }
    if (seq_len > config_.max_seq_len) {
        throw std::runtime_error("forwardInit: seq_len exceeds max_seq_len");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // Copy tokens to device
    cudaMemcpyAsync(reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data),
                    prompt_tokens.data(),
                    seq_len * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    // Copy numeric side-channel (zero when empty to prevent stale bleed)
    if (!prompt_numeric_values.empty()) {
        cudaMemcpyAsync(training_state_.cached_token_numeric_values.data,
                        prompt_numeric_values.data(),
                        seq_len * sizeof(float),
                        cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemsetAsync(training_state_.cached_token_numeric_values.data,
                        0, static_cast<size_t>(seq_len) * sizeof(float), stream);
    }
    if (!prompt_atom_mask.empty()) {
        cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_atom_mask.data),
                        prompt_atom_mask.data(),
                        seq_len * sizeof(uint8_t),
                        cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemsetAsync(training_state_.cached_token_atom_mask.data,
                        0, static_cast<size_t>(seq_len) * sizeof(uint8_t), stream);
    }

    copyTokenSlotMapH2D(training_state_, stream, seq_len, prompt_token_to_slot_map);
    
    // Store sequence length for subsequent forwardStep() calls
    training_state_.kv_cache_len = seq_len;
    
    // Prefill: populate KV cache from autograd intermediates
    Vector logits = executeInferenceForward_(seq_len, /*populate_kv_cache=*/true);

    // Prepare execution trace state for inference decode (no autograd needed)
    if (!training_state_.trace_state_by_row.empty()) {
        training_state_.trace_state_by_row[0].requires_grad = false;
    }

    return logits;
}

//======================================================//
//  forwardStep - Decode phase: single token with KV cache
//======================================================//
Vector LanguageModel::forwardStep(int new_token, float numeric_value, uint8_t atom_mask,
                                  int32_t new_token_slot_id) {
    if (!training_state_.initialized) {
        throw std::runtime_error("forwardStep: training state not initialized");
    }
    if (training_state_.kv_cache_len == 0) {
        throw std::runtime_error("forwardStep: call forwardInit first");
    }

    const int token_pos = training_state_.kv_cache_len;
    const int new_seq_len = token_pos + 1;
    
    if (new_seq_len > config_.max_seq_len) {
        throw std::runtime_error("forwardStep: sequence exceeds max_seq_len");
    }
    if (training_state_.kv_cache_k.empty()) {
        throw std::runtime_error("forwardStep: KV cache not allocated — call initInferenceState first");
    }

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // Append new token to device buffer (needed for embedding lookup)
    int* token_ids_ptr = reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data);
    cudaMemcpyAsync(token_ids_ptr + token_pos,
                    &new_token, sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    
    // Append numeric side-channel
    cudaMemcpyAsync(training_state_.cached_token_numeric_values.data + token_pos,
                    &numeric_value, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(reinterpret_cast<uint8_t*>(training_state_.cached_token_atom_mask.data) + token_pos,
                    &atom_mask, sizeof(uint8_t),
                    cudaMemcpyHostToDevice, stream);

    if (training_state_.cached_token_to_slot_map.data) {
        auto* slot_dst = reinterpret_cast<int32_t*>(training_state_.cached_token_to_slot_map.data) + token_pos;
        cudaError_t serr = cudaMemcpyAsync(slot_dst, &new_token_slot_id, sizeof(int32_t),
                                           cudaMemcpyHostToDevice, stream);
        if (serr != cudaSuccess) {
            throw std::runtime_error(std::string("forwardStep slot map: ") + cudaGetErrorString(serr));
        }
    }

    training_state_.kv_cache_len = new_seq_len;
    
    // Use KV-cached decode (O(1) per layer instead of O(n²))
    return executeDecodeForward_(token_pos);
}

//======================================================//
//  executeDecodeForward_ - KV-cached single-token decode
//  Processes one token through all encoder layers using
//  cached K,V from prior tokens. O(n) per layer instead of O(n²).
//======================================================//
Vector LanguageModel::executeDecodeForward_(int token_pos) {
    if (!training_state_.initialized) {
        throw std::runtime_error("executeDecodeForward_: training state not initialized");
    }
    if (training_state_.kv_cache_k.empty()) {
        throw std::runtime_error("executeDecodeForward_: KV cache not allocated");
    }

    using namespace TensorContract;
    namespace ag = autograd;

    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    const auto& cfg = config_;
    auto& ts = training_state_;

    const int d_model = cfg.d_model;
    const int num_heads = cfg.num_heads;
    const int num_kv_heads = ts.num_kv_heads;
    const int head_dim = d_model / num_heads;
    const int kv_dim = num_kv_heads * head_dim;
    const int num_layers = cfg.num_layers;
    const float rms_eps = cfg.rms_epsilon;
    const int seqlen_k = token_pos + 1;  // K cache has [0..token_pos] inclusive

    // Set cuBLAS handle for autograd matmul calls
    ag::set_autograd_cublas_handle(ts.cublas_handle);

    // ── Step 1: Embedding lookup for the single new token ──
    EmbeddingLayer* emb_layer = getEmbeddingLayer();
    if (!emb_layer) {
        throw std::runtime_error("executeDecodeForward_: embedding layer is NULL");
    }
    Tensor& emb_weights = emb_layer->tokenWeights();
    emb_weights.requires_grad = false;

    // Token ID is at device offset token_pos
    const int* token_id_ptr = reinterpret_cast<const int*>(ts.cached_token_ids_tensor.data) + token_pos;

    Tensor hidden = ag::embedding(emb_weights, token_id_ptr, 1, stream, 1.0f);
    // hidden: [1, d_model]

    // ── Step 2: PBM validation ──
    if (!ts.pbm_spec.valid || !ts.pbm_spec.rope_inv_freq || !ts.pbm_spec.alibi_slopes) {
        throw std::runtime_error("executeDecodeForward_: PBM (RoPE/ALiBi) not initialized");
    }

    // ── ExecutionBlock setup ──
    ExecutionBlockLayer* exec_block = getExecutionBlockLayer();
    ScratchBlockLayer* scratch_block = getScratchBlockLayer();
    const bool exec_block_configured = cfg.execution_block_enabled && exec_block != nullptr;
    if (exec_block_configured && (!scratch_block || !scratch_block->isEnabled())) {
        throw std::runtime_error(
            "executeDecodeForward_: execution_block_enabled requires an enabled ScratchBlock for row-local decode atom views");
    }
    const bool exec_block_active = cfg.execution_block_enabled
        && exec_block != nullptr
        && scratch_block != nullptr
        && scratch_block->isEnabled()
        && ts.has_inference_exec_memory
        && !ts.trace_state_by_row.empty()
        && !ts.execution_trace_by_row.empty();

    int exec_layer = -1;
    int exec_K = 0;
    if (exec_block_active) {
        exec_layer = cfg.execution_block_layer;
        if (exec_layer < 0) exec_layer = num_layers - 2;
        if (exec_layer < 0) exec_layer = 0;
        if (exec_layer >= num_layers) exec_layer = num_layers - 1;
        exec_K = cfg.execution_block_num_steps;
    }

    // ── Step 3: Process through all encoder layers ──
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        auto* enc = getGpuEncoder().getLayer(layer_idx);
        if (!enc) {
            throw std::runtime_error("executeDecodeForward_: encoder layer " +
                                     std::to_string(layer_idx) + " is NULL");
        }

        // 3a. RMSNorm1
        enc->rms1Gamma().requires_grad = false;
        Tensor ln1_out = ag::rms_norm(hidden, enc->rms1Gamma(), rms_eps, stream);

        // 3b. QKV projection: [1, d_model] @ W_qkv^T → [1, qkv_dim]
        enc->attnWqkv().requires_grad = false;
        Tensor qkv = ag::matmul(ln1_out, enc->attnWqkv(), stream,
                                nullptr, nullptr, true);
        if (enc->attnBqkv().data) {
            enc->attnBqkv().requires_grad = false;
            qkv = ag::broadcast_add(qkv, enc->attnBqkv(), stream);
        }

        // 3c. Split QKV → Q [1,nh,1,hd], K [1,nkv,1,hd], V [1,nkv,1,hd]
        auto [Q, K, V] = ag::split_and_reshape_qkv(
            qkv, 1, 1, num_heads, num_kv_heads, head_dim, stream);

        // 3d. RoPE rotation with position offset
        // For S=1, Q is [1,nh,1,hd] and K is [1,nkv,1,hd]
        PBM::launchRoPERotationGQA(
            Q.data, K.data, ts.pbm_spec.rope_inv_freq,
            1, num_heads, num_kv_heads, 1, head_dim,
            ts.pbm_spec.rotary_dim, stream, token_pos);

        // 3e. Convert K,V to BF16 and write to KV cache at position token_pos
        // For S=1: BHSD [1,H,1,D] = BSHD [1,1,H,D] — same contiguous layout [H*D]
        // Cache layout: [kv_capacity, nkv, hd] BSHD
        const size_t kv_slot_offset = static_cast<size_t>(token_pos) * num_kv_heads * head_dim;
        auto* k_cache = static_cast<__nv_bfloat16*>(ts.kv_cache_k[layer_idx]);
        auto* v_cache = static_cast<__nv_bfloat16*>(ts.kv_cache_v[layer_idx]);

        TensorConversion::convert_BHSD_to_BSHD_bf16(
            K.data, k_cache + kv_slot_offset,
            1, num_kv_heads, 1, head_dim, stream);
        TensorConversion::convert_BHSD_to_BSHD_bf16(
            V.data, v_cache + kv_slot_offset,
            1, num_kv_heads, 1, head_dim, stream);

        // 3f. Convert Q to BF16 for flash attention
        auto* q_bf16 = static_cast<__nv_bfloat16*>(ts.decode_q_bf16);
        TensorConversion::convert_BHSD_to_BSHD_bf16(
            Q.data, q_bf16, 1, num_heads, 1, head_dim, stream);

        // 3g. Flash attention with KV cache
        // Q: [1, 1, nh, hd] BSHD BF16 (seqlen_q=1)
        // K cache: [1, kv_capacity, nkv, hd] BSHD BF16 (seqlen_k=token_pos+1)
        auto* attn_out_bf16 = static_cast<__nv_bfloat16*>(ts.decode_attn_out_bf16);
        flash_attn_fwd_kvcache(
            q_bf16,
            k_cache,
            v_cache,
            attn_out_bf16,
            ts.kv_cache_softmax_lse,
            ts.pbm_spec.alibi_slopes,
            1,              // batch=1
            1,              // seqlen_q=1
            seqlen_k,       // cached tokens including current
            num_heads,
            num_kv_heads,
            head_dim,
            true,           // causal
            true,           // is_bf16
            stream);

        // 3h. Convert attention output BF16 → FP32
        // [1,1,nh,hd] BSHD → [1,nh,1,hd] BHSD (same memory for S=1)
        TensorConversion::convert_BSHD_bf16_to_BHSD(
            attn_out_bf16, ts.decode_attn_out_fp32,
            1, 1, num_heads, head_dim, stream);

        // 3i. Wrap attention output as non-owning Tensor [1, d_model]
        Tensor attn_flat;
        attn_flat.data = ts.decode_attn_out_fp32;
        attn_flat.shape = TensorShape::make_BSM(1, d_model);
        attn_flat.requires_grad = false;
        attn_flat.owns_data = false;
        attn_flat.stream = stream;

        // 3j. Output projection: [1, d_model] @ W_o^T
        enc->attnWo().requires_grad = false;
        Tensor proj = ag::matmul(attn_flat, enc->attnWo(), stream,
                                 nullptr, nullptr, true);
        if (enc->attnBo().data) {
            enc->attnBo().requires_grad = false;
            proj = ag::broadcast_add(proj, enc->attnBo(), stream);
        }

        // 3k. LayerScale + residual
        if (enc->layerScale1().data) {
            enc->layerScale1().requires_grad = false;
            proj = ag::layer_scale(proj, enc->layerScale1(), stream);
        }
        hidden = ag::add(hidden, proj, stream);
        // Skip center_columns for S=1 (centering a single vector is a no-op)

        // 3l. RMSNorm2
        enc->rms2Gamma().requires_grad = false;
        Tensor ln2_out = ag::rms_norm(hidden, enc->rms2Gamma(), rms_eps, stream);

        // 3m. FFN (SwiGLU)
        ForwardIntermediates ffn_ints;
        auto* ffn_layer = enc->getFfnLayer();
        if (!ffn_layer) {
            throw std::runtime_error("executeDecodeForward_: FFN layer is NULL at layer " +
                                     std::to_string(layer_idx));
        }
        Tensor ffn_out = ffn_layer->forward(ln2_out, ffn_ints, 0, layer_idx);

        // 3n. LayerScale + residual
        if (enc->layerScale2().data) {
            enc->layerScale2().requires_grad = false;
            ffn_out = ag::layer_scale(ffn_out, enc->layerScale2(), stream);
        }
        hidden = ag::add(hidden, ffn_out, stream);
        // Skip center_columns for S=1

        // ── ExecutionBlock: K execution steps at exec_layer ──
        if (exec_block_active && layer_idx == exec_layer) {
            const float T = cfg.execution_block_temp_start;
            const int32_t* slot_ptr = ts.cached_token_to_slot_map.data
                ? reinterpret_cast<const int32_t*>(ts.cached_token_to_slot_map.data) + token_pos
                : nullptr;
            auto row_atom_view = buildDecodeExecutionAtomView(
                *scratch_block, cfg, ts, stream, token_pos);

            // Bootstrap the new token's slot binding into execution memory.
            // During prefill, bootstrap runs inside AutogradTraining. During
            // decode, the single new token's numeric_value + slot_id must be
            // injected here so the execution block can operate on it.
            if (slot_ptr && ts.cached_token_numeric_values.data) {
                exec_block->bootstrapMemoryFromSlotMap(
                    ts.inference_exec_memory,
                    ts.cached_token_numeric_values.data + token_pos,
                    slot_ptr,
                    1, stream);
            }

            ExecutionBlockStepOutput last_step_diag;
            for (int step = 0; step < exec_K; ++step) {
                ExecutionBlockStepOutput step_diag;
                exec_block->executeStep(
                    hidden, ts.inference_exec_memory,
                    reinterpret_cast<const int*>(row_atom_view.atom_positions.data),
                    slot_ptr,  // device pointer to current token's slot_id
                    row_atom_view.num_atoms,
                    1,         // total_tokens (single token)
                    step, T, stream,
                    &step_diag,
                    0, 1,      // token_offset, row_tokens
                    ts.trace_state_by_row[0],
                    ts.execution_trace_by_row[0],
                    nullptr);
                ts.execution_trace_by_row[0].push_back(step_diag.record);
                if (step == exec_K - 1) {
                    last_step_diag = std::move(step_diag);
                }
            }

        }

        // ── ExecutionBlock: cross-attention read at layers >= exec_layer ──
        if (exec_block_active && layer_idx >= exec_layer) {
            exec_block->crossAttentionRead(
                hidden, ts.inference_exec_memory,
                1, stream, 0, 1);
        }
    }

    // ── Step 4: LM Head (RMSNorm → optional centering → W @ hidden^T) ──
    LMHeadLayer* lm_head = getLmHeadLayer();
    if (!lm_head) {
        throw std::runtime_error("executeDecodeForward_: LM head is NULL");
    }
    lm_head->setStream(stream);
    lm_head->setCublasHandle(ts.cublas_handle);

    Tensor centered_hidden;
    Tensor logits_tensor = lm_head->forward(hidden, centered_hidden);

    // ── Step 5: Extract logits to host ──
    Vector logits(cfg.vocab_size);
    if (!logits_tensor.data) {
        throw std::runtime_error("executeDecodeForward_: logits_tensor.data is NULL after LM head");
    }
    // logits_tensor is [1, vocab_size] — copy the single row
    cudaMemcpyAsync(logits.data.data(), logits_tensor.data,
                    cfg.vocab_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return logits;
}

//======================================================//
//  resetKVCache - Clear sequence state
//======================================================//
void LanguageModel::resetKVCache() {
    if (training_state_.initialized) {
        training_state_.kv_cache_len = 0;
        training_state_.has_inference_exec_memory = false;

        // Clear execution trace state (persisted across decode steps)
        training_state_.execution_trace_by_row.clear();
        training_state_.trace_state_by_row.clear();

        // Zero BF16 KV cache buffers
        const auto& cfg = getConfig();
        const int num_kv_heads = training_state_.num_kv_heads;
        const int head_dim = cfg.d_model / cfg.num_heads;
        const size_t kv_bytes_per_layer = static_cast<size_t>(num_kv_heads) *
            training_state_.kv_cache_capacity * head_dim * sizeof(__nv_bfloat16);
        cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

        for (size_t l = 0; l < training_state_.kv_cache_k.size(); ++l) {
            if (training_state_.kv_cache_k[l])
                cudaMemsetAsync(training_state_.kv_cache_k[l], 0, kv_bytes_per_layer, stream);
            if (training_state_.kv_cache_v[l])
                cudaMemsetAsync(training_state_.kv_cache_v[l], 0, kv_bytes_per_layer, stream);
        }
    }
}

//======================================================//
//  getKVCacheLength - Query current sequence length
//======================================================//
int LanguageModel::getKVCacheLength() const {
    return training_state_.kv_cache_len;
}

//======================================================//
//  predictNumericValue — removed (execution-first spec; no value head)
//======================================================//
float LanguageModel::predictNumericValue() const {
    throw std::runtime_error(
        "predictNumericValue: NumericHead removed; numeric values must come from ExecutionBlock "
        "and slot binding (spec Step Z), not hidden-state regression");
}

} // namespace GRIM

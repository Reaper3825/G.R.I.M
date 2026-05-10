//======================================================//
//  InitInferenceState.cu
//  Lightweight inference-only state initialization
//  
//  Minimal GPU buffer allocation for forward-pass only
//  Implemented as LanguageModel::initInferenceState()
//  
//  Author: GRIM Development Team
//  Date: December 9, 2025
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../Embedding/Embedding_GPU.hpp"
#include "../Encoding/Encoding_GPU.hpp"
#include "../ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

using GRIM::Tensor;

namespace GRIM {

#ifdef USE_CUDA

void LanguageModel::initInferenceState() {
    // RULE 20: double-init is a caller-order bug, not a recoverable condition.
    if (training_state_.initialized) {
        throw std::runtime_error(
            "[InitInferenceState] FATAL: training_state_.initialized is already true. "
            "Caller invoked initInferenceState() twice (or after initTrainingState). "
            "This is a call-order bug.");
    }
    
    const auto& model_cfg = getConfig();
    const auto cache_hp = HyperParameters::inferenceCacheConstructionHP(model_cfg);
    std::cout << "[InitInferenceState] Initializing INFERENCE-ONLY state..." << std::endl;
    std::cout << "  → Skipping gradient buffers" << std::endl;
    std::cout << "  → Skipping optimizer state" << std::endl;
    std::cout << "  → Minimal activation caches only" << std::endl;
    
    // 1. Initialize StreamController and cuBLAS handle
    // StreamController manages CUDA streams - initialize it first
    GRIM::StreamControllerConfig stream_config;
    stream_config.verbose = false;
    
    if (!training_state_.stream_ctrl.initialize(stream_config)) {
        throw std::runtime_error("[InitInferenceState] Failed to initialize StreamController");
    }
    
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    StreamController::fatalIfDefaultStream(primary_stream, "InitInferenceState primary stream");

    cublasStatus_t cublas_err = cublasCreate(training_state_.cublas_handle.outParam());
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("[InitInferenceState] Failed to create cuBLAS handle, cublasStatus=" +
                                 std::to_string(static_cast<int>(cublas_err)));
    }
    // Enable Tensor Core acceleration for Ampere+ GPUs
    cublasSetMathMode(training_state_.cublas_handle.get(), CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSetStream(training_state_.cublas_handle.get(), primary_stream);
    std::cout << "  ✓ Created cuBLAS handle with Tensor Core acceleration" << std::endl;
    
    training_state_.cached_num_layers = cache_hp.num_layers;
    const int num_kv_heads = cache_hp.num_kv_heads;
    
    // TensorContract shape helpers
    using TC = TensorContract::TensorShape;
    
    // 2. Create EmbeddingLayer (Pattern B: self-allocating, inference mode)
    {
        const auto emb_hp = HyperParameters::embeddingLayerConstructionHP(model_cfg);

        embedding_layer_ = std::make_unique<EmbeddingLayer>(
            emb_hp, /*seed=*/0, primary_stream, false
        );
        std::cout << "  ✓ EmbeddingLayer initialized (inference, Pattern B)" << std::endl;
    }

    // 3. Setup LM head layer (Pattern B: self-allocating)
    // LMHeadLayer owns weights, bias, and final_rms_gamma.
    // Weight tying is handled by passing tied embedding pointer to constructor.
    {
        const auto lm_hp = HyperParameters::lmHeadLayerConstructionHP(model_cfg);
        
        Tensor* tied_ptr = lm_hp.tie_embeddings ? &embedding_layer_->tokenWeights() : nullptr;
        
        lm_head_layer_ = std::make_unique<LMHeadLayer>(
            lm_hp, /*seed=*/0, primary_stream, tied_ptr
        );
        std::cout << "  ✓ LMHeadLayer initialized (inference, tie=" 
                  << (lm_hp.tie_embeddings ? "true" : "false") << ")" << std::endl;
    }

    // ReasoningHead layer (inference — loaded from checkpoint if present)
    const auto reasoning_hp = HyperParameters::reasoningHeadConstructionHP(model_cfg);
    if (reasoning_hp.enabled) {

        reasoning_head_layer_ = std::make_unique<ReasoningHeadLayer>(reasoning_hp, /*seed=*/0, primary_stream);
        std::cout << "  ✓ ReasoningHeadLayer initialized (inference, d_total="
                  << (reasoning_hp.d_model + reasoning_hp.atom_embedding_dim)
                  << ", num_ops=" << reasoning_hp.num_ops << ")" << std::endl;
    }

    // ExecutionBlock layer (inference — loaded from checkpoint if present)
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(model_cfg);
    if (execution_hp.enabled) {
        execution_block_layer_ = std::make_unique<ExecutionBlockLayer>(execution_hp, /*seed=*/0, primary_stream);
        std::cout << "  ✓ ExecutionBlockLayer initialized (inference, V="
                  << execution_hp.num_slots << ", K=" << execution_hp.num_exec_steps << ")" << std::endl;

        // Decode-time slot selector (inference — loaded from checkpoint)
        const auto selector_hp = HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
        if (selector_hp.enabled) {

            decode_time_slot_selector_layer_ = std::make_unique<DecodeTimeSlotSelectorLayer>(
                selector_hp, /*seed=*/0, primary_stream);

            decode_time_num_policy_ = std::make_unique<DecodeTimeNumPolicy>(selector_hp);

            std::cout << "  ✓ DecodeTimeSlotSelector initialized (inference, d_selector="
                      << selector_hp.d_selector << ")" << std::endl;
        }
    }

    // MTP heads (inference: allocate so load() can fill from .mtp sidecar; not used in forward)
    const auto mtp_hp = HyperParameters::mtpConstructionHP(model_cfg);
    if (mtp_hp.enabled) {
        mtp_heads_.resize(static_cast<size_t>(mtp_hp.k));
        for (int k = 0; k < mtp_hp.k; ++k) {
            auto& head = mtp_heads_[static_cast<size_t>(k)];
            head.weight = Tensor::zeros({mtp_hp.vocab_size, mtp_hp.d_model}, primary_stream, ("mtp_inf_" + std::to_string(k) + "_w").c_str());
            head.bias = Tensor::zeros({mtp_hp.vocab_size}, primary_stream, ("mtp_inf_" + std::to_string(k) + "_b").c_str());
        }
        std::cout << "  ✓ MTP " << mtp_hp.k << " heads allocated (weights loaded from .mtp if present)" << std::endl;
    }

    // 3b. Allocate minimal activation caches
    const size_t max_batch_size = cache_hp.max_batch_size;
    const size_t max_seq_len_cache = cache_hp.max_seq_len_cache;
    size_t max_tokens = cache_hp.max_tokens;
    
    std::cout << "  ℹ Allocating activation caches: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache 
              << ", total_tokens=" << max_tokens << std::endl;
    
    // NOTE: Tensor::zeros sets all bytes to 0 = UNK_TOKEN_ID when read as int32.
    // Harmless: forwardInit() writes prompt tokens, forwardStep() appends one at a time.
    // Only positions [0..generation_state_.kv_cache_len-1] are ever read. If a fill kernel is added, use PAD=1.
    training_state_.cached_token_ids_tensor = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for inference
        primary_stream,
        "cached_token_ids_tensor_inf"
    );
    // Mark as int32 type via allocation size (Tensor internally tracks)
    std::cout << "  ✓ Allocated token ID cache (Tensor API)" << std::endl;
    
    training_state_.cached_token_numeric_values = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for inference
        primary_stream,
        "cached_token_numeric_values_inf"
    );
    std::cout << "  ✓ Allocated numeric values cache (Tensor API)" << std::endl;
    
    training_state_.cached_token_atom_mask = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for inference
        primary_stream,
        "cached_token_atom_mask_inf"
    );
    std::cout << "  ✓ Allocated atom mask cache (Tensor API)" << std::endl;

    training_state_.cached_token_to_slot_map = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for inference
        primary_stream,
        "cached_token_to_slot_map_inf"
    );
    std::cout << "  ✓ Allocated token-to-slot map cache (Tensor API)" << std::endl;
    
    // DELETED: cached_embeddings_tensor - not used in inference (encoder output computed on-the-fly)
    // DELETED: encoder_layer_caches - intermediate tensor caching moved to AutogradIntermediates
    
    // DELETED: FA bf16 buffers — FlashAttentionLayer::ensureScratch() self-manages.
    // Autograd ScaledDotProductAttentionGradFn also self-allocates backward buffers.
    
    // Encoder outputs cache - use Tensor API
    training_state_.cached_encoder_output = Tensor::zeros(
        TC::make_BSM(static_cast<int>(max_tokens), cache_hp.d_model),
        false,  // no grad for inference
        primary_stream,
        "cached_encoder_output_inf"
    );
    std::cout << "  ✓ Allocated encoder output cache (Tensor API)" << std::endl;
    
    // Logits cache - use Tensor API
    training_state_.cached_logits_tensor = Tensor::zeros(
        TC::make_BSM(static_cast<int>(max_tokens), cache_hp.vocab_size),
        false,  // no grad for inference
        primary_stream,
        "cached_logits_tensor_inf"
    );
    std::cout << "  ✓ Allocated logits cache (Tensor API)" << std::endl;

    
    // 4. Allocate single-token buffers for incremental generation (KV cache)
    generation_state_.single_token_embedding = Tensor::zeros(
        TC::make_BSM(1, cache_hp.d_model),
        false,  // no grad for inference
        primary_stream,
        "single_token_embedding_inf"
    );
    
    generation_state_.single_token_hidden = Tensor::zeros(
        TC::make_BSM(1, cache_hp.d_model),
        false,  // no grad for inference
        primary_stream,
        "single_token_hidden_inf"
    );
    
    generation_state_.single_token_logits = Tensor::zeros(
        TC::make_BSM(1, cache_hp.vocab_size),
        false,  // no grad for inference
        primary_stream,
        "single_token_logits_inf"
    );
    
    generation_state_.resetSession();
    std::cout << "  ✓ Allocated single-token buffers for incremental generation" << std::endl;
    std::cout << "    KV cache capacity: " << max_seq_len_cache << " tokens" << std::endl;

    // 5. Allocate per-layer KV cache (BF16, BSHD layout for FlashAttention direct use)
    {
        const int head_dim = cache_hp.head_dim;
        const int n_layers = cache_hp.num_layers;
        const size_t kv_elems_per_layer = static_cast<size_t>(num_kv_heads) * max_seq_len_cache * head_dim;
        const size_t kv_bytes_per_layer = kv_elems_per_layer * sizeof(__nv_bfloat16);

        generation_state_.kv_cache.shape.num_layers = n_layers;
        generation_state_.kv_cache.shape.num_kv_heads = num_kv_heads;
        generation_state_.kv_cache.shape.head_dim = head_dim;
        generation_state_.kv_cache.shape.capacity_tokens = static_cast<int>(max_seq_len_cache);
        generation_state_.kv_cache.k.resize(n_layers);
        generation_state_.kv_cache.v.resize(n_layers);

        for (int l = 0; l < n_layers; ++l) {
            generation_state_.kv_cache.k[l].allocate(kv_bytes_per_layer, "kv_cache_k");
            generation_state_.kv_cache.v[l].allocate(kv_bytes_per_layer, "kv_cache_v");
            cudaMemsetAsync(generation_state_.kv_cache.k[l], 0, kv_bytes_per_layer, primary_stream);
            cudaMemsetAsync(generation_state_.kv_cache.v[l], 0, kv_bytes_per_layer, primary_stream);
        }

        // Shared softmax LSE scratch for decode: [num_heads, generation KV capacity rounded]
        const int kv_cap_rounded = ((static_cast<int>(max_seq_len_cache) + 127) / 128) * 128;
        const size_t lse_bytes = static_cast<size_t>(cache_hp.num_heads) * kv_cap_rounded * sizeof(float);
        generation_state_.kv_cache.softmax_lse.allocate(lse_bytes, "kv_cache_lse");
        cudaMemsetAsync(generation_state_.kv_cache.softmax_lse, 0, lse_bytes, primary_stream);

        const size_t total_kv_bytes = n_layers * 2 * kv_bytes_per_layer + lse_bytes;
        std::cout << "  ✓ Allocated per-layer KV cache: " << n_layers << " layers × "
                  << (kv_bytes_per_layer / 1024.0 / 1024.0) << " MB each (K+V) = "
                  << (total_kv_bytes / 1024.0 / 1024.0) << " MB total (BF16)" << std::endl;

        // Decode scratch buffers (tiny, reused per layer per decode step)
          const size_t q_bf16_bytes = static_cast<size_t>(cache_hp.num_heads) * head_dim * sizeof(__nv_bfloat16);
        generation_state_.decode_scratch.q_bf16.allocate(q_bf16_bytes, "decode_q_bf16");
        generation_state_.decode_scratch.attn_out_bf16.allocate(q_bf16_bytes, "decode_attn_out_bf16");
          generation_state_.decode_scratch.attn_out_fp32.allocate(static_cast<size_t>(cache_hp.d_model) * sizeof(float), "decode_attn_out_fp32");
        std::cout << "  ✓ Allocated decode scratch buffers ("
              << ((q_bf16_bytes * 2 + cache_hp.d_model * sizeof(float)) / 1024.0) << " KB)" << std::endl;
    }
    
    // NOTE: encoder_workspace DELETED (Rule 20/26) — autograd forward creates its own Tensors.

    // 6. Initialize ScratchBlock reasoning layer (if enabled)
    if (cache_hp.use_scratch_block) {
        try {
            // ScratchBlock layer owns its own buffers now (no external caches needed).
            // ScratchBlockLayer constructor handles weight allocation + initialization.
            
            if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
                std::cout << "  ✓ ScratchBlock reasoning layer enabled (d_model="
                          << cache_hp.d_model << ", atom_dim=" << cache_hp.scratch_block_atom_embedding_dim
                          << ", max_atoms=" << cache_hp.scratch_block_max_atoms << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("[InitInferenceState] ScratchBlock init failed (config says use_scratch_block=true): "
                                     + std::string(e.what()));
        }
    }
    
    training_state_.initialized = true;
    std::cout << "[InitInferenceState] ✓ Inference state initialized successfully" << std::endl;
    std::cout << "  Memory allocated for: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache
              << ", tokens=" << max_tokens << std::endl;
    
    // Compute actual allocated activation buffer sizes (weights excluded — loaded separately)
    const size_t token_cache_bytes = max_tokens * sizeof(float) * 3;  // token_ids + numeric_values + numeric_mask (all float Tensors)
    const size_t encoder_out_bytes = max_tokens * cache_hp.d_model * sizeof(float);
    const size_t logits_bytes = max_tokens * cache_hp.vocab_size * sizeof(float);
    const size_t single_token_bytes = (2 * cache_hp.d_model + cache_hp.vocab_size) * sizeof(float);  // embedding + hidden + logits
    const size_t workspace_bytes = static_cast<size_t>(cache_hp.d_ff) * max_seq_len_cache * 4 * sizeof(float);
    
    const size_t total_bytes = token_cache_bytes + encoder_out_bytes + logits_bytes +
                               single_token_bytes + workspace_bytes;
    
    std::cout << "  📊 Total GPU activation memory: ~" << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "      (excludes model weights loaded from checkpoint)" << std::endl;
     
    // Ensure all initialization is complete before returning
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        throw std::runtime_error("[InitInferenceState] cudaDeviceSynchronize failed: " +
                                 std::string(cudaGetErrorString(sync_err)));
    }
    
    std::cout << "[InitInferenceState] ✓ GPU synchronization complete" << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM

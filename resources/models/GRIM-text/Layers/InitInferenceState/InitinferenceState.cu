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

#define USE_CUDA

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
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
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

using GRIM::Tensor;

namespace GRIM {

#ifdef USE_CUDA

void LanguageModel::initInferenceState() {
    if (training_state_.initialized) {
        std::cout << "[InitInferenceState] Already initialized, skipping" << std::endl;
        return;
    }
    
    const auto& cfg = getConfig();
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

    cublasStatus_t cublas_err = cublasCreate(&training_state_.cublas_handle);
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("[InitInferenceState] Failed to create cuBLAS handle, cublasStatus=" +
                                 std::to_string(static_cast<int>(cublas_err)));
    }
    // Enable Tensor Core acceleration for Ampere+ GPUs
    cublasSetMathMode(training_state_.cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSetStream(training_state_.cublas_handle, primary_stream);
    std::cout << "  ✓ Created cuBLAS handle with Tensor Core acceleration" << std::endl;
    
    training_state_.cached_num_layers = cfg.num_layers;

    const int num_kv_heads = (cfg.num_kv_heads > 0) ? cfg.num_kv_heads : cfg.num_heads;
    if (cfg.num_heads % num_kv_heads != 0) {
        throw std::runtime_error("[InitInferenceState] Invalid GQA config: num_heads=" +
                                 std::to_string(cfg.num_heads) + " not divisible by num_kv_heads=" +
                                 std::to_string(num_kv_heads));
    }
    training_state_.num_heads = cfg.num_heads;
    training_state_.num_kv_heads = num_kv_heads;
    
    // TensorContract shape helpers
    using TC = TensorContract::TensorShape;
    
    // 2. Create EmbeddingLayer (Pattern B: self-allocating, inference mode)
    {
        EmbeddingLayerConfig emb_cfg{};
        emb_cfg.vocab_size = cfg.vocab_size;
        emb_cfg.d_model = cfg.d_model;
        emb_cfg.max_seq_len = cfg.max_seq_len;
        emb_cfg.positional_encoding = cfg.positional_encoding;
        emb_cfg.embedding_scale = 1.0f;  // Issue #140: no scaling for ALiBi/RoPE
        emb_cfg.requires_grad = false;   // Inference only — no gradients

        embedding_layer_ = std::make_unique<EmbeddingLayer>(
            emb_cfg, /*seed=*/0, primary_stream
        );
        std::cout << "  ✓ EmbeddingLayer initialized (inference, Pattern B)" << std::endl;
    }

    // 3. Setup LM head layer (Pattern B: self-allocating)
    // LMHeadLayer owns weights, bias, and final_rms_gamma.
    // Weight tying is handled by passing tied embedding pointer to constructor.
    {
        LMHeadLayerConfig lm_cfg{};
        lm_cfg.d_model = cfg.d_model;
        lm_cfg.vocab_size = cfg.vocab_size;
        lm_cfg.use_bias = cfg.use_bias;
        lm_cfg.has_final_rms_norm = true;
        lm_cfg.stream = primary_stream;
        lm_cfg.cublas_handle = training_state_.cublas_handle;
        
        Tensor* tied_ptr = cfg.tie_embeddings ? &embedding_layer_->tokenWeights() : nullptr;
        
        lm_head_layer_ = std::make_unique<LMHeadLayer>(
            lm_cfg, /*seed=*/0, primary_stream, tied_ptr
        );
        std::cout << "  ✓ LMHeadLayer initialized (inference, tie=" 
                  << (cfg.tie_embeddings ? "true" : "false") << ")" << std::endl;
    }
    
    // 3b. Allocate minimal activation caches
    const size_t max_batch_size = static_cast<size_t>(std::max(1, cfg.max_cached_batch));
    const size_t max_seq_len_cache = static_cast<size_t>(std::max(1, std::min(cfg.max_seq_len, cfg.max_cached_seq_len)));
    size_t max_tokens = max_batch_size * max_seq_len_cache;

    training_state_.max_cached_batch = static_cast<int>(max_batch_size);
    training_state_.max_cached_seq_len = static_cast<int>(max_seq_len_cache);
    training_state_.max_cached_tokens = max_tokens;
    training_state_.max_logit_tokens = max_tokens;
    
    std::cout << "  ℹ Allocating activation caches: batch=" << max_batch_size
              << ", seq_len=" << max_seq_len_cache 
              << ", total_tokens=" << max_tokens << std::endl;
    
    // NOTE: Tensor::zeros sets all bytes to 0 = UNK_TOKEN_ID when read as int32.
    // Harmless: forwardInit() writes prompt tokens, forwardStep() appends one at a time.
    // Only positions [0..kv_cache_len-1] are ever read. If a fill kernel is added, use PAD=1.
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
    
    training_state_.cached_token_numeric_mask = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for inference
        primary_stream,
        "cached_token_numeric_mask_inf"
    );
    std::cout << "  ✓ Allocated numeric mask cache (Tensor API)" << std::endl;
    
    // DELETED: cached_embeddings_tensor - not used in inference (encoder output computed on-the-fly)
    // DELETED: encoder_layer_caches - intermediate tensor caching moved to AutogradIntermediates
    
    // DELETED: FA bf16 buffers — FlashAttentionLayer::ensureScratch() self-manages.
    // Autograd ScaledDotProductAttentionGradFn also self-allocates backward buffers.
    
    // Encoder outputs cache - use Tensor API
    training_state_.cached_encoder_output = Tensor::zeros(
        TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model),
        false,  // no grad for inference
        primary_stream,
        "cached_encoder_output_inf"
    );
    std::cout << "  ✓ Allocated encoder output cache (Tensor API)" << std::endl;
    
    // Logits cache - use Tensor API
    training_state_.cached_logits_tensor = Tensor::zeros(
        TC::make_BSM(static_cast<int>(max_tokens), cfg.vocab_size),
        false,  // no grad for inference
        primary_stream,
        "cached_logits_tensor_inf"
    );
    std::cout << "  ✓ Allocated logits cache (Tensor API)" << std::endl;

    
    // 4. Allocate single-token buffers for incremental generation (KV cache)
    training_state_.single_token_embedding = Tensor::zeros(
        TC::make_BSM(1, cfg.d_model),
        false,  // no grad for inference
        primary_stream,
        "single_token_embedding_inf"
    );
    
    training_state_.single_token_hidden = Tensor::zeros(
        TC::make_BSM(1, cfg.d_model),
        false,  // no grad for inference
        primary_stream,
        "single_token_hidden_inf"
    );
    
    training_state_.single_token_logits = Tensor::zeros(
        TC::make_BSM(1, cfg.vocab_size),
        false,  // no grad for inference
        primary_stream,
        "single_token_logits_inf"
    );
    
    training_state_.kv_cache_len = 0;
    training_state_.kv_cache_capacity = max_seq_len_cache;
    std::cout << "  ✓ Allocated single-token buffers for incremental generation" << std::endl;
    std::cout << "    KV cache capacity: " << training_state_.kv_cache_capacity << " tokens" << std::endl;
    
    // NOTE: encoder_workspace DELETED (Rule 20/26) — autograd forward creates its own Tensors.

    // 6. Initialize ScratchBlock reasoning layer (if enabled)
    if (cfg.use_scratch_block) {
        try {
            // ScratchBlock layer owns its own buffers now (no external caches needed).
            // ScratchBlockLayer constructor handles weight allocation + initialization.
            // scratch_pool remains nullptr for inference (only allocated in InitTrainingState).
            
            if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
                std::cout << "  ✓ ScratchBlock reasoning layer enabled (d_model="
                          << cfg.d_model << ", atom_dim=" << cfg.scratch_block_atom_embedding_dim
                          << ", max_atoms=" << cfg.scratch_block_max_atoms << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            throw std::runtime_error("[InitInferenceState] ScratchBlock init failed (config says use_scratch_block=true): "
                                     + std::string(e.what()));
        }
    }
    
    // 7. Check if activation quantization is enabled
    const auto& quant_cfg = cfg.activation_quantization;
    if (quant_cfg.enabled) {
        std::cout << "  ℹ Activation quantization ENABLED:" << std::endl;
        std::cout << "    - Scale: " << quant_cfg.scale << std::endl;
        std::cout << "    - Range: [" << quant_cfg.clip_min << ", " << quant_cfg.clip_max << "]" << std::endl;
        std::cout << "    - Apply to embeddings: " << (quant_cfg.apply_to_embeddings ? "YES" : "NO") << std::endl;
        std::cout << "    - Apply to encoder outputs: " << (quant_cfg.apply_to_encoder_outputs ? "YES" : "NO") << std::endl;
        std::cout << "    - Apply to logits: " << (quant_cfg.apply_to_logits ? "YES" : "NO") << std::endl;
        
        // Quantization layer will be initialized on first use in executeInferenceForward_
    } else {
        std::cout << "  ℹ Activation quantization DISABLED" << std::endl;
    }
    
    training_state_.initialized = true;
    std::cout << "[InitInferenceState] ✓ Inference state initialized successfully" << std::endl;
    std::cout << "  Memory allocated for: batch=" << training_state_.max_cached_batch
              << ", seq_len=" << training_state_.max_cached_seq_len
              << ", tokens=" << training_state_.max_cached_tokens << std::endl;
    
    // Compute actual allocated activation buffer sizes (weights excluded — loaded separately)
    const size_t token_cache_bytes = max_tokens * sizeof(float) * 3;  // token_ids + numeric_values + numeric_mask (all float Tensors)
    const size_t encoder_out_bytes = max_tokens * cfg.d_model * sizeof(float);
    const size_t logits_bytes = max_tokens * cfg.vocab_size * sizeof(float);
    const size_t single_token_bytes = (2 * cfg.d_model + cfg.vocab_size) * sizeof(float);  // embedding + hidden + logits
    const size_t workspace_bytes = static_cast<size_t>(cfg.d_ff) * max_seq_len_cache * 4 * sizeof(float);
    const size_t numeric_pred_bytes = 0;  // NumericHead deleted (Issue #143)
    
    const size_t total_bytes = token_cache_bytes + encoder_out_bytes + logits_bytes +
                               single_token_bytes + workspace_bytes + numeric_pred_bytes;
    
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

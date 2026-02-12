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
#include "../Encoding/Encoding_GPU.hpp"
#include "../ScratchBlock/ScratchBlock_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/TrainingState/TrainingTensors.hpp"

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
    
    cudaError_t err;  // Declare error variable for CUDA calls
    
    // 1. Initialize StreamController and cuBLAS handle
    // StreamController manages CUDA streams - initialize it first
    GRIM::StreamControllerConfig stream_config;
    stream_config.verbose = false;
    
    if (!training_state_.stream_ctrl.initialize(stream_config)) {
        std::cerr << "[InitInferenceState] Failed to initialize StreamController" << std::endl;
        return;
    }
    
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    StreamController::fatalIfDefaultStream(primary_stream, "InitInferenceState primary stream");

    cublasStatus_t cublas_err = cublasCreate(&training_state_.cublas_handle);
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[InitInferenceState] Failed to create cuBLAS handle" << std::endl;
        return;
    }
    // Enable Tensor Core acceleration for Ampere+ GPUs
    cublasSetMathMode(training_state_.cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSetStream(training_state_.cublas_handle, primary_stream);
    std::cout << "  ✓ Created cuBLAS handle with Tensor Core acceleration" << std::endl;
    
    training_state_.cached_num_layers = cfg.num_layers;

    const int num_kv_heads = (cfg.num_kv_heads > 0) ? cfg.num_kv_heads : cfg.num_heads;
    if (cfg.num_heads % num_kv_heads != 0) {
        std::cerr << "[InitInferenceState] ERROR: Invalid GQA config: num_heads="
                  << cfg.num_heads << " num_kv_heads=" << num_kv_heads << std::endl;
        return;
    }
    training_state_.num_heads = cfg.num_heads;
    training_state_.num_kv_heads = num_kv_heads;
    
    // TensorContract shape helpers
    using TC = TensorContract::TensorShape;
    
    // Create TrainingTensors container FIRST — all weight tensors live here
    // (no gradients for inference, but same container as training for uniform access)
    training_state_.tensors_ = std::make_unique<TrainingTensors>();
    
    // 2. Allocate embedding weights (will be populated by model load)
    training_state_.tensors_->embedding_weights = Tensor::zeros(
        TC::make_BSM(cfg.vocab_size, cfg.d_model),
        false,  // no grad for inference
        primary_stream,
        "embedding_weights_inf"
    );
    std::cout << "  ✓ Allocated embedding weights (Tensor API)" << std::endl;
    
    // 3. Setup LM head weights (tied to embeddings or separate allocation)
    if (cfg.tie_embeddings) {
        if (!training_state_.tensors_->embedding_weights.data) {
            throw std::runtime_error("[InitInferenceState] embedding_weights allocation failed — cannot tie");
        }
        // from_ptr: wraps embedding buffer, doesn't own data
        training_state_.tensors_->lm_head_weights = Tensor::from_ptr(
            training_state_.tensors_->embedding_weights.data,
            TC::make_BSM(cfg.vocab_size, cfg.d_model),
            false,  // doesn't own data
            false,   // no grad for inference
            "lm_head_weights_tied_inference"
        );
        training_state_.tensors_->lm_head_weights.owns_data = false;
        std::cout << "  ✓ LM head weights tied to embeddings (Tensor API)" << std::endl;
    } else {
        // Allocate separate LM head weights (will be loaded from model file)
        training_state_.tensors_->lm_head_weights = Tensor::zeros(
            TC::make_BSM(cfg.vocab_size, cfg.d_model),
            false,  // no grad for inference
            primary_stream,
            "lm_head_weights_inf"
        );
        std::cout << "  ✓ Allocated LM head weights (Tensor API)" << std::endl;
    }
    
    // Optional: LM head bias
    if (cfg.use_bias) {
        training_state_.tensors_->lm_head_bias = Tensor::zeros(
            TC::make_BSM(1, cfg.vocab_size),
            false,  // no grad for inference
            primary_stream,
            "lm_head_bias_inf"
        );
        std::cout << "  Allocated LM head bias (Tensor API)" << std::endl;
    }

    if (cfg.numeric_head_enabled) {
        training_state_.tensors_->numeric_head_weights = Tensor::zeros(
            TC::make_BSM(1, cfg.d_model),
            false,  // no grad for inference
            primary_stream,
            "numeric_head_weights_inf"
        );
        if (cfg.use_bias) {
            training_state_.tensors_->numeric_head_bias = Tensor::zeros(
                TC::make_BSM(1, 1),
                false,  // no grad for inference
                primary_stream,
                "numeric_head_bias_inf"
            );
        }
        std::cout << "  Allocated numeric head weights (Tensor API)" << std::endl;
    }
    
    // Final RMSNorm gamma [d_model] - needed for inference forward pass
    training_state_.tensors_->final_rms_gamma = Tensor::zeros(
        TC::make_BSM(1, cfg.d_model),
        false,  // no grad for inference
        primary_stream,
        "final_rms_gamma_inf"
    );
    {
        std::vector<float> ones(cfg.d_model, 1.0f);
        cudaMemcpyAsync(training_state_.tensors_->final_rms_gamma.data, ones.data(),
                        cfg.d_model * sizeof(float), cudaMemcpyHostToDevice, primary_stream);
    }
    std::cout << "  Allocated final RMSNorm gamma (Tensor API)" << std::endl;
    
    // 3. Allocate minimal activation caches
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
    
    // Token IDs cache - use Tensor API
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

    if (cfg.numeric_head_enabled) {
        training_state_.cached_numeric_predictions = Tensor::zeros(
            TC::make_BSM(1, static_cast<int>(max_tokens)),
            false,  // no grad for inference
            primary_stream,
            "cached_numeric_predictions_inf"
        );
        std::cout << "  ✓ Allocated numeric predictions cache (Tensor API)" << std::endl;
    }
    
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
    
    // 5. Allocate encoder workspace (reused across forward passes for cuBLAS)
    // Conservative estimate: 4x the max intermediate size (d_ff * max_seq)
    size_t workspace_elems = static_cast<size_t>(cfg.d_ff) * max_seq_len_cache * 4;
    training_state_.encoder_workspace = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(workspace_elems)),
        false,  // no grad for inference
        primary_stream,
        "encoder_workspace_inf"
    );
    training_state_.encoder_workspace_size = workspace_elems * sizeof(float);
    std::cout << "  ✓ Allocated encoder workspace (Tensor API): " << (training_state_.encoder_workspace_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // 6. Initialize ScratchBlock reasoning layer (if enabled)
    if (cfg.use_scratch_block) {
        try {
            // Allocate scratch block cache buffers using Tensor API
            const size_t max_atoms = cfg.scratch_block_max_atoms;
            const size_t atom_emb_dim = cfg.scratch_block_atom_embedding_dim;
            
            training_state_.cached_scratch_block_embeddings = Tensor::zeros(
                TC::make_BSM(static_cast<int>(max_tokens), static_cast<int>(atom_emb_dim)),
                false,  // no grad for inference
                primary_stream,
                "cached_scratch_block_embeddings_inf"
            );
            
            training_state_.cached_scratch_block_positions = Tensor::zeros(
                TC::make_BSM(1, static_cast<int>(max_tokens)),
                false,  // no grad for inference
                primary_stream,
                "cached_scratch_block_positions_inf"
            );
            
            training_state_.cached_scratch_block_num_atoms = Tensor::zeros(
                TC::make_BSM(1, static_cast<int>(max_batch_size)),
                false,  // no grad for inference
                primary_stream,
                "cached_scratch_block_num_atoms_inf"
            );
            
            // Initialize scratch pool for pinned memory transfers (simplified for inference)
            training_state_.scratch_enabled = true;
            training_state_.scratch_pool = nullptr;  // Pool will be initialized on first use if needed
            
            std::cout << "  ✓ ScratchBlock cache buffers allocated (Tensor API)" << std::endl;
            
            // Create ScratchBlock layer instance
            if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
                std::cout << "  ✓ ScratchBlock reasoning layer enabled (d_model="
                          << cfg.d_model << ", atom_dim=" << atom_emb_dim
                          << ", max_atoms=" << max_atoms << ")" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "[InitInferenceState] WARNING: ScratchBlock init failed: " << e.what() << std::endl;
            // Non-fatal, continue without ScratchBlock
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
        
        // Quantization layer will be initialized on first use in forwardWithCache
    } else {
        std::cout << "  ℹ Activation quantization DISABLED" << std::endl;
    }
    
    training_state_.initialized = true;
    std::cout << "[InitInferenceState] ✓ Inference state initialized successfully" << std::endl;
    std::cout << "  Memory allocated for: batch=" << training_state_.max_cached_batch
              << ", seq_len=" << training_state_.max_cached_seq_len
              << ", tokens=" << training_state_.max_cached_tokens << std::endl;
    const size_t token_cache_bytes = max_tokens * sizeof(int);
    const size_t embedding_cache_bytes = max_tokens * cfg.d_model * sizeof(float);
    const size_t layer_cache_bytes = cfg.num_layers * max_tokens * (
        5 * cfg.d_model +  // ln1, attn_out, residual1, ln2, layer_out
        2 * cfg.d_ff +     // ffn_pre_gelu, ffn_out
        5 * cfg.d_model    // Q, K, V, attn_in, attn_raw_out
    ) * sizeof(float);
    const size_t encoder_out_bytes = max_tokens * cfg.d_model * sizeof(float);
    const size_t logits_bytes = max_tokens * cfg.vocab_size * sizeof(float);
    const size_t lm_head_bytes = cfg.tie_embeddings ? 0 : (cfg.vocab_size * cfg.d_model * sizeof(float));
    
    const size_t total_bytes = token_cache_bytes + embedding_cache_bytes + layer_cache_bytes +
                               encoder_out_bytes + logits_bytes + lm_head_bytes;
    
    std::cout << "  📊 Total GPU memory: ~" << (total_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "      (excludes model weights and workspace)" << std::endl;
    
    // Ensure all initialization is complete before returning
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "[InitInferenceState] ERROR: cudaDeviceSynchronize failed: " 
                  << cudaGetErrorString(sync_err) << std::endl;
        training_state_.initialized = false;
        return;
    }
    
    std::cout << "[InitInferenceState] ✓ GPU synchronization complete" << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM

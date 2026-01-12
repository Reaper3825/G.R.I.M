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
#include "../Embedding/Embedding_GPU.hpp"
#include "../ScratchBlock/ScratchBlock_GPU.hpp"
#include "../FlashAttention/Flash_Attention_Kernal.hpp"
#include "../../Shared/StreamController/StreamController_GPU.hpp"

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
    stream_config.create_transfer_stream = false;  // Inference doesn't need transfer stream
    stream_config.create_auxiliary_stream = false;
    
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
    
    // 2. Setup LM head weights (tied to embeddings for inference)
    if (cfg.tie_embeddings) {
        auto* embedding_runtime = &getGpuEmbedder();
        if (embedding_runtime && embedding_runtime->token_buffer) {
            training_state_.lm_head_weights = embedding_runtime->token_buffer;
            std::cout << "  ✓ LM head weights tied to embeddings" << std::endl;
        } else {
            std::cerr << "[InitInferenceState] ERROR: Cannot tie embeddings, buffer not available" << std::endl;
            return;
        }
    } else {
        // Allocate separate LM head weights (will be loaded from model file)
        size_t lm_head_weight_size = cfg.vocab_size * cfg.d_model;
        err = cudaMalloc(&training_state_.lm_head_weights, lm_head_weight_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "[InitInferenceState] Failed to allocate LM head weights: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        std::cout << "  ✓ Allocated LM head weights" << std::endl;
    }
    
    // Optional: LM head bias
    if (cfg.use_bias) {
        err = cudaMalloc(&training_state_.lm_head_bias, cfg.vocab_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "[InitInferenceState] Failed to allocate LM head bias: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        cudaMemsetAsync(training_state_.lm_head_bias, 0, cfg.vocab_size * sizeof(float), primary_stream);
        std::cout << "  ✓ Allocated LM head bias" << std::endl;
    }

    if (cfg.numeric_head_enabled) {
        err = cudaMalloc(&training_state_.numeric_head_weights, cfg.d_model * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "[InitInferenceState] Failed to allocate numeric head weights: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        if (cfg.use_bias) {
            err = cudaMalloc(&training_state_.numeric_head_bias, sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "[InitInferenceState] Failed to allocate numeric head bias: " << cudaGetErrorString(err) << std::endl;
                return;
            }
            cudaMemsetAsync(training_state_.numeric_head_bias, 0, sizeof(float), primary_stream);
        }
        std::cout << "  ✓ Allocated numeric head weights" << std::endl;
    }
    
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
    
    // Token IDs cache
    err = cudaMalloc(&training_state_.cached_token_ids, max_tokens * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate token cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&training_state_.cached_token_numeric_values, max_tokens * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate numeric value cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    cudaMemsetAsync(training_state_.cached_token_numeric_values, 0, max_tokens * sizeof(float), primary_stream);
    
    err = cudaMalloc(&training_state_.cached_token_numeric_mask, max_tokens * sizeof(uint8_t));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate numeric mask cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    cudaMemsetAsync(training_state_.cached_token_numeric_mask, 0, max_tokens * sizeof(uint8_t), primary_stream);
    
    // Embeddings cache
    err = cudaMalloc(&training_state_.cached_embeddings, max_tokens * cfg.d_model * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate embedding cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Per-layer activation caches
    training_state_.cached_ln1_outputs.resize(cfg.num_layers, nullptr);
    training_state_.cached_attn_outputs.resize(cfg.num_layers, nullptr);
    training_state_.cached_residual1_outputs.resize(cfg.num_layers, nullptr);
    training_state_.cached_ln2_outputs.resize(cfg.num_layers, nullptr);
    training_state_.cached_ffn_pre_gelu.resize(cfg.num_layers, nullptr);
    training_state_.cached_ffn_outputs.resize(cfg.num_layers, nullptr);
    training_state_.cached_layer_outputs.resize(cfg.num_layers, nullptr);
    
    training_state_.cached_Q.resize(cfg.num_layers, nullptr);
    training_state_.cached_K.resize(cfg.num_layers, nullptr);
    training_state_.cached_V.resize(cfg.num_layers, nullptr);
    training_state_.cached_attn_inputs.resize(cfg.num_layers, nullptr);
    training_state_.cached_attn_bhsd.resize(cfg.num_layers, nullptr);
    training_state_.cached_softmax_lse.resize(cfg.num_layers, nullptr);
    
    const size_t softmax_lse_elems = static_cast<size_t>(max_batch_size) *
                                     static_cast<size_t>(cfg.num_heads) *
                                     static_cast<size_t>(max_seq_len_cache);

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        cudaMalloc(&training_state_.cached_ln1_outputs[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_attn_outputs[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_residual1_outputs[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_ln2_outputs[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_ffn_pre_gelu[layer], max_tokens * cfg.d_ff * sizeof(float));
        cudaMalloc(&training_state_.cached_ffn_outputs[layer], max_tokens * cfg.d_ff * sizeof(float));
        cudaMalloc(&training_state_.cached_layer_outputs[layer], max_tokens * cfg.d_model * sizeof(float));
        
        cudaMalloc(&training_state_.cached_Q[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_K[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_V[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_attn_inputs[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_attn_bhsd[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_softmax_lse[layer], softmax_lse_elems * sizeof(float));
    }
    std::cout << "  ✓ Allocated per-layer activation caches (" << cfg.num_layers << " layers)" << std::endl;

    if (!training_state_.forward_layer_caches) {
        training_state_.forward_layer_cache_count = cfg.num_layers;
        training_state_.forward_layer_caches = new EncoderLayerCache[cfg.num_layers]();
    }
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        auto& cache = training_state_.forward_layer_caches[layer];
        cache.ln1_output = training_state_.cached_ln1_outputs[layer];
        cache.attn_input = training_state_.cached_attn_inputs[layer];
        cache.attn_bhsd = training_state_.cached_attn_bhsd[layer];
        cache.softmax_lse = training_state_.cached_softmax_lse[layer];
        cache.attn_output = training_state_.cached_attn_outputs[layer];
        cache.residual1 = training_state_.cached_residual1_outputs[layer];
        cache.ln2_input = training_state_.cached_residual1_outputs[layer];
        cache.ln2_output = training_state_.cached_ln2_outputs[layer];
        cache.ffn_input = training_state_.cached_ln2_outputs[layer];
        cache.ffn_pre_gelu = training_state_.cached_ffn_pre_gelu[layer];
        cache.ffn_output = training_state_.cached_ffn_outputs[layer];
        cache.layer_output = training_state_.cached_layer_outputs[layer];
        cache.q = training_state_.cached_Q[layer];
        cache.k = training_state_.cached_K[layer];
        cache.v = training_state_.cached_V[layer];
    }

    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    const size_t fa_q_elems = static_cast<size_t>(max_batch_size) *
                              static_cast<size_t>(cfg.num_heads) *
                              static_cast<size_t>(max_seq_len_cache) *
                              static_cast<size_t>(head_dim);
    const size_t fa_kv_elems = static_cast<size_t>(max_batch_size) *
                               static_cast<size_t>(training_state_.num_kv_heads) *
                               static_cast<size_t>(max_seq_len_cache) *
                               static_cast<size_t>(head_dim);
    training_state_.fa_q_bf16_elems = fa_q_elems;
    training_state_.fa_kv_bf16_elems = fa_kv_elems;

    auto alloc_bf16 = [&](const char* label, __nv_bfloat16** ptr, size_t elems) -> bool {
        cudaError_t berr = cudaMalloc(ptr, elems * sizeof(__nv_bfloat16));
        if (berr != cudaSuccess) {
            std::cerr << "[InitInferenceState] Failed to allocate " << label << ": "
                      << cudaGetErrorString(berr) << std::endl;
            return false;
        }
        return true;
    };

    if (!alloc_bf16("fa_q_bf16", &training_state_.fa_q_bf16, fa_q_elems)) return;
    if (!alloc_bf16("fa_k_bf16", &training_state_.fa_k_bf16, fa_kv_elems)) return;
    if (!alloc_bf16("fa_v_bf16", &training_state_.fa_v_bf16, fa_kv_elems)) return;
    if (!alloc_bf16("fa_out_bf16", &training_state_.fa_out_bf16, fa_q_elems)) return;
    if (!alloc_bf16("fa_dout_bf16", &training_state_.fa_dout_bf16, fa_q_elems)) return;
    if (!alloc_bf16("fa_dq_bf16", &training_state_.fa_dq_bf16, fa_q_elems)) return;
    if (!alloc_bf16("fa_dk_bf16", &training_state_.fa_dk_bf16, fa_kv_elems)) return;
    if (!alloc_bf16("fa_dv_bf16", &training_state_.fa_dv_bf16, fa_kv_elems)) return;
    
    // Encoder outputs cache
    err = cudaMalloc(&training_state_.cached_encoder_outputs, max_tokens * cfg.d_model * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate encoder output cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Logits cache
    err = cudaMalloc(&training_state_.cached_logits, max_tokens * cfg.vocab_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate logits cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "  ✓ Allocated encoder output and logits caches" << std::endl;

    if (cfg.numeric_head_enabled) {
        err = cudaMalloc(&training_state_.cached_numeric_predictions, max_tokens * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "[InitInferenceState] Failed to allocate numeric prediction cache: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
    
    // 4. Allocate single-token buffers for incremental generation (KV cache)
    err = cudaMalloc(&training_state_.single_token_embedding, cfg.d_model * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate single_token_embedding: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&training_state_.single_token_hidden, cfg.d_model * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate single_token_hidden: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&training_state_.single_token_logits, cfg.vocab_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate single_token_logits: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    training_state_.kv_cache_len = 0;
    training_state_.kv_cache_capacity = max_seq_len_cache;
    std::cout << "  ✓ Allocated single-token buffers for incremental generation" << std::endl;
    std::cout << "    KV cache capacity: " << training_state_.kv_cache_capacity << " tokens" << std::endl;
    
    // 5. Allocate encoder workspace (reused across forward passes for cuBLAS)
    // Conservative estimate: 4x the max intermediate size (d_ff * max_seq)
    size_t workspace_bytes = static_cast<size_t>(cfg.d_ff) * max_seq_len_cache * sizeof(float) * 4;
    
    err = cudaMalloc(&training_state_.encoder_workspace, workspace_bytes);
    if (err != cudaSuccess) {
        std::cerr << "[InitInferenceState] Failed to allocate encoder workspace: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    training_state_.encoder_workspace_size = workspace_bytes;
    std::cout << "  ✓ Allocated encoder workspace: " << (workspace_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    
    // 6. Initialize ScratchBlock reasoning layer (if enabled)
    if (cfg.use_scratch_block) {
        try {
            // Allocate scratch block cache buffers
            const size_t max_atoms = cfg.scratch_block_max_atoms;
            const size_t atom_emb_dim = cfg.scratch_block_atom_embedding_dim;
            
            err = cudaMalloc(&training_state_.cached_scratch_block_embeddings,
                           max_tokens * atom_emb_dim * sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "[InitInferenceState] Failed to allocate scratch block embeddings" << std::endl;
                return;
            }
            
            err = cudaMalloc(&training_state_.cached_scratch_block_positions,
                           max_tokens * sizeof(int));
            if (err != cudaSuccess) {
                std::cerr << "[InitInferenceState] Failed to allocate scratch block positions" << std::endl;
                return;
            }
            
            err = cudaMalloc(&training_state_.cached_scratch_block_num_atoms,
                           max_batch_size * sizeof(int));
            if (err != cudaSuccess) {
                std::cerr << "[InitInferenceState] Failed to allocate scratch block atom counts" << std::endl;
                return;
            }
            
            // Initialize scratch pool for pinned memory transfers (simplified for inference)
            training_state_.scratch_enabled = true;
            training_state_.scratch_pool = nullptr;  // Pool will be initialized on first use if needed
            
            std::cout << "  ✓ ScratchBlock cache buffers allocated (inference mode)" << std::endl;
            
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

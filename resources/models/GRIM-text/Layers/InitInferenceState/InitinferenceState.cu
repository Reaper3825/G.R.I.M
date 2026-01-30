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
    
    // TensorContract shape helpers
    using TC = TensorContract::TensorShape;
    
    // 2. Setup LM head weights (tied to embeddings for inference)
    // Tensor API: Use from_ptr for tied embeddings, zeros for separate allocation
    if (cfg.tie_embeddings) {
        auto* embedding_runtime = &getGpuEmbedder();
        if (embedding_runtime && embedding_runtime->token_buffer) {
            // Use from_ptr: doesn't own the data, just wraps the embedding buffer
            training_state_.lm_head_weights = Tensor::from_ptr(
                embedding_runtime->token_buffer,
                TC::make_BSM(cfg.vocab_size, cfg.d_model),
                false,  // doesn't own data
                false   // no grad for inference
            );
            std::cout << "  ✓ LM head weights tied to embeddings (Tensor API)" << std::endl;
        } else {
            std::cerr << "[InitInferenceState] ERROR: Cannot tie embeddings, buffer not available" << std::endl;
            return;
        }
    } else {
        // Allocate separate LM head weights (will be loaded from model file)
        training_state_.lm_head_weights = Tensor::zeros(
            TC::make_BSM(cfg.vocab_size, cfg.d_model),
            false,  // no grad for inference
            primary_stream
        );
        std::cout << "  ✓ Allocated LM head weights (Tensor API)" << std::endl;
    }
    
    // Optional: LM head bias
    if (cfg.use_bias) {
        training_state_.lm_head_bias = Tensor::zeros(
            TC::make_BSM(1, cfg.vocab_size),
            false,  // no grad for inference
            primary_stream
        );
        std::cout << "  ✓ Allocated LM head bias (Tensor API)" << std::endl;
    }

    if (cfg.numeric_head_enabled) {
        training_state_.numeric_head_weights = Tensor::zeros(
            TC::make_BSM(1, cfg.d_model),
            false,  // no grad for inference
            primary_stream
        );
        if (cfg.use_bias) {
            training_state_.numeric_head_bias = Tensor::zeros(
                TC::make_BSM(1, 1),
                false,  // no grad for inference
                primary_stream
            );
        }
        std::cout << "  ✓ Allocated numeric head weights (Tensor API)" << std::endl;
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
    
    // Token IDs cache - use Tensor API
    training_state_.cached_token_ids_tensor = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for inference
        primary_stream
    );
    // Mark as int32 type via allocation size (Tensor internally tracks)
    std::cout << "  ✓ Allocated token ID cache (Tensor API)" << std::endl;
    
    training_state_.cached_token_numeric_values = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for inference
        primary_stream
    );
    std::cout << "  ✓ Allocated numeric values cache (Tensor API)" << std::endl;
    
    training_state_.cached_token_numeric_mask = Tensor::zeros(
        TC::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for inference
        primary_stream
    );
    std::cout << "  ✓ Allocated numeric mask cache (Tensor API)" << std::endl;
    
    // Embeddings cache - use Tensor API
    training_state_.cached_embeddings_tensor = Tensor::zeros(
        TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model),
        false,  // no grad for inference
        primary_stream
    );
    std::cout << "  ✓ Allocated embeddings cache (Tensor API)" << std::endl;
    
    // Per-layer activation caches using Tensor-based EncoderLayerCacheTensors
    training_state_.encoder_layer_caches.resize(cfg.num_layers);
    
    const size_t softmax_lse_elems = static_cast<size_t>(max_batch_size) *
                                     static_cast<size_t>(cfg.num_heads) *
                                     static_cast<size_t>(max_seq_len_cache);
    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    const size_t attn_bhsd_elems = static_cast<size_t>(max_batch_size) *
                                   static_cast<size_t>(cfg.num_heads) *
                                   static_cast<size_t>(max_seq_len_cache) *
                                   static_cast<size_t>(head_dim);
    const size_t kv_bhsd_elems = static_cast<size_t>(max_batch_size) *
                                 static_cast<size_t>(training_state_.num_kv_heads) *
                                 static_cast<size_t>(max_seq_len_cache) *
                                 static_cast<size_t>(head_dim);

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        auto& cache = training_state_.encoder_layer_caches[layer];
        
        cache.ln1_output = Tensor::zeros(TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model), false, primary_stream);
        cache.attn_input = Tensor::zeros(TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model), false, primary_stream);
        cache.attn_bhsd = Tensor::zeros(TC::make_BSM(1, static_cast<int>(attn_bhsd_elems)), false, primary_stream);
        cache.softmax_lse = Tensor::zeros(TC::make_BSM(1, static_cast<int>(softmax_lse_elems)), false, primary_stream);
        cache.attn_output = Tensor::zeros(TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model), false, primary_stream);
        cache.residual1 = Tensor::zeros(TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model), false, primary_stream);
        cache.ln2_output = Tensor::zeros(TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model), false, primary_stream);
        cache.ffn_pre_gelu = Tensor::zeros(TC::make_BSM(static_cast<int>(max_tokens), cfg.d_ff), false, primary_stream);
        cache.ffn_output = Tensor::zeros(TC::make_BSM(static_cast<int>(max_tokens), cfg.d_ff), false, primary_stream);
        cache.layer_output = Tensor::zeros(TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model), false, primary_stream);
        
        cache.Q = Tensor::zeros(TC::make_BSM(1, static_cast<int>(attn_bhsd_elems)), false, primary_stream);
        cache.K = Tensor::zeros(TC::make_BSM(1, static_cast<int>(kv_bhsd_elems)), false, primary_stream);
        cache.V = Tensor::zeros(TC::make_BSM(1, static_cast<int>(kv_bhsd_elems)), false, primary_stream);
    }
    std::cout << "  ✓ Allocated per-layer activation caches using Tensor API (" << cfg.num_layers << " layers)" << std::endl;

    // Set up forward_layer_caches to point into Tensor data for CUDA kernel compatibility
    // Note: GRIM::EncoderLayerCache is defined in grim_language_model_cuda.hpp (inside namespace GRIM)
    if (!training_state_.forward_layer_caches) {
        training_state_.forward_layer_cache_count = cfg.num_layers;
        training_state_.forward_layer_caches = new GRIM::EncoderLayerCache[cfg.num_layers]();
    }
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& tc = training_state_.encoder_layer_caches[layer];
        GRIM::EncoderLayerCache& cache = training_state_.forward_layer_caches[layer];
        cache.ln1_output = tc.ln1_output.data;
        cache.attn_input = tc.attn_input.data;
        cache.attn_bhsd = tc.attn_bhsd.data;
        cache.softmax_lse = tc.softmax_lse.data;
        cache.attn_output = tc.attn_output.data;
        cache.residual1 = tc.residual1.data;
        cache.ln2_input = tc.residual1.data;  // Same as residual1 (pre-LN2)
        cache.ln2_output = tc.ln2_output.data;
        cache.ffn_input = tc.ln2_output.data;  // Same as ln2_output (post-LN2)
        cache.ffn_pre_gelu = tc.ffn_pre_gelu.data;
        cache.ffn_output = tc.ffn_output.data;
        cache.layer_output = tc.layer_output.data;
        cache.q = tc.Q.data;
        cache.k = tc.K.data;
        cache.v = tc.V.data;
    }

    // Flash Attention BF16 buffers - use Tensor API
    // Note: Tensor manages bf16 via raw allocation + element count tracking
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

    // Allocate BF16 Tensors - use bf16 allocation helper
    // Note: Using Tensor::zeros for float, then bf16 will be allocated separately below
    // For now, allocate via cudaMalloc to BF16 Tensor's data pointer
    auto alloc_bf16_tensor = [&](Tensor& tensor, const char* label, size_t elems) -> bool {
        __nv_bfloat16* ptr = nullptr;
        cudaError_t berr = cudaMalloc(&ptr, elems * sizeof(__nv_bfloat16));
        if (berr != cudaSuccess) {
            std::cerr << "[InitInferenceState] Failed to allocate " << label << ": "
                      << cudaGetErrorString(berr) << std::endl;
            return false;
        }
        // Create tensor wrapper around bf16 allocation
        tensor.data = reinterpret_cast<float*>(ptr);  // Store bf16 ptr cast to float*
        tensor.shape = TC::make_BSM(1, static_cast<int>(elems));
        tensor.owns_data = true;
        tensor.requires_grad = false;
        return true;
    };

    if (!alloc_bf16_tensor(training_state_.fa_q_bf16, "fa_q_bf16", fa_q_elems)) return;
    if (!alloc_bf16_tensor(training_state_.fa_k_bf16, "fa_k_bf16", fa_kv_elems)) return;
    if (!alloc_bf16_tensor(training_state_.fa_v_bf16, "fa_v_bf16", fa_kv_elems)) return;
    if (!alloc_bf16_tensor(training_state_.fa_out_bf16, "fa_out_bf16", fa_q_elems)) return;
    if (!alloc_bf16_tensor(training_state_.fa_dout_bf16, "fa_dout_bf16", fa_q_elems)) return;
    if (!alloc_bf16_tensor(training_state_.fa_dq_bf16, "fa_dq_bf16", fa_q_elems)) return;
    if (!alloc_bf16_tensor(training_state_.fa_dk_bf16, "fa_dk_bf16", fa_kv_elems)) return;
    if (!alloc_bf16_tensor(training_state_.fa_dv_bf16, "fa_dv_bf16", fa_kv_elems)) return;
    std::cout << "  ✓ Allocated Flash Attention BF16 buffers (Tensor API)" << std::endl;
    
    // Encoder outputs cache - use Tensor API
    training_state_.cached_encoder_output = Tensor::zeros(
        TC::make_BSM(static_cast<int>(max_tokens), cfg.d_model),
        false,  // no grad for inference
        primary_stream
    );
    std::cout << "  ✓ Allocated encoder output cache (Tensor API)" << std::endl;
    
    // Logits cache - use Tensor API
    training_state_.cached_logits_tensor = Tensor::zeros(
        TC::make_BSM(static_cast<int>(max_tokens), cfg.vocab_size),
        false,  // no grad for inference
        primary_stream
    );
    std::cout << "  ✓ Allocated logits cache (Tensor API)" << std::endl;

    if (cfg.numeric_head_enabled) {
        training_state_.cached_numeric_predictions = Tensor::zeros(
            TC::make_BSM(1, static_cast<int>(max_tokens)),
            false,  // no grad for inference
            primary_stream
        );
        std::cout << "  ✓ Allocated numeric predictions cache (Tensor API)" << std::endl;
    }
    
    // 4. Allocate single-token buffers for incremental generation (KV cache)
    training_state_.single_token_embedding = Tensor::zeros(
        TC::make_BSM(1, cfg.d_model),
        false,  // no grad for inference
        primary_stream
    );
    
    training_state_.single_token_hidden = Tensor::zeros(
        TC::make_BSM(1, cfg.d_model),
        false,  // no grad for inference
        primary_stream
    );
    
    training_state_.single_token_logits = Tensor::zeros(
        TC::make_BSM(1, cfg.vocab_size),
        false,  // no grad for inference
        primary_stream
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
        primary_stream
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
                primary_stream
            );
            
            training_state_.cached_scratch_block_positions = Tensor::zeros(
                TC::make_BSM(1, static_cast<int>(max_tokens)),
                false,  // no grad for inference
                primary_stream
            );
            
            training_state_.cached_scratch_block_num_atoms = Tensor::zeros(
                TC::make_BSM(1, static_cast<int>(max_batch_size)),
                false,  // no grad for inference
                primary_stream
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

#define USE_CUDA

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Layers/ScratchBlock/ScratchBlock_GPU.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Shared/StreamController/StreamController_GPU.hpp"
#include "../Shared/Activations/Xavier/Xavier.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"

using GRIM::launchXavierInit;

namespace GRIM {

#ifdef USE_CUDA

//======================================================//
//  Early GPU Resource Initialization
//======================================================//
void LanguageModel::initCuBLASHandle() {
    // Initialize cuBLAS handle that encoder layers need during initGPU()
    // MUST be called after StreamController is initialized
    
    if (!training_state_.stream_ctrl.isInitialized()) {
        std::cerr << "FATAL: StreamController must be initialized before creating cuBLAS handle" << std::endl;
        throw std::runtime_error("StreamController not initialized");
    }
    
    if (training_state_.cublas_handle != nullptr) {
        std::cout << "✓ cuBLAS handle already initialized" << std::endl;
        return;  // Already created
    }
    
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    StreamController::fatalIfDefaultStream(primary_stream, "LanguageModel::initCuBLASHandle");

    cublasStatus_t cublas_err = cublasCreate(&training_state_.cublas_handle);
    if (cublas_err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Failed to create cuBLAS handle: " << cublas_err << std::endl;
        throw std::runtime_error("cuBLAS handle creation failed");
    }
    
    // Enable Tensor Core acceleration for Ampere+ GPUs (RTX 30xx, 40xx)
    cublasSetMathMode(training_state_.cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSetStream(training_state_.cublas_handle, primary_stream);
    std::cout << "✓ cuBLAS handle bound to Primary stream with Tensor Core acceleration" << std::endl;
}

//======================================================//
//  Unified PBM (ALiBi+RoPE Hybrid) Initialization
//  Call this BEFORE initGPU() to ensure encoder gets both position encodings
//======================================================//
void LanguageModel::initPBM() {
    if (training_state_.pbm_initialized) {
        std::cout << "✓ PBM (ALiBi+RoPE) already initialized" << std::endl;
        return;
    }
    
    if (!training_state_.stream_ctrl.isInitialized()) {
        std::cerr << "FATAL: StreamController must be initialized before PBM" << std::endl;
        throw std::runtime_error("StreamController not initialized");
    }
    
    const auto& cfg = getConfig();
    const int head_dim = cfg.d_model / cfg.num_heads;
    
    PBM::PBMConfig pbm_config{};  // Uses HyperParameters defaults
    pbm_config.num_heads = cfg.num_heads;
    pbm_config.num_kv_heads = cfg.num_kv_heads;
    // alibi_slope_exponent uses default from HyperParameters::ALIBI_SLOPE_EXPONENT
    pbm_config.head_dim = head_dim;
    pbm_config.rotary_dim = head_dim;  // Full rotation
    // rope_theta/rope_scaling use defaults from HyperParameters
    pbm_config.verbose = true;
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    StreamController::fatalIfDefaultStream(primary_stream, "LanguageModel::initPBM");
    pbm_config.stream = primary_stream;
    
    if (!PBM::initializePBM(pbm_config, training_state_.pbm_state)) {
        std::cerr << "FATAL: Failed to initialize PBM (ALiBi+RoPE)" << std::endl;
        throw std::runtime_error("PBM initialization failed");
    }
    
    // Build the spec that encoder layers will use
    training_state_.pbm_spec = PBM::getPBMSpec(training_state_.pbm_state);
    
    training_state_.pbm_initialized = true;
    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) initialized:" << std::endl;
    std::cout << "    ALiBi: " << cfg.num_heads << " heads with slopes" << std::endl;
    std::cout << "    RoPE:  head_dim=" << head_dim 
              << ", rotary_dim=" << training_state_.pbm_spec.rotary_dim
              << ", theta=10000" << std::endl;
}

void LanguageModel::initTrainingState() {
    if (training_state_.initialized) return;
    
    const auto& cfg = getConfig();
    std::cout << "[DEBUG initTrainingState] Entry point" << std::endl;

    
    // ═══════════════════════════════════════════════════════════════════════
    //  STEP 1: Initialize StreamController if not already done
    //  (May be pre-initialized by Phase1_Startup before initGPU)
    // ═══════════════════════════════════════════════════════════════════════
    std::cout << "[DEBUG initTrainingState] Checking StreamController..." << std::endl;
    if (!training_state_.stream_ctrl.isInitialized()) {
        StreamControllerConfig stream_config;
        stream_config.verbose = true;
        stream_config.create_transfer_stream = true;   // For async H2D/D2H
        stream_config.create_auxiliary_stream = false; // On-demand if needed
        
        if (!training_state_.stream_ctrl.initialize(stream_config)) {
            std::cerr << "FATAL: Failed to initialize StreamController" << std::endl;
            return;
        }
        std::cout << "✓ StreamController initialized (Primary + Transfer streams)" << std::endl;
    } else {
        std::cout << "✓ StreamController already initialized" << std::endl;
    }
    StreamController::fatalIfDefaultStream(training_state_.stream_ctrl.getPrimaryStream(),
                                           "LanguageModel::initTrainingState");
    
    // ═══════════════════════════════════════════════════════════════════════
    //  STEP 2: cuBLAS handle (should already be initialized by initCuBLASHandle)
    // ═══════════════════════════════════════════════════════════════════════
    if (training_state_.cublas_handle == nullptr) {
        std::cerr << "FATAL: cuBLAS handle not initialized. Call initCuBLASHandle() first!" << std::endl;
        throw std::runtime_error("cuBLAS handle not initialized");
    }
    std::cout << "✓ Using pre-initialized cuBLAS handle" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════════════
    //  STEP 3: Initialize PBM (Unified ALiBi+RoPE Hybrid)
    //  CRITICAL: Without positional encoding, attention has no position info!
    //  - RoPE: Rotary Position Embedding rotates Q,K to encode position
    //  - ALiBi: Attention-Linear-Biases adds position-dependent bias to scores
    //  Missing PBM causes position-blind attention → training plateau!
    // ═══════════════════════════════════════════════════════════════════════
    if (!training_state_.pbm_initialized) {
        const int head_dim = cfg.d_model / cfg.num_heads;
        
        PBM::PBMConfig pbm_config{};  // Uses HyperParameters defaults
        pbm_config.num_heads = cfg.num_heads;
        pbm_config.num_kv_heads = cfg.num_kv_heads;
        // alibi_slope_exponent uses default from HyperParameters::ALIBI_SLOPE_EXPONENT
        pbm_config.head_dim = head_dim;
        pbm_config.rotary_dim = head_dim;  // Full rotation
        // rope_theta/rope_scaling use defaults from HyperParameters
        pbm_config.verbose = true;
        pbm_config.stream = training_state_.stream_ctrl.getPrimaryStream();
        
        if (!PBM::initializePBM(pbm_config, training_state_.pbm_state)) {
            std::cerr << "FATAL: Failed to initialize PBM (ALiBi+RoPE)" << std::endl;
            throw std::runtime_error("PBM initialization failed");
        }
        
        // Build the spec that will be passed to encoder layers
        training_state_.pbm_spec = PBM::getPBMSpec(training_state_.pbm_state);
        
        training_state_.pbm_initialized = true;
        std::cout << "✓ PBM (Hybrid ALiBi+RoPE) initialized:" << std::endl;
        std::cout << "    ALiBi: " << cfg.num_heads << " heads with position-decaying slopes" << std::endl;
        std::cout << "    RoPE:  head_dim=" << head_dim 
                  << ", rotary_dim=" << training_state_.pbm_spec.rotary_dim
                  << ", theta=10000" << std::endl;
    }

    training_state_.cached_num_layers = cfg.num_layers;
    
    // WEIGHT TYING FIX: When tie_embeddings=true, both embedding and LM head share
    // the SAME grad buffer. This ensures:
    // 1. LM head backward writes dense grads (cublasSgemm with beta=0)
    // 2. Embedding backward accumulates sparse grads (atomicAdd)
    // 3. Single optimizer state (m, v) for tied weights
    // 4. Weight decay applied once, not twice
    
    training_state_.embedding_grad_size = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
    size_t lm_head_weight_size = training_state_.embedding_grad_size;
    
    // Always allocate ONE grad buffer for the tied embedding/LM head
    cudaError_t err = cudaMalloc(&training_state_.lm_head_weight_grads, lm_head_weight_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate LM head weight gradients: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    if (cfg.tie_embeddings) {
        // TIED: embedding_grads is an alias to lm_head_weight_grads
        training_state_.embedding_grads = training_state_.lm_head_weight_grads;
        std::cout << "🔗 Embedding grads tied to LM head grads at " << (void*)training_state_.embedding_grads << std::endl;
    } else {
        // UNTIED: separate grad buffers
        err = cudaMalloc(&training_state_.embedding_grads, training_state_.embedding_grad_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate embedding gradients: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        std::cout << "📦 Separate embedding grads at " << (void*)training_state_.embedding_grads << std::endl;
    }
    
    // CRITICAL: When tie_embeddings=true, LM head shares embedding buffer
    // NOTE: Embeddings are initialized DIRECTLY on GPU via launchXavierInit() to avoid CPU->GPU copy overhead.
    // The embedding buffer may already contain weights from checkpoint loading (via Serialization_GPU.cu),
    // but we reinitialize to ensure proper Xavier scaling for training stability.
    if (cfg.tie_embeddings) {
        auto* embedding_runtime = &getGpuEmbedder();
        if (embedding_runtime && embedding_runtime->token_buffer) {
            training_state_.lm_head_weights = embedding_runtime->token_buffer;
            training_state_.lm_head_weights_owned = false;  // Aliased to EmbeddingRuntime - don't free!
            std::cout << "🔗 LM head weights tied to embeddings at " << (void*)training_state_.lm_head_weights << std::endl;
            
            // Initialize embeddings directly on GPU with proper scaling
            // BUG FIX: Was using Xavier formula sqrt(2/(d_model+vocab_size)) = 0.00625 - WAY TOO SMALL!
            // Embeddings are lookup tables, not dense layers. Each row is independent.
            // PyTorch uses N(0, 1) by default, GPT-2 uses 0.02.
            // We use 0.1 for stronger initial signal - can adjust if training unstable.
            constexpr float embedding_stddev = 0.1f;  // Reasonable for embedding init
            std::cout << "🎲 Initializing embedding weights directly on GPU (stddev=" << embedding_stddev << ")" << std::endl;
            launchXavierInit(training_state_.lm_head_weights, lm_head_weight_size, embedding_stddev, 42, training_state_.stream_ctrl.getPrimaryStream());
            
            cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
            
            // Verify initialization quality
            std::vector<float> sample_weights(std::min<size_t>(100, lm_head_weight_size));
            cudaMemcpy(sample_weights.data(), training_state_.lm_head_weights,
                       sample_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
            float weight_sum_sq = 0.0f;
            float weight_max = 0.0f;
            for (float w : sample_weights) {
                weight_sum_sq += w * w;
                weight_max = std::max(weight_max, std::abs(w));
            }
            float weight_norm = std::sqrt(weight_sum_sq / sample_weights.size());
            std::cout << "✓ Embedding weights initialized on GPU: avg_norm=" << weight_norm
                      << ", max_abs=" << weight_max << " (expected ~" << embedding_stddev << ")" << std::endl;
        } else {
            std::cerr << "ERROR: tie_embeddings=true but embedding buffer not available!" << std::endl;
            return;
        }
    } else {
        // Untied embeddings: allocate separate LM head weights on GPU
        err = cudaMalloc(&training_state_.lm_head_weights, lm_head_weight_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate LM head weights: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        training_state_.lm_head_weights_owned = true;  // We allocated - must free!
        
        // Initialize LM head weights with proper scaling (not Xavier - embeddings are lookup tables)
        // BUG FIX: Was using sqrt(2/(d_model+vocab_size)) = 0.00625 - WAY TOO SMALL!
        constexpr float embedding_stddev = 0.1f;  // Reasonable for embedding init
        std::cout << "🎲 Initializing LM head weights directly on GPU (stddev=" << embedding_stddev << ")" << std::endl;
        launchXavierInit(training_state_.lm_head_weights, lm_head_weight_size, embedding_stddev, 42, training_state_.stream_ctrl.getPrimaryStream());
        
        cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
        
        // Verify initialization quality
        std::vector<float> sample_weights(std::min<size_t>(100, lm_head_weight_size));
        cudaMemcpy(sample_weights.data(), training_state_.lm_head_weights,
                   sample_weights.size() * sizeof(float), cudaMemcpyDeviceToHost);
        float weight_sum_sq = 0.0f;
        float weight_max = 0.0f;
        for (float w : sample_weights) {
            weight_sum_sq += w * w;
            weight_max = std::max(weight_max, std::abs(w));
        }
        float weight_norm = std::sqrt(weight_sum_sq / sample_weights.size());
        std::cout << "✓ LM head weights initialized on GPU: avg_norm=" << weight_norm
                  << ", max_abs=" << weight_max << " (expected ~" << embedding_stddev << ")" << std::endl;
    }
    
    if (cfg.use_bias) {
        err = cudaMalloc(&training_state_.lm_head_bias_grads, cfg.vocab_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate LM head bias gradients: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        err = cudaMalloc(&training_state_.lm_head_bias, cfg.vocab_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate LM head bias: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        // Use centralized stream per Rule 22
        cudaMemsetAsync(training_state_.lm_head_bias, 0, cfg.vocab_size * sizeof(float),
                        training_state_.stream_ctrl.getPrimaryStream());
    }

    if (cfg.numeric_head_enabled) {
        err = cudaMalloc(&training_state_.numeric_head_weight_grads, cfg.d_model * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate numeric head weight gradients: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        err = cudaMalloc(&training_state_.numeric_head_weights, cfg.d_model * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate numeric head weights: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        const float xavier_stddev = std::sqrt(2.0f / (cfg.d_model + 1.0f));
        std::cout << "🎲 Initializing numeric head weights on GPU with Xavier (stddev="
                  << xavier_stddev << ")" << std::endl;
        launchXavierInit(training_state_.numeric_head_weights,
                         cfg.d_model,
                         xavier_stddev,
                         1337,
                         training_state_.stream_ctrl.getPrimaryStream());

        if (cfg.use_bias) {
            err = cudaMalloc(&training_state_.numeric_head_bias_grads, sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate numeric head bias gradients: " << cudaGetErrorString(err) << std::endl;
                return;
            }
            err = cudaMalloc(&training_state_.numeric_head_bias, sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate numeric head bias: " << cudaGetErrorString(err) << std::endl;
                return;
            }
            cudaMemsetAsync(training_state_.numeric_head_bias, 0, sizeof(float),
                            training_state_.stream_ctrl.getPrimaryStream());
        }
    }
    
    training_state_.rms1_gamma_grads.resize(cfg.num_layers, nullptr);
    training_state_.rms2_gamma_grads.resize(cfg.num_layers, nullptr);
    training_state_.attn_qkv_weight_grads.resize(cfg.num_layers, nullptr);
    training_state_.attn_qkv_bias_grads.resize(cfg.num_layers, nullptr);
    training_state_.attn_out_weight_grads.resize(cfg.num_layers, nullptr);
    training_state_.attn_out_bias_grads.resize(cfg.num_layers, nullptr);
    training_state_.ffn_w1_grads.resize(cfg.num_layers, nullptr);
    training_state_.ffn_b1_grads.resize(cfg.num_layers, nullptr);
    training_state_.ffn_w2_grads.resize(cfg.num_layers, nullptr);
    training_state_.ffn_b2_grads.resize(cfg.num_layers, nullptr);
    
    // Learnable QK-norm scales (nGPT-style)
    training_state_.attn_alpha_q.resize(cfg.num_layers, nullptr);
    training_state_.attn_alpha_k.resize(cfg.num_layers, nullptr);
    training_state_.attn_alpha_q_grads.resize(cfg.num_layers, nullptr);
    training_state_.attn_alpha_k_grads.resize(cfg.num_layers, nullptr);
    
    // GQA configuration: determine number of KV heads
    // Use HyperParameters default if config doesn't specify, or equal to num_heads for MHA
    const int num_kv_heads = HyperParameters::GQA_ENABLED ? 
                             HyperParameters::DEFAULT_NUM_KV_HEADS : cfg.num_heads;
    
    // Validate GQA configuration
    if (!HyperParameters::isValidGQAConfig(cfg.num_heads, num_kv_heads)) {
        std::cerr << "ERROR: Invalid GQA config: num_heads=" << cfg.num_heads 
                  << " num_kv_heads=" << num_kv_heads << std::endl;
        return;
    }
    
    // Store GQA config in training state
    training_state_.num_heads = cfg.num_heads;
    training_state_.num_kv_heads = num_kv_heads;
    
    std::cout << "🔧 GQA Configuration: num_heads=" << cfg.num_heads 
              << " num_kv_heads=" << num_kv_heads 
              << " (heads_per_kv_group=" << (cfg.num_heads / num_kv_heads) << ")" << std::endl;
    
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        cudaMalloc(&training_state_.rms1_gamma_grads[layer], cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.rms2_gamma_grads[layer], cfg.d_model * sizeof(float));
        
        // GQA: QKV weight size changes
        // Q projection: [d_model, d_model] 
        // K projection: [kv_dim, d_model] where kv_dim = num_kv_heads * head_dim
        // V projection: [kv_dim, d_model]
        const int head_dim = cfg.d_model / cfg.num_heads;
        const int kv_dim = num_kv_heads * head_dim;
        const size_t q_weight_size = cfg.d_model * cfg.d_model;
        const size_t k_weight_size = kv_dim * cfg.d_model;
        const size_t v_weight_size = kv_dim * cfg.d_model;
        const size_t qkv_weight_size = q_weight_size + k_weight_size + v_weight_size;
        const size_t qkv_bias_size = cfg.d_model + kv_dim + kv_dim;
        
        cudaMalloc(&training_state_.attn_qkv_weight_grads[layer], qkv_weight_size * sizeof(float));
        cudaMalloc(&training_state_.attn_qkv_bias_grads[layer], qkv_bias_size * sizeof(float));
        cudaMalloc(&training_state_.attn_out_weight_grads[layer], cfg.d_model * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.attn_out_bias_grads[layer], cfg.d_model * sizeof(float));
        
        cudaMalloc(&training_state_.ffn_w1_grads[layer], cfg.d_model * cfg.d_ff * sizeof(float));
        cudaMalloc(&training_state_.ffn_b1_grads[layer], cfg.d_ff * sizeof(float));
        cudaMalloc(&training_state_.ffn_w2_grads[layer], cfg.d_ff * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.ffn_b2_grads[layer], cfg.d_model * sizeof(float));
        
        // Allocate and initialize learnable QK-norm scales
        // GQA: alpha_q has num_heads entries, alpha_k has num_kv_heads entries
        cudaMalloc(&training_state_.attn_alpha_q[layer], cfg.num_heads * sizeof(float));
        cudaMalloc(&training_state_.attn_alpha_k[layer], num_kv_heads * sizeof(float));
        cudaMalloc(&training_state_.attn_alpha_q_grads[layer], cfg.num_heads * sizeof(float));
        cudaMalloc(&training_state_.attn_alpha_k_grads[layer], num_kv_heads * sizeof(float));
        
        // Initialize alpha_q to 1.0 (num_heads entries)
        std::vector<float> alpha_q_init(cfg.num_heads, HyperParameters::QK_NORM_ALPHA_INIT);
        cudaMemcpy(training_state_.attn_alpha_q[layer], alpha_q_init.data(), 
                   cfg.num_heads * sizeof(float), cudaMemcpyHostToDevice);
        
        // Initialize alpha_k to 1.0 (num_kv_heads entries for GQA)
        std::vector<float> alpha_k_init(num_kv_heads, HyperParameters::QK_NORM_ALPHA_INIT);
        cudaMemcpy(training_state_.attn_alpha_k[layer], alpha_k_init.data(), 
                   num_kv_heads * sizeof(float), cudaMemcpyHostToDevice);
    }

    // NOTE: Encoder layer weight initialization is handled in TrainingOps.cu::initGPU()
    // with proper GQA-aware dimensions and GPT-2 residual scaling.
    // DO NOT duplicate Xavier init here per Rule 20 (no backwards compatibility shims).
    
    const size_t max_batch_size = static_cast<size_t>(std::max(1, cfg.max_cached_batch));
    const size_t max_seq_len_cache = static_cast<size_t>(std::max(1, std::min(cfg.max_seq_len, cfg.max_cached_seq_len)));
    size_t max_tokens = max_batch_size * max_seq_len_cache;
    const size_t max_logit_tokens = (cfg.max_tokens_per_batch > 0)
        ? std::min(max_tokens, static_cast<size_t>(cfg.max_tokens_per_batch))
        : max_tokens;

    training_state_.max_cached_batch = static_cast<int>(max_batch_size);
    training_state_.max_cached_seq_len = static_cast<int>(max_seq_len_cache);
    training_state_.max_cached_tokens = max_tokens;
    training_state_.max_logit_tokens = max_logit_tokens;
    
    // BUG FIX: Set kv_cache_capacity for inference sampling during training
    // Previously missing - caused forwardInit() to fail with capacity=0
    training_state_.kv_cache_capacity = static_cast<int>(max_seq_len_cache);
    training_state_.kv_cache_len = 0;  // Start with empty cache
    
    // BUG FIX: Allocate single-token inference buffers for training-time sampling
    // These are required by forwardInit() for incremental generation during training
    // Rule 20: Fail loud if allocation fails - no silent fallbacks
    err = cudaMalloc(&training_state_.single_token_hidden, cfg.d_model * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate single_token_hidden: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMalloc(&training_state_.single_token_logits, cfg.vocab_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate single_token_logits: " + std::string(cudaGetErrorString(err)));
    }
    std::cout << "✓ Allocated single-token inference buffers for training sampling" << std::endl;
    
    std::cout << "📊 Allocating activation caches for max_tokens=" << max_tokens
              << " (batch=" << max_batch_size << ", seq_len=" << max_seq_len_cache << ")" << std::endl;
    
    err = cudaMalloc(&training_state_.cached_embeddings, max_tokens * cfg.d_model * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate embedding cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
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
    
    // GQA: K and V caches use num_kv_heads, Q uses num_heads
    const int head_dim_cache = cfg.d_model / cfg.num_heads;
    const int kv_dim_cache = training_state_.num_kv_heads * head_dim_cache;
    
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
        
        // GQA: Q has full d_model (num_heads * head_dim)
        // K and V have reduced dimension (num_kv_heads * head_dim)
        cudaMalloc(&training_state_.cached_Q[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_K[layer], max_tokens * kv_dim_cache * sizeof(float));
        cudaMalloc(&training_state_.cached_V[layer], max_tokens * kv_dim_cache * sizeof(float));
        cudaMalloc(&training_state_.cached_attn_inputs[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_attn_bhsd[layer], max_tokens * cfg.d_model * sizeof(float));
        cudaMalloc(&training_state_.cached_softmax_lse[layer], softmax_lse_elems * sizeof(float));
    }

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
    
    err = cudaMalloc(&training_state_.cached_encoder_outputs, max_tokens * cfg.d_model * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate encoder output cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&training_state_.cached_logits, max_logit_tokens * cfg.vocab_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate logits cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    if (cfg.numeric_head_enabled) {
        err = cudaMalloc(&training_state_.cached_numeric_predictions, max_logit_tokens * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate numeric prediction cache: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
    
    err = cudaMalloc(&training_state_.cached_targets, max_logit_tokens * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate targets cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // BUG FIX: Token IDs cache must be sized by max_tokens (full cache capacity)
    // not max_logit_tokens. Inference sampling requires full buffer.
    err = cudaMalloc(&training_state_.cached_token_ids, max_tokens * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate token IDs cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // BUG FIX: Numeric buffers must be sized by max_tokens (full cache capacity)
    // not max_logit_tokens (training optimization). Inference sampling requires
    // the full buffer for sequences up to max_cached_seq_len.
    // BUG FIX: Always allocate numeric/text buffers even when ScratchBlock is disabled
    // because prepareLossBatchInputs() always populates these fields from tokenizer
    err = cudaMalloc(&training_state_.cached_token_numeric_values, max_tokens * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate token numeric value cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&training_state_.cached_token_numeric_mask, max_tokens * sizeof(uint8_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate token numeric mask cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // GRMT v4: Allocate text feature buffers
    constexpr int kTextFeatureDim = 16;  // Must match GRIM::Tokenizer::kTextFeatureDim
    err = cudaMalloc(&training_state_.cached_token_text_features, max_tokens * kTextFeatureDim * sizeof(uint16_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate token text feature cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMalloc(&training_state_.cached_token_text_mask, max_tokens * sizeof(uint8_t));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate token text mask cache: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    std::cout << "✓ Allocated numeric/text feature buffers (" 
              << (max_tokens * (sizeof(float) + sizeof(uint8_t) + kTextFeatureDim * sizeof(uint16_t) + sizeof(uint8_t)) / 1024 / 1024) 
              << " MB)" << std::endl;

    // Allocate entropy output buffer (per-layer, per-batch, per-head)
    // Size: num_layers * max_batch_size * num_heads
    const size_t entropy_size = cfg.num_layers * max_batch_size * cfg.num_heads;
    training_state_.entropy_output_capacity = entropy_size;
    err = cudaMalloc(&training_state_.d_entropy_output, entropy_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate entropy output buffer: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "📊 Allocated entropy output buffer: " << entropy_size 
              << " floats (" << (entropy_size * sizeof(float) / 1024.0 / 1024.0) << " MB)" << std::endl;

    err = cudaMalloc(&training_state_.sequence_weights, max_batch_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate sequence weight buffer: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    // Use centralized stream per Rule 22
    cudaMemsetAsync(training_state_.sequence_weights, 0, max_batch_size * sizeof(float),
                    training_state_.stream_ctrl.getPrimaryStream());
    training_state_.sequence_weight_capacity = static_cast<int>(max_batch_size);
    training_state_.sequence_weight_count = 0;
    
    err = cudaMalloc(&training_state_.grad_logits, max_logit_tokens * cfg.vocab_size * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate logits gradient buffer: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    if (cfg.numeric_head_enabled) {
        err = cudaMalloc(&training_state_.grad_numeric_predictions, max_logit_tokens * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate numeric prediction gradient buffer: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
    
    err = cudaMalloc(&training_state_.grad_encoder_out, max_tokens * cfg.d_model * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate encoder gradient buffer: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    const size_t tokens_per_batch = max_batch_size * max_seq_len_cache;
    EncodingConfig enc_cfg{};
    enc_cfg.d_model = cfg.d_model;
    enc_cfg.num_heads = cfg.num_heads;
    enc_cfg.num_kv_heads = training_state_.num_kv_heads;  // Use calculated GQA value from training_state
    enc_cfg.d_ff = cfg.d_ff;  // Use actual d_ff from config
    enc_cfg.rms_epsilon = 1e-5f;
    enc_cfg.causal_mask = cfg.causal_mask;
    enc_cfg.use_flash_attention = false;
    enc_cfg.stream = training_state_.stream_ctrl.getPrimaryStream();
    enc_cfg.cublas_handle = training_state_.cublas_handle;  // Rule 22: centralized handle
    EncodingLayer enc_layer(enc_cfg);
    // CRITICAL: requiredWorkspaceBytes computes batch_size = total_tokens / seq_len
    // We must use max_seq_len_cache (the actual max seq len we'll process), NOT cfg.max_seq_len (8192)
    // Otherwise batch_size = tokens_per_batch / 8192 = 0, causing undersized workspace allocation!
    const int seq_len_for_workspace = static_cast<int>(max_seq_len_cache);
    size_t workspace_bytes = enc_layer.requiredWorkspaceBytes(static_cast<int>(tokens_per_batch), seq_len_for_workspace);
    
    std::cout << "📊 Encoder workspace: tokens_per_batch=" << tokens_per_batch 
              << " seq_len=" << seq_len_for_workspace
              << " batch_size=" << (tokens_per_batch / seq_len_for_workspace)
              << " workspace=" << (workspace_bytes / (1024*1024)) << "MB" << std::endl;

    err = cudaMalloc(&training_state_.encoder_workspace, workspace_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate encoder workspace: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    training_state_.encoder_workspace_size = workspace_bytes;

    // Allocate reusable backward temporaries (avoid per-layer cudaMalloc/free)
    const int head_dim = cfg.d_model / cfg.num_heads;
    const size_t qkv_elems = max_tokens * static_cast<size_t>(cfg.num_heads) * static_cast<size_t>(head_dim); // = max_tokens * d_model
    const size_t model_elems = max_tokens * static_cast<size_t>(cfg.d_model);
    const size_t ffn_elems = max_tokens * static_cast<size_t>(cfg.d_ff);
    const size_t qkv_concat_elems = max_tokens * static_cast<size_t>(3 * cfg.d_model);

    auto alloc_or_fail = [&](float** ptr, size_t elems, const char* label) -> bool {
        cudaError_t aerr = cudaMalloc(ptr, elems * sizeof(float));
        if (aerr != cudaSuccess) {
            std::cerr << "Failed to allocate " << label << ": " << cudaGetErrorString(aerr) << std::endl;
            return false;
        }
        return true;
    };

    if (!alloc_or_fail(&training_state_.grad_ffn_input, model_elems, "grad_ffn_input")) return;
    if (!alloc_or_fail(&training_state_.grad_ffn_hidden, ffn_elems, "grad_ffn_hidden")) return;
    if (!alloc_or_fail(&training_state_.grad_attn_input, model_elems, "grad_attn_input")) return;
    if (!alloc_or_fail(&training_state_.grad_attn_out_before_proj, model_elems, "grad_attn_out_before_proj")) return;
    if (!alloc_or_fail(&training_state_.grad_attn_out_reshaped, qkv_elems, "grad_attn_out_reshaped")) return;
    if (!alloc_or_fail(&training_state_.grad_q, qkv_elems, "grad_Q")) return;
    if (!alloc_or_fail(&training_state_.grad_k, qkv_elems, "grad_K")) return;
    if (!alloc_or_fail(&training_state_.grad_v, qkv_elems, "grad_V")) return;
    if (!alloc_or_fail(&training_state_.grad_qkv_concat, qkv_concat_elems, "grad_qkv_concat")) return;
    if (!alloc_or_fail(&training_state_.grad_qkv_input, model_elems, "grad_qkv_input")) return;
    // DEDICATED scratch buffer for attention output BSM conversion (W_o gradient computation)
    // CRITICAL: Do NOT reuse grad_qkv_input - prevents temporal aliasing bugs
    if (!alloc_or_fail(&training_state_.grad_attn_bsm_scratch, model_elems, "grad_attn_bsm_scratch")) return;

    training_state_.fa_dq_accum_bytes = flash_attn_dq_accum_bytes(
        static_cast<int>(max_batch_size),
        static_cast<int>(max_seq_len_cache),
        cfg.num_heads,
        head_dim);
    training_state_.fa_dsoftmax_sum_bytes = flash_attn_dsoftmax_sum_bytes(
        static_cast<int>(max_batch_size),
        static_cast<int>(max_seq_len_cache),
        cfg.num_heads);

    if (training_state_.fa_dq_accum_bytes > 0) {
        err = cudaMalloc(&training_state_.fa_dq_accum, training_state_.fa_dq_accum_bytes);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate fa_dq_accum: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
    if (training_state_.fa_dsoftmax_sum_bytes > 0) {
        err = cudaMalloc(&training_state_.fa_dsoftmax_sum, training_state_.fa_dsoftmax_sum_bytes);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate fa_dsoftmax_sum: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }

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

    auto alloc_bf16 = [&](__nv_bfloat16** ptr, size_t elems, const char* label) -> bool {
        cudaError_t berr = cudaMalloc(ptr, elems * sizeof(__nv_bfloat16));
        if (berr != cudaSuccess) {
            std::cerr << "Failed to allocate " << label << ": " << cudaGetErrorString(berr) << std::endl;
            return false;
        }
        return true;
    };

    if (!alloc_bf16(&training_state_.fa_q_bf16, fa_q_elems, "fa_q_bf16")) return;
    if (!alloc_bf16(&training_state_.fa_k_bf16, fa_kv_elems, "fa_k_bf16")) return;
    if (!alloc_bf16(&training_state_.fa_v_bf16, fa_kv_elems, "fa_v_bf16")) return;
    if (!alloc_bf16(&training_state_.fa_out_bf16, fa_q_elems, "fa_out_bf16")) return;
    if (!alloc_bf16(&training_state_.fa_dout_bf16, fa_q_elems, "fa_dout_bf16")) return;
    if (!alloc_bf16(&training_state_.fa_dq_bf16, fa_q_elems, "fa_dq_bf16")) return;
    if (!alloc_bf16(&training_state_.fa_dk_bf16, fa_kv_elems, "fa_dk_bf16")) return;
    if (!alloc_bf16(&training_state_.fa_dv_bf16, fa_kv_elems, "fa_dv_bf16")) return;
    
    err = cudaMallocAsync(&training_state_.d_loss_scratch,
                          max_logit_tokens * sizeof(float),
                          training_state_.stream_ctrl.getPrimaryStream());
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate loss scratch buffer: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaMallocAsync(&training_state_.d_loss_sum_scratch,
                          sizeof(float),
                          training_state_.stream_ctrl.getPrimaryStream());
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate loss sum scratch buffer: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    training_state_.loss_scratch_capacity = max_logit_tokens;

    if (cfg.numeric_head_enabled) {
        err = cudaMallocAsync(&training_state_.d_numeric_loss_sum,
                              sizeof(float),
                              training_state_.stream_ctrl.getPrimaryStream());
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate numeric loss sum buffer: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        err = cudaMallocAsync(&training_state_.d_numeric_loss_count,
                              sizeof(int),
                              training_state_.stream_ctrl.getPrimaryStream());
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate numeric loss count buffer: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
    
    // Initialize scratch block pool for pinned memory transfers (if enabled in config)
    // Load configuration from ai_config.json
    training_state_.scratch_enabled = true;  // Default enabled
    size_t scratch_max_tokens = 16384;       // Default
    size_t scratch_num_blocks = 2;           // Default (double buffer)
    bool scratch_write_combined = false;     // Default
    
    // Note: Config loading happens in train_gpu.cu, so we just use defaults here
    // The training loop can update training_state_.scratch_enabled at runtime
    
    if (training_state_.scratch_enabled) {
        ScratchBlock::ScratchBlockConfig scratch_config;
        scratch_config.enabled = true;
        scratch_config.max_tokens_per_block = scratch_max_tokens;
        scratch_config.num_blocks = scratch_num_blocks;
        scratch_config.use_write_combined = scratch_write_combined;
        
        training_state_.scratch_pool = new ScratchBlock::ScratchBlockPool(scratch_config);
        
        if (training_state_.scratch_pool && training_state_.scratch_pool->isInitialized()) {
            size_t total_bytes = training_state_.scratch_pool->getTotalPinnedMemoryBytes();
            double mb = total_bytes / (1024.0 * 1024.0);
            std::cout << "✓ Scratch block pool initialized ("
                      << scratch_num_blocks << " blocks × "
                      << scratch_max_tokens << " tokens = "
                      << std::fixed << std::setprecision(2) << mb
                      << " MB pinned memory)" << std::endl;
        } else {
            std::cerr << "FATAL: Scratch block pool initialization failed" << std::endl;
            delete training_state_.scratch_pool;
            training_state_.scratch_pool = nullptr;
            throw std::runtime_error("InitTrainingState: Scratch block pool initialization failed");
        }
    } else {
        std::cout << "ℹ Scratch blocks disabled (using pageable memory)" << std::endl;
        training_state_.scratch_pool = nullptr;
    }
    
    // Initialize ScratchBlock reasoning layer
    if (cfg.use_scratch_block) {
        std::cout << "🧠 Initializing ScratchBlock reasoning layer..." << std::endl;
        
        ScratchBlockConfig sb_config;
        sb_config.d_model = cfg.d_model;
        sb_config.max_atoms = cfg.scratch_block_max_atoms;
        sb_config.atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
        sb_config.enabled = true;
        sb_config.inject_atom_embeddings = true;
        sb_config.atom_scale = cfg.scratch_block_atom_scale;
        sb_config.atom_token_start = GRIM::Tokenizer::ATOM_TOKEN_OFFSET;
        sb_config.atom_token_end = GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET;
        sb_config.stream = training_state_.stream_ctrl.getPrimaryStream();
        
        scratch_block_layer_ = std::make_unique<ScratchBlockLayer>(sb_config);
        
        // Allocate activation caches for ScratchBlock forward/backward pass
        // These store intermediate values (atom positions, types, etc.) - NOT gradients
        const size_t atom_emb_size = cfg.scratch_block_max_atoms * cfg.scratch_block_atom_embedding_dim;
        cudaMalloc(&training_state_.cached_scratch_block_embeddings, atom_emb_size * sizeof(float));
        cudaMalloc(&training_state_.cached_scratch_block_positions, cfg.scratch_block_max_atoms * sizeof(int));
        cudaMalloc(&training_state_.cached_scratch_block_types, cfg.scratch_block_max_atoms * sizeof(int));
        cudaMalloc(&training_state_.cached_scratch_block_num_atoms, sizeof(int));
        
        std::cout << "✓ ScratchBlock reasoning layer initialized (d_model="
                  << cfg.d_model << ", atom_dim=" << cfg.scratch_block_atom_embedding_dim
                  << ", max_atoms=" << cfg.scratch_block_max_atoms << ")" << std::endl;
    } else {
        std::cout << "ℹ ScratchBlock reasoning layer disabled" << std::endl;
        scratch_block_layer_ = nullptr;
    }
    
    training_state_.initialized = true;
    std::cout << "✓ Training state initialized with full gradient buffers" << std::endl;
    std::cout << "[InitTrainingState] max_cached_batch=" << training_state_.max_cached_batch
              << " max_cached_seq_len=" << training_state_.max_cached_seq_len
              << " max_cached_tokens=" << training_state_.max_cached_tokens
              << " max_logit_tokens=" << training_state_.max_logit_tokens << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM

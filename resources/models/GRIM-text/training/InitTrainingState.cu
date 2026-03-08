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
// NOTE: Encoding_GPU.hpp include DELETED — was only needed for requiredWorkspaceBytes() (now deleted)
#include "../Layers/ScratchBlock/ScratchBlockReasoning_GPU.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Shared/StreamController/StreamController_GPU.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/PBM/PositionalBiasMethod.hpp"

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
    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    
    PBM::PBMConfig pbm_config{};  // Uses HyperParameters defaults
    pbm_config.num_heads = cfg.num_heads;
    pbm_config.num_kv_heads = cfg.num_kv_heads;
    pbm_config.max_seq_len = cfg.max_seq_len;  // CRITICAL: Set from model config for context-aware scaling
    // alibi_slope_exponent uses default from HyperParameters::ALIBI_SLOPE_EXPONENT
    // rope_theta/rope_scaling use defaults from HyperParameters (NTK scaling auto-applies if max_seq_len > 2048)
    pbm_config.head_dim = head_dim;
    pbm_config.rotary_dim = head_dim;  // Full rotation
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
    if (training_state_.initialized) {
        fprintf(stderr, "[initTrainingState] WARNING: Already initialized, skipping re-init\n");
        return;
    }
    
    const auto& cfg = getConfig();
    
    // ═══════════════════════════════════════════════════════════════════════
    //  STEP 1: Initialize StreamController if not already done
    //  (May be pre-initialized by Phase1_Startup before initGPU)
    // ═══════════════════════════════════════════════════════════════════════
    if (!training_state_.stream_ctrl.isInitialized()) {
        StreamControllerConfig stream_config;
        stream_config.verbose = true;
        
        if (!training_state_.stream_ctrl.initialize(stream_config)) {
            throw std::runtime_error("[initTrainingState] Failed to initialize StreamController - CUDA device may be unavailable");
        }
        std::cout << "✓ StreamController initialized (Primary stream)" << std::endl;
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
    
    std::cout << "[DEBUG-INIT-1] After cuBLAS, before PBM check" << std::endl << std::flush;
    
    // ═══════════════════════════════════════════════════════════════════════
    //  STEP 3: Verify PBM (Unified ALiBi+RoPE Hybrid) is initialized
    //  RULE 20: PBM MUST be initialized by initPBM() before this point.
    //  NO silent fallback — if PBM is missing, it's a bug in the call order.
    // ═══════════════════════════════════════════════════════════════════════
    if (!training_state_.pbm_initialized) {
        throw std::runtime_error("[initTrainingState] PBM not initialized! "
            "Call initPBM() before initTrainingState() — Rule 20: no silent fallbacks");
    }
    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) pre-initialized" << std::endl;

     training_state_.cached_num_layers = cfg.num_layers;
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    
    std::cout << "[DEBUG-INIT-2] After PBM, before autograd check." << std::endl << std::flush;
    
    // ═══════════════════════════════════════════════════════════════
    // PARAMETER TENSORS: Preallocate once, reuse throughout training
    // ═══════════════════════════════════════════════════════════════
    // Using GRIM::Tensor with requires_grad=true allocates both data and grad buffers.
    // Weight tying: When tie_embeddings=true, lm_head_weights.data points to embedding buffer
    // and lm_head_weights.grad is shared with embedding tokenWeights().grad via share_grad().
    
    using TC = TensorContract::TensorShape;
    
    // ═══════════════════════════════════════════════════════════════
    // RULE 20: autograd MUST be initialized (seed stored)
    // ═══════════════════════════════════════════════════════════════
    // Phase1_Startup step 2.75 calls initializeAutogradSeed() which stores
    // the weight init seed. If not set, it's a bug - fail loud!
    
    if (!training_state_.seed_initialized_) {
        throw std::runtime_error(
            "[InitTrainingState] FATAL: Autograd seed not initialized!\n"
            "Phase1_Startup must call initializeAutogradSeed() in step 2.75 before initTrainingState().");
    }
    
    // All weights are owned by Pattern B layers (EmbeddingLayer, LMHeadLayer, EncodingLayer, ScratchBlockLayer)
    std::cout << "[DEBUG-INIT-4] autograd initialized, checking layer pointers..." << std::endl << std::flush;
    
    // CRASH DEBUG: Step-by-step pointer access to find exact crash point
    // ISSUE #59: Use grad_data() accessor
    // Session 6: Embedding weights now owned by EmbeddingLayer (Pattern B)
    if (!embedding_layer_) throw std::runtime_error("[InitTrainingState] FATAL: embedding_layer_ is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    std::cout << "[DEBUG-INIT-4a] About to read embedding tokenWeights().data..." << std::endl << std::flush;
    float* emb_data = embedding_layer_->tokenWeights().data;
    std::cout << "[DEBUG-INIT-4b] emb_data=" << (void*)emb_data << std::endl << std::flush;
    
    std::cout << "[DEBUG-INIT-4c] About to read embedding tokenWeights().grad_data()..." << std::endl << std::flush;
    float* emb_grad = embedding_layer_->tokenWeights().grad_data();
    std::cout << "[DEBUG-INIT-4d] emb_grad=" << (void*)emb_grad << std::endl << std::flush;
    
    float* pos_data = nullptr;
    float* pos_grad = nullptr;
    if (embedding_layer_->hasPositionEmbeddings()) {
        std::cout << "[DEBUG-INIT-4e] About to read embedding positionWeights().data..." << std::endl << std::flush;
        pos_data = embedding_layer_->positionWeights().data;
        std::cout << "[DEBUG-INIT-4f] pos_data=" << (void*)pos_data << std::endl << std::flush;
        
        std::cout << "[DEBUG-INIT-4g] About to read embedding positionWeights().grad_data()..." << std::endl << std::flush;
        pos_grad = embedding_layer_->positionWeights().grad_data();
        std::cout << "[DEBUG-INIT-4h] pos_grad=" << (void*)pos_grad << std::endl << std::flush;
    } else {
        std::cout << "[DEBUG-INIT-4e] No position embeddings (ALiBi/RoPE mode)" << std::endl << std::flush;
    }
    
    std::cout << "[DEBUG-INIT-4i] About to read lm_head_layer_->weights().data..." << std::endl << std::flush;
    if (!lm_head_layer_) throw std::runtime_error("[InitTrainingState] FATAL: lm_head_layer_ is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    float* lm_data = lm_head_layer_->weights().data;
    std::cout << "[DEBUG-INIT-4j] lm_data=" << (void*)lm_data << std::endl << std::flush;
    
    std::cout << "[DEBUG-INIT-4k] About to read lm_head_layer_->weights().grad_data()..." << std::endl << std::flush;
    float* lm_grad = lm_head_layer_->weights().grad_data();
    std::cout << "[DEBUG-INIT-4l] lm_grad=" << (void*)lm_grad << std::endl << std::flush;
    
    std::cout << "✓ Embeddings initialized via EmbeddingLayer (Pattern B ownership)\n";
    std::cout << "  tokenWeights.data=" << (void*)emb_data
              << " grad=" << (void*)emb_grad << "\n";
    std::cout << "  positionWeights.data=" << (void*)pos_data
              << " grad=" << (void*)pos_grad << "\n";
    std::cout << "  lm_head weights.data=" << (void*)lm_data
              << " grad=" << (void*)lm_grad << "\n";
    
    // Weight tying verification is handled by LMHeadLayer constructor (Pattern B).
    // LMHeadLayer validates shape match and pointer aliasing at construction time.
    // lm_head_bias, final_rms_gamma are owned by LMHeadLayer (Pattern B).
    // Access via lm_head_layer_->bias() / lm_head_layer_->finalRmsGamma().
    
    // ═══════════════════════════════════════════════════════════════
    // AUTOGRAD MIGRATION COMPLETE - Legacy vectors REMOVED
    // ═══════════════════════════════════════════════════════════════
    // FFN/Attention/RMSNorm gradients flow through encoder's Tensor& accessors:
    //   - enc->ffnWGateUp().grad_data(), enc->ffnWDown().grad_data()
    //   - enc->attnWqkv().grad_data(), enc->attnWo().grad_data()
    //   - enc->rms1Gamma().grad_data(), enc->rms2Gamma().grad_data()
    // Allocated via ensure_grad() in EncodingLayer::allocateWeights().
    
    // GQA configuration: source from JSON config (NOT compile-time HyperParameters)
    const int num_kv_heads = cfg.num_kv_heads;
    
    // Validate GQA configuration
    if (!HyperParameters::isValidGQAConfig(cfg.num_heads, num_kv_heads)) {
        throw std::runtime_error("[initTrainingState] Invalid GQA config: num_heads=" + std::to_string(cfg.num_heads) 
                                 + " num_kv_heads=" + std::to_string(num_kv_heads));
    }
    
    // Store GQA config in training state
    training_state_.num_heads = cfg.num_heads;
    training_state_.num_kv_heads = num_kv_heads;
    
    std::cout << "GQA Configuration: num_heads=" << cfg.num_heads 
              << " num_kv_heads=" << num_kv_heads 
              << " (heads_per_kv_group=" << (cfg.num_heads / num_kv_heads) << ")" << std::endl;
    
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
    
    // DELETED: batch_prep_* lazy allocation (Rule 20) — replaced by BatchPayload struct
    
    // BUG FIX: Set kv_cache_capacity for inference sampling during training
    // Previously missing - caused forwardInit() to fail with capacity=0
    training_state_.kv_cache_capacity = static_cast<int>(max_seq_len_cache);
    training_state_.kv_cache_len = 0;  // Start with empty cache
    
    // NOTE: single_token_hidden/logits/embedding are inference-only buffers.
    // Allocated in InitInferenceState.cu when inference is initialized.
    // NOT needed during training — removed dead allocations (Finding 3).
    
    std::cout << "📊 Allocating activation caches for max_tokens=" << max_tokens
              << " (batch=" << max_batch_size << ", seq_len=" << max_seq_len_cache << ")" << std::endl;
    
    // Output layer cache - using Tensor API (actively used by inference and Phase2 diagnostics)
    training_state_.cached_encoder_output = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream, "cached_encoder_output");
    
    // DELETED: cached_final_rms_input - DEAD CODE (Rule 20)
    // Was allocated but never read by any code.
    
    // Allocate logits cache with LOGITS layout tracking (TensorContract integration)
    training_state_.cached_logits_tensor = Tensor::empty(
        TensorContract::TensorShape::make_LOGITS(max_logit_tokens, cfg.vocab_size), false, primary_stream, "cached_logits");
    std::cout << "✓ Allocated cached_logits [" << max_logit_tokens << " x " << cfg.vocab_size << "] LOGITS layout" << std::endl;

    training_state_.cached_targets_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_logit_tokens, 1), false, primary_stream, "cached_targets");
    
    // NOTE: Using empty() not zeros() - ComputeLossBatch fully overwrites this buffer
    // via cudaMemcpyAsync before every forward pass. No need to waste bandwidth zero-filling.
    training_state_.cached_token_ids_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for token IDs
        primary_stream,
        "cached_token_ids"
    );
    std::cout << "✓ Allocated token IDs cache (Tensor API) [" << max_tokens << "]" << std::endl;
    
    // BUG FIX: Numeric buffers must be sized by max_tokens (full cache capacity)
    // not max_logit_tokens (training optimization). Inference sampling requires
    // the full buffer for sequences up to max_cached_seq_len.
    // BUG FIX: Always allocate numeric/text buffers even when ScratchBlock is disabled
    // because buildBatchPayload() always populates these fields from tokenizer
    // Rule 20: Use Tensor API instead of raw cudaMalloc
    // NOTE: Using empty() not zeros() - buffers fully overwritten by ComputeLossBatch
    training_state_.cached_token_numeric_values = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad
        primary_stream,
        "cached_token_numeric_values"
    );
    std::cout << "✓ Allocated token numeric values cache (Tensor API)" << std::endl;
    
    training_state_.cached_token_atom_mask = Tensor::empty(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad
        primary_stream,
        "cached_token_atom_mask"
    );
    std::cout << "✓ Allocated token atom mask cache (Tensor API)" << std::endl;

    // Allocate text feature buffers - Rule 20: Tensor API
    constexpr int kTextFeatureDim = Batching::BatchPayload::kTextFeatureDim;
    training_state_.cached_token_text_features = Tensor::empty(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_tokens), kTextFeatureDim),
        false,  // no grad
        primary_stream,
        "cached_token_text_features"
    );
    std::cout << "✓ Allocated token text features cache (Tensor API)" << std::endl;
    
    training_state_.cached_token_atom_flags = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad
        primary_stream,
        "cached_token_atom_flags"
    );
    std::cout << "✓ Allocated token atom flags cache (Tensor API)" << std::endl;
    
    std::cout << "✓ Allocated atom mask + text feature + atom flags buffers (" 
              << (max_tokens * (sizeof(float) + sizeof(uint8_t) + sizeof(uint32_t) + kTextFeatureDim * sizeof(uint16_t)) / 1024 / 1024) 
              << " MB)" << std::endl;

    training_state_.sequence_weights_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_batch_size), 1), false, primary_stream, "sequence_weights");
    training_state_.sequence_weight_capacity = static_cast<int>(max_batch_size);
    training_state_.sequence_weight_count = 0;
    
    // ═══════════════════════════════════════════════════════════════
    //  INTERMEDIATE GRADIENT TENSORS (Issue #45 FIX: Proper autograd)
    // ═══════════════════════════════════════════════════════════════
    // Using Tensor::zeros() instead of raw cudaMalloc for proper lifecycle management.
    // Tensors own their memory and provide zero_grad(stream) for gradient zeroing.
    
    cudaStream_t grad_stream = training_state_.stream_ctrl.getPrimaryStream();
    
    // grad_logits: [max_logit_tokens, vocab_size] LOGITS layout
    training_state_.grad_logits_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_LOGITS(static_cast<int>(max_logit_tokens), cfg.vocab_size),
        false, grad_stream, "grad_logits");
    std::cout << "✓ Allocated grad_logits_tensor [" << max_logit_tokens << " x " << cfg.vocab_size << "] LOGITS layout" << std::endl;

    // grad_encoder: [max_tokens, d_model]
    training_state_.grad_encoder_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_tokens), cfg.d_model),
        false, grad_stream, "grad_encoder_out");
    
    const size_t tokens_per_batch = max_batch_size * max_seq_len_cache;

    // ═══════════════════════════════════════════════════════════════
    //  ENCODER BACKWARD TEMPORARIES (Issue #45 FIX: Tensor allocation)
    // ═══════════════════════════════════════════════════════════════
    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    const int max_tokens_int = static_cast<int>(max_tokens);
    
    // FFN backward temporaries
    training_state_.grad_ffn_input_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_ffn_input");
    
    training_state_.grad_ffn_hidden_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_ff),
        false, grad_stream, "grad_ffn_hidden");
    
    // Attention backward temporaries (model-width)
    training_state_.grad_attn_input_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_attn_input");
    
    training_state_.grad_attn_out_proj_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_attn_out_before_proj");
    
    training_state_.grad_attn_out_bhsd_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_attn_out_reshaped");
    
    // QKV gradients (need 4D shape for attention, but stored flat for now)
    // Full shape: [batch, heads, seq, head_dim] - using BSM as [tokens, d_model]
    training_state_.grad_q_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_Q");
    
    training_state_.grad_k_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_K");
    
    training_state_.grad_v_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_V");
    
    // QKV fused [tokens, 3*d_model]
    training_state_.grad_qkv_concat_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_QKV_FUSED(max_tokens_int, 3 * cfg.d_model),
        false, grad_stream, "grad_qkv_concat");
    
    training_state_.grad_qkv_input_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_qkv_input");
    
    // DEDICATED scratch buffer for attention output BSM conversion (W_o gradient computation)
    // CRITICAL: Do NOT reuse grad_qkv_input - prevents temporal aliasing bugs
    training_state_.grad_attn_bsm_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream, "grad_attn_bsm_scratch");
    
    // Issue #43 FIX: Centering scratch buffer for encoder weight gradients
    // Size: max(d_model, d_ff) to handle both model and FFN width activations
    const int centering_width = std::max(cfg.d_model, cfg.d_ff);
    training_state_.centering_scratch_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, centering_width),
        false, grad_stream, "centered_activation_scratch");
    std::cout << "📐 Issue #43: Allocated centering scratch tensor (" 
              << (training_state_.centering_scratch_tensor.numel() * sizeof(float) / (1024*1024)) << " MB)" << std::endl;

    // Rule 20: NO BACKWARDS COMPATIBILITY - callers must use tensor.data directly
    // Removed raw pointer alias assignments

    // DELETED: FA bf16/dq_accum/dsoftmax_sum buffers — FlashAttentionLayer::ensureScratch() self-manages
    // (was ~56MB dead GPU allocation). Autograd ScaledDotProductAttentionGradFn also self-allocates backward buffers.
    
    // Loss scratch buffers using Tensor API (Rule 20: no raw cudaMalloc)
    training_state_.d_loss_scratch = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_logit_tokens), 1),
        false, primary_stream, "d_loss_scratch");
    training_state_.d_loss_sum_scratch = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, 1),  // Scalar
        false, primary_stream, "d_loss_sum_scratch");

    // Initialize scratch block pool for pinned memory batch transfers
    // Block size derived from max_cached_tokens — the largest per-batch transfer is
    // text_features at max_tokens * kTextFeatureDim * sizeof(uint16_t).
    training_state_.scratch_enabled = true;
    
    // kTextFeatureDim already declared above from BatchPayload::kTextFeatureDim
    const size_t max_transfer_bytes = training_state_.max_cached_tokens
                                    * static_cast<size_t>(kTextFeatureDim) * sizeof(uint16_t);
    const size_t tokens_per_block = (max_transfer_bytes + sizeof(int) - 1) / sizeof(int);
    if (tokens_per_block == 0) {
        throw std::runtime_error("InitTrainingState: scratch pool tokens_per_block computed as 0");
    }
    
    {
        ScratchBlock::ScratchBlockConfig scratch_config;
        scratch_config.enabled = true;
        scratch_config.max_tokens_per_block = tokens_per_block;
        scratch_config.num_blocks = 2;  // Double buffer
        scratch_config.use_write_combined = false;
        
        training_state_.scratch_pool = new ScratchBlock::ScratchBlockPool(scratch_config);
        
        if (!training_state_.scratch_pool || !training_state_.scratch_pool->isInitialized()) {
            throw std::runtime_error("InitTrainingState: Scratch block pool initialization failed");
        }
        
        size_t total_bytes = training_state_.scratch_pool->getTotalPinnedMemoryBytes();
        double mb = total_bytes / (1024.0 * 1024.0);
        std::cout << "✓ Scratch block pool initialized ("
                  << scratch_config.num_blocks << " blocks × "
                  << max_transfer_bytes << " bytes (" << tokens_per_block << " tokens) = "
                  << std::fixed << std::setprecision(2) << mb
                  << " MB pinned memory)" << std::endl;
    }
    
    // Initialize ScratchBlock reasoning layer
    if (cfg.use_scratch_block) {
        std::cout << "🧠 Initializing ScratchBlock reasoning layer..." << std::endl;
        
        ScratchBlockConfig sb_config;
        sb_config.d_model = cfg.d_model;
        sb_config.max_atoms = cfg.scratch_block_max_atoms;
        sb_config.atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
        sb_config.enabled = true;
        sb_config.atom_scale = cfg.scratch_block_atom_scale;
        sb_config.atom_token_start = GRIM::Tokenizer::ATOM_TOKEN_OFFSET;
        sb_config.atom_token_end = GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET;
        sb_config.stream = training_state_.stream_ctrl.getPrimaryStream();
        
        scratch_block_layer_ = std::make_unique<ScratchBlockLayer>(sb_config);
        
        std::cout << "✓ ScratchBlock reasoning layer initialized (d_model="
                  << cfg.d_model << ", atom_dim=" << cfg.scratch_block_atom_embedding_dim
                  << ", max_atoms=" << cfg.scratch_block_max_atoms << ")" << std::endl;
    } else {
        std::cout << "ℹ ScratchBlock reasoning layer disabled" << std::endl;
        scratch_block_layer_ = nullptr;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP FINAL: Confirm initialization complete
    // ═══════════════════════════════════════════════════════════════════════════
    
    std::cout << "✓ Verified: Autograd seed initialized (from Phase1_Startup)" << std::endl;
    
    training_state_.initialized = true;
    std::cout << "✓ Training state initialized with full gradient buffers" << std::endl;
    std::cout << "[InitTrainingState] max_cached_batch=" << training_state_.max_cached_batch
              << " max_cached_seq_len=" << training_state_.max_cached_seq_len
              << " max_cached_tokens=" << training_state_.max_cached_tokens
              << " max_logit_tokens=" << training_state_.max_logit_tokens << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM



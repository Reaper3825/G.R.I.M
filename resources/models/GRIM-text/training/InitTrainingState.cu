#ifndef USE_CUDA
#define USE_CUDA
#endif

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "../GRIM/grim_language_model_cuda.hpp"
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

    cublasStatus_t cublas_err = cublasCreate(training_state_.cublas_handle.outParam());
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
bool LanguageModel::isPBMInitialized() const {
    return pbm_spec_initialized_ && pbm_spec_.valid &&
           pbm_spec_.rope_inv_freq != nullptr &&
           pbm_spec_.alibi_slopes != nullptr;
}

const PBM::PBMSpec& LanguageModel::getPBMSpec() const {
    if (!isPBMInitialized()) {
        throw std::runtime_error("LanguageModel::getPBMSpec: PBM is not initialized");
    }
    return pbm_spec_;
}

void LanguageModel::initPBM() {
    if (isPBMInitialized()) {
        std::cout << "✓ PBM (ALiBi+RoPE) already initialized" << std::endl;
        return;
    }

    if (!embedder_) {
        throw std::runtime_error("LanguageModel::initPBM: embedder is NULL");
    }

    const auto& cfg = getConfig();

    const auto* alibi_bias = embedder_->getALiBiBias();
    if (!alibi_bias || !alibi_bias->isInitialized()) {
        throw std::runtime_error(
            "LanguageModel::initPBM: positional bias is not initialized on GrimEmbeddingStack");
    }

    // Non-owning view into GrimEmbeddingStack::ALiBiPositionalBias PBM buffers.
    PBM::PBMSpec pbm_spec = PBM::getPBMSpec(alibi_bias->getPBMState());
    if (!pbm_spec.valid || !pbm_spec.rope_inv_freq || !pbm_spec.alibi_slopes) {
        throw std::runtime_error("LanguageModel::initPBM: PBM spec is invalid");
    }
    pbm_spec_ = pbm_spec;
    pbm_spec_initialized_ = true;

    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) initialized:" << std::endl;
    std::cout << "    ALiBi: " << cfg.num_heads << " heads with slopes" << std::endl;
    std::cout << "    RoPE:  head_dim=" << cfg.head_dim
              << ", rotary_dim=" << pbm_spec_.rotary_dim
              << ", theta=10000" << std::endl;
}

void LanguageModel::initTrainingState() {
    // RULE 20: double-init is a caller-order bug, not a recoverable condition.
    if (training_state_.initialized) {
        throw std::runtime_error(
            "[initTrainingState] FATAL: training_state_.initialized is already true. "
            "Caller invoked initTrainingState() twice (or after initInferenceState). "
            "This is a call-order bug.");
    }
    
    const auto& cfg = getConfig();
    
    // ═══════════════════════════════════════════════════════════════════════
    //  STEP 1: Verify StreamController is initialized
    //  RULE 20: stream_ctrl MUST be initialized by Phase1_Startup before this
    //  point. NO silent fallback — if it's missing, it's a bug in call order.
    // ═══════════════════════════════════════════════════════════════════════
    if (!training_state_.stream_ctrl.isInitialized()) {
        throw std::runtime_error(
            "[initTrainingState] StreamController not initialized! "
            "Phase1_Startup must call stream_ctrl.initialize() before "
            "initTrainingState() — Rule 20: no silent fallbacks");
    }
    StreamController::fatalIfDefaultStream(training_state_.stream_ctrl.getPrimaryStream(),
                                           "LanguageModel::initTrainingState");
    std::cout << "✓ StreamController pre-initialized" << std::endl;
    
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
    if (!isPBMInitialized()) {
        throw std::runtime_error("[initTrainingState] PBM not initialized! "
            "Call initPBM() before initTrainingState() — Rule 20: no silent fallbacks");
    }
    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) pre-initialized" << std::endl;

     training_state_.cached_num_layers = cfg.num_layers;
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    
    std::cout << "[DEBUG-INIT-2] After PBM, checking layer pointers." << std::endl << std::flush;
    
    // ═══════════════════════════════════════════════════════════════
    // PARAMETER TENSORS: Preallocate once, reuse throughout training
    // ═══════════════════════════════════════════════════════════════
    // Using GRIM::Tensor with requires_grad=true allocates both data and grad buffers.
    // Weight tying: When tie_embeddings=true, lm_head_weights.data points to embedding buffer
    // and lm_head_weights.grad is shared with embedding tokenWeights().grad via share_grad().
    
    using TC = TensorContract::TensorShape;
    
    // All weights are owned by Pattern B layers (EmbeddingLayer, LMHeadLayer, EncodingLayer, ScratchBlockLayer)
    std::cout << "[DEBUG-INIT-4] checking Pattern B layer pointers..." << std::endl << std::flush;
    
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
    //   - enc->ffnW1().grad_data(), enc->ffnW2().grad_data()
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
    
    std::cout << "GQA Configuration: num_heads=" << cfg.num_heads 
              << " num_kv_heads=" << num_kv_heads 
              << " (heads_per_kv_group=" << (cfg.num_heads / num_kv_heads) << ")" << std::endl;
    
    // NOTE: Encoder layer weight initialization is handled in TrainingOps.cu::initGPU()
    // with proper GQA-aware dimensions and GPT-2 residual scaling.
    // DO NOT duplicate Xavier init here per Rule 20 (no backwards compatibility shims).
    
    // Rule 20: capacity is authored upstream (RunCapacity -> LanguageModelConfig mirrors).
    // This layer must not silently clamp mismatches; it must throw.
    if (cfg.max_cached_batch <= 0) {
        throw std::runtime_error("[initTrainingState] FATAL: cfg.max_cached_batch <= 0 (" +
                                 std::to_string(cfg.max_cached_batch) + ")");
    }
    if (cfg.max_seq_len <= 0 || cfg.max_cached_seq_len <= 0) {
        throw std::runtime_error("[initTrainingState] FATAL: invalid seq lens (max_seq_len=" +
                                 std::to_string(cfg.max_seq_len) + " max_cached_seq_len=" +
                                 std::to_string(cfg.max_cached_seq_len) + ")");
    }
    if (cfg.max_seq_len != cfg.max_cached_seq_len) {
        throw std::runtime_error("[initTrainingState] FATAL: cfg.max_seq_len != cfg.max_cached_seq_len (max_seq_len=" +
                                 std::to_string(cfg.max_seq_len) + " max_cached_seq_len=" +
                                 std::to_string(cfg.max_cached_seq_len) + ")");
    }
    if (cfg.max_tokens_per_batch <= 0) {
        throw std::runtime_error("[initTrainingState] FATAL: cfg.max_tokens_per_batch <= 0 (" +
                                 std::to_string(cfg.max_tokens_per_batch) + ")");
    }

    const size_t max_batch_size = static_cast<size_t>(cfg.max_cached_batch);
    const size_t max_seq_len_cache = static_cast<size_t>(cfg.max_cached_seq_len);

    const uint64_t max_tokens_u64 =
        static_cast<uint64_t>(max_batch_size) * static_cast<uint64_t>(max_seq_len_cache);
    if (max_tokens_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("[initTrainingState] FATAL: max_tokens overflow (batch=" +
                                 std::to_string(max_batch_size) + " seq_len=" +
                                 std::to_string(max_seq_len_cache) + " product=" +
                                 std::to_string(max_tokens_u64) + ")");
    }
    const size_t max_tokens = static_cast<size_t>(max_tokens_u64);

    if (static_cast<size_t>(cfg.max_tokens_per_batch) != max_tokens) {
        throw std::runtime_error("[initTrainingState] FATAL: cfg.max_tokens_per_batch does not match cache rectangle (cfg=" +
                                 std::to_string(cfg.max_tokens_per_batch) + " expected=" +
                                 std::to_string(max_tokens) + " batch=" +
                                 std::to_string(max_batch_size) + " seq_len=" +
                                 std::to_string(max_seq_len_cache) + ")");
    }

    const size_t logit_token_capacity = max_tokens;
    
    // DELETED: batch_prep_* lazy allocation (Rule 20) — replaced by BatchPayload struct
    
    // Generation/KV-cache state is intentionally NOT initialized here.
    // Training-time sampling must explicitly call ensureKVCacheAllocated(),
    // which creates GenerationState from the authored config capacity.

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
        TensorContract::TensorShape::make_LOGITS(logit_token_capacity, cfg.vocab_size), false, primary_stream, "cached_logits");
    std::cout << "✓ Allocated cached_logits [" << logit_token_capacity << " x " << cfg.vocab_size << "] LOGITS layout" << std::endl;

    training_state_.cached_targets_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(logit_token_capacity, 1), false, primary_stream, "cached_targets");
    // MTP shifted targets are intentionally NOT TrainingState-owned: each MTP
    // head needs a distinct target tensor in AutogradIntermediates because
    // NLLLossGradFn stores raw target pointers through backward.
    
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
    // not the logits-only capacity. Inference sampling requires
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

    training_state_.cached_token_to_slot_map = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad — runtime substrate metadata
        primary_stream,
        "cached_token_to_slot_map"
    );
    std::cout << "✓ Allocated token-to-slot map cache (Tensor API)" << std::endl;
    
    std::cout << "✓ Allocated atom mask + text feature + atom flags buffers (" 
              << (max_tokens * (sizeof(float) + sizeof(uint8_t) + sizeof(uint32_t) + kTextFeatureDim * sizeof(uint16_t)) / 1024 / 1024) 
              << " MB)" << std::endl;

    training_state_.sequence_weights_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_batch_size), 1), false, primary_stream, "sequence_weights");
    training_state_.sequence_weight_capacity = static_cast<int>(max_batch_size);
    training_state_.sequence_weight_count = 0;
    
    // Rule 20: NO BACKWARDS COMPATIBILITY - callers must use tensor.data directly
    // Removed raw pointer alias assignments
    // centering_scratch_tensor DELETED — cached_encoder_output is now overwritten
    // with centered data after LM head forward (single source of truth)

    // DELETED: FA bf16/dq_accum/dsoftmax_sum buffers — FlashAttentionLayer::ensureScratch() self-manages
    // (was ~56MB dead GPU allocation). Autograd ScaledDotProductAttentionGradFn also self-allocates backward buffers.
    
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
    
    std::cout << "✓ Verified: Pattern B layers initialized by initGPU()" << std::endl;
    
    training_state_.initialized = true;
    std::cout << "✓ Training state initialized with full gradient buffers" << std::endl;
    std::cout << "[InitTrainingState] max_cached_batch=" << max_batch_size
              << " max_cached_seq_len=" << max_seq_len_cache
              << " token_capacity=" << max_tokens
              << " logit_token_capacity=" << logit_token_capacity << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM



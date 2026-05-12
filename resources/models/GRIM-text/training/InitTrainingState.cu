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
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Shared/StreamController/StreamController_GPU.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "../Shared/PBM/PBMStateOwner.hpp"

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
    
    if (training_state_.cublas_handle.get() != nullptr) {
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
    cublasSetMathMode(training_state_.cublas_handle.get(), CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSetStream(training_state_.cublas_handle.get(), primary_stream);
    std::cout << "✓ cuBLAS handle bound to Primary stream with Tensor Core acceleration" << std::endl;
}

//======================================================//
//  Unified PBM (ALiBi+RoPE Hybrid) Initialization
//  Call this BEFORE initGPU() to ensure encoder gets both position encodings
//======================================================//
bool LanguageModel::isPBMInitialized() const {
    return pbm_owner_.initialized() &&
           pbm_spec_initialized_ && pbm_spec_.valid &&
           pbm_spec_.rope_inv_freq != nullptr &&
           pbm_spec_.alibi_slopes != nullptr;
}

const PBM::PBMSpec& LanguageModel::getPBMSpec() const {
    if (!isPBMInitialized()) {
        throw std::runtime_error("LanguageModel::getPBMSpec: PBM is not initialized");
    }
    return pbm_spec_;
}

const PBM::PBMState& LanguageModel::getPBMState() const {
    if (!isPBMInitialized()) {
        throw std::runtime_error("LanguageModel::getPBMState: PBM is not initialized");
    }
    return pbm_owner_.state();
}

void LanguageModel::initPBM() {
    if (isPBMInitialized()) {
        std::cout << "✓ PBM (ALiBi+RoPE) already initialized" << std::endl;
        return;
    }

    if (!training_state_.stream_ctrl.isInitialized()) {
        throw std::runtime_error(
            "LanguageModel::initPBM: StreamController is not initialized - "
            "ModelAllocationState must initialize stream_ctrl before PBM");
    }
    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    StreamController::fatalIfDefaultStream(stream, "LanguageModel::initPBM");

    const auto pbm_hp = HyperParameters::pbmConstructionHP(config_);
    if (config_.positional_encoding != HyperParameters::PositionalEncodingType::ALIBI_ROPE &&
        config_.positional_encoding != HyperParameters::PositionalEncodingType::ALIBI) {
        throw std::runtime_error(
            std::string("LanguageModel::initPBM: unified PBM requires ALIBI or ALIBI_ROPE, got ") +
            HyperParameters::positionalEncodingTypeToString(config_.positional_encoding));
    }

    const int expected_d_model = pbm_hp.num_heads * pbm_hp.head_dim;
    if (config_.d_model != expected_d_model) {
        throw std::runtime_error(
            "LanguageModel::initPBM: grouped PBM geometry does not match d_model (d_model=" +
            std::to_string(config_.d_model) + " expected=" + std::to_string(expected_d_model) + ")");
    }

    pbm_owner_.initialize(pbm_hp, stream);

    // Non-owning view into the model-level PBMStateOwner buffers.
    PBM::PBMSpec pbm_spec = pbm_owner_.spec();
    if (!pbm_spec.valid || !pbm_spec.rope_inv_freq || !pbm_spec.alibi_slopes) {
        throw std::runtime_error("LanguageModel::initPBM: PBM spec is invalid");
    }
    pbm_spec_ = pbm_spec;
    pbm_spec_initialized_ = true;

    std::cout << "✓ PBM (Hybrid ALiBi+RoPE) initialized" << std::endl;
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
    if (training_state_.cublas_handle.get() == nullptr) {
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

    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    
    std::cout << "[DEBUG-INIT-2] After PBM, checking layer pointers." << std::endl << std::flush;
    
    // ═══════════════════════════════════════════════════════════════
    // PARAMETER TENSORS: Preallocate once, reuse throughout training
    // ═══════════════════════════════════════════════════════════════
    // Using GRIM::Tensor with requires_grad=true allocates both data and grad buffers.
    // Weight tying: When tie_embeddings=true, lm_head_weights.data points to embedding buffer
    // and lm_head_weights.grad is shared with embedding tokenWeights().grad via share_grad().
    
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
    
    // NOTE: Encoder layer weight initialization is handled by Startup/Model GPU assembly.
    // with proper GQA-aware dimensions and GPT-2 residual scaling.
    // DO NOT duplicate Xavier init here per Rule 20 (single initialization owner).
    
    // DELETED: batch_prep_* lazy allocation (Rule 20) — replaced by BatchPayload struct
    
    // Generation/KV-cache state is intentionally NOT initialized here.
    // Training-time sampling must explicitly call ensureKVCacheAllocated(),
    // which creates GenerationState from the authored config capacity.

    // NOTE: single-token decode scratch is generation-owned.
    // Allocated in GenerationState by InitInferenceState.cu or ensureKVCacheAllocated().
    // NOT needed by the training state cache rectangle.
    
    // Capacity is already authored on LanguageModelConfig. Per-batch
    // geometry/semantics come from BatchPayload at upload/forward time,
    // never from this init path.
    std::cout << "📊 Allocating TrainingState step workspaces" << std::endl;
    
    training_state_.allocateStepDeviceWorkspaces(cfg, primary_stream);
    
    // Rule 20: callers must use tensor.data directly
    // Removed raw pointer alias assignments
    // centering_scratch_tensor DELETED — cached_encoder_output is now overwritten
    // with centered data after LM head forward (single source of truth)

    // DELETED: FA bf16/dq_accum/dsoftmax_sum buffers — FlashAttentionLayer::ensureScratch() self-manages
    // (was ~56MB dead GPU allocation). Autograd ScaledDotProductAttentionGradFn also self-allocates backward buffers.
    // ScratchBlockLayer is durable model topology and is assembled in initGPU().
    // initTrainingState() only verifies startup order and asks TrainingState to allocate
    // its own reusable runtime cache tensors from the authored model config.
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP FINAL: Confirm initialization complete
    // ═══════════════════════════════════════════════════════════════════════════
    
    std::cout << "✓ Verified: Pattern B layers initialized by initGPU()" << std::endl;
    
    training_state_.initialized = true;
    std::cout << "✓ Training state initialized with full gradient buffers" << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM



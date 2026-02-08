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
    if (training_state_.initialized) return;
    
    const auto& cfg = getConfig();
    
    // NOTE: batch_prep_* vectors may be corrupted due to memory overwrite bug.
    // We use local vectors in prepareLossBatchInputs() instead (Issue #59+ workaround).
    // Do NOT attempt to resize/use batch_prep_* vectors until root cause is found.

    
    // ═══════════════════════════════════════════════════════════════════════
    //  STEP 1: Initialize StreamController if not already done
    //  (May be pre-initialized by Phase1_Startup before initGPU)
    // ═══════════════════════════════════════════════════════════════════════
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
    
    std::cout << "[DEBUG-INIT-1] After cuBLAS, before PBM check" << std::endl << std::flush;
    
    // ═══════════════════════════════════════════════════════════════════════
    //  STEP 3: Initialize PBM (Unified ALiBi+RoPE Hybrid)
    //  CRITICAL: Without positional encoding, attention has no position info!
    //  - RoPE: Rotary Position Embedding rotates Q,K to encode position
    //  - ALiBi: Attention-Linear-Biases adds position-dependent bias to scores
    //  Missing PBM causes position-blind attention → training plateau!
    // ═══════════════════════════════════════════════════════════════════════
    if (!training_state_.pbm_initialized) {
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
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    
    std::cout << "[DEBUG-INIT-2] After PBM, before tensors check. tensors_=" << (void*)training_state_.tensors_.get() << std::endl << std::flush;
    
    // ═══════════════════════════════════════════════════════════════
    // PARAMETER TENSORS: Preallocate once, reuse throughout training
    // ═══════════════════════════════════════════════════════════════
    // Using GRIM::Tensor with requires_grad=true allocates both data and grad buffers.
    // Weight tying: When tie_embeddings=true, lm_head_weights.data points to embedding buffer
    // but lm_head_weights.grad is shared with embedding_weights.grad.
    
    const size_t embedding_size = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
    const size_t position_size = static_cast<size_t>(cfg.max_seq_len) * cfg.d_model;
    using TC = TensorContract::TensorShape;
    
    // ═══════════════════════════════════════════════════════════════
    // RULE 20: NO BACKWARDS COMPATIBILITY - TrainingTensors MUST be initialized
    // ═══════════════════════════════════════════════════════════════
    // Phase1_Startup step 2.75 calls initializeAutogradTensors() which creates
    // TrainingTensors. If not set, it's a bug - fail loud!
    
    if (!training_state_.tensors_) {
        std::cout << "[DEBUG-INIT-3] tensors_ is NULL - about to throw" << std::endl << std::flush;
        throw std::runtime_error(
            "[InitTrainingState] FATAL: TrainingTensors not initialized!\n"
            "Phase1_Startup must call initializeAutogradTensors() in step 2.75 before initTrainingState().\n"
            "Legacy wrapping code has been DELETED per Rule 20 - no backwards compatibility.");
    }
    
    // TrainingTensors owns the memory - tensors already set up by initializeAutogradTensors()
    std::cout << "[DEBUG-INIT-4] tensors_ is SET, checking pointers..." << std::endl << std::flush;
    
    // CRASH DEBUG: Step-by-step pointer access to find exact crash point
    // ISSUE #59: Use grad_data() accessor
    std::cout << "[DEBUG-INIT-4a] About to read embedding_weights.data..." << std::endl << std::flush;
    float* emb_data = training_state_.embedding_weights.data;
    std::cout << "[DEBUG-INIT-4b] emb_data=" << (void*)emb_data << std::endl << std::flush;
    
    std::cout << "[DEBUG-INIT-4c] About to read embedding_weights.grad_data()..." << std::endl << std::flush;
    float* emb_grad = training_state_.embedding_weights.grad_data();
    std::cout << "[DEBUG-INIT-4d] emb_grad=" << (void*)emb_grad << std::endl << std::flush;
    
    std::cout << "[DEBUG-INIT-4e] About to read position_embedding_weights.data..." << std::endl << std::flush;
    float* pos_data = training_state_.position_embedding_weights.data;
    std::cout << "[DEBUG-INIT-4f] pos_data=" << (void*)pos_data << std::endl << std::flush;
    
    std::cout << "[DEBUG-INIT-4g] About to read position_embedding_weights.grad_data()..." << std::endl << std::flush;
    float* pos_grad = training_state_.position_embedding_weights.grad_data();
    std::cout << "[DEBUG-INIT-4h] pos_grad=" << (void*)pos_grad << std::endl << std::flush;
    
    std::cout << "[DEBUG-INIT-4i] About to read lm_head_weights.data..." << std::endl << std::flush;
    float* lm_data = training_state_.lm_head_weights.data;
    std::cout << "[DEBUG-INIT-4j] lm_data=" << (void*)lm_data << std::endl << std::flush;
    
    std::cout << "[DEBUG-INIT-4k] About to read lm_head_weights.grad_data()..." << std::endl << std::flush;
    float* lm_grad = training_state_.lm_head_weights.grad_data();
    std::cout << "[DEBUG-INIT-4l] lm_grad=" << (void*)lm_grad << std::endl << std::flush;
    
    std::cout << "✓ Embeddings initialized via TrainingTensors (proper ownership)\n";
    std::cout << "  embedding_weights.data=" << (void*)emb_data
              << " grad=" << (void*)emb_grad << "\n";
    std::cout << "  position_embedding_weights.data=" << (void*)pos_data
              << " grad=" << (void*)pos_grad << "\n";
    std::cout << "  lm_head_weights.data=" << (void*)lm_data
              << " grad=" << (void*)lm_grad << "\n";
    
    // Verify weight tying aliasing if enabled
    // ISSUE #59: Use grad_data() accessor
    if (cfg.tie_embeddings) {
        if (training_state_.embedding_weights.data != training_state_.lm_head_weights.data) {
            throw std::runtime_error(
                "[InitTrainingState] FATAL: tie_embeddings=true but data pointers NOT aliased!\n"
                "embedding_weights.data=" + std::to_string(reinterpret_cast<uintptr_t>(training_state_.embedding_weights.data)) +
                " lm_head_weights.data=" + std::to_string(reinterpret_cast<uintptr_t>(training_state_.lm_head_weights.data)));
        }
        if (training_state_.embedding_weights.grad_data() != training_state_.lm_head_weights.grad_data()) {
            throw std::runtime_error(
                "[InitTrainingState] FATAL: tie_embeddings=true but grad pointers NOT aliased!\n"
                "embedding_weights.grad=" + std::to_string(reinterpret_cast<uintptr_t>(training_state_.embedding_weights.grad_data())) +
                " lm_head_weights.grad=" + std::to_string(reinterpret_cast<uintptr_t>(training_state_.lm_head_weights.grad_data())));
        }
        std::cout << "✓ Weight tying verified: embedding and lm_head share data AND grad pointers\n";
    }
    
    // LM head bias [vocab_size] - optional
    if (cfg.use_bias) {
        training_state_.lm_head_bias = Tensor::zeros(
            TC::make_BSM(1, cfg.vocab_size), true, primary_stream);
        training_state_.lm_head_bias.ensure_grad();  // Allocate grad buffer NOW
        std::cout << "📦 LM head bias allocated: " << cfg.vocab_size << " elements" << std::endl;
    }
    
    // Numeric head [d_model] - optional
    if (cfg.numeric_head_enabled) {
        training_state_.numeric_head_weights = Tensor::zeros(
            TC::make_BSM(1, cfg.d_model), true, primary_stream);
        training_state_.numeric_head_weights.ensure_grad();  // Allocate grad buffer NOW
        training_state_.numeric_head_bias = Tensor::zeros(
            TC::make_BSM(1, 1), true, primary_stream);
        training_state_.numeric_head_bias.ensure_grad();  // Allocate grad buffer NOW
        
        // Initialize numeric head with Xavier uniform using deterministic seed
        // Shape [1, d_model]: fan_in = d_model (columns), fan_out = 1 (rows)
        // Use seed + 100 to avoid collision with encoder weight seeds
        const uint64_t numeric_head_seed = 12345ULL + 100;
        Tensor::xavier_uniform_(training_state_.numeric_head_weights, numeric_head_seed, primary_stream);
        // expected_rms = sqrt(6 / (fan_in + fan_out)) / sqrt(3) where fan_in=d_model, fan_out=1
        const float expected_rms = std::sqrt(6.0f / static_cast<float>(cfg.d_model + 1)) / std::sqrt(3.0f);
        std::cout << "📦 Numeric head allocated (expected_rms=" << expected_rms << ")" << std::endl;
        
        // Learned loss weighting (homoscedastic uncertainty)
        // Formula: weight = 0.5 * exp(-log_var)
        // To get weight = 1.0 (no scaling), need log_var = -ln(2) ≈ -0.693
        // BUG FIX: Was initialized to 0 → weight = 0.5 → HALVED the loss!
        const float init_log_var = -std::log(2.0f);  // -0.693 → weight = 1.0
        training_state_.log_var_text = Tensor::zeros(TC::make_BSM(1, 1), true, primary_stream);
        training_state_.log_var_text.ensure_grad();
        cudaMemcpyAsync(training_state_.log_var_text.data, &init_log_var, sizeof(float), 
                        cudaMemcpyHostToDevice, primary_stream);
        training_state_.log_var_numeric = Tensor::zeros(TC::make_BSM(1, 1), true, primary_stream);
        training_state_.log_var_numeric.ensure_grad();
        cudaMemcpyAsync(training_state_.log_var_numeric.data, &init_log_var, sizeof(float),
                        cudaMemcpyHostToDevice, primary_stream);
        std::cout << "📦 Learned loss weights allocated (log_var=" << init_log_var << " → weight=1.0)" << std::endl;
    }
    
    // Final RMSNorm gamma [d_model] - Issue #33 FIX
    training_state_.final_rms_gamma = Tensor::zeros(
        TC::make_BSM(1, cfg.d_model), true, primary_stream);
    training_state_.final_rms_gamma.ensure_grad();  // Allocate grad buffer NOW
    // Initialize gamma to 1.0 (standard for RMSNorm)
    std::vector<float> ones(cfg.d_model, 1.0f);
    cudaMemcpyAsync(training_state_.final_rms_gamma.data, ones.data(),
                    cfg.d_model * sizeof(float), cudaMemcpyHostToDevice, primary_stream);
    std::cout << "📦 Final RMSNorm gamma allocated: " << cfg.d_model << " elements" << std::endl;
    
    // ═══════════════════════════════════════════════════════════════
    // AUTOGRAD MIGRATION COMPLETE - Legacy vectors REMOVED
    // ═══════════════════════════════════════════════════════════════
    // FFN/Attention/RMSNorm gradients now use encoder's Tensor& accessors:
    //   - enc->ffnW1().grad_data(), enc->ffnW2().grad_data()
    //   - enc->attnWqkv().grad_data(), enc->attnWo().grad_data()
    //   - enc->rms1Gamma().grad_data(), enc->rms2Gamma().grad_data()
    // Allocated via ensure_grad() in encoder's allocateWeights().
    
    // Learnable QK-norm scales (nGPT-style) - now Tensor-based with autograd
    training_state_.attn_alpha_q.resize(cfg.num_layers);
    training_state_.attn_alpha_k.resize(cfg.num_layers);
    
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
    std::cout.flush();
    
    std::cout << "[DEBUG-LAYER-ALLOC] About to allocate QK-norm scales for " << cfg.num_layers << " layers..." << std::endl;
    std::cout.flush();
    
    // CHECK: Are batch_prep vectors intact BEFORE the QK-norm loop?
    std::cout << "[DEBUG-CORRUPTION-CHECK] BEFORE QK-norm loop:" << std::endl;
    std::cout << "[DEBUG-CORRUPTION-CHECK] target_ids: size=" << training_state_.batch_prep_target_ids.size()
              << " capacity=" << training_state_.batch_prep_target_ids.capacity()
              << " data=" << (void*)training_state_.batch_prep_target_ids.data() << std::endl;
    std::cout.flush();
    
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        // AUTOGRAD MIGRATION COMPLETE: All encoder layer gradients now use encoder's Tensor.grad
        // (allocated via ensure_grad() in encoder's allocateWeights())
        
        // Allocate and initialize learnable QK-norm scales using Tensor
        // GQA: alpha_q has num_heads entries, alpha_k has num_kv_heads entries
        // Use BSM shape with dim=1 for 1D vectors
        training_state_.attn_alpha_q[layer] = Tensor::empty(
            TensorContract::TensorShape::make_BSM(cfg.num_heads, 1), true, primary_stream);
        training_state_.attn_alpha_q[layer].ensure_grad();
        
        training_state_.attn_alpha_k[layer] = Tensor::empty(
            TensorContract::TensorShape::make_BSM(num_kv_heads, 1), true, primary_stream);
        training_state_.attn_alpha_k[layer].ensure_grad();
        
        // Initialize alpha_q to 1.0 (num_heads entries)
        std::vector<float> alpha_q_init(cfg.num_heads, HyperParameters::QK_NORM_ALPHA_INIT);
        cudaMemcpy(training_state_.attn_alpha_q[layer].data, alpha_q_init.data(), 
                   cfg.num_heads * sizeof(float), cudaMemcpyHostToDevice);
        
        // Initialize alpha_k to 1.0 (num_kv_heads entries for GQA)
        std::vector<float> alpha_k_init(num_kv_heads, HyperParameters::QK_NORM_ALPHA_INIT);
        cudaMemcpy(training_state_.attn_alpha_k[layer].data, alpha_k_init.data(), 
                   num_kv_heads * sizeof(float), cudaMemcpyHostToDevice);
        
        if (layer == 0 || layer == cfg.num_layers - 1) {
            std::cout << "[DEBUG-LAYER-ALLOC] Layer " << layer << " QK-norm allocated OK" << std::endl;
            std::cout.flush();
        }
    }
    std::cout << "[DEBUG-LAYER-ALLOC] All " << cfg.num_layers << " layers allocated successfully" << std::endl;
    std::cout.flush();
    
    // CHECK: Are the batch_prep vectors already corrupt after the QK-norm loop?
    std::cout << "[DEBUG-CORRUPTION-CHECK] After QK-norm loop, checking batch_prep vectors..." << std::endl;
    std::cout << "[DEBUG-CORRUPTION-CHECK] input_ids: size=" << training_state_.batch_prep_input_ids.size()
              << " capacity=" << training_state_.batch_prep_input_ids.capacity()
              << " data=" << (void*)training_state_.batch_prep_input_ids.data() << std::endl;
    std::cout << "[DEBUG-CORRUPTION-CHECK] target_ids: size=" << training_state_.batch_prep_target_ids.size()
              << " capacity=" << training_state_.batch_prep_target_ids.capacity()
              << " data=" << (void*)training_state_.batch_prep_target_ids.data() << std::endl;
    std::cout.flush();

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
    
    // ========== REMOVED: batch_prep pre-allocation (CAUSES CRASH) ==========
    // batch_prep_target_ids vector is corrupted BEFORE initTrainingState() is called
    // (capacity = garbage value 1113289828100734976).
    // Something is writing over TrainingState memory. 
    // WORKAROUND: Let prepareLossBatchInputs() allocate on first use.
    // The 0.02s allocation cost is acceptable vs crash.
    // =========================================================================
    training_state_.batch_prep_capacity = 0;  // Signal to prepareLossBatchInputs: allocate on first use
    
    // BUG FIX: Set kv_cache_capacity for inference sampling during training
    // Previously missing - caused forwardInit() to fail with capacity=0
    training_state_.kv_cache_capacity = static_cast<int>(max_seq_len_cache);
    training_state_.kv_cache_len = 0;  // Start with empty cache
    
    // BUG FIX: Allocate single-token inference buffers for training-time sampling
    // These are required by forwardInit() for incremental generation during training
    // Rule 20: Fail loud if allocation fails - no silent fallbacks
    
    // Use BSM shape with dim=1 for 1D vectors
    training_state_.single_token_hidden = Tensor::empty(
        TensorContract::TensorShape::make_BSM(cfg.d_model, 1), false, primary_stream);
    
    training_state_.single_token_logits = Tensor::empty(
        TensorContract::TensorShape::make_BSM(cfg.vocab_size, 1), false, primary_stream);
    
    std::cout << "✓ Allocated single-token inference buffers for training sampling" << std::endl;
    
    std::cout << "📊 Allocating activation caches for max_tokens=" << max_tokens
              << " (batch=" << max_batch_size << ", seq_len=" << max_seq_len_cache << ")" << std::endl;
    
    // Embedding cache - now uses Tensor
    training_state_.cached_embeddings_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
    
    // Per-layer encoder activation caches using new Tensor-based architecture
    // GQA: K and V caches use num_kv_heads, Q uses num_heads
    const int head_dim_cache = cfg.head_dim;  // Use pre-computed value from config
    const int kv_dim_cache = training_state_.num_kv_heads * head_dim_cache;
    
    const size_t softmax_lse_elems = static_cast<size_t>(max_batch_size) *
                                     static_cast<size_t>(cfg.num_heads) *
                                     static_cast<size_t>(max_seq_len_cache);

    // Initialize encoder_layer_caches vector with Tensors for each layer
    training_state_.encoder_layer_caches.resize(cfg.num_layers);
    
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        auto& cache = training_state_.encoder_layer_caches[layer];
        
        // BSM layout for [batch*seq, feature_dim] shaped tensors
        cache.ln1_output = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
        cache.attn_input = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
        cache.attn_output = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
        cache.residual1 = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
        cache.ln2_output = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
        cache.ffn_pre_gelu = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_ff), false, primary_stream);
        cache.ffn_output = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_ff), false, primary_stream);
        cache.layer_output = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
        
        // BHSD layout for attention tensors [batch, heads, seq, head_dim]
        cache.attn_bhsd = Tensor::empty(
            TensorContract::TensorShape::make_BHSD(max_batch_size, cfg.num_heads, max_seq_len_cache, head_dim_cache), false, primary_stream);
        cache.softmax_lse = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_batch_size * cfg.num_heads, max_seq_len_cache), false, primary_stream);
        
        // Q/K/V caches - Q uses full num_heads, K/V use num_kv_heads (GQA)
        cache.Q = Tensor::empty(
            TensorContract::TensorShape::make_BHSD(max_batch_size, cfg.num_heads, max_seq_len_cache, head_dim_cache), false, primary_stream);
        cache.K = Tensor::empty(
            TensorContract::TensorShape::make_BHSD(max_batch_size, training_state_.num_kv_heads, max_seq_len_cache, head_dim_cache), false, primary_stream);
        cache.V = Tensor::empty(
            TensorContract::TensorShape::make_BHSD(max_batch_size, training_state_.num_kv_heads, max_seq_len_cache, head_dim_cache), false, primary_stream);
    }
    
    // Output layer caches - using Tensor API
    training_state_.cached_encoder_output = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
    
    // Issue #33: Final RMSNorm input cache for backward pass
    // Stores encoder output BEFORE final norm (used to compute gamma gradients)
    training_state_.cached_final_rms_input = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_tokens, cfg.d_model), false, primary_stream);
    
    // Allocate logits cache with LOGITS layout tracking (TensorContract integration)
    training_state_.cached_logits_tensor = Tensor::empty(
        TensorContract::TensorShape::make_LOGITS(max_logit_tokens, cfg.vocab_size), false, primary_stream);
    std::cout << "✓ Allocated cached_logits [" << max_logit_tokens << " x " << cfg.vocab_size << "] LOGITS layout" << std::endl;

    if (cfg.numeric_head_enabled) {
        training_state_.cached_numeric_predictions = Tensor::empty(
            TensorContract::TensorShape::make_BSM(max_logit_tokens, 1), false, primary_stream);
    }
    
    training_state_.cached_targets_tensor = Tensor::empty(
        TensorContract::TensorShape::make_BSM(max_logit_tokens, 1), false, primary_stream);
    
    // BUG FIX: Token IDs cache must be sized by max_tokens (full cache capacity)
    // not max_logit_tokens. Inference sampling requires full buffer.
    // Rule 20: Use Tensor API instead of raw cudaMalloc
    training_state_.cached_token_ids_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad for token IDs
        primary_stream
    );
    std::cout << "✓ Allocated token IDs cache (Tensor API) [" << max_tokens << "]" << std::endl;
    
    // BUG FIX: Numeric buffers must be sized by max_tokens (full cache capacity)
    // not max_logit_tokens (training optimization). Inference sampling requires
    // the full buffer for sequences up to max_cached_seq_len.
    // BUG FIX: Always allocate numeric/text buffers even when ScratchBlock is disabled
    // because prepareLossBatchInputs() always populates these fields from tokenizer
    // Rule 20: Use Tensor API instead of raw cudaMalloc
    training_state_.cached_token_numeric_values = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad
        primary_stream
    );
    std::cout << "✓ Allocated token numeric values cache (Tensor API)" << std::endl;
    
    training_state_.cached_token_numeric_mask = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad
        primary_stream
    );
    std::cout << "✓ Allocated token numeric mask cache (Tensor API)" << std::endl;

    // GRMT v4: Allocate text feature buffers - Rule 20: Tensor API
    constexpr int kTextFeatureDim = 16;  // Must match GRIM::Tokenizer::kTextFeatureDim
    training_state_.cached_token_text_features = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_tokens), kTextFeatureDim),
        false,  // no grad
        primary_stream
    );
    std::cout << "✓ Allocated token text features cache (Tensor API)" << std::endl;
    
    training_state_.cached_token_text_mask = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, static_cast<int>(max_tokens)),
        false,  // no grad
        primary_stream
    );
    std::cout << "✓ Allocated token text mask cache (Tensor API)" << std::endl;
    
    std::cout << "✓ Allocated numeric/text feature buffers (" 
              << (max_tokens * (sizeof(float) + sizeof(uint8_t) + kTextFeatureDim * sizeof(uint16_t) + sizeof(uint8_t)) / 1024 / 1024) 
              << " MB)" << std::endl;

    // Allocate entropy output buffer (per-layer, per-batch, per-head)
    // Size: num_layers * max_batch_size * num_heads
    const size_t entropy_size = cfg.num_layers * max_batch_size * cfg.num_heads;
    training_state_.d_entropy_output = Tensor::empty(
        TensorContract::TensorShape::make_BSM(static_cast<int>(entropy_size), 1), false, primary_stream);
    std::cout << "📊 Allocated entropy output buffer: " << entropy_size 
              << " floats (" << (entropy_size * sizeof(float) / 1024.0 / 1024.0) << " MB)" << std::endl;

    training_state_.sequence_weights_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_batch_size), 1), false, primary_stream);
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
        false, grad_stream);
    training_state_.grad_logits_tensor.name = "grad_logits";
    std::cout << "✓ Allocated grad_logits_tensor [" << max_logit_tokens << " x " << cfg.vocab_size << "] LOGITS layout" << std::endl;

    // grad_numeric: [max_logit_tokens] for numeric head
    if (cfg.numeric_head_enabled) {
        training_state_.grad_numeric_tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(static_cast<int>(max_logit_tokens), 1),
            false, grad_stream);
        training_state_.grad_numeric_tensor.name = "grad_numeric";
    }
    
    // grad_encoder: [max_tokens, d_model]
    training_state_.grad_encoder_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_tokens), cfg.d_model),
        false, grad_stream);
    training_state_.grad_encoder_tensor.name = "grad_encoder_out";
    
    const size_t tokens_per_batch = max_batch_size * max_seq_len_cache;
    EncodingConfig enc_cfg{};
    enc_cfg.d_model = cfg.d_model;
    enc_cfg.num_heads = cfg.num_heads;
    enc_cfg.num_kv_heads = training_state_.num_kv_heads;  // Use calculated GQA value from training_state
    enc_cfg.d_ff = cfg.d_ff;  // Use actual d_ff from config
    enc_cfg.rms_epsilon = 1e-5f;
    enc_cfg.causal_mask = cfg.causal_mask;
    enc_cfg.use_flash_attention = true;
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

    training_state_.encoder_workspace = Tensor::empty(
        TensorContract::TensorShape::make_BSM(static_cast<int>(workspace_bytes / sizeof(float)), 1), false, primary_stream);
    training_state_.encoder_workspace_size = workspace_bytes;

    // ═══════════════════════════════════════════════════════════════
    //  ENCODER BACKWARD TEMPORARIES (Issue #45 FIX: Tensor allocation)
    // ═══════════════════════════════════════════════════════════════
    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    const int max_tokens_int = static_cast<int>(max_tokens);
    
    // FFN backward temporaries
    training_state_.grad_ffn_input_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_ffn_input_tensor.name = "grad_ffn_input";
    
    training_state_.grad_ffn_hidden_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_ff),
        false, grad_stream);
    training_state_.grad_ffn_hidden_tensor.name = "grad_ffn_hidden";
    
    // Attention backward temporaries (model-width)
    training_state_.grad_attn_input_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_attn_input_tensor.name = "grad_attn_input";
    
    training_state_.grad_attn_out_proj_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_attn_out_proj_tensor.name = "grad_attn_out_before_proj";
    
    training_state_.grad_attn_out_bhsd_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_attn_out_bhsd_tensor.name = "grad_attn_out_reshaped";
    
    // QKV gradients (need 4D shape for attention, but stored flat for now)
    // Full shape: [batch, heads, seq, head_dim] - using BSM as [tokens, d_model]
    training_state_.grad_q_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_q_tensor.name = "grad_Q";
    
    training_state_.grad_k_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_k_tensor.name = "grad_K";
    
    training_state_.grad_v_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_v_tensor.name = "grad_V";
    
    // QKV fused [tokens, 3*d_model]
    training_state_.grad_qkv_concat_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_QKV_FUSED(max_tokens_int, 3 * cfg.d_model),
        false, grad_stream);
    training_state_.grad_qkv_concat_tensor.name = "grad_qkv_concat";
    
    training_state_.grad_qkv_input_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_qkv_input_tensor.name = "grad_qkv_input";
    
    // DEDICATED scratch buffer for attention output BSM conversion (W_o gradient computation)
    // CRITICAL: Do NOT reuse grad_qkv_input - prevents temporal aliasing bugs
    training_state_.grad_attn_bsm_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, cfg.d_model),
        false, grad_stream);
    training_state_.grad_attn_bsm_tensor.name = "grad_attn_bsm_scratch";
    
    // Issue #43 FIX: Centering scratch buffer for encoder weight gradients
    // Size: max(d_model, d_ff) to handle both model and FFN width activations
    const int centering_width = std::max(cfg.d_model, cfg.d_ff);
    training_state_.centering_scratch_tensor = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(max_tokens_int, centering_width),
        false, grad_stream);
    training_state_.centering_scratch_tensor.name = "centered_activation_scratch";
    std::cout << "📐 Issue #43: Allocated centering scratch tensor (" 
              << (training_state_.centering_scratch_tensor.numel() * sizeof(float) / (1024*1024)) << " MB)" << std::endl;

    // Rule 20: NO BACKWARDS COMPATIBILITY - callers must use tensor.data directly
    // Removed raw pointer alias assignments

    // Flash Attention workspace allocation (using Tensor API)
    const size_t fa_dq_accum_elems = flash_attn_dq_accum_bytes(
        static_cast<int>(max_batch_size),
        static_cast<int>(max_seq_len_cache),
        cfg.num_heads,
        head_dim) / sizeof(float);  // Convert bytes to float elems
    const size_t fa_dsoftmax_sum_elems = flash_attn_dsoftmax_sum_bytes(
        static_cast<int>(max_batch_size),
        static_cast<int>(max_seq_len_cache),
        cfg.num_heads) / sizeof(float);

    if (fa_dq_accum_elems > 0) {
        training_state_.fa_dq_accum = Tensor::zeros(
            TensorContract::TensorShape::make_BHSD(
                static_cast<int>(max_batch_size), cfg.num_heads, 
                static_cast<int>(max_seq_len_cache), head_dim),
            false, primary_stream);
    }
    if (fa_dsoftmax_sum_elems > 0) {
        training_state_.fa_dsoftmax_sum = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(
                static_cast<int>(max_batch_size * cfg.num_heads),
                static_cast<int>(max_seq_len_cache)),
            false, primary_stream);
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

    // BF16 buffers for FlashAttention v2 - using Tensor API with BSHD layout
    // Note: Tensors use float but these need bf16 - must cast at usage site
    training_state_.fa_q_bf16 = Tensor::empty(
        TensorContract::TensorShape::make_BSHD(
            static_cast<int>(max_batch_size), static_cast<int>(max_seq_len_cache),
            cfg.num_heads, head_dim), false, primary_stream);
    training_state_.fa_k_bf16 = Tensor::empty(
        TensorContract::TensorShape::make_BSHD(
            static_cast<int>(max_batch_size), static_cast<int>(max_seq_len_cache),
            training_state_.num_kv_heads, head_dim), false, primary_stream);
    training_state_.fa_v_bf16 = Tensor::empty(
        TensorContract::TensorShape::make_BSHD(
            static_cast<int>(max_batch_size), static_cast<int>(max_seq_len_cache),
            training_state_.num_kv_heads, head_dim), false, primary_stream);
    training_state_.fa_out_bf16 = Tensor::empty(
        TensorContract::TensorShape::make_BSHD(
            static_cast<int>(max_batch_size), static_cast<int>(max_seq_len_cache),
            cfg.num_heads, head_dim), false, primary_stream);
    training_state_.fa_dout_bf16 = Tensor::empty(
        TensorContract::TensorShape::make_BSHD(
            static_cast<int>(max_batch_size), static_cast<int>(max_seq_len_cache),
            cfg.num_heads, head_dim), false, primary_stream);
    training_state_.fa_dq_bf16 = Tensor::empty(
        TensorContract::TensorShape::make_BSHD(
            static_cast<int>(max_batch_size), static_cast<int>(max_seq_len_cache),
            cfg.num_heads, head_dim), false, primary_stream);
    training_state_.fa_dk_bf16 = Tensor::empty(
        TensorContract::TensorShape::make_BSHD(
            static_cast<int>(max_batch_size), static_cast<int>(max_seq_len_cache),
            training_state_.num_kv_heads, head_dim), false, primary_stream);
    training_state_.fa_dv_bf16 = Tensor::empty(
        TensorContract::TensorShape::make_BSHD(
            static_cast<int>(max_batch_size), static_cast<int>(max_seq_len_cache),
            training_state_.num_kv_heads, head_dim), false, primary_stream);
    
    // Loss scratch buffers using Tensor API (Rule 20: no raw cudaMalloc)
    training_state_.d_loss_scratch = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(static_cast<int>(max_logit_tokens), 1),
        false, primary_stream);
    training_state_.d_loss_sum_scratch = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(1, 1),  // Scalar
        false, primary_stream);

    if (cfg.numeric_head_enabled) {
        training_state_.d_numeric_loss_sum = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, 1),  // Scalar
            false, primary_stream);
        training_state_.d_numeric_loss_count = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, 1),  // Scalar (int stored as float, cast at usage)
            false, primary_stream);
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
        
        // Allocate activation caches for ScratchBlock forward/backward pass (Rule 20: Tensor API)
        const size_t atom_emb_size = cfg.scratch_block_max_atoms * cfg.scratch_block_atom_embedding_dim;
        training_state_.cached_scratch_block_embeddings = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(
                static_cast<int>(cfg.scratch_block_max_atoms),
                static_cast<int>(cfg.scratch_block_atom_embedding_dim)),
            false, primary_stream);
        training_state_.cached_scratch_block_positions = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(static_cast<int>(cfg.scratch_block_max_atoms), 1),
            false, primary_stream);  // int32 stored as float, cast at usage
        training_state_.cached_scratch_block_types = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(static_cast<int>(cfg.scratch_block_max_atoms), 1),
            false, primary_stream);  // int32 stored as float, cast at usage
        training_state_.cached_scratch_block_num_atoms = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, 1),
            false, primary_stream);  // Scalar int32 stored as float
        
        std::cout << "✓ ScratchBlock reasoning layer initialized (d_model="
                  << cfg.d_model << ", atom_dim=" << cfg.scratch_block_atom_embedding_dim
                  << ", max_atoms=" << cfg.scratch_block_max_atoms << ")" << std::endl;
    } else {
        std::cout << "ℹ ScratchBlock reasoning layer disabled" << std::endl;
        scratch_block_layer_ = nullptr;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    //  STEP FINAL: Verify Autograd Tensors were initialized
    //  Phase1_Startup (step 2.75) already called initializeAutogradTensors() to
    //  set up TrainingTensors with proper memory ownership. Here we just verify
    //  the flag is set and print confirmation.
    //  
    //  WHY THIS MOVED: Previously called initializeAutogradTensors() here, but
    //  that caused problems because EmbeddingRuntime already allocated memory.
    //  Now tensors_ OWNS memory BEFORE initGPU(), so no duplicate allocation.
    // ═══════════════════════════════════════════════════════════════════════════
    if (!training_state_.use_autograd_tensors || !training_state_.tensors_) {
        throw std::runtime_error(
            "[InitTrainingState] FATAL: use_autograd_tensors not enabled! "
            "Phase1_Startup should have called initializeAutogradTensors() in step 2.75. "
            "This indicates incorrect initialization order."
        );
    }
    std::cout << "✓ Verified: Autograd tensor system already enabled (from Phase1_Startup)" << std::endl;
    
    training_state_.initialized = true;
    std::cout << "✓ Training state initialized with full gradient buffers" << std::endl;
    std::cout << "[InitTrainingState] max_cached_batch=" << training_state_.max_cached_batch
              << " max_cached_seq_len=" << training_state_.max_cached_seq_len
              << " max_cached_tokens=" << training_state_.max_cached_tokens
              << " max_logit_tokens=" << training_state_.max_logit_tokens << std::endl;
}

#endif // USE_CUDA

} // namespace GRIM

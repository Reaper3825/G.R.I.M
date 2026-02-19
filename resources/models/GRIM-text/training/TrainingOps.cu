#define USE_CUDA

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Common/grim_scale_buffer.hpp"
#include "../Shared/StreamController/StreamController_GPU.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"
#include "module_logger.hpp"

namespace {
using ForwardLog = ModuleLogger<GRIM::Logging::ModuleId::ForwardPass>;

inline void cudaFail(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

inline void cudaFailLast(const char* where) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

} // namespace

namespace GRIM {

#ifdef USE_CUDA

// Legacy computeLoss() function DELETED per Rule 20 (backwards compatibility forbidden).
// Production code uses computeLossBatch() which uses autograd loss (AutogradLoss.cu).
// This function was using deleted ComputeLossHost_GPU which relied on deleted UnifiedLoss_GPU.

LanguageModel::ModelStats LanguageModel::getModelStats() const {
    ModelStats stats;

    // Count actual allocated parameters from parameter groups (ground truth)
    for (const auto& group : parameter_groups_) {
        if (group.name == "embedding" || group.name == "embedding_lm_head_tied") {
            stats.embedding_params += group.size();
        } else if (group.name == "position_embedding") {
            stats.position_embedding_params += group.size();
        } else if (group.name.find("lm_head") != std::string::npos) {
            stats.lm_head_params += group.size();
        } else {
            stats.encoder_params += group.size();
        }
    }

    if (parameter_groups_.empty()) {
        const auto& cfg = config_;
        const int head_dim = cfg.head_dim;
        const int kv_dim = cfg.num_kv_heads * head_dim;
        const int total_qkv_dim = cfg.d_model + 2 * kv_dim;

        stats.embedding_params = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;

        if (!cfg.tie_embeddings) {
            stats.lm_head_params = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
        }
        if (cfg.use_bias) {
            stats.lm_head_params += cfg.vocab_size;  // lm_head_bias
        }

        size_t per_layer = 0;
        per_layer += static_cast<size_t>(total_qkv_dim) * cfg.d_model;  // W_qkv
        per_layer += static_cast<size_t>(cfg.d_model) * cfg.d_model;    // W_o
        per_layer += static_cast<size_t>(cfg.d_model) * cfg.d_ff;       // W1
        per_layer += static_cast<size_t>(cfg.d_ff) * cfg.d_model;       // W2
        per_layer += cfg.d_model; // RMSNorm1 gamma
        per_layer += cfg.d_model; // RMSNorm2 gamma
        // Issue #148: Sandwich norm gammas REMOVED (no post-residual normalization)
        if (cfg.use_bias) {
            per_layer += total_qkv_dim;  // b_qkv
            per_layer += cfg.d_model;    // b_o
            per_layer += cfg.d_ff;       // b1
            per_layer += cfg.d_model;    // b2
        }

        stats.encoder_params = per_layer * cfg.num_layers;
        stats.encoder_params += cfg.d_model;  // Final RMSNorm gamma
    } else {
        const auto& cfg = config_;
        const int head_dim = cfg.head_dim;
        const int kv_dim = cfg.num_kv_heads * head_dim;
        const int total_qkv_dim = cfg.d_model + 2 * kv_dim;

        TensorContract::GQADims gqa_dims{cfg.num_heads, cfg.num_kv_heads, head_dim};
        if (!gqa_dims.is_valid()) {
            std::cerr << "[WARNING] TensorContract GQA validation failed!\n";
            std::cerr << "  d_model=" << cfg.d_model
                      << " num_heads=" << cfg.num_heads
                      << " head_dim=" << head_dim << "\n";
            assert(false && "GQA dimension validation failed");
        }

        const int expected_qkv_dim = gqa_dims.total_qkv_dim();
        if (total_qkv_dim != expected_qkv_dim) {
            std::cerr << "[WARNING] QKV dimension mismatch: computed=" << total_qkv_dim
                      << " expected=" << expected_qkv_dim << "\n";
            assert(false && "QKV dimension formula drift from TensorContract");
        }

        const size_t est_embedding = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;

        // Position embeddings: only when actually allocated (ALiBi/RoPE = 0)
        const size_t est_position_embedding = stats.position_embedding_params;

        // LM head: weight (0 when tied — counted under embedding) + bias when use_bias
        size_t est_lm_head = cfg.tie_embeddings ? 0 : static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
        if (cfg.use_bias) {
            est_lm_head += cfg.vocab_size;  // lm_head_bias [vocab_size]
        }

        // Per-layer encoder: weights + biases + pre-norm
        size_t per_layer = 0;
        per_layer += static_cast<size_t>(total_qkv_dim) * cfg.d_model;  // W_qkv
        per_layer += static_cast<size_t>(cfg.d_model) * cfg.d_model;    // W_o
        per_layer += static_cast<size_t>(cfg.d_model) * cfg.d_ff;       // W1
        per_layer += static_cast<size_t>(cfg.d_ff) * cfg.d_model;       // W2
        per_layer += cfg.d_model;  // RMSNorm1 gamma (pre-attn)
        per_layer += cfg.d_model;  // RMSNorm2 gamma (pre-ffn)
        // Issue #148: Sandwich norm gammas REMOVED (no post-residual normalization)
        if (cfg.use_bias) {
            per_layer += total_qkv_dim;  // b_qkv
            per_layer += cfg.d_model;    // b_o
            per_layer += cfg.d_ff;       // b1
            per_layer += cfg.d_model;    // b2
        }

        size_t est_encoder = per_layer * cfg.num_layers;
        est_encoder += cfg.d_model; // Final RMSNorm gamma (not per-layer)

        // ScratchBlock params are classified as encoder_params by parameter_groups
        if (cfg.use_scratch_block) {
            constexpr int NUM_ATOM_TYPES = HyperParameters::NUM_ATOM_TYPES;
            const int atom_dim = cfg.scratch_block_atom_embedding_dim;
            constexpr int kTextFeatureDim = 16;  // Matches ScratchBlockReasoning_GPU.cu
            est_encoder += static_cast<size_t>(NUM_ATOM_TYPES) * atom_dim;  // atom_type_embeddings
            est_encoder += static_cast<size_t>(atom_dim) * cfg.d_model;     // atom_projection
            est_encoder += static_cast<size_t>(kTextFeatureDim) * cfg.d_model; // text_feature_projection
            est_encoder += cfg.d_model;  // value_extraction_weight
            est_encoder += 1;            // value_extraction_bias
        }

        const size_t est_total = est_embedding + est_position_embedding + est_encoder + est_lm_head;
        const size_t actual_total =
            stats.embedding_params + stats.position_embedding_params + stats.encoder_params +
            stats.lm_head_params;

        if (actual_total > 0) {
            const float drift_pct =
                100.0f * std::abs(static_cast<int64_t>(est_total - actual_total)) /
                static_cast<float>(actual_total);

            if (drift_pct > 0.1f) {
                std::cerr << "[WARNING] getModelStats formula drift detected: "
                          << "actual=" << actual_total << " estimate=" << est_total
                          << " (" << drift_pct << "%)\n";
                std::cerr << "  Embedding: actual=" << stats.embedding_params << " est=" << est_embedding << "\n";
                std::cerr << "  Pos Embed: actual=" << stats.position_embedding_params << " est=" << est_position_embedding << "\n";
                std::cerr << "  Encoder:   actual=" << stats.encoder_params << " est=" << est_encoder << "\n";
                std::cerr << "  LM Head:   actual=" << stats.lm_head_params << " est=" << est_lm_head << "\n";
                assert(false && "Parameter count formula drifted from actual allocations");
            }
        }
    }

    stats.total_params = stats.embedding_params + stats.position_embedding_params + stats.encoder_params +
                         stats.lm_head_params;

    // ScratchBlock stats for reporting (already counted in encoder_params via parameter_groups)
    if (config_.use_scratch_block) {
        constexpr int NUM_ATOM_TYPES = HyperParameters::NUM_ATOM_TYPES;
        const int atom_dim = config_.scratch_block_atom_embedding_dim;
        constexpr int kTextFeatureDim = 16;
        stats.scratchblock_params =
            static_cast<size_t>(NUM_ATOM_TYPES) * atom_dim +
            static_cast<size_t>(atom_dim) * config_.d_model +
            static_cast<size_t>(kTextFeatureDim) * config_.d_model +
            config_.d_model +  // value_extraction_weight
            1;                 // value_extraction_bias
        // NOTE: scratchblock_params NOT added to total_params — already in encoder_params
    }

    stats.model_size_mb = (stats.total_params * sizeof(float)) / (1024.0f * 1024.0f);
    return stats;
}

//======================================================//
//  GPU Initialization in Constructor
//======================================================//

void LanguageModel::initGPU() {
    const auto& cfg = getConfig();

    // NOTE: Cannot use ForwardLog here - LogRecorder not initialized yet during Phase1 startup.
    std::cout << "[initGPU] Entry, use_gpu=" << (cfg.use_gpu ? "true" : "false") << std::endl;
    if (!cfg.use_gpu) {
        throw std::runtime_error("[initGPU] use_gpu=false but GRIM-text REQUIRES GPU - fix ai_config.json");
    }

    try {
        std::cout << "[initGPU] Initializing GPU-accelerated transformer layers..." << std::endl;

        //======================================================//
        //  1) Initialize CUDA device (must be first CUDA work)
        //======================================================//
        int device_count = 0;
        cudaFail(cudaGetDeviceCount(&device_count), "[initGPU] cudaGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("No CUDA devices found - GPU backend required");
        }

        cudaFail(cudaSetDevice(0), "[initGPU] cudaSetDevice(0)");

        cudaDeviceProp prop{};
        cudaError_t prop_err = cudaGetDeviceProperties(&prop, 0);
        if (prop_err == cudaSuccess) {
            std::cout << "✓ CUDA Device initialized: " << prop.name << "\n";
            std::cout << "  - Compute capability: " << prop.major << "." << prop.minor << "\n";
            std::cout << "  - Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        } else {
            std::cout << "⚠ Failed to query device properties: " << cudaGetErrorString(prop_err) << "\n";
        }

        //======================================================//
        //  2) Stream + cuBLAS prerequisites
        //======================================================//
        if (!training_state_.stream_ctrl.isInitialized()) {
            throw std::runtime_error(
                "FATAL: StreamController not initialized. "
                "Initialize stream_ctrl before initGPU() (TrainingState owns streams)."
            );
        }

        cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
        StreamController::fatalIfDefaultStream(primary_stream, "LanguageModel::initGPU");

        if (!training_state_.cublas_handle) {
            throw std::runtime_error("FATAL: cuBLAS handle not initialized (training_state_.cublas_handle == NULL)");
        }

        //======================================================//
        //  3) Build GPU encoder
        //======================================================//
        EncoderConfig enc_config;
        enc_config.d_model = cfg.d_model;
        enc_config.num_heads = cfg.num_heads;
        enc_config.num_kv_heads = cfg.num_kv_heads;
        enc_config.head_dim = cfg.head_dim;
        enc_config.d_ff = cfg.d_ff;
        enc_config.num_layers = cfg.num_layers;
        enc_config.dropout_rate = cfg.dropout_rate;
        enc_config.attention_dropout = cfg.attention_dropout;
        enc_config.use_pre_norm = cfg.use_pre_norm;
        enc_config.use_simd = cfg.use_simd;
        enc_config.num_threads = cfg.num_threads;

        // Must propagate flash attention toggles
        enc_config.use_flash_attention = cfg.use_flash_attention;
        enc_config.min_seq_len_for_flash = cfg.min_seq_len_for_flash;
        enc_config.causal_mask = cfg.causal_mask;
        enc_config.max_seq_len = cfg.max_seq_len;
        enc_config.max_cached_batch = cfg.max_cached_batch;
        enc_config.max_cached_seq_len = cfg.max_cached_seq_len;

        // Issue #109 FIX: Propagate LayerScale config from LanguageModelConfig → EncoderConfig
        enc_config.use_layer_scale = cfg.use_layer_scale;
        enc_config.layer_scale_init = cfg.layer_scale_init;
        enc_config.center_encoder_residuals = cfg.center_encoder_residuals;
        enc_config.use_bias = cfg.use_bias;

        // Pattern B: Enable layer self-allocation. Encoder layers allocate and Xavier-init
        // their own weights in the constructor. No god object involved.
        {
            enc_config.weight_seed = training_state_.weight_init_seed;
            // Issue #142: 1/sqrt(2*num_layers) for residual projection init
            enc_config.residual_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(cfg.num_layers));
            enc_config.layer_scale_init_value = cfg.layer_scale_init;
            fprintf(stdout, "[initGPU] Layers will self-allocate weights "
                    "(seed=%llu, residual_scale=%.6f)\n",
                    static_cast<unsigned long long>(enc_config.weight_seed),
                    enc_config.residual_scale);
        }
        
        enc_config.stream = primary_stream;
        enc_config.cublas_handle = training_state_.cublas_handle;

        // PBM must be initialized before encoder creation
        if (!training_state_.pbm_initialized) {
            throw std::runtime_error(
                "[initGPU] FATAL: PBM not initialized before encoder construction! "
                "Call initPBM() BEFORE createGPUEncoder()");
        }
        enc_config.pos_encoding = &training_state_.pbm_spec;

        std::cout << "✓ Encoder using TrainingState primary stream (handle=" << training_state_.cublas_handle
                  << ", stream=" << primary_stream << ")\n";

        auto* encoder_ptr = new GPUGrimEncoder(enc_config);
        gpu_encoder_.reset(encoder_ptr);

        //======================================================//
        //  5) Verify self-allocated encoder layers are ready
        //======================================================//
        if (!training_state_.seed_initialized_) {
            throw std::runtime_error("[initGPU] FATAL: autograd seed not initialized - Phase1_Startup must call "
                                     "initializeAutogradSeed() BEFORE initGPU()");
        }

        // Layers self-allocated their own weights in the constructor. Verify all are ready.
        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            auto* gpu_layer = encoder_ptr->getLayer(layer);
            if (!gpu_layer || !gpu_layer->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: Encoder layer " + std::to_string(layer) +
                                         " not ready after self-allocation!");
            }
        }
        std::cout << "✓ " << cfg.num_layers << " encoder layers self-allocated weights\n";

        //======================================================//
        //  6a) Build persistent Embedding layer (Pattern B)
        //  
        //  Self-allocates token weights [vocab_size, d_model] + optional
        //  position weights [max_seq_len, d_model] for learned mode.
        //  Must be created BEFORE LMHeadLayer (LM head aliases embedding for tied config).
        //======================================================//
        {
            EmbeddingLayerConfig emb_config;
            emb_config.vocab_size = cfg.vocab_size;
            emb_config.d_model = cfg.d_model;
            emb_config.max_seq_len = cfg.max_seq_len;
            emb_config.positional_encoding = cfg.positional_encoding;
            emb_config.embedding_scale = 1.0f;  // Issue #140: No scaling for ALiBi/RoPE

            // Seed convention: embedding uses weight_init_seed + 0
            const uint64_t emb_seed = training_state_.weight_init_seed;

            embedding_layer_ = std::make_unique<EmbeddingLayer>(emb_config, emb_seed, primary_stream);

            if (!embedding_layer_->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: EmbeddingLayer not ready after construction!");
            }
            std::cout << "✓ Embedding layer created (vocab=" << cfg.vocab_size
                      << ", d_model=" << cfg.d_model
                      << ", pos_emb=" << (embedding_layer_->hasPositionEmbeddings() ? "learned" : "none")
                      << ")\n";
        }

        //======================================================//
        //  6b) Build persistent LM Head layer (Pattern B)
        //  
        //  Self-allocates weights (or aliases embedding for tied config).
        //  Owns final_rms_gamma. Created AFTER EmbeddingLayer (needs embedding
        //  weights for tying) and AFTER encoder (needs stream/cublas).
        //======================================================//
        {
            LMHeadLayerConfig lm_config;
            lm_config.d_model = cfg.d_model;
            lm_config.vocab_size = cfg.vocab_size;
            lm_config.use_bias = cfg.use_bias;
            lm_config.center_hidden_states = cfg.lm_head_center_hidden_states;
            lm_config.project_out_pc1 = cfg.project_out_pc1;
            lm_config.pc1_power_iters = cfg.pc1_power_iters;
            lm_config.center_logits = cfg.center_logits;
            lm_config.has_final_rms_norm = true;
            lm_config.rms_epsilon = cfg.rms_epsilon;
            lm_config.stream = primary_stream;
            lm_config.cublas_handle = training_state_.cublas_handle;

            // Seed convention: lm_head uses weight_init_seed + 1
            const uint64_t lm_head_seed = training_state_.weight_init_seed + 1;

            // For tied weights, pass pointer to embedding token weights (owned by EmbeddingLayer)
            Tensor* tied_emb = cfg.tie_embeddings ? &embedding_layer_->tokenWeights() : nullptr;

            lm_head_layer_ = std::make_unique<LMHeadLayer>(lm_config, lm_head_seed, primary_stream, tied_emb);

            if (!lm_head_layer_->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: LMHeadLayer not ready after construction!");
            }
            std::cout << "✓ LM Head layer created ("
                      << (cfg.tie_embeddings ? "tied to embedding" : "separate weights")
                      << ", final_rms_gamma owned, bias=" << (cfg.use_bias ? "yes" : "no") << ")\n";
        }

        std::cout << "✓ GPU encoder initialized with " << cfg.num_layers << " layers\n";
        std::cout << "  - Attention: GPU-accelerated\n";
        std::cout << "  - FFN: GPU-accelerated with fused GELU\n";
        std::cout << "  - Layer Norm: GPU-accelerated\n";

        if (cfg.use_flash_attention) {
            std::cout << "⚡ Enabling Flash Attention 2...\n";
            encoder_ptr->setFlashAttention(true, cfg.min_seq_len_for_flash);
            std::cout << "✓ Flash Attention enabled (min_seq_len=" << cfg.min_seq_len_for_flash << ")\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ EXCEPTION in initGPU(): " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_CUDA

} // namespace GRIM

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
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Common/grim_scale_buffer.hpp"
#include "../Shared/Loss/NumericLoss/NumericLoss_GPU.hpp"
#include "../Shared/StreamController/StreamController_GPU.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../Shared/Activations/Xavier/Xavier.hpp"
#include "../Shared/TrainingState/TrainingTensors.hpp"  // For proper memory ownership
#include "module_logger.hpp"

// External kernel declaration (C++ linkage - can throw exceptions)
void launchBiasSumGradient(const float* grad_output, float* grad_bias,
                           int total_tokens, int hidden_dim, cudaStream_t stream);

using GRIM::launchXavierInit;

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

// EmbeddingRuntime is managed by destroyEmbeddingRuntime()
struct EmbeddingRuntimeDeleter {
    void operator()(GRIM::EmbeddingRuntime* p) const noexcept {
        if (p) GRIM::destroyEmbeddingRuntime(p);
    }
};
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
            stats.embedding_params += group.size;
        } else if (group.name == "position_embedding") {
            stats.position_embedding_params += group.size;
        } else if (group.name.find("lm_head") != std::string::npos) {
            stats.lm_head_params += group.size;
        } else if (group.name == "numeric_head_weight") {
            stats.numeric_head_params += group.size;
        } else {
            stats.encoder_params += group.size;
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

        if (cfg.numeric_head_enabled) {
            stats.numeric_head_params = static_cast<size_t>(cfg.d_model);
        }

        size_t per_layer = 0;
        per_layer += static_cast<size_t>(total_qkv_dim) * cfg.d_model;
        per_layer += static_cast<size_t>(cfg.d_model) * cfg.d_model;
        per_layer += static_cast<size_t>(cfg.d_model) * cfg.d_ff;
        per_layer += static_cast<size_t>(cfg.d_ff) * cfg.d_model;
        per_layer += cfg.d_model; // RMSNorm1 gamma
        per_layer += cfg.d_model; // RMSNorm2 gamma

        stats.encoder_params = per_layer * cfg.num_layers;
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
        const size_t est_position_embedding = static_cast<size_t>(cfg.max_seq_len) * cfg.d_model;
        const size_t est_lm_head = cfg.tie_embeddings ? 0 : static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
        const size_t est_numeric_head = cfg.numeric_head_enabled ? static_cast<size_t>(cfg.d_model) : 0;

        size_t per_layer = 0;
        per_layer += static_cast<size_t>(total_qkv_dim) * cfg.d_model;
        per_layer += static_cast<size_t>(cfg.d_model) * cfg.d_model;
        per_layer += static_cast<size_t>(cfg.d_model) * cfg.d_ff;
        per_layer += static_cast<size_t>(cfg.d_ff) * cfg.d_model;
        per_layer += cfg.d_model;
        per_layer += cfg.d_model;

        size_t est_encoder = per_layer * cfg.num_layers;
        est_encoder += cfg.d_model; // Final RMSNorm gamma (not per-layer)

        const size_t est_total = est_embedding + est_position_embedding + est_encoder + est_lm_head + est_numeric_head;
        const size_t actual_total =
            stats.embedding_params + stats.position_embedding_params + stats.encoder_params +
            stats.lm_head_params + stats.numeric_head_params;

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
                std::cerr << "  Num Head:  actual=" << stats.numeric_head_params << " est=" << est_numeric_head << "\n";
                assert(false && "Parameter count formula drifted from actual allocations");
            }
        }
    }

    stats.total_params = stats.embedding_params + stats.position_embedding_params + stats.encoder_params +
                         stats.lm_head_params + stats.numeric_head_params;

    if (config_.use_scratch_block) {
        constexpr int NUM_ATOM_TYPES = HyperParameters::NUM_ATOM_TYPES;
        const int atom_embedding_dim = config_.scratch_block_atom_embedding_dim;
        stats.scratchblock_params =
            static_cast<size_t>(NUM_ATOM_TYPES) * atom_embedding_dim +
            static_cast<size_t>(atom_embedding_dim) * config_.d_model;
        stats.total_params += stats.scratchblock_params;
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
    if (!cfg.use_gpu) return;

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
        //  3) Build GPU embedding runtime (RAII-managed)
        //======================================================//
        std::unique_ptr<EmbeddingRuntime, EmbeddingRuntimeDeleter> embedding_runtime(new EmbeddingRuntime());

        // Keep config consistent with your prior behavior
        embedding_runtime->config.vocab_size = cfg.vocab_size;
        embedding_runtime->config.max_position = cfg.max_seq_len;
        embedding_runtime->config.d_model = cfg.d_model;
        embedding_runtime->config.apply_rms_norm = true;
        embedding_runtime->config.rms_epsilon = 1e-5f;

        embedding_runtime->stream = primary_stream;
        embedding_runtime->owns_stream = false; // TrainingState owns it
        embedding_runtime->config.stream = embedding_runtime->stream;

        std::cout << "✓ Embedding runtime using TrainingState primary stream\n";

        // Helper: fail-loud cleanup with message
        auto fail_embedding = [&](const std::string& msg) -> void {
            std::cerr << msg << std::endl;
            embedding_runtime.reset(); // calls destroyEmbeddingRuntime()
            gpu_embedder_.reset();
            throw std::runtime_error("Failed to initialize GPU embeddings");
        };

        const GrimEmbeddingStack& cpu_embedder = *getEmbedderPtr();

        // ═══════════════════════════════════════════════════════════════
        // PROPER OWNERSHIP FIX: Use TrainingTensors if already initialized
        // ═══════════════════════════════════════════════════════════════
        // TrainingTensors owns GPU memory via Tensor::zeros().
        // EmbeddingRuntime now just POINTS to that memory (doesn't own it).
        // This eliminates the Tensor::from_ptr() wrapping pattern.
        
        const bool use_training_tensors = (training_state_.tensors_ != nullptr);
        
        // --- Token embeddings ---
        const size_t token_elements = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
        const size_t token_bytes = token_elements * sizeof(float);

        if (use_training_tensors) {
            // TrainingTensors owns the memory - just point to it (no ownership)
            embedding_runtime->token_buffer = training_state_.tensors_->embedding_weights.data;
            embedding_runtime->owns_token_buffer = false;  // CRITICAL: Don't free on destroy!
            std::cout << "✓ Token embeddings using TrainingTensors memory (no duplicate allocation)\n";
        } else {
            // Legacy path: allocate new memory (takes ownership)
            embedding_runtime->owns_token_buffer = true;
            if (cudaMalloc(&embedding_runtime->token_buffer, token_bytes) != cudaSuccess) {
                fail_embedding("❌ FATAL: Failed to allocate GPU memory for token embeddings");
            }
        }

        embedding_runtime->weights.token_embeddings = TensorContract::TensorView::make_BSM(
            embedding_runtime->token_buffer, cfg.vocab_size, cfg.d_model, "token_embeddings");

        // Load CPU embedder data into the buffer (whether owned by TrainingTensors or self)
        std::vector<float> token_data;
        token_data.reserve(token_elements);
        for (int i = 0; i < cfg.vocab_size; ++i) {
            const auto& row = cpu_embedder.token_embed.rows[i];
            if (row.data.size() != static_cast<size_t>(cfg.d_model)) {
                fail_embedding("❌ FATAL: Token embedding size mismatch");
            }
            token_data.insert(token_data.end(), row.data.begin(), row.data.end());
        }

        cudaFail(cudaMemcpyAsync(embedding_runtime->token_buffer,
                                 token_data.data(),
                                 token_bytes,
                                 cudaMemcpyHostToDevice,
                                 primary_stream),
                 "[initGPU] cudaMemcpyAsync(token_embeddings)");

        // --- Position embeddings ---
        const size_t pos_elements = static_cast<size_t>(cfg.max_seq_len) * cfg.d_model;
        const size_t pos_bytes = pos_elements * sizeof(float);

        if (use_training_tensors) {
            // TrainingTensors owns the memory - just point to it (no ownership)
            embedding_runtime->position_buffer = training_state_.tensors_->position_embedding_weights.data;
            embedding_runtime->owns_position_buffer = false;  // CRITICAL: Don't free on destroy!
            std::cout << "✓ Position embeddings using TrainingTensors memory (no duplicate allocation)\n";
            
            // NOTE: TrainingTensors already did Xavier init, but we still do it here for consistency
            // This overwrites the TrainingTensors Xavier init with the same algorithm
        } else {
            embedding_runtime->owns_position_buffer = true;
            if (cudaMalloc(&embedding_runtime->position_buffer, pos_bytes) != cudaSuccess) {
                fail_embedding("❌ FATAL: Failed to allocate GPU memory for positional encodings");
            }
        }

        embedding_runtime->weights.position_embeddings = TensorContract::TensorView::make_BSM(
            embedding_runtime->position_buffer, cfg.max_seq_len, cfg.d_model, "position_embeddings");

        // Learned position embeddings: Xavier init on GPU
        const float pos_embedding_stddev =
            std::sqrt(2.0f / static_cast<float>(cfg.max_seq_len + cfg.d_model));

        std::cout << "🎲 Initializing position embeddings on GPU (stddev=" << pos_embedding_stddev << ")\n";

        launchXavierInit(embedding_runtime->position_buffer,
                         static_cast<int>(pos_elements),
                         pos_embedding_stddev,
                         43,
                         primary_stream);
        cudaFailLast("[initGPU] launchXavierInit(position_embeddings)");

        // --- RMSNorm gamma ---
        const size_t ln_bytes = static_cast<size_t>(cfg.d_model) * sizeof(float);

        if (use_training_tensors && training_state_.tensors_->final_rms_gamma.data) {
            // TrainingTensors owns the memory (no ownership)
            embedding_runtime->gamma_buffer = training_state_.tensors_->final_rms_gamma.data;
            embedding_runtime->owns_gamma_buffer = false;  // CRITICAL: Don't free on destroy!
            std::cout << "✓ RMSNorm gamma using TrainingTensors memory (no duplicate allocation)\n";
        } else {
            embedding_runtime->owns_gamma_buffer = true;
            if (cudaMalloc(&embedding_runtime->gamma_buffer, ln_bytes) != cudaSuccess) {
                fail_embedding("❌ FATAL: Failed to allocate GPU memory for RMSNorm gamma");
            }
        }

        embedding_runtime->weights.gamma = TensorContract::TensorView::make_BSM(
            embedding_runtime->gamma_buffer, 1, cfg.d_model, "rmsnorm_gamma");

        cudaFail(cudaMemcpyAsync(embedding_runtime->gamma_buffer,
                                 cpu_embedder.rms_gamma.data.data(),
                                 ln_bytes,
                                 cudaMemcpyHostToDevice,
                                 primary_stream),
                 "[initGPU] cudaMemcpyAsync(rmsnorm_gamma)");

        // Ensure embedding uploads complete before exposing runtime
        cudaFail(cudaStreamSynchronize(primary_stream), "[initGPU] cudaStreamSynchronize(embedding uploads)");

        gpu_embedder_.reset(embedding_runtime.release()); // transfer ownership to class’ unique_ptr (existing design)
        ForwardLog::info("✓ GPU embeddings initialized");

        //======================================================//
        //  4) Build GPU encoder
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

        enc_config.stream = primary_stream;
        enc_config.cublas_handle = training_state_.cublas_handle;

        // PBM must be initialized before encoder creation
        if (!training_state_.pbm_initialized) {
            fprintf(stderr, "\n[LanguageModel::initGPU] FATAL: PBM not initialized before encoder construction!\n");
            fprintf(stderr, "[LanguageModel::initGPU] Call initPBM() BEFORE createGPUEncoder().\n");
            fprintf(stderr, "[LanguageModel::initGPU] Fix initialization order in TrainingOps.cu\n");
            std::abort();
        }
        enc_config.pos_encoding = &training_state_.pbm_spec;

        std::cout << "✓ Encoder using TrainingState primary stream (handle=" << training_state_.cublas_handle
                  << ", stream=" << primary_stream << ")\n";

        auto* encoder_ptr = new GPUGrimEncoder(enc_config);
        gpu_encoder_.reset(encoder_ptr);

        //======================================================//
        //  5) Initialize encoder weights (Xavier + GPT-2 residual scaling)
        //======================================================//
        const float residual_scale = 1.0f / std::sqrt(2.0f * cfg.num_layers);

        std::cout << "🎲 Initializing encoder layer weights...\n";
        std::cout << "   Using GPT-2 residual scaling: " << residual_scale << "\n";

        for (int layer = 0; layer < cfg.num_layers; ++layer) {
            auto* gpu_layer = encoder_ptr->getLayer(layer);
            if (!gpu_layer) {
                ForwardLog::error("    ERROR: Could not get layer " + std::to_string(layer) + " for initialization!");
                continue;
            }

            // Attention weights
            {
                float* w_qkv_ptr = gpu_layer->getAttnWqkv();
                float* w_o_ptr = gpu_layer->getAttnWo();

                if (!w_qkv_ptr || !w_o_ptr) {
                    ForwardLog::error("    ERROR: Null attention weight pointers!");
                } else {
                    TensorContract::GQADims gqa_dims{cfg.num_heads, cfg.num_kv_heads, cfg.head_dim};
                    const int total_qkv_dim = gqa_dims.total_qkv_dim();
                    const size_t qkv_size = static_cast<size_t>(total_qkv_dim) * cfg.d_model;

                    const float xavier_qkv_stddev =
                        std::sqrt(2.0f / (cfg.d_model + static_cast<float>(total_qkv_dim)));

                    launchXavierInit(w_qkv_ptr,
                                     static_cast<int>(qkv_size),
                                     xavier_qkv_stddev,
                                     42 + layer * 4,
                                     primary_stream);
                    cudaFailLast("[initGPU] launchXavierInit(W_qkv)");

                    const float xavier_wo_stddev = xavier_qkv_stddev * residual_scale;
                    launchXavierInit(w_o_ptr,
                                     cfg.d_model * cfg.d_model,
                                     xavier_wo_stddev,
                                     43 + layer * 4,
                                     primary_stream);
                    cudaFailLast("[initGPU] launchXavierInit(W_o)");
                }
            }

            // FFN weights
            {
                float* w1_ptr = gpu_layer->getFFNW1();
                float* w2_ptr = gpu_layer->getFFNW2();

                if (!w1_ptr || !w2_ptr) {
                    ForwardLog::error("    ERROR: Null FFN weight pointers!");
                } else {
                    const float xavier_w1_stddev =
                        std::sqrt(2.0f / (cfg.d_model + cfg.d_ff));

                    launchXavierInit(w1_ptr,
                                     cfg.d_model * cfg.d_ff,
                                     xavier_w1_stddev,
                                     44 + layer * 4,
                                     primary_stream);
                    cudaFailLast("[initGPU] launchXavierInit(W1)");

                    const float xavier_w2_stddev =
                        std::sqrt(2.0f / (cfg.d_ff + cfg.d_model)) * residual_scale;

                    launchXavierInit(w2_ptr,
                                     cfg.d_ff * cfg.d_model,
                                     xavier_w2_stddev,
                                     45 + layer * 4,
                                     primary_stream);
                    cudaFailLast("[initGPU] launchXavierInit(W2)");
                }
            }

            // LayerNorm/RMSNorm gamma init handled by layer constructors (per your architecture)
        }

        // Stream sync (NOT device-wide) to guarantee weights ready
        cudaFail(cudaStreamSynchronize(primary_stream), "[initGPU] cudaStreamSynchronize(weight init)");

        {
            std::ostringstream oss;
            oss << "✓ Encoder layer weights initialized with Xavier (layers 0-" << (cfg.num_layers - 1) << ")";
            std::cout << oss.str() << "\n";
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

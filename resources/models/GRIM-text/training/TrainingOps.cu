#define USE_CUDA

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <cstdint>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/FlashAttention/Flash_Attention_Kernal.hpp"
#include "../Layers/Embedding/Embedding_GPU.hpp"
#include "../Common/grim_scale_buffer.hpp"
#include "../Shared/Loss/ComputeLoss/ComputeLossHost_GPU.hpp"
#include "../Shared/Loss/NumericLoss/NumericLoss_GPU.hpp"
#include "../Shared/StreamController/StreamController_GPU.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../Shared/Activations/Xavier/Xavier.hpp"
#include "module_logger.hpp"

// External kernel declaration (C++ linkage - can throw exceptions)
void launchBiasSumGradient(const float* grad_output, float* grad_bias,
                          int total_tokens, int hidden_dim, cudaStream_t stream);

using GRIM::launchXavierInit;

namespace {
using ForwardLog = ModuleLogger<GRIM::Logging::ModuleId::ForwardPass>;
}

namespace GRIM {

#ifdef USE_CUDA

float LanguageModel::computeLoss(const std::vector<int>& input_ids,
                                const std::vector<int>& target_ids,
                                const std::vector<float>& token_numeric_values,
                                const std::vector<uint8_t>& token_numeric_mask) {
    // Ensure training state is initialized before computing loss
    if (!training_state_.initialized) {
        try {
            ForwardLog::warn("[computeLoss] Training state not initialized, attempting initTrainingState()");
            const_cast<LanguageModel*>(this)->initTrainingState();
            if (!training_state_.initialized) {
                ForwardLog::error("[computeLoss] FATAL: initTrainingState() completed but flag still false");
                throw std::runtime_error("computeLoss: training state not initialized after initTrainingState()");
            }
            ForwardLog::warn("[computeLoss] Training state initialized successfully");
        } catch (const std::exception& e) {
            ForwardLog::error(std::string("[computeLoss] FATAL: Failed to initialize training state: ") + e.what());
            throw;
        } catch (...) {
            ForwardLog::error("[computeLoss] FATAL: Unknown error during training state initialization");
            throw std::runtime_error("computeLoss: unknown error during initTrainingState()");
        }
    }

    const auto& cfg = getConfig();
    const size_t total_tokens = std::max(input_ids.size(), target_ids.size());
    const size_t logit_limit = training_state_.max_logit_tokens > 0
        ? training_state_.max_logit_tokens
        : training_state_.max_cached_tokens;
    if (total_tokens > logit_limit) {
        ForwardLog::error("[computeLoss] FATAL: total_tokens=" + std::to_string(total_tokens) +
                          " exceeds logit buffer capacity " + std::to_string(logit_limit));
        throw std::runtime_error("computeLoss: token count exceeds logit buffer capacity");
    }
    if (token_numeric_values.size() != input_ids.size() ||
        token_numeric_mask.size() != input_ids.size()) {
        throw std::runtime_error("computeLoss: numeric side-channel length mismatch");
    }

    cudaMemcpyAsync(training_state_.cached_targets,
                    target_ids.data(),
                    target_ids.size() * sizeof(int),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());

    auto logits = forwardWithCache(input_ids,
                                   token_numeric_values,
                                   token_numeric_mask);
    (void)logits;

    if (training_state_.cached_seq_len > 0) {
        std::vector<float> sample_output(std::min(5, config_.vocab_size));
        cudaMemcpyAsync(sample_output.data(), training_state_.cached_logits,
                       sample_output.size() * sizeof(float), cudaMemcpyDeviceToHost,
                       training_state_.stream_ctrl.getPrimaryStream());
        training_state_.stream_ctrl.syncPrimaryStream();

        bool has_invalid = false;
        for (float val : sample_output) {
            if (!std::isfinite(val) || std::abs(val) > 1e6f) {
                has_invalid = true;
                break;
            }
        }

        if (has_invalid) {
            ForwardLog::error("[computeLoss] FATAL: Invalid logits from forward pass");
            ForwardLog::error("  seq_len=" + std::to_string(training_state_.cached_seq_len));
            std::ostringstream logits;
            for (float val : sample_output) {
                logits << val << ' ';
            }
            ForwardLog::error("  Sample logits: " + logits.str());
            throw std::runtime_error("computeLoss: invalid logits from forward pass");
        }
    }

    if (training_state_.cached_seq_len > cfg.max_seq_len) {
        ForwardLog::error("ERROR: Sequence length " +
                          std::to_string(training_state_.cached_seq_len) +
                          " exceeds max cache size " + std::to_string(cfg.max_seq_len) + "!");
        throw std::runtime_error("computeLoss: sequence length exceeds max cache size");
    }

    LossScratch scratch{
        training_state_.d_loss_scratch,
        training_state_.d_loss_sum_scratch,
        training_state_.loss_scratch_capacity
    };

    LossComputationInputs loss_inputs{};

    int valid_token_count = 0;
    for (int t : target_ids) {
        if (t >= 0) {
            ++valid_token_count;
        }
    }
    if (valid_token_count == 0) {
        ForwardLog::error("[computeLoss] FATAL: valid_token_count=0");
        throw std::runtime_error("computeLoss: valid_token_count is zero");
    }
    loss_inputs.valid_token_count = static_cast<size_t>(valid_token_count);
    training_state_.cached_valid_tokens = valid_token_count;

    Loss::LossContext loss_ctx{};
    loss_ctx.logits = training_state_.cached_logits;
    loss_ctx.targets = training_state_.cached_targets;
    loss_ctx.batch_size = training_state_.cached_batch_size;
    loss_ctx.seq_len = training_state_.cached_seq_len;
    loss_ctx.valid_tokens = valid_token_count;
    loss_ctx.vocab_size = cfg.vocab_size;
    // Only pass sequence_weights if we actually have weights set (count > 0)
    // Otherwise pass nullptr so kernel uses default sample_weight=1.0f
    loss_ctx.sequence_weights = (training_state_.sequence_weight_count > 0)
                                ? training_state_.sequence_weights
                                : nullptr;
    loss_ctx.sequence_weight_count = training_state_.sequence_weight_count;
    // Issue #38 FIX: Per-token class weighting to prevent mode collapse
    // Weight indexed by TARGET token ID - frequent tokens get lower weight
    loss_ctx.token_weights = (training_state_.token_weights_count > 0)
                             ? training_state_.token_weights
                             : nullptr;
    // Issue #39 FIX: Output logit bias correction to prevent mode collapse
    // Subtracts running EMA of mean logit per token BEFORE softmax
    static int s_issue39_diag = 0;
    if (++s_issue39_diag <= 3) {
        fprintf(stderr, "[Issue39-TrainingOps] batch=%d logit_bias_count=%d logit_bias=%p logit_bias_update=%p\n",
                s_issue39_diag, training_state_.logit_bias_count, 
                training_state_.logit_bias, training_state_.logit_bias_update);
    }
    loss_ctx.logit_bias = (training_state_.logit_bias_count > 0)
                          ? training_state_.logit_bias
                          : nullptr;
    loss_ctx.logit_bias_update = (training_state_.logit_bias_count > 0)
                                 ? training_state_.logit_bias_update
                                 : nullptr;
    loss_ctx.logit_bias_ema_alpha = 0.05f;  // 5% new data per batch - slow adaptation
    loss_ctx.stream = training_state_.stream_ctrl.getPrimaryStream();

    loss_inputs.context = loss_ctx;
    loss_inputs.config.limits.max_tokens = logit_limit;
    loss_inputs.grad_logits = training_state_.grad_logits;  // Pass pre-allocated buffer

    const auto loss_result = computeLossHost(loss_inputs, scratch);

    training_state_.d_loss_scratch = scratch.loss_values;
    training_state_.d_loss_sum_scratch = scratch.loss_accumulator;
    training_state_.loss_scratch_capacity = scratch.capacity;

    if (!loss_result.success) {
        ForwardLog::error("[computeLoss] FATAL: computeLossHost failed");
        throw std::runtime_error("computeLoss: computeLossHost failed");
    }

    if (!std::isfinite(loss_result.average_loss)) {
        ForwardLog::error("[computeLoss] FATAL: Invalid loss detected");
        std::ostringstream loss_msg;
        loss_msg << "  total_loss=" << loss_result.total_loss
                 << ", seq_len=" << training_state_.cached_seq_len;
        ForwardLog::error(loss_msg.str());
        ForwardLog::error("  avg_loss=" + std::to_string(loss_result.average_loss));

        std::vector<float> sample_logits(std::min(10, config_.vocab_size));
        cudaMemcpyAsync(sample_logits.data(), training_state_.cached_logits,
                       sample_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                       training_state_.stream_ctrl.getPrimaryStream());
        training_state_.stream_ctrl.syncPrimaryStream();

        std::ostringstream logits;
        logits << "  Sample logits: ";
        for (float value : sample_logits) {
            logits << value << ' ';
        }
        ForwardLog::error(logits.str());
        throw std::runtime_error("computeLoss: non-finite loss");
    }

    float numeric_loss_sum = 0.0f;
    int numeric_loss_count = 0;
    if (cfg.numeric_head_enabled) {
        if (!training_state_.cached_numeric_predictions ||
            !training_state_.grad_numeric_predictions ||
            !training_state_.d_numeric_loss_sum ||
            !training_state_.d_numeric_loss_count) {
            throw std::runtime_error("computeLoss: numeric head enabled but buffers missing");
        }

        NumericLossInputs num_inputs{};
        num_inputs.predictions = training_state_.cached_numeric_predictions;
        num_inputs.token_numeric_values = training_state_.cached_token_numeric_values;
        num_inputs.token_numeric_mask = training_state_.cached_token_numeric_mask;
        num_inputs.targets = training_state_.cached_targets;
        num_inputs.total_tokens = static_cast<int>(total_tokens);
        num_inputs.seq_len = static_cast<int>(training_state_.cached_seq_len);
        num_inputs.huber_delta = cfg.numeric_head_huber_delta;
        num_inputs.log_scale = cfg.numeric_head_log_scale;
        num_inputs.loss_weight = cfg.numeric_head_loss_weight;

        NumericLossOutputs num_outputs{};
        num_outputs.loss_sum = training_state_.d_numeric_loss_sum;
        num_outputs.count = training_state_.d_numeric_loss_count;
        num_outputs.grad_predictions = training_state_.grad_numeric_predictions;

        if (!launchNumericLoss(num_inputs, num_outputs, training_state_.stream_ctrl.getPrimaryStream())) {
            throw std::runtime_error("computeLoss: numeric loss kernel launch failed");
        }

        cudaMemcpyAsync(&numeric_loss_sum, training_state_.d_numeric_loss_sum,
                        sizeof(float), cudaMemcpyDeviceToHost,
                        training_state_.stream_ctrl.getPrimaryStream());
        cudaMemcpyAsync(&numeric_loss_count, training_state_.d_numeric_loss_count,
                        sizeof(int), cudaMemcpyDeviceToHost,
                        training_state_.stream_ctrl.getPrimaryStream());
        training_state_.stream_ctrl.syncPrimaryStream();
        if (!std::isfinite(numeric_loss_sum)) {
            numeric_loss_sum = 0.0f;
            numeric_loss_count = 0;
        }
    }

    const float weighted_numeric_loss = (numeric_loss_count > 0)
        ? cfg.numeric_head_loss_weight * numeric_loss_sum
        : 0.0f;
    const float avg_loss = (loss_result.total_loss + weighted_numeric_loss) /
        static_cast<float>(valid_token_count);
    return avg_loss;
}

LanguageModel::ModelStats LanguageModel::getModelStats() const {
    ModelStats stats;
    
    // Count actual allocated parameters from parameter groups
    // This is the ground truth - no formulas that can drift out of sync
    for (const auto& group : parameter_groups_) {
        if (group.name == "embedding") {
            stats.embedding_params += group.size;
        } else if (group.name.find("lm_head") != std::string::npos) {
            stats.lm_head_params += group.size;
        } else if (group.name == "numeric_head_weight") {
            stats.numeric_head_params += group.size;
        } else {
            stats.encoder_params += group.size;
        }
    }
    
    // If parameter groups not yet built, fall back to config-based estimate
    // (this can happen during early initialization before initTrainingState)
    if (parameter_groups_.empty()) {
        const auto& cfg = config_;
        const int head_dim = cfg.head_dim;  // Use pre-computed value from config
        const int kv_dim = cfg.num_kv_heads * head_dim;
        const int total_qkv_dim = cfg.d_model + 2 * kv_dim;  // Q + K + V with GQA
        
        // Embedding: vocab_size * d_model
        stats.embedding_params = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
        
        // LM head: only if not tied
        if (!cfg.tie_embeddings) {
            stats.lm_head_params = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
        }

        if (cfg.numeric_head_enabled) {
            stats.numeric_head_params = static_cast<size_t>(cfg.d_model);
        }
        
        // Per-layer encoder params (matches buildParameterGroups exactly)
        size_t per_layer = 0;
        per_layer += total_qkv_dim * cfg.d_model;       // QKV projection (GQA-aware)
        per_layer += cfg.d_model * cfg.d_model;          // Wo projection
        per_layer += cfg.d_model * cfg.d_ff;             // FFN W1
        per_layer += cfg.d_ff * cfg.d_model;             // FFN W2
        per_layer += cfg.d_model;                        // RMSNorm1 gamma
        per_layer += cfg.d_model;                        // RMSNorm2 gamma
        // Note: No biases in this architecture, no beta (RMSNorm has no beta)
        
        stats.encoder_params = per_layer * cfg.num_layers;
    } else {
        // Debug assert: verify fallback formula matches actual allocations
        // This catches drift between buildParameterGroups and getModelStats formula
        const auto& cfg = config_;
        const int head_dim = cfg.head_dim;  // Use pre-computed value from config
        const int kv_dim = cfg.num_kv_heads * head_dim;
        const int total_qkv_dim = cfg.d_model + 2 * kv_dim;
        
        // Validate GQA dimensions using TensorContract
        TensorContract::GQADims gqa_dims{cfg.num_heads, cfg.num_kv_heads, head_dim};
        
        // TensorContract validates: d_model == num_heads * head_dim
        if (!gqa_dims.is_valid()) {
            std::cerr << "[WARNING] TensorContract GQA validation failed!" << std::endl;
            std::cerr << "  d_model=" << cfg.d_model 
                      << " num_heads=" << cfg.num_heads 
                      << " head_dim=" << head_dim << std::endl;
            assert(false && "GQA dimension validation failed");
        }
        
        // Verify total_qkv_dim formula matches TensorContract expectation
        const int expected_qkv_dim = gqa_dims.total_qkv_dim();
        if (total_qkv_dim != expected_qkv_dim) {
            std::cerr << "[WARNING] QKV dimension mismatch: computed=" << total_qkv_dim 
                      << " expected=" << expected_qkv_dim << std::endl;
            assert(false && "QKV dimension formula drift from TensorContract");
        }
        
        size_t est_embedding = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
        size_t est_lm_head = cfg.tie_embeddings ? 0 : static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
        size_t est_numeric_head = cfg.numeric_head_enabled ? static_cast<size_t>(cfg.d_model) : 0;
        
        size_t per_layer = 0;
        per_layer += total_qkv_dim * cfg.d_model;
        per_layer += cfg.d_model * cfg.d_model;
        per_layer += cfg.d_model * cfg.d_ff;
        per_layer += cfg.d_ff * cfg.d_model;
        per_layer += cfg.d_model;
        per_layer += cfg.d_model;
        size_t est_encoder = per_layer * cfg.num_layers;
        
        size_t est_total = est_embedding + est_encoder + est_lm_head + est_numeric_head;
        size_t actual_total = stats.embedding_params + stats.encoder_params + stats.lm_head_params +
                              stats.numeric_head_params;
        
        // Check for drift > 0.1%
        if (actual_total > 0) {
            float drift_pct = 100.0f * std::abs(static_cast<int64_t>(est_total - actual_total)) / static_cast<float>(actual_total);
            if (drift_pct > 0.1f) {
                std::cerr << "[WARNING] getModelStats formula drift detected: "
                          << "actual=" << actual_total << " estimate=" << est_total 
                          << " (" << drift_pct << "%)" << std::endl;
                std::cerr << "  Embedding: actual=" << stats.embedding_params << " est=" << est_embedding << std::endl;
                std::cerr << "  Encoder:   actual=" << stats.encoder_params << " est=" << est_encoder << std::endl;
                std::cerr << "  LM Head:   actual=" << stats.lm_head_params << " est=" << est_lm_head << std::endl;
                std::cerr << "  Num Head:  actual=" << stats.numeric_head_params << " est=" << est_numeric_head << std::endl;
                assert(false && "Parameter count formula drifted from actual allocations");
            }
        }
    }
    
    stats.total_params = stats.embedding_params + stats.encoder_params + stats.lm_head_params +
                         stats.numeric_head_params;
    
    // Add ScratchBlock parameters if enabled
    if (config_.use_scratch_block) {
        constexpr int NUM_ATOM_TYPES = HyperParameters::NUM_ATOM_TYPES;
        const int atom_embedding_dim = config_.scratch_block_atom_embedding_dim;
        stats.scratchblock_params = NUM_ATOM_TYPES * atom_embedding_dim +           // atom type embeddings
                                    atom_embedding_dim * config_.d_model;            // atom projection
        stats.total_params += stats.scratchblock_params;
    }
    
    stats.model_size_mb = (stats.total_params * sizeof(float)) / (1024.0f * 1024.0f);
    
    return stats;
}

// Note: save() and load() implementations are in grim_language_model_serialization.cpp
// (separate compilation unit to avoid mixing CUDA with FlatBuffers headers)

//======================================================//
//  GPU Initialization in Constructor
//======================================================//

void LanguageModel::initGPU() {
    const auto& cfg = getConfig();
    // NOTE: Cannot use ForwardLog here - LogRecorder not initialized yet during Phase1 startup
    // initGPU() is called from initializeModel() which happens BEFORE GRIM::Logging::InitLogRecorder()
    std::cout << "[initGPU] Entry, use_gpu=" << (cfg.use_gpu ? "true" : "false") << std::endl;
    if (!cfg.use_gpu) return;
    
    try {
        std::cout << "[initGPU] Initializing GPU-accelerated transformer layers..." << std::endl;
    
    // Initialize CUDA device (CRITICAL: Must be done before any other CUDA API calls)
    {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to query CUDA device count: ") +
                                     cudaGetErrorString(err));
        }
        
        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices found - GPU backend required");
        }
        
        // Set device 0 (or cfg.gpu_device if specified in future)
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("Failed to set CUDA device 0: ") +
                                     cudaGetErrorString(err));
        }
        
        // Get device properties
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) {
            std::cout << "⚠ Failed to query device properties: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "✓ CUDA Device initialized: " << prop.name << std::endl;
            std::cout << "  - Compute capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "  - Memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        }
    }
    
    // Create GPU embedding runtime
    GPUConfig gpu_config;
    gpu_config.use_tensor_cores = true;
    gpu_config.use_cuda_graphs = true;
    gpu_config.use_pinned_memory = true;
    
    auto* embedding_runtime = new EmbeddingRuntime();
    embedding_runtime->config.vocab_size = cfg.vocab_size;
    embedding_runtime->config.max_position = cfg.max_seq_len;
    embedding_runtime->config.d_model = cfg.d_model;
    embedding_runtime->config.apply_rms_norm = true;
    embedding_runtime->config.rms_epsilon = 1e-5f;
    
    // Use centralized stream from TrainingState
    // NOTE: training_state_.stream_ctrl MUST be initialized before calling initGPU()
    if (!training_state_.stream_ctrl.isInitialized()) {
        throw std::runtime_error("FATAL: StreamController not initialized. Call initTrainingState() first or initialize stream_ctrl before initGPU()");
    }
    cudaStream_t primary_stream = training_state_.stream_ctrl.getPrimaryStream();
    StreamController::fatalIfDefaultStream(primary_stream, "LanguageModel::initGPU");
    
    embedding_runtime->stream = primary_stream;
    embedding_runtime->owns_stream = false;  // TrainingState owns it
    std::cout << "✓ Embedding runtime using TrainingState primary stream" << std::endl;
    embedding_runtime->config.stream = embedding_runtime->stream;
    
    auto cleanup_runtime = [&](const char* msg) {
        if (msg) {
            std::cerr << msg << std::endl;
        }
        destroyEmbeddingRuntime(embedding_runtime);
        gpu_embedder_.reset();
        throw std::runtime_error("Failed to initialize GPU embeddings");
    };
    
    const GrimEmbeddingStack& cpu_embedder = *getEmbedderPtr();
    const size_t token_elements = static_cast<size_t>(cfg.vocab_size) * cfg.d_model;
    const size_t token_bytes = token_elements * sizeof(float);
    if (cudaMalloc(&embedding_runtime->token_buffer, token_bytes) != cudaSuccess) {
        cleanup_runtime("❌ FATAL: Failed to allocate GPU memory for token embeddings");
    }
    embedding_runtime->weights.token_embeddings = embedding_runtime->token_buffer;
    
    std::vector<float> token_data;
    token_data.reserve(token_elements);
    for (int i = 0; i < cfg.vocab_size; ++i) {
        const auto& row = cpu_embedder.token_embed.rows[i];
        if (row.data.size() != static_cast<size_t>(cfg.d_model)) {
            cleanup_runtime("❌ FATAL: Token embedding size mismatch");
        }
        token_data.insert(token_data.end(), row.data.begin(), row.data.end());
    }
    cudaMemcpyAsync(embedding_runtime->token_buffer,
                    token_data.data(),
                    token_bytes,
                    cudaMemcpyHostToDevice,
                    embedding_runtime->stream);
    
    const size_t pos_elements = static_cast<size_t>(cfg.max_seq_len) * cfg.d_model;
    const size_t pos_bytes = pos_elements * sizeof(float);
    if (cudaMalloc(&embedding_runtime->position_buffer, pos_bytes) != cudaSuccess) {
        cleanup_runtime("❌ FATAL: Failed to allocate GPU memory for positional encodings");
    }
    embedding_runtime->weights.position_embeddings = embedding_runtime->position_buffer;
    
    // LEARNED POSITION EMBEDDINGS: Initialize directly on GPU with Xavier
    // This matches PyTorch baseline (nn.Embedding) and token embedding initialization.
    // Xavier stddev = sqrt(2/(max_seq_len + d_model)) for proper scale matching.
    const float pos_embedding_stddev = std::sqrt(2.0f / static_cast<float>(cfg.max_seq_len + cfg.d_model));
    std::cout << "🎲 Initializing position embeddings on GPU (stddev=" << pos_embedding_stddev 
              << ", matches token emb scale)" << std::endl;
    launchXavierInit(embedding_runtime->position_buffer, 
                     static_cast<int>(pos_elements), 
                     pos_embedding_stddev, 
                     43,  // Different seed from token embeddings (42)
                     embedding_runtime->stream);
    
    const size_t ln_bytes = static_cast<size_t>(cfg.d_model) * sizeof(float);
    if (cudaMalloc(&embedding_runtime->gamma_buffer, ln_bytes) != cudaSuccess) {
        cleanup_runtime("❌ FATAL: Failed to allocate GPU memory for RMSNorm gamma");
    }
    embedding_runtime->weights.gamma = embedding_runtime->gamma_buffer;
    
    cudaMemcpyAsync(embedding_runtime->gamma_buffer,
                    cpu_embedder.rms_gamma.data.data(),
                    ln_bytes,
                    cudaMemcpyHostToDevice,
                    embedding_runtime->stream);
    
    cudaError_t sync_err = cudaStreamSynchronize(embedding_runtime->stream);
    if (sync_err != cudaSuccess) {
        cleanup_runtime("❌ FATAL: Failed to synchronize embedding stream after upload");
    }
    
    gpu_embedder_.reset(embedding_runtime);
    ForwardLog::info("✓ GPU embeddings initialized");
    
    // Create GPU encoder
    EncoderConfig enc_config;
    enc_config.d_model = cfg.d_model;
    enc_config.num_heads = cfg.num_heads;
    enc_config.num_kv_heads = cfg.num_kv_heads;  // GQA support
    enc_config.head_dim = cfg.head_dim;          // Use pre-computed value from LanguageModelConfig
    enc_config.d_ff = cfg.d_ff;
    enc_config.num_layers = cfg.num_layers;
    enc_config.dropout_rate = cfg.dropout_rate;
    enc_config.attention_dropout = cfg.attention_dropout;
    enc_config.use_pre_norm = cfg.use_pre_norm;
    enc_config.use_simd = cfg.use_simd;
    enc_config.num_threads = cfg.num_threads;
    
    // CRITICAL BUG FIX: Must propagate use_flash_attention from model config!
    // Without this, EncoderConfig defaults to true while model config could be false,
    // causing forward pass to use Flash Attention but backward pass to fail every batch.
    enc_config.use_flash_attention = cfg.use_flash_attention;
    enc_config.min_seq_len_for_flash = cfg.min_seq_len_for_flash;
    enc_config.causal_mask = cfg.causal_mask;
    enc_config.max_seq_len = cfg.max_seq_len;
    enc_config.max_cached_batch = cfg.max_cached_batch;
    enc_config.max_cached_seq_len = cfg.max_cached_seq_len;
    
    // Use TrainingState stream for encoder (already validated above)
    enc_config.stream = primary_stream;
    enc_config.cublas_handle = training_state_.cublas_handle;  // Centralized handle (Rule 22)
    
    // CRITICAL: PBM must be initialized BEFORE encoder construction
    if (!training_state_.pbm_initialized) {
        fprintf(stderr, "\n[LanguageModel::initGPU] FATAL: PBM not initialized before encoder construction!\n");
        fprintf(stderr, "[LanguageModel::initGPU] Call initPBM() BEFORE createGPUEncoder().\n");
        fprintf(stderr, "[LanguageModel::initGPU] Fix initialization order in TrainingOps.cu\n");
        std::abort();
    }
    enc_config.pos_encoding = &training_state_.pbm_spec;  // ALiBi+RoPE positional encoding (validated above)
    
    // DEBUG: Validate handle before creating encoder
    if (!training_state_.cublas_handle) {
        std::cerr << "FATAL: training_state_.cublas_handle is NULL before GPUGrimEncoder construction!" << std::endl;
        throw std::runtime_error("cuBLAS handle not initialized");
    }
    if (!primary_stream) {
        std::cerr << "FATAL: primary_stream is NULL before GPUGrimEncoder construction!" << std::endl;
        throw std::runtime_error("Primary stream not initialized");
    }
    std::cout << "✓ Encoder using TrainingState primary stream (handle=" << training_state_.cublas_handle 
              << ", stream=" << primary_stream << ")" << std::endl;
    
    auto* encoder_ptr = new GPUGrimEncoder(enc_config);  // GPU encoder manages its own weights independently
    gpu_encoder_.reset(encoder_ptr);
    
    // GPT-2 style residual scaling to prevent gradient explosion in deep networks
    // Scale output projections by 1/sqrt(2*num_layers)
    const float residual_scale = 1.0f / std::sqrt(2.0f * cfg.num_layers);
    
    // CRITICAL: Initialize encoder layer weights on GPU with Xavier initialization
    std::cout << "🎲 Initializing encoder layer weights..." << std::endl;
    std::cout << "   Using GPT-2 residual scaling: " << residual_scale << std::endl;
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        auto* gpu_layer = encoder_ptr->getLayer(layer);
        if (!gpu_layer) {
            std::cerr << "ERROR: Could not get layer " << layer << " for initialization!" << std::endl;
            continue;
        }

        // Initialize attention weights (W_qkv, W_o)
        auto* attn_enc = gpu_layer; // GPUEncoderLayer is an alias of EncodingLayer
        if (attn_enc) {
            float* w_qkv_ptr = attn_enc->getAttnWqkv();
            float* w_o_ptr = attn_enc->getAttnWo();
            
            if (w_qkv_ptr && w_o_ptr) {
                // GQA QKV dimension calculation using TensorContract
                const int head_dim = cfg.head_dim;  // Use pre-computed value from config
                TensorContract::GQADims gqa_dims{cfg.num_heads, cfg.num_kv_heads, head_dim};
                const int total_qkv_dim = gqa_dims.total_qkv_dim();
                const size_t qkv_size = static_cast<size_t>(total_qkv_dim) * cfg.d_model;
                
                // Xavier init for QKV: stddev = sqrt(2 / (d_model + total_qkv_dim))
                float xavier_qkv_stddev = std::sqrt(2.0f / (cfg.d_model + static_cast<float>(total_qkv_dim)));
                launchXavierInit(w_qkv_ptr, static_cast<int>(qkv_size), xavier_qkv_stddev, 42 + layer * 4, primary_stream);
                
                // Initialize output projection with GPT-2 residual scaling
                // This prevents gradient explosion by making residual contributions smaller
                float xavier_wo_stddev = xavier_qkv_stddev * residual_scale;
                launchXavierInit(w_o_ptr, cfg.d_model * cfg.d_model, xavier_wo_stddev, 43 + layer * 4, primary_stream);
            } else {
                ForwardLog::error("    ERROR: Null attention weight pointers!");
            }
        }
        
        // Initialize FFN weights (W1, W2)
        auto* ffn_enc = gpu_layer; // GPUEncoderLayer is an alias of EncodingLayer
        if (ffn_enc) {
            float* w1_ptr = ffn_enc->getFFNW1();
            float* w2_ptr = ffn_enc->getFFNW2();
            
            if (w1_ptr && w2_ptr) {
                // Xavier init for W1: stddev = sqrt(2 / (d_model + d_ff))
                float xavier_w1_stddev = std::sqrt(2.0f / (cfg.d_model + cfg.d_ff));
                launchXavierInit(w1_ptr, cfg.d_model * cfg.d_ff, xavier_w1_stddev, 44 + layer * 4, primary_stream);
                
                // Xavier init for W2 with GPT-2 residual scaling
                // W2 feeds into residual connection, so scale it down
                float xavier_w2_stddev = std::sqrt(2.0f / (cfg.d_ff + cfg.d_model)) * residual_scale;
                launchXavierInit(w2_ptr, cfg.d_ff * cfg.d_model, xavier_w2_stddev, 45 + layer * 4, primary_stream);
            } else {
                ForwardLog::error("    ERROR: Null FFN weight pointers!");
            }
        }
        
        // Layer norm is already initialized in GPULayerNorm constructor to gamma=1, beta=0
        // No need to re-initialize
    }
    cudaDeviceSynchronize();
    {
        std::ostringstream oss;
        oss << "✓ Encoder layer weights initialized with Xavier (layers 0-" << (cfg.num_layers - 1) << ")";
std::cout << oss.str() << std::endl;
    }
    
    std::cout << "✓ GPU encoder initialized with " << cfg.num_layers << " layers" << std::endl;
    std::cout << "  - Attention: GPU-accelerated" << std::endl;
    std::cout << "  - FFN: GPU-accelerated with fused GELU" << std::endl;
    std::cout << "  - Layer Norm: GPU-accelerated" << std::endl;
    
    // Configure Flash Attention if enabled
    if (cfg.use_flash_attention) {
        std::cout << "⚡ Enabling Flash Attention 2..." << std::endl;
        encoder_ptr->setFlashAttention(true, cfg.min_seq_len_for_flash);
        std::cout << "✓ Flash Attention enabled (min_seq_len=" << cfg.min_seq_len_for_flash << ")" << std::endl;
    }
    
    } catch (const std::exception& e) {
        std::cerr << "❌ EXCEPTION in initGPU(): " << e.what() << std::endl;
        throw;  // Re-throw to trigger cleanup
    }
}

#endif // USE_CUDA

} // namespace GRIM

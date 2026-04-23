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
#include "../Shared/StreamController/StreamController_GPU.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"

namespace {

inline void cudaFail(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(err));
    }
}

[[maybe_unused]] inline void cudaFailLast(const char* where) {
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
    // Rule 20 / Rule 26: parameter_groups_ is the single source of truth for parameter
    // counts. The previous formula-based estimate duplicated that data and silently
    // drifted whenever a new subsystem (ExecutionBlock, ReasoningHead, SlotSelector,
    // MTP heads, ...) was added. The formula has been deleted; counting walks the
    // registered parameter groups directly.
    if (parameter_groups_.empty()) {
        throw std::runtime_error(
            "getModelStats called before buildParameterGroups — parameter_groups_ is empty at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    ModelStats stats;
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

        // Sub-bucket: scratch_block_* params are also reported separately for diagnostics.
        // They remain counted under encoder_params above (do NOT double-count in total_params).
        if (group.name.rfind("scratch_block_", 0) == 0) {
            stats.scratchblock_params += group.size();
        }
    }

    stats.total_params = stats.embedding_params + stats.position_embedding_params +
                         stats.encoder_params + stats.lm_head_params;
    stats.model_size_mb = (stats.total_params * sizeof(float)) / (1024.0f * 1024.0f);
    return stats;
}

//======================================================//
//  GPU Initialization in Constructor
//======================================================//

void LanguageModel::initGPU() {
    const auto& cfg = getConfig();

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
        // Copy architecture fields from LanguageModelConfig (both inherit ModelArchitecture)
        static_cast<GRIM::HyperParameters::ModelArchitecture&>(enc_config) =
            static_cast<const GRIM::HyperParameters::ModelArchitecture&>(cfg);
        enc_config.use_pre_norm = cfg.use_pre_norm;
        enc_config.use_simd = cfg.use_simd;
        enc_config.num_threads = cfg.num_threads;

        // Must propagate flash attention toggles
        enc_config.use_flash_attention = cfg.use_flash_attention;
        enc_config.min_seq_len_for_flash = cfg.min_seq_len_for_flash;
        enc_config.causal_mask = cfg.causal_mask;
        enc_config.max_cached_batch = cfg.max_cached_batch;
        enc_config.max_cached_seq_len = cfg.max_cached_seq_len;

        // Issue #109 FIX: Propagate LayerScale config from LanguageModelConfig → EncoderConfig
        enc_config.use_layer_scale = cfg.use_layer_scale;
        enc_config.layer_scale_init = cfg.layer_scale_init;
        enc_config.center_encoder_residuals = cfg.center_encoder_residuals;
        enc_config.use_bias = cfg.use_bias;
        enc_config.qk_norm_enabled = cfg.qk_norm_enabled;

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
            lm_config.freeze_final_rms_gamma = cfg.lm_head_freeze_final_rms_gamma;
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

        // ReasoningHead layer
        if (cfg.reasoning_head_enabled) {
            ReasoningHeadConfig rh_config;
            rh_config.d_model = cfg.d_model;
            rh_config.atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
            rh_config.num_ops = cfg.reasoning_num_ops;
            const uint64_t rh_seed = training_state_.weight_init_seed + 10;
            reasoning_head_layer_ = std::make_unique<ReasoningHeadLayer>(rh_config, rh_seed, primary_stream);
            std::cout << "✓ ReasoningHead layer created (d_total="
                      << rh_config.d_total() << ", num_ops=" << rh_config.num_ops << ")\n";
        }

        // ExecutionBlock layer (differentiable register machine)
        if (cfg.execution_block_enabled) {
            ExecutionBlockConfig eb_config;
            eb_config.d_model = cfg.d_model;
            eb_config.atom_embedding_dim = cfg.scratch_block_atom_embedding_dim;
            eb_config.num_ops = cfg.execution_block_num_ops;
            eb_config.num_slots = cfg.execution_block_num_slots;
            eb_config.num_exec_steps = cfg.execution_block_num_steps;
            eb_config.d_key = cfg.execution_block_d_key;
            eb_config.d_type = cfg.execution_block_d_type;
            eb_config.cross_attn_head_dim = cfg.execution_block_cross_attn_head_dim;
            eb_config.cross_attn_topk = cfg.execution_block_cross_attn_topk;
            eb_config.usage_decay = cfg.execution_block_usage_decay;
            eb_config.transition_hard_threshold = cfg.execution_block_transition_hard_threshold;
            eb_config.div_invalid_penalty_weight = cfg.div_invalid_penalty_weight;
            eb_config.div_magnitude_penalty_weight = cfg.div_magnitude_penalty_weight;
            eb_config.arg_reinforce_weight = cfg.arg_reinforce_weight;
            eb_config.arg_reinforce_baseline_decay = cfg.arg_reinforce_baseline_decay;

            const uint64_t eb_seed = training_state_.weight_init_seed + 20;
            execution_block_layer_ = std::make_unique<ExecutionBlockLayer>(eb_config, eb_seed, primary_stream);
            std::cout << "✓ ExecutionBlock layer created (V=" << eb_config.num_slots
                      << ", K=" << eb_config.num_exec_steps
                      << ", ops=" << eb_config.num_ops << ")\n";

            // Decode-time slot selector (pointer-selector baseline)
            if (cfg.selector_enabled) {
                DecodeTimeSlotSelectorConfig sel_config;
                sel_config.d_model = cfg.d_model;
                sel_config.d_selector = cfg.selector_d_selector;
                sel_config.d_slot_features = kSlotFeatureDim;
                sel_config.cublas_handle = training_state_.cublas_handle;

                const uint64_t sel_seed = training_state_.weight_init_seed + 30;
                decode_time_slot_selector_layer_ = std::make_unique<DecodeTimeSlotSelectorLayer>(
                    sel_config, sel_seed, primary_stream);

                NumPolicyConfig pol_config;
                pol_config.selection_margin = cfg.selector_selection_margin;
                pol_config.num_slots = cfg.execution_block_num_slots;
                pol_config.scratch_slots = 0; // num_scratch_slots defaults to 0
                decode_time_num_policy_ = std::make_unique<DecodeTimeNumPolicy>(pol_config);

                std::cout << "✓ DecodeTimeSlotSelector created (d_selector=" << sel_config.d_selector
                          << ", margin=" << pol_config.selection_margin << ")\n";
            }
        }

        // Multi-token prediction (MTP) auxiliary heads: K independent linear heads (not tied to embedding)
        if (cfg.mtp_enabled && cfg.mtp_k > 0) {
            mtp_heads_.resize(static_cast<size_t>(cfg.mtp_k));
            for (int k = 0; k < cfg.mtp_k; ++k) {
                auto& head = mtp_heads_[static_cast<size_t>(k)];
                const std::string w_name = "mtp_head_" + std::to_string(k) + ".weight";
                const std::string b_name = "mtp_head_" + std::to_string(k) + ".bias";
                head.weight = Tensor::zeros({cfg.vocab_size, cfg.d_model}, primary_stream, w_name.c_str());
                head.weight.requires_grad_();
                head.weight.ensure_grad();
                const uint64_t mtp_seed = training_state_.weight_init_seed + 3 + static_cast<uint64_t>(k);
                Tensor::xavier_uniform_(head.weight, mtp_seed, primary_stream);
                head.bias = Tensor::zeros({cfg.vocab_size}, primary_stream, b_name.c_str());
                head.bias.requires_grad_();
                head.bias.ensure_grad();
            }
            std::cout << "✓ MTP " << cfg.mtp_k << " auxiliary heads created (alpha=" << cfg.mtp_alpha
                      << ", warmup_steps=" << cfg.mtp_alpha_warmup_steps << ")\n";
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

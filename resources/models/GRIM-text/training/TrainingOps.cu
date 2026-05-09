#ifndef USE_CUDA
#define USE_CUDA
#endif
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
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"
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

    stats.total_params = stats.embedding_params +
                         stats.encoder_params + stats.lm_head_params;
    stats.model_size_mb = (stats.total_params * sizeof(float)) / (1024.0f * 1024.0f);
    return stats;
}

//======================================================//
//  GPU Initialization in Constructor
//======================================================//

void LanguageModel::initGPU(uint64_t weight_init_seed) {
    const auto& model_cfg = getConfig();
    const auto init_hp = HyperParameters::gpuModelInitializationHP(model_cfg);

    std::cout << "[initGPU] Entry, use_gpu=" << (init_hp.use_gpu ? "true" : "false") << std::endl;
    if (!init_hp.use_gpu) {
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
        //
        //  Hyperparameters come from EncoderLayerConstructionHP, the grouped
        //  read view owned by HyperparameterGroupings.hpp. Runtime device
        //  handles — positional encoding, stream, cuBLAS, init seed — come from
        //  EncoderRuntimeBindings. Runtime and hyperparameter ownership stay
        //  separate; no caller-local cfg slicing.
        //======================================================//
        if (!isPBMInitialized()) {
            throw std::runtime_error(
                "[initGPU] FATAL: PBM not initialized before encoder construction! "
                "Call initPBM() BEFORE createGPUEncoder()");
        }

        const auto encoder_hp = HyperParameters::encoderLayerConstructionHP(model_cfg);
        EncoderRuntimeBindings enc_bindings;
        enc_bindings.pos_encoding = &getPBMSpec();
        enc_bindings.stream = primary_stream;
        enc_bindings.cublas_handle = training_state_.cublas_handle;
        enc_bindings.weight_seed = weight_init_seed;
        // Issue #142: 1/sqrt(2*num_layers) for residual projection init
        enc_bindings.residual_scale = init_hp.residual_scale;

        fprintf(stdout, "[initGPU] Layers will self-allocate weights "
                "(seed=%llu, residual_scale=%.6f)\n",
                static_cast<unsigned long long>(enc_bindings.weight_seed),
                enc_bindings.residual_scale);

        std::cout << "✓ Encoder using TrainingState primary stream (handle="
                  << training_state_.cublas_handle
                  << ", stream=" << primary_stream << ")\n";

        auto* encoder_ptr = new GPUGrimEncoder(encoder_hp, enc_bindings);
        gpu_encoder_.reset(encoder_ptr);

        // Layers self-allocated their own weights in the constructor. Verify all are ready.
        for (int layer = 0; layer < init_hp.num_layers; ++layer) {
            auto* gpu_layer = encoder_ptr->getLayer(layer);
            if (!gpu_layer || !gpu_layer->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: Encoder layer " + std::to_string(layer) +
                                         " not ready after self-allocation!");
            }
        }
        std::cout << "✓ " << init_hp.num_layers << " encoder layers self-allocated weights\n";

        //======================================================//
        //  6a) Build persistent Embedding layer (Pattern B)
        //
        //  Self-allocates token weights [vocab_size, d_model]. Position
        //  information is injected inside attention via ALiBi/RoPE, so no
        //  separate position-embedding table is allocated (Rule 26).
        //  Must be created BEFORE LMHeadLayer (LM head aliases embedding for tied config).
        //======================================================//
        {
            const auto emb_hp = HyperParameters::embeddingLayerConstructionHP(model_cfg);
            // Seed convention: embedding uses weight_init_seed + 0
            const uint64_t emb_seed = weight_init_seed;

            embedding_layer_ = std::make_unique<EmbeddingLayer>(emb_hp, emb_seed, primary_stream, true);

            if (!embedding_layer_->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: EmbeddingLayer not ready after construction!");
            }
            std::cout << "✓ Embedding layer created (vocab=" << emb_hp.vocab_size
                      << ", d_model=" << emb_hp.d_model
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
            const auto lm_hp = HyperParameters::lmHeadLayerConstructionHP(model_cfg);
            // Seed convention: lm_head uses weight_init_seed + 1
            const uint64_t lm_head_seed = weight_init_seed + 1;

            // For tied weights, pass pointer to embedding token weights (owned by EmbeddingLayer)
            Tensor* tied_emb = lm_hp.tie_embeddings ? &embedding_layer_->tokenWeights() : nullptr;

            lm_head_layer_ = std::make_unique<LMHeadLayer>(
                lm_hp, lm_head_seed, primary_stream, training_state_.cublas_handle, tied_emb);

            if (!lm_head_layer_->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: LMHeadLayer not ready after construction!");
            }
            std::cout << "✓ LM Head layer created ("
                      << (lm_hp.tie_embeddings ? "tied to embedding" : "separate weights")
                      << ", final_rms_gamma owned, bias=" << (lm_hp.use_bias ? "yes" : "no") << ")\n";
        }

        // ReasoningHead layer
        const auto reasoning_hp = HyperParameters::reasoningHeadConstructionHP(model_cfg);
        if (reasoning_hp.enabled) {
            const uint64_t rh_seed = weight_init_seed + 10;
            reasoning_head_layer_ = std::make_unique<ReasoningHeadLayer>(reasoning_hp, rh_seed, primary_stream);
            std::cout << "✓ ReasoningHead layer created (d_total="
                      << (reasoning_hp.d_model + reasoning_hp.atom_embedding_dim)
                      << ", num_ops=" << reasoning_hp.num_ops << ")\n";
        }

        // ExecutionBlock layer (differentiable register machine)
        const auto execution_hp = HyperParameters::executionBlockConstructionHP(model_cfg);
        if (execution_hp.enabled) {
            const uint64_t eb_seed = weight_init_seed + 20;
            execution_block_layer_ = std::make_unique<ExecutionBlockLayer>(execution_hp, eb_seed, primary_stream);
            std::cout << "✓ ExecutionBlock layer created (V=" << execution_hp.num_slots
                      << ", K=" << execution_hp.num_exec_steps
                      << ", ops=" << execution_hp.num_ops << ")\n";

            // Decode-time slot selector (pointer-selector baseline)
            const auto selector_hp = HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
            if (selector_hp.enabled) {
                const uint64_t sel_seed = weight_init_seed + 30;
                decode_time_slot_selector_layer_ = std::make_unique<DecodeTimeSlotSelectorLayer>(
                    selector_hp, sel_seed, primary_stream, training_state_.cublas_handle);

                decode_time_num_policy_ = std::make_unique<DecodeTimeNumPolicy>(selector_hp);

                std::cout << "✓ DecodeTimeSlotSelector created (d_selector=" << selector_hp.d_selector
                          << ", margin=" << selector_hp.selection_margin << ")\n";
            }
        }

        // Multi-token prediction (MTP) auxiliary heads: K independent linear heads (not tied to embedding)
        const auto mtp_hp = HyperParameters::mtpConstructionHP(model_cfg);
        if (mtp_hp.enabled) {
            mtp_heads_.resize(static_cast<size_t>(mtp_hp.k));
            for (int k = 0; k < mtp_hp.k; ++k) {
                auto& head = mtp_heads_[static_cast<size_t>(k)];
                const std::string w_name = "mtp_head_" + std::to_string(k) + ".weight";
                const std::string b_name = "mtp_head_" + std::to_string(k) + ".bias";
                head.weight = Tensor::zeros({mtp_hp.vocab_size, mtp_hp.d_model}, primary_stream, w_name.c_str());
                head.weight.requires_grad_();
                head.weight.ensure_grad();
                const uint64_t mtp_seed = weight_init_seed + 3 + static_cast<uint64_t>(k);
                Tensor::xavier_uniform_(head.weight, mtp_seed, primary_stream);
                head.bias = Tensor::zeros({mtp_hp.vocab_size}, primary_stream, b_name.c_str());
                head.bias.requires_grad_();
                head.bias.ensure_grad();
            }
            std::cout << "✓ MTP " << mtp_hp.k << " auxiliary heads created (alpha=" << mtp_hp.alpha
                      << ", warmup_steps=" << mtp_hp.alpha_warmup_steps << ")\n";
        }

        std::cout << "✓ GPU encoder initialized with " << init_hp.num_layers << " layers\n";
        std::cout << "  - Attention: GPU-accelerated\n";
        std::cout << "  - FFN: GPU-accelerated with fused GELU\n";
        std::cout << "  - Layer Norm: GPU-accelerated\n";

        if (init_hp.use_flash_attention) {
            std::cout << "⚡ Enabling Flash Attention 2...\n";
            encoder_ptr->setFlashAttention(true, init_hp.min_seq_len_for_flash);
            std::cout << "✓ Flash Attention enabled (min_seq_len=" << init_hp.min_seq_len_for_flash << ")\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ EXCEPTION in initGPU(): " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_CUDA

} // namespace GRIM

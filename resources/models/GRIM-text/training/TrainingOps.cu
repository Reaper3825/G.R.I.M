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

namespace GRIM {

#ifdef USE_CUDA

//======================================================//
//  GPU Layer Initialization
//======================================================//

void LanguageModel::initGPU(uint64_t weight_init_seed) {
    const auto& model_cfg = getConfig();
    const auto init_hp = HyperParameters::gpuModelInitializationHP(model_cfg);

    std::cout << "[initGPU] Verifying grouped GPU initialization config..." << std::endl;
    if (!init_hp.use_gpu) {
        throw std::runtime_error("[initGPU] use_gpu=false but GRIM-text REQUIRES GPU - fix ai_config.json");
    }

    try {
        std::cout << "[initGPU] Initializing GPU layer objects..." << std::endl;

        //======================================================//
        //  1) Verify startup-owned runtime prerequisites
        //
        //  CUDA device/context setup is owned by Phase1 startup. This method
        //  consumes only already-initialized TrainingState runtime resources
        //  and grouped construction views.
        //======================================================//

        //======================================================//
        //  2) Stream + cuBLAS prerequisites
        //======================================================//
        if (!training_state_.stream_ctrl.isInitialized()) {
            throw std::runtime_error(
                "FATAL: StreamController not initialized. "
                "Initialize stream_ctrl before initGPU() (TrainingState owns streams)."
            );
        }

        if (!training_state_.cublas_handle.get()) {
            throw std::runtime_error("FATAL: cuBLAS handle not initialized (training_state_.cublas_handle.get() == NULL)");
        }

        //======================================================//
        //  3) Build GPU encoder
        //
        //  Hyperparameters come from EncoderLayerConstructionHP, the grouped
        //  read view owned by HyperparameterGroupings.hpp. Construction inputs
        //  — positional encoding, startup init stream, init seed — come from
        //  EncoderConstructionBindings. Forward stream/cuBLAS live on the
        //  forward payload/request, not on layer configs.
        //======================================================//
        if (!isPBMInitialized()) {
            throw std::runtime_error(
                "[initGPU] FATAL: PBM not initialized before encoder construction! "
                "Call initPBM() BEFORE createGPUEncoder()");
        }

        const auto encoder_hp = HyperParameters::encoderLayerConstructionHP(model_cfg);
        EncoderConstructionBindings enc_bindings;
        enc_bindings.pos_encoding = &getPBMSpec();
        enc_bindings.init_stream = training_state_.stream_ctrl.getPrimaryStream();
        StreamController::fatalIfDefaultStream(enc_bindings.init_stream, "LanguageModel::initGPU");
        enc_bindings.weight_seed = weight_init_seed;
        // Issue #142: 1/sqrt(2*num_layers) for residual projection init
        enc_bindings.residual_scale = init_hp.residual_scale;

        std::cout << "[initGPU] Encoder construction bindings prepared" << std::endl;

        std::cout << "✓ Encoder using TrainingState construction bindings\n";

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
        std::cout << "✓ Encoder layers self-allocated weights\n";

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

            embedding_layer_ = std::make_unique<EmbeddingLayer>(emb_hp, emb_seed, enc_bindings.init_stream, true);

            if (!embedding_layer_->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: EmbeddingLayer not ready after construction!");
            }
            std::cout << "✓ Embedding layer created\n";
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
                lm_hp, lm_head_seed, enc_bindings.init_stream, tied_emb);

            if (!lm_head_layer_->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: LMHeadLayer not ready after construction!");
            }
            std::cout << "✓ LM Head layer created\n";
        }

        // ReasoningHead layer
        const auto reasoning_hp = HyperParameters::reasoningHeadConstructionHP(model_cfg);
        if (reasoning_hp.enabled) {
            const uint64_t rh_seed = weight_init_seed + 10;
            reasoning_head_layer_ = std::make_unique<ReasoningHeadLayer>(reasoning_hp, rh_seed, enc_bindings.init_stream);
            std::cout << "✓ ReasoningHead layer created\n";
        }

        // ExecutionBlock layer (differentiable register machine)
        const auto execution_hp = HyperParameters::executionBlockConstructionHP(model_cfg);
        if (execution_hp.enabled) {
            const uint64_t eb_seed = weight_init_seed + 20;
            execution_block_layer_ = std::make_unique<ExecutionBlockLayer>(execution_hp, eb_seed, enc_bindings.init_stream);
            std::cout << "✓ ExecutionBlock layer created\n";

            // Decode-time slot selector (pointer-selector baseline)
            const auto selector_hp = HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
            if (selector_hp.enabled) {
                const uint64_t sel_seed = weight_init_seed + 30;
                decode_time_slot_selector_layer_ = std::make_unique<DecodeTimeSlotSelectorLayer>(
                    selector_hp, sel_seed, enc_bindings.init_stream);

                decode_time_num_policy_ = std::make_unique<DecodeTimeNumPolicy>(selector_hp);

                std::cout << "✓ DecodeTimeSlotSelector created\n";
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
                head.weight = Tensor::zeros({mtp_hp.vocab_size, mtp_hp.d_model}, enc_bindings.init_stream, w_name.c_str());
                head.weight.requires_grad_();
                head.weight.ensure_grad();
                const uint64_t mtp_seed = weight_init_seed + 3 + static_cast<uint64_t>(k);
                Tensor::xavier_uniform_(head.weight, mtp_seed, enc_bindings.init_stream);
                head.bias = Tensor::zeros({mtp_hp.vocab_size}, enc_bindings.init_stream, b_name.c_str());
                head.bias.requires_grad_();
                head.bias.ensure_grad();
            }
            std::cout << "✓ MTP auxiliary heads created\n";
        }

        std::cout << "✓ GPU layer initialization complete\n";
        std::cout << "  - Attention: GPU-accelerated\n";
        std::cout << "  - FFN: GPU-accelerated with fused GELU\n";
        std::cout << "  - Layer Norm: GPU-accelerated\n";

        if (init_hp.use_flash_attention) {
            std::cout << "✓ Flash Attention configured from grouped encoder HP\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ EXCEPTION in initGPU(): " << e.what() << std::endl;
        throw;
    }
}

#endif // USE_CUDA

} // namespace GRIM

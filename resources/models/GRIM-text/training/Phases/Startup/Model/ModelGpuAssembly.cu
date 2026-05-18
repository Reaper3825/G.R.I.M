#ifndef USE_CUDA
#define USE_CUDA
#endif

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../../../GRIM/grim_language_model_cuda.hpp"
#include "../../../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/StreamController/StreamController_GPU.hpp"
#include "../../../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {

#ifdef USE_CUDA

//======================================================//
//  Startup Model GPU Assembly
//
//  Ownership boundary:
//    - Phase1/Startup/Model owns deciding WHEN this runs.
//    - LanguageModel owns the durable layer members being assembled here.
//
//  Consumes already-initialized startup prerequisites:
//    - CUDA context selected by ModelAllocationState.cu
//    - TrainingState StreamController
//    - TrainingState cuBLAS handle
//    - PBM positional-bias spec
//    - Phase1 RNG weight_init_seed
//
//  Creates durable model topology:
//    - GPU encoder layers
//    - embedding layer
//    - LM head
//    - optional reasoning/execution/decode-time/MTP heads
//
//  Does NOT own activation caches, optimizer state, parameter-group
//  registration, checkpoint loading, forward/backward, or Phase2 training.
//======================================================//

void LanguageModel::initGPU(uint64_t weight_init_seed) {
    const auto& model_cfg = config_;
    const auto init_hp = HyperParameters::gpuModelInitializationHP(model_cfg);

    std::cout << "[initGPU] Verifying grouped GPU initialization config..." << std::endl;
    if (!init_hp.use_gpu) {
        throw std::runtime_error("[initGPU] use_gpu=false but GRIM-text REQUIRES GPU - fix ai_config.json");
    }

    try {
        std::cout << "[initGPU] Assembling GPU model layer objects..." << std::endl;

        //======================================================//
        //  1) Verify startup-owned runtime prerequisites
        //======================================================//

        if (!training_state_.stream_ctrl.isInitialized()) {
            throw std::runtime_error(
                "FATAL: StreamController not initialized. "
                "ModelAllocationState must initialize stream_ctrl before initGPU()."
            );
        }

        if (!training_state_.cublas_handle.get()) {
            throw std::runtime_error("FATAL: cuBLAS handle not initialized (training_state_.cublas_handle.get() == NULL)");
        }

        if (!isPBMInitialized()) {
            throw std::runtime_error(
                "[initGPU] FATAL: PBM not initialized before encoder construction. "
                "ModelAllocationState must call initPBM() before initGPU().");
        }

        //======================================================//
        //  2) Build GPU encoder
        //
        //  Hyperparameters come from EncoderLayerConstructionHP, the grouped
        //  read view owned by HyperparameterGroupings.hpp. Construction resource
        //  bindings — positional encoding and startup init stream — come from
        //  EncoderConstructionBindings. The init seed is an explicit Phase1 RNG
        //  input. Forward stream/cuBLAS live on the forward payload/request,
        //  not on layer configs.
        //======================================================//

        const auto encoder_hp = HyperParameters::encoderLayerConstructionHP(model_cfg);
        EncoderConstructionBindings enc_bindings;
        enc_bindings.pos_encoding = &getPBMSpec();
        enc_bindings.init_stream = training_state_.stream_ctrl.getPrimaryStream();
        StreamController::fatalIfDefaultStream(enc_bindings.init_stream, "LanguageModel::initGPU");

        std::cout << "[initGPU] Encoder construction bindings prepared" << std::endl;
        std::cout << "✓ Encoder using TrainingState construction bindings\n";

        auto* encoder_ptr = new GPUGrimEncoder(encoder_hp, enc_bindings, weight_init_seed);
        gpu_encoder_.reset(encoder_ptr);

        // Layers self-allocate their own weights in the constructor. Verify all are ready.
        for (int layer = 0; layer < init_hp.num_layers; ++layer) {
            auto* gpu_layer = encoder_ptr->getLayer(layer);
            if (!gpu_layer || !gpu_layer->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: Encoder layer " + std::to_string(layer) +
                                         " not ready after self-allocation!");
            }
        }
        std::cout << "✓ Encoder layers self-allocated weights\n";

        //======================================================//
        //  3) Build persistent Embedding layer (Pattern B)
        //
        //  Self-allocates token weights [vocab_size, d_model]. Position
        //  information is injected inside attention via ALiBi/RoPE, so no
        //  separate position-embedding table is allocated (Rule 26).
        //  Must be created BEFORE LMHeadLayer (LM head aliases embedding for tied config).
        //======================================================//
        {
            const auto emb_hp = HyperParameters::embeddingLayerConstructionHP(model_cfg);
            const uint64_t emb_seed = weight_init_seed;

            embedding_layer_ = std::make_unique<EmbeddingLayer>(emb_hp, emb_seed, enc_bindings.init_stream, true);

            if (!embedding_layer_->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: EmbeddingLayer not ready after construction!");
            }
            std::cout << "✓ Embedding layer created\n";
        }

        //======================================================//
        //  4) Build persistent LM Head layer (Pattern B)
        //
        //  Self-allocates weights or aliases embedding for tied config.
        //  Owns final_rms_gamma. Created AFTER EmbeddingLayer because tied
        //  weights need embedding token storage.
        //======================================================//
        {
            const auto lm_hp = HyperParameters::lmHeadLayerConstructionHP(model_cfg);
            const uint64_t lm_head_seed = weight_init_seed + 1;

            Tensor* tied_emb = nullptr;
            if (lm_hp.tie_embeddings) {
                tied_emb = &embedding_layer_->tokenWeights();
                if (!tied_emb->data) {
                    throw std::runtime_error("[initGPU] FATAL: tied embedding token weights have NULL data");
                }
            }

            lm_head_layer_ = std::make_unique<LMHeadLayer>(
                lm_hp, lm_head_seed, enc_bindings.init_stream, tied_emb);

            if (!lm_head_layer_->weightsReady()) {
                throw std::runtime_error("[initGPU] FATAL: LMHeadLayer not ready after construction!");
            }
            std::cout << "✓ LM Head layer created\n";
        }

        //======================================================//
        //  5) Build optional ScratchBlock reasoning layer (Pattern B)
        //
        //  HyperparameterGroupings owns the static construction contract;
        //  startup model assembly supplies only the init stream. TrainingState
        //  must not allocate or configure this durable model layer.
        //======================================================//
        const auto scratch_hp = HyperParameters::scratchBlockConstructionHP(model_cfg);
        if (scratch_hp.enabled) {
            scratch_block_layer_ = std::make_unique<ScratchBlockLayer>(
                scratch_hp, enc_bindings.init_stream);

            if (!scratch_block_layer_->isEnabled()) {
                throw std::runtime_error("[initGPU] FATAL: ScratchBlockLayer constructed disabled while config.use_scratch_block=true");
            }
            if (!scratch_block_layer_->atomTypeEmbeddings().data ||
                !scratch_block_layer_->atomProjection().data) {
                throw std::runtime_error("[initGPU] FATAL: ScratchBlockLayer tensors not ready after construction");
            }
            std::cout << "✓ ScratchBlock layer created\n";
        }

        //======================================================//
        //  6) Build optional model heads/subsystems
        //======================================================//
        const auto reasoning_hp = HyperParameters::reasoningHeadConstructionHP(model_cfg);
        if (reasoning_hp.enabled) {
            const uint64_t rh_seed = weight_init_seed + 10;
            reasoning_head_layer_ = std::make_unique<ReasoningHeadLayer>(reasoning_hp, rh_seed, enc_bindings.init_stream);
            std::cout << "✓ ReasoningHead layer created\n";
        }

        const auto execution_hp = HyperParameters::executionBlockConstructionHP(model_cfg);
        if (execution_hp.enabled) {
            const uint64_t eb_seed = weight_init_seed + 20;
            execution_block_layer_ = std::make_unique<ExecutionBlockLayer>(execution_hp, eb_seed, enc_bindings.init_stream);
            std::cout << "✓ ExecutionBlock layer created\n";

            const auto selector_hp = HyperParameters::decodeTimeSelectorConstructionHP(model_cfg);
            if (selector_hp.enabled) {
                const uint64_t sel_seed = weight_init_seed + 30;
                decode_time_slot_selector_layer_ = std::make_unique<DecodeTimeSlotSelectorLayer>(
                    selector_hp, sel_seed, enc_bindings.init_stream);

                decode_time_num_policy_ = std::make_unique<DecodeTimeNumPolicy>(selector_hp);

                std::cout << "✓ DecodeTimeSlotSelector created\n";
            }
        }

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

        std::cout << "✓ GPU model layer assembly complete\n";
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

/**
 * @file GradAccumulationController_Integration.cu
 * @brief Implementation of ModelGradAccumulationController binding logic
 */

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "GradAccumulationController_Integration.hpp"
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/ScratchBlock/ScratchBlock_GPU.hpp"  // Issue #29: ScratchBlock gradient registration
#include "../LogRecorder/LogRecorder.hpp"

namespace {
constexpr auto kModuleGradBind = GRIM::Logging::ModuleId::Optimizer;
}

namespace GRIM {

void ModelGradAccumulationController::bindToModel(LanguageModel& model) {
    std::cout << "[BIND-A] bindToModel ENTER" << std::endl << std::flush;
    const auto& cfg = model.getConfig();
    std::cout << "[BIND-B] Got config, vocab_size=" << cfg.vocab_size << " d_model=" << cfg.d_model << std::endl << std::flush;
    auto& ts = model.getTrainingState();
    std::cout << "[BIND-C] Got TrainingState ref" << std::endl << std::flush;
    
#ifdef DEBUG_GRAD_BINDING
    // DEBUG: Verify pointer integrity at entry to bindToModel
    std::ostringstream oss_debug;
    oss_debug << "bindToModel Entry:\n"
              << "  model address=" << &model << "\n"
              << "  training_state offset=" << (reinterpret_cast<const char*>(&ts) - reinterpret_cast<const char*>(&model)) << " bytes\n"
              << "  embedding_grads=" << ts.embedding_grads() << "\n"
              << "  lm_head_weight_grads=" << ts.lm_head_weight_grads() << "\n"
              << "  embedding_grad_size=" << (cfg.vocab_size * cfg.d_model);
    GRIM::Logging::EmitModuleInfo(kModuleGradBind, oss_debug.str());
#endif
    
    // GQA dimensions
    std::cout << "[BIND-D] Computing GQA dimensions..." << std::endl << std::flush;
    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    const int kv_dim = ts.num_kv_heads * head_dim;
    const int total_qkv_dim = cfg.d_model + 2 * kv_dim;
    std::cout << "[BIND-E] GQA: head_dim=" << head_dim << " kv_dim=" << kv_dim << " total_qkv_dim=" << total_qkv_dim << std::endl << std::flush;
    
    // Clear any existing buffers
    std::cout << "[BIND-F] Clearing existing buffers..." << std::endl << std::flush;
    controller_.clearBuffers();
    std::cout << "[BIND-G] Buffers cleared" << std::endl << std::flush;
    
    // ========== Global Gradients ==========
    
    // WEIGHT TYING: When tie_embeddings=true, embedding_weights.grad == lm_head_weights.grad
    // (same pointer). Only register ONCE to avoid double-zeroing and corruption.
    const bool grads_are_tied = cfg.tie_embeddings;
    std::cout << "[BIND-H] tie_embeddings=" << (grads_are_tied ? "true" : "false") << std::endl << std::flush;
    
    // Get the grad pointers via accessor methods
    std::cout << "[BIND-I] Getting embedding_grads pointer..." << std::endl << std::flush;
    float* emb_grads = ts.embedding_grads();
    std::cout << "[BIND-J] emb_grads=" << emb_grads << std::endl << std::flush;
    std::cout << "[BIND-K] Getting lm_head_weight_grads pointer..." << std::endl << std::flush;
    float* lm_grads = ts.lm_head_weight_grads();
    std::cout << "[BIND-L] lm_grads=" << lm_grads << std::endl << std::flush;
    
    if (grads_are_tied) {
        std::cout << "[BIND-M] Tied path: checking pointer aliasing..." << std::endl << std::flush;
        // VALIDATION: Verify pointer aliasing was set up correctly by initTrainingState
        if (emb_grads != lm_grads) {
            std::ostringstream oss;
            oss << "FATAL: cfg.tie_embeddings=true but pointers are NOT aliased!\n"
                << "  embedding_grads=" << emb_grads << "\n"
                << "  lm_head_weight_grads=" << lm_grads << "\n"
                << "  This indicates initTrainingState() was not called before bindToModel().";
            GRIM::Logging::EmitModuleError(kModuleGradBind, oss.str());
            std::exit(1);
        }
        
        std::cout << "[BIND-N] Pointers aliased correctly, registering tied buffer..." << std::endl << std::flush;
        // Tied: register single combined buffer
        if (lm_grads) {
            controller_.registerGradientBuffer(
                "embedding_lm_head_tied_grads",
                lm_grads,
                static_cast<std::size_t>(cfg.vocab_size) * cfg.d_model
            );
            std::cout << "[BIND-O] Registered tied embedding/lm_head buffer" << std::endl << std::flush;
        }
    } else {
        std::cout << "[BIND-M2] Untied path: checking NOT aliased..." << std::endl << std::flush;
        // VALIDATION: Verify pointers are NOT aliased when tying is disabled
        if (emb_grads == lm_grads && emb_grads != nullptr) {
            std::ostringstream oss;
            oss << "FATAL: cfg.tie_embeddings=false but pointers ARE aliased!\n"
                << "  embedding_grads=" << emb_grads << "\n"
                << "  lm_head_weight_grads=" << lm_grads << "\n"
                << "  Configuration mismatch detected.";
            GRIM::Logging::EmitModuleError(kModuleGradBind, oss.str());
            std::exit(1);
        }
        
        std::cout << "[BIND-N2] Registering separate embedding buffer..." << std::endl << std::flush;
        // Untied: register both separately
        const std::size_t embedding_grad_size = static_cast<std::size_t>(cfg.vocab_size) * cfg.d_model;
        if (emb_grads && embedding_grad_size > 0) {
            controller_.registerGradientBuffer(
                "embedding_grads",
                emb_grads,
                embedding_grad_size
            );
            std::cout << "[BIND-O2] Registered embedding_grads" << std::endl << std::flush;
        }
        
        std::cout << "[BIND-P2] Registering separate lm_head buffer..." << std::endl << std::flush;
        if (lm_grads) {
            controller_.registerGradientBuffer(
                "lm_head_weight_grads",
                lm_grads,
                static_cast<std::size_t>(cfg.vocab_size) * cfg.d_model
            );
            std::cout << "[BIND-Q2] Registered lm_head_weight_grads" << std::endl << std::flush;
        }
    }
    
    std::cout << "[BIND-P] Checking lm_head_bias..." << std::endl << std::flush;
    // LM head bias grads (Tensor member access)
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (ts.lm_head_bias.has_grad()) {
        std::cout << "[BIND-Q] Registering lm_head_bias_grads=" << ts.lm_head_bias.grad_data() << std::endl << std::flush;
        controller_.registerGradientBuffer(
            "lm_head_bias_grads",
            ts.lm_head_bias.grad_data(),
            static_cast<std::size_t>(cfg.vocab_size)
        );
    }

    std::cout << "[BIND-R] Checking numeric_head_weights..." << std::endl << std::flush;
    // Numeric head grads (Tensor member access)
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (ts.numeric_head_weights.has_grad()) {
        std::cout << "[BIND-S] Registering numeric_head_weight_grads=" << ts.numeric_head_weights.grad_data() << std::endl << std::flush;
        controller_.registerGradientBuffer(
            "numeric_head_weight_grads",
            ts.numeric_head_weights.grad_data(),
            static_cast<std::size_t>(cfg.d_model)
        );
    }
    std::cout << "[BIND-T] Checking numeric_head_bias..." << std::endl << std::flush;
    // ISSUE #59: Use has_grad() and grad_data() accessors
    if (ts.numeric_head_bias.has_grad()) {
        std::cout << "[BIND-U] Registering numeric_head_bias_grads=" << ts.numeric_head_bias.grad_data() << std::endl << std::flush;
        controller_.registerGradientBuffer(
            "numeric_head_bias_grads",
            ts.numeric_head_bias.grad_data(),
            static_cast<std::size_t>(1)
        );
    }
    
    std::cout << "[BIND-V] Checking final_rms_gamma_grads..." << std::endl << std::flush;
    // Issue #33: Final RMSNorm gamma gradients (normalizes encoder output before LM head)
    if (ts.final_rms_gamma_grads()) {
        std::cout << "[BIND-W] Registering final_rms_gamma_grads=" << ts.final_rms_gamma_grads() << std::endl << std::flush;
        controller_.registerGradientBuffer(
            "final_rms_gamma_grads",
            ts.final_rms_gamma_grads(),
            static_cast<std::size_t>(cfg.d_model)
        );
    }
    
    std::cout << "[BIND-X] Global gradients done, starting per-layer..." << std::endl << std::flush;
    
    // ========== MIGRATED TO TENSOR SYSTEM ==========
    // Issue #45 fix: The following intermediate gradient buffers are now managed by the
    // Tensor system in TrainingState_GPU.hpp. Zeroing is handled by zeroIntermediateGrads()
    // which is called at the start of each accumulation window.
    //
    // Migrated buffers (NO LONGER registered here):
    // - grad_ffn_input, grad_ffn_hidden, grad_attn_input
    // - grad_q, grad_k, grad_v
    // - grad_encoder_out, grad_logits
    // - position_embedding_grads
    //
    // The Tensor::zero_grad(stream) method provides proper CUDA stream synchronization
    // and eliminates the gradient explosion bug that occurred when these buffers
    // were not zeroed between accumulation micro-steps.
    
    // ========== Per-Layer Gradients ==========
    
    // Get the GPU encoder to access the actual RMSNorm gradient buffers
    // IMPORTANT: The encoder owns its own gamma_grad buffers which are different
    // from TrainingState's rms1_gamma_grads/rms2_gamma_grads vectors!
    // We must register the encoder's actual buffers so they get zeroed properly.
    std::cout << "[BIND-Y] Getting GPU encoder..." << std::endl << std::flush;
    auto* gpu_encoder = &model.getGpuEncoder();
    std::cout << "[BIND-Z] gpu_encoder=" << gpu_encoder << std::endl << std::flush;
    
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        std::cout << "[BIND-LAYER-" << layer << "] Processing layer " << layer << std::endl << std::flush;
        std::string prefix = "layer" + std::to_string(layer) + "_";
        
        // RMSNorm gamma gradients - use encoder's actual buffers!
        // The encoder's getRMS1GammaGrad()/getRMS2GammaGrad() are what backward() writes to,
        // NOT the TrainingState vectors (which are a separate unused allocation).
        auto* enc = gpu_encoder ? gpu_encoder->getLayer(layer) : nullptr;
        if (!enc) {
            std::cout << "[BIND-LAYER-" << layer << "] WARNING: enc is nullptr!" << std::endl << std::flush;
            continue;
        }
        
        if (enc->getRMS1GammaGrad()) {
            controller_.registerGradientBuffer(
                prefix + "rms1_gamma",
                enc->getRMS1GammaGrad(),
                static_cast<std::size_t>(cfg.d_model)
            );
        }
        if (enc->getRMS2GammaGrad()) {
            controller_.registerGradientBuffer(
                prefix + "rms2_gamma",
                enc->getRMS2GammaGrad(),
                static_cast<std::size_t>(cfg.d_model)
            );
        }
        
        // Attention QKV weight/bias gradients (GQA-aware)
        // AUTOGRAD MIGRATION: Use encoder's Tensor.grad instead of TrainingState raw pointers
        if (enc->getAttnWqkvGrad()) {
            controller_.registerGradientBuffer(
                prefix + "attn_qkv_weight",
                enc->getAttnWqkvGrad(),
                static_cast<std::size_t>(total_qkv_dim) * cfg.d_model
            );
        }
        if (enc->getAttnBqkvGrad()) {
            controller_.registerGradientBuffer(
                prefix + "attn_qkv_bias",
                enc->getAttnBqkvGrad(),
                static_cast<std::size_t>(total_qkv_dim)
            );
        }
        
        // Attention output projection gradients
        // AUTOGRAD MIGRATION: Use encoder's Tensor.grad
        if (enc && enc->getAttnWoGrad()) {
            controller_.registerGradientBuffer(
                prefix + "attn_o_weight",
                enc->getAttnWoGrad(),
                static_cast<std::size_t>(cfg.d_model) * cfg.d_model
            );
        }
        if (enc && enc->getAttnBoGrad()) {
            controller_.registerGradientBuffer(
                prefix + "attn_o_bias",
                enc->getAttnBoGrad(),
                static_cast<std::size_t>(cfg.d_model)
            );
        }
        
        // FFN W1 gradients
        // AUTOGRAD MIGRATION: Use encoder's Tensor.grad
        if (enc && enc->getFFNW1Grad()) {
            controller_.registerGradientBuffer(
                prefix + "ffn_w1_weight",
                enc->getFFNW1Grad(),
                static_cast<std::size_t>(cfg.d_ff) * cfg.d_model
            );
        }
        if (enc && enc->getFFNB1Grad()) {
            controller_.registerGradientBuffer(
                prefix + "ffn_w1_bias",
                enc->getFFNB1Grad(),
                static_cast<std::size_t>(cfg.d_ff)
            );
        }
        
        // FFN W2 gradients
        // AUTOGRAD MIGRATION: Use encoder's Tensor.grad
        if (enc && enc->getFFNW2Grad()) {
            controller_.registerGradientBuffer(
                prefix + "ffn_w2_weight",
                enc->getFFNW2Grad(),
                static_cast<std::size_t>(cfg.d_model) * cfg.d_ff
            );
        }
        if (enc && enc->getFFNB2Grad()) {
            controller_.registerGradientBuffer(
                prefix + "ffn_w2_bias",
                enc->getFFNB2Grad(),
                static_cast<std::size_t>(cfg.d_model)
            );
        }
        std::cout << "[BIND-LAYER-" << layer << "] Done" << std::endl << std::flush;
    }
    std::cout << "[BIND-LAYERS-DONE] All " << cfg.num_layers << " layers processed" << std::endl << std::flush;
    
    // ========== ScratchBlock Gradients (Issue #29: CRITICAL FIX) ==========
    // These buffers were NOT registered previously, causing infinite gradient
    // accumulation and explosion after ~490 batches!
    std::cout << "[BIND-SB-A] Getting ScratchBlock..." << std::endl << std::flush;
    auto* scratch_block = model.getScratchBlockLayer();
    std::cout << "[BIND-SB-B] scratch_block=" << scratch_block << " use_scratch_block=" << cfg.use_scratch_block << std::endl << std::flush;
    if (scratch_block && cfg.use_scratch_block) {
        const auto& sb_config = scratch_block->getConfig();
        std::cout << "[BIND-SB-C] Got ScratchBlock config" << std::endl << std::flush;
        
        // Atom type embeddings gradients: [NUM_ATOM_TYPES, atom_embedding_dim]
        if (scratch_block->getAtomTypeEmbeddingsGrad()) {
            // NUM_ATOM_TYPES is a compile-time constant in HyperParameters
            const int num_atom_types = 16;  // From HyperParameters::NUM_ATOM_TYPES
            controller_.registerGradientBuffer(
                "scratch_block_atom_type_embeddings_grad",
                scratch_block->getAtomTypeEmbeddingsGrad(),
                static_cast<std::size_t>(num_atom_types) * sb_config.atom_embedding_dim
            );
        }
        
        // Atom projection gradients: [atom_embedding_dim, d_model]
        if (scratch_block->getAtomProjectionGrad()) {
            controller_.registerGradientBuffer(
                "scratch_block_atom_projection_grad",
                scratch_block->getAtomProjectionGrad(),
                static_cast<std::size_t>(sb_config.atom_embedding_dim) * cfg.d_model
            );
        }
        
        // Text feature projection gradients: [16, d_model] (FP16 text features → d_model)
        if (scratch_block->getTextFeatureProjectionGrad()) {
            const int text_feature_dim = 16;  // kTextFeatureDim in ScratchBlock_GPU.cu
            controller_.registerGradientBuffer(
                "scratch_block_text_projection_grad",
                scratch_block->getTextFeatureProjectionGrad(),
                static_cast<std::size_t>(text_feature_dim) * cfg.d_model
            );
        }
        std::cout << "[BIND-SB-D] ScratchBlock grads registered" << std::endl << std::flush;
    }
    
    std::cout << "[BIND-CALLBACK] Setting up zeroIntermediateGrads callback..." << std::endl << std::flush;
    // Issue #45 FIX: Set up callback to zero Tensor-managed intermediate gradients
    // These buffers were migrated from raw float* registrations to TensorContract::Tensor.
    // The callback invokes TrainingState::zeroIntermediateGrads() which calls zero_grad()
    // on all intermediate gradient tensors, fixing the gradient explosion bug.
    controller_.setAdditionalZeroCallback([&ts](cudaStream_t stream) {
        ts.zeroIntermediateGrads(stream);
    });
    std::cout << "[BIND-CALLBACK-DONE] Callback set" << std::endl << std::flush;
    
    // Store model reference for later use
    std::cout << "[BIND-STORE] Storing model pointer..." << std::endl << std::flush;
    model_ = &model;
    std::cout << "[BIND-EXIT] bindToModel complete" << std::endl << std::flush;
}

} // namespace GRIM

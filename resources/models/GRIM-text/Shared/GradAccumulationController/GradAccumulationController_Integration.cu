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
    const auto& cfg = model.getConfig();
    auto& ts = model.getTrainingState();
    
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
    const int head_dim = cfg.head_dim;  // Use pre-computed value from config
    const int kv_dim = ts.num_kv_heads * head_dim;
    const int total_qkv_dim = cfg.d_model + 2 * kv_dim;
    
    // Clear any existing buffers
    controller_.clearBuffers();
    
    // ========== Global Gradients ==========
    
    // WEIGHT TYING: When tie_embeddings=true, embedding_weights.grad == lm_head_weights.grad
    // (same pointer). Only register ONCE to avoid double-zeroing and corruption.
    const bool grads_are_tied = cfg.tie_embeddings;
    
    // Get the grad pointers via accessor methods
    float* emb_grads = ts.embedding_grads();
    float* lm_grads = ts.lm_head_weight_grads();
    
    if (grads_are_tied) {
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
        
        // Tied: register single combined buffer
        if (lm_grads) {
            controller_.registerGradientBuffer(
                "embedding_lm_head_tied_grads",
                lm_grads,
                static_cast<std::size_t>(cfg.vocab_size) * cfg.d_model
            );
        }
    } else {
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
        
        // Untied: register both separately
        const std::size_t embedding_grad_size = static_cast<std::size_t>(cfg.vocab_size) * cfg.d_model;
        if (emb_grads && embedding_grad_size > 0) {
            controller_.registerGradientBuffer(
                "embedding_grads",
                emb_grads,
                embedding_grad_size
            );
        }
        
        if (lm_grads) {
            controller_.registerGradientBuffer(
                "lm_head_weight_grads",
                lm_grads,
                static_cast<std::size_t>(cfg.vocab_size) * cfg.d_model
            );
        }
    }
    
    // LM head bias grads (Tensor member access)
    if (ts.lm_head_bias.grad) {
        controller_.registerGradientBuffer(
            "lm_head_bias_grads",
            ts.lm_head_bias.grad,
            static_cast<std::size_t>(cfg.vocab_size)
        );
    }

    // Numeric head grads (Tensor member access)
    if (ts.numeric_head_weights.grad) {
        controller_.registerGradientBuffer(
            "numeric_head_weight_grads",
            ts.numeric_head_weights.grad,
            static_cast<std::size_t>(cfg.d_model)
        );
    }
    if (ts.numeric_head_bias.grad) {
        controller_.registerGradientBuffer(
            "numeric_head_bias_grads",
            ts.numeric_head_bias.grad,
            static_cast<std::size_t>(1)
        );
    }
    
    // Issue #33: Final RMSNorm gamma gradients (normalizes encoder output before LM head)
    if (ts.final_rms_gamma_grads()) {
        controller_.registerGradientBuffer(
            "final_rms_gamma_grads",
            ts.final_rms_gamma_grads(),
            static_cast<std::size_t>(cfg.d_model)
        );
    }
    
    // ========== Temporary Gradient Buffers (CRITICAL) ==========
    // These buffers are reused across layers during backward pass.
    // They MUST be registered so they're zeroed at window start,
    // otherwise they accumulate indefinitely causing gradient explosion!
    
    if (ts.grad_ffn_input) {
        controller_.registerGradientBuffer(
            "grad_ffn_input_temp",
            ts.grad_ffn_input,
            static_cast<std::size_t>(ts.max_cached_tokens) * cfg.d_model
        );
    }
    
    if (ts.grad_ffn_hidden) {
        controller_.registerGradientBuffer(
            "grad_ffn_hidden_temp",
            ts.grad_ffn_hidden,
            static_cast<std::size_t>(ts.max_cached_tokens) * cfg.d_ff
        );
    }
    
    if (ts.grad_attn_input) {
        controller_.registerGradientBuffer(
            "grad_attn_input_temp",
            ts.grad_attn_input,
            static_cast<std::size_t>(ts.max_cached_tokens) * cfg.d_model
        );
    }
    
    // Grad Q/K/V are already zeroed per-layer in backward(), but register for completeness
    if (ts.grad_q) {
        const int head_dim_qkv = cfg.head_dim;  // Use pre-computed value from config
        const int q_size = ts.max_cached_tokens * cfg.num_heads * head_dim_qkv;
        controller_.registerGradientBuffer(
            "grad_q_temp",
            ts.grad_q,
            static_cast<std::size_t>(q_size)
        );
    }
    if (ts.grad_k) {
        const int head_dim_qkv = cfg.head_dim;  // Use pre-computed value from config
        const int kv_size = ts.max_cached_tokens * ts.num_kv_heads * head_dim_qkv;
        controller_.registerGradientBuffer(
            "grad_k_temp",
            ts.grad_k,
            static_cast<std::size_t>(kv_size)
        );
    }
    if (ts.grad_v) {
        const int head_dim_qkv = cfg.head_dim;  // Use pre-computed value from config
        const int kv_size = ts.max_cached_tokens * ts.num_kv_heads * head_dim_qkv;
        controller_.registerGradientBuffer(
            "grad_v_temp",
            ts.grad_v,
            static_cast<std::size_t>(kv_size)
        );
    }
    
    // ========== Per-Layer Gradients ==========
    
    // Get the GPU encoder to access the actual RMSNorm gradient buffers
    // IMPORTANT: The encoder owns its own gamma_grad buffers which are different
    // from TrainingState's rms1_gamma_grads/rms2_gamma_grads vectors!
    // We must register the encoder's actual buffers so they get zeroed properly.
    auto* gpu_encoder = &model.getGpuEncoder();
    
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        std::string prefix = "layer" + std::to_string(layer) + "_";
        
        // RMSNorm gamma gradients - use encoder's actual buffers!
        // The encoder's getRMS1GammaGrad()/getRMS2GammaGrad() are what backward() writes to,
        // NOT the TrainingState vectors (which are a separate unused allocation).
        auto* enc = gpu_encoder ? gpu_encoder->getLayer(layer) : nullptr;
        if (enc && enc->getRMS1GammaGrad()) {
            controller_.registerGradientBuffer(
                prefix + "rms1_gamma",
                enc->getRMS1GammaGrad(),
                static_cast<std::size_t>(cfg.d_model)
            );
        }
        if (enc && enc->getRMS2GammaGrad()) {
            controller_.registerGradientBuffer(
                prefix + "rms2_gamma",
                enc->getRMS2GammaGrad(),
                static_cast<std::size_t>(cfg.d_model)
            );
        }
        
        // Attention QKV weight/bias gradients (GQA-aware)
        if (layer < static_cast<int>(ts.attn_qkv_weight_grads.size()) && ts.attn_qkv_weight_grads[layer]) {
            controller_.registerGradientBuffer(
                prefix + "attn_qkv_weight",
                ts.attn_qkv_weight_grads[layer],
                static_cast<std::size_t>(total_qkv_dim) * cfg.d_model
            );
        }
        if (layer < static_cast<int>(ts.attn_qkv_bias_grads.size()) && ts.attn_qkv_bias_grads[layer]) {
            controller_.registerGradientBuffer(
                prefix + "attn_qkv_bias",
                ts.attn_qkv_bias_grads[layer],
                static_cast<std::size_t>(total_qkv_dim)
            );
        }
        
        // Attention output projection gradients
        if (layer < static_cast<int>(ts.attn_out_weight_grads.size()) && ts.attn_out_weight_grads[layer]) {
            controller_.registerGradientBuffer(
                prefix + "attn_o_weight",
                ts.attn_out_weight_grads[layer],
                static_cast<std::size_t>(cfg.d_model) * cfg.d_model
            );
        }
        if (layer < static_cast<int>(ts.attn_out_bias_grads.size()) && ts.attn_out_bias_grads[layer]) {
            controller_.registerGradientBuffer(
                prefix + "attn_o_bias",
                ts.attn_out_bias_grads[layer],
                static_cast<std::size_t>(cfg.d_model)
            );
        }
        
        // FFN W1 gradients
        if (layer < static_cast<int>(ts.ffn_w1_grads.size()) && ts.ffn_w1_grads[layer]) {
            controller_.registerGradientBuffer(
                prefix + "ffn_w1_weight",
                ts.ffn_w1_grads[layer],
                static_cast<std::size_t>(cfg.d_ff) * cfg.d_model
            );
        }
        if (layer < static_cast<int>(ts.ffn_b1_grads.size()) && ts.ffn_b1_grads[layer]) {
            controller_.registerGradientBuffer(
                prefix + "ffn_w1_bias",
                ts.ffn_b1_grads[layer],
                static_cast<std::size_t>(cfg.d_ff)
            );
        }
        
        // FFN W2 gradients
        if (layer < static_cast<int>(ts.ffn_w2_grads.size()) && ts.ffn_w2_grads[layer]) {
            controller_.registerGradientBuffer(
                prefix + "ffn_w2_weight",
                ts.ffn_w2_grads[layer],
                static_cast<std::size_t>(cfg.d_model) * cfg.d_ff
            );
        }
        if (layer < static_cast<int>(ts.ffn_b2_grads.size()) && ts.ffn_b2_grads[layer]) {
            controller_.registerGradientBuffer(
                prefix + "ffn_w2_bias",
                ts.ffn_b2_grads[layer],
                static_cast<std::size_t>(cfg.d_model)
            );
        }
    }
    
    // ========== ScratchBlock Gradients (Issue #29: CRITICAL FIX) ==========
    // These buffers were NOT registered previously, causing infinite gradient
    // accumulation and explosion after ~490 batches!
    auto* scratch_block = model.getScratchBlockLayer();
    if (scratch_block && cfg.use_scratch_block) {
        const auto& sb_config = scratch_block->getConfig();
        
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
    }
    
    // Store model reference for later use
    model_ = &model;
}

} // namespace GRIM

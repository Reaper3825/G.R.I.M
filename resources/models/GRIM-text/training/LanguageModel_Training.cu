#define USE_CUDA

//======================================================//
//  LanguageModel_Training.cu - Parameter Group Management
//======================================================//
//
//  This file contains LanguageModel methods for parameter group
//  registration and debug probes.
//
//  Methods in this file:
//  - buildParameterGroups() - Collect learnable parameters from layers
//  - dumpGradientValues() - Debug gradient dump
//
//  MOVED to AdamW_Kernal_GPU.{hpp,cu} (ownership cleanup):
//  - updateWeights() → launchAdamWStep() free function
//  - resetOptimizerMoments() → resetAdamWMoments() free function
//  - scaleOptimizerMoments() → scaleAdamWMoments() free function
//  Rationale: AdamW stepping is training infrastructure, not model logic.
//  The model owns its parameter groups (via buildParameterGroups()),
//  but the optimizer operates ON those groups from outside.
//
//======================================================//

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <set>
#include <mutex>
#include <fstream>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"
#include "../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../Shared/HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM {

#ifdef USE_CUDA

//======================================================//
//  Logging Setup
//======================================================//

namespace {
constexpr auto kBwdModule = GRIM::Logging::ModuleId::BackwardPass;

#define BWD_INFO(msg) do { std::ostringstream _oss; _oss << msg; GRIM::Logging::EmitModuleInfo(kBwdModule, _oss.str()); } while (0)
#define BWD_WARN(msg) do { std::ostringstream _oss; _oss << msg; GRIM::Logging::EmitModuleWarning(kBwdModule, _oss.str()); } while (0)
#define BWD_ERROR(msg) do { std::ostringstream _oss; _oss << msg; GRIM::Logging::EmitModuleError(kBwdModule, _oss.str()); } while (0)
std::string g_gradcheck_log_path;
std::mutex g_gradcheck_mutex;

// NonFiniteScanResult, scanNonFiniteKernel, endsWith, getGroupShape, scanFirstNonFinite
// ALL DELETED (Rule 26). They were only used by computeGradNorm() which is now deleted.
// Gradient norm measurement is done via free functions in GradNormGPU.{cu,hpp}.
}

void setGradCheckLogPath(const std::string& path) {
    std::lock_guard<std::mutex> lock(g_gradcheck_mutex);
    g_gradcheck_log_path = path;
}

// zeroGrad() and backward() DELETED (Rule 20).
// Gradient zeroing is handled by executeAutogradBackward() when accumulate=false.
// Backward pass is called via autogradTrainingStep() which does forward+loss+backward.


//======================================================//
//  buildParameterGroups - Initialize Parameter Groups for Optimizer
//======================================================//

void LanguageModel::buildParameterGroups() {
    fprintf(stderr, "[buildParameterGroups] ENTER\n"); fflush(stderr);
    // Clear parameter group metadata (optimizer states are managed by TrainingState)
    parameter_groups_.clear();
    
    const auto& cfg = config_;
    const int num_kv_heads = training_state_.num_kv_heads;
    fprintf(stderr, "[buildParameterGroups] cfg: num_layers=%d num_heads=%d head_dim=%d num_kv_heads=%d d_model=%d vocab=%u tie=%d\n",
            cfg.num_layers, cfg.num_heads, cfg.head_dim, num_kv_heads, cfg.d_model, cfg.vocab_size, (int)cfg.tie_embeddings); fflush(stderr);
    
    // Validate GQA dimensions using TensorContract
    TensorContract::GQADims gqa_dims{cfg.num_heads, num_kv_heads, cfg.head_dim};
    if (!gqa_dims.is_valid()) {
        BWD_ERROR("[buildParameterGroups] TensorContract GQA validation failed!");
    }

    GPUGrimEncoder& gpu_encoder = getGpuEncoder();

    // ── Helper: Register a Tensor as a parameter group ───────────────────────
    // Eliminates manual size computation — uses Tensor::numel() directly.
    // Rule 20: tensor MUST have data and grad, otherwise skip with warning.
    auto registerTensor = [&](const std::string& name, Tensor& tensor,
                              ParamGroupType type, int layer = -1,
                              float wd_mult = 1.0f, float lr_mult = 1.0f) {
        if (!tensor.data || !tensor.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] " + name + " has no data or grad! data=" 
                                     + std::to_string(reinterpret_cast<uintptr_t>(tensor.data)) 
                                     + " has_grad=" + std::to_string(tensor.has_grad()) 
                                     + " layer=" + std::to_string(layer));
        }
        ParameterGroup group;
        group.name = name;
        group.tensor = &tensor;
        group.m_tensor = nullptr;
        group.v_tensor = nullptr;
        group.type = type;
        group.layer_index = layer;
        group.weight_decay_multiplier = wd_mult;
        group.lr_multiplier = lr_mult;
        if (layer >= 0) {
            group.upsilon = HyperParameters::UPSILON_BASE * 
                sqrtf(static_cast<float>(HyperParameters::UPSILON_REFERENCE_LAYERS) / static_cast<float>(layer + 1));
        }
        parameter_groups_.push_back(group);
    };

    // Helper: Register a bias or norm tensor (weight_decay = 0.0)
    // Standard AdamW practice: biases and normalization parameters should NOT have weight decay
    auto registerNonDecayTensor = [&](const std::string& name, Tensor& tensor,
                                      ParamGroupType type, int layer = -1) {
        registerTensor(name, tensor, type, layer, 0.0f);
    };

    // Helper: Try to register a bias tensor if it has been allocated (guards use_bias=false case)
    auto tryRegisterBias = [&](const std::string& name, Tensor& tensor,
                               ParamGroupType type, int layer = -1) {
        if (tensor.data && tensor.has_grad()) {
            registerNonDecayTensor(name, tensor, type, layer);
        } else {
            fprintf(stderr, "[buildParameterGroups] %s skipped (use_bias=false or not allocated)\n", name.c_str());
        }
    };

    // ── Top-level parameters ─────────────────────────────────────────────────
    
    // Embedding weights - SKIP when tied (handled by LM head group below)
    // When tie_embeddings=true, embedding grad buffer == lm_head grad buffer (same pointer)
    // Registering both would double-count gradients and corrupt optimizer state
    if (!cfg.tie_embeddings) {
        registerTensor("embedding", embedding_layer_->tokenWeights(), ParamGroupType::EMBEDDING);
    }

    // Issue #113: Sinusoidal position embeddings are FIXED (not learned).
    // Only register if EmbeddingLayer allocated them (i.e., learned position embeddings are in use).
    if (embedding_layer_->hasPositionEmbeddings() && embedding_layer_->positionWeights().has_grad()) {
        registerTensor("position_embedding", embedding_layer_->positionWeights(), ParamGroupType::EMBEDDING);
    } else {
        fprintf(stderr, "[buildParameterGroups] position_embedding skipped (sinusoidal/fixed, not learned)\n");
    }

    // LM head weights (includes tied embedding grads when tie_embeddings=true)
    fprintf(stderr, "[buildParameterGroups] DIAG-A: about to register LM head\n"); fflush(stderr);
    {
        std::string lm_name = cfg.tie_embeddings ? "embedding_lm_head_tied" : "lm_head_weight";
        registerTensor(lm_name, lm_head_layer_->weights(), ParamGroupType::LM_HEAD);
    }
    // LM head bias — no weight decay
    tryRegisterBias("lm_head_bias", lm_head_layer_->bias(), ParamGroupType::LM_HEAD);
    fprintf(stderr, "[buildParameterGroups] DIAG-B: LM head done, starting %d layers\n", cfg.num_layers); fflush(stderr);
    // ── Per-layer encoder parameters ─────────────────────────────────────────
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        GPUEncoderLayer* enc = gpu_encoder.getLayer(layer);
        if (!enc) {
            BWD_WARN("[buildParameterGroups] Layer " << layer << " is NULL - skipping");
            continue;
        }
        
        std::string prefix = "layer" + std::to_string(layer);
        
        // Attention weights/biases
        registerTensor(prefix + "_qkv_weight", enc->attnWqkv(), ParamGroupType::ATTENTION, layer);
        tryRegisterBias(prefix + "_qkv_bias",  enc->attnBqkv(), ParamGroupType::ATTENTION, layer);
        registerTensor(prefix + "_wo_weight",   enc->attnWo(),   ParamGroupType::ATTENTION, layer);
        tryRegisterBias(prefix + "_wo_bias",    enc->attnBo(),   ParamGroupType::ATTENTION, layer);
        
        // FFN weights (SwiGLU: W_gate, W1, W2, b2)
        registerTensor(prefix + "_ffn_w_gate", enc->ffnWGate(), ParamGroupType::FFN, layer);
        registerTensor(prefix + "_ffn_w1", enc->ffnW1(), ParamGroupType::FFN, layer);
        registerTensor(prefix + "_ffn_w2", enc->ffnW2(), ParamGroupType::FFN, layer);
        tryRegisterBias(prefix + "_ffn_b2", enc->ffnB2(), ParamGroupType::FFN, layer);
        
        // RMSNorm gamma (pre-norm) — no weight decay
        registerNonDecayTensor(prefix + "_rms1_gamma", enc->rms1Gamma(), ParamGroupType::RMSNORM, layer);
        registerNonDecayTensor(prefix + "_rms2_gamma", enc->rms2Gamma(), ParamGroupType::RMSNORM, layer);
        // Issue #148: Sandwich norm gammas REMOVED (no post-residual normalization)
        
        // LayerScale (Issue #109) — learnable scalars, no weight decay
        // BUG FIX: These were zero_grad'd but NEVER registered with optimizer — frozen since creation!
        // Grouped under RMSNORM since they're normalization-adjacent (24 total scalars, negligible norm)
        tryRegisterBias(prefix + "_layer_scale1", enc->layerScale1(), ParamGroupType::RMSNORM, layer);
        tryRegisterBias(prefix + "_layer_scale2", enc->layerScale2(), ParamGroupType::RMSNORM, layer);
    }

    fprintf(stderr, "[buildParameterGroups] DIAG-D: all %d layers done, registering scratchblock/final_rms\n", cfg.num_layers); fflush(stderr);
    // ── ScratchBlock parameters ──
    fprintf(stderr, "[buildParameterGroups] DIAG-D0: scratch_block_layer_=%p\n", (void*)scratch_block_layer_.get()); fflush(stderr);
    if (scratch_block_layer_) {
        fprintf(stderr, "[buildParameterGroups] DIAG-D0a: isEnabled=%d\n", (int)scratch_block_layer_->isEnabled()); fflush(stderr);
    }
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        try {
            auto& ate = scratch_block_layer_->atomTypeEmbeddings();
            fprintf(stderr, "[buildParameterGroups] DIAG-D1: atom_type_embeddings data=%p grad=%d numel=%zu\n",
                    (void*)ate.data, (int)ate.has_grad(), ate.numel()); fflush(stderr);
            registerTensor("scratch_block_atom_type_embeddings", ate, ParamGroupType::SCRATCHBLOCK);
            fprintf(stderr, "[buildParameterGroups] DIAG-D1: registered OK\n"); fflush(stderr);

            auto& ap = scratch_block_layer_->atomProjection();
            fprintf(stderr, "[buildParameterGroups] DIAG-D2: atom_projection data=%p grad=%d numel=%zu\n",
                    (void*)ap.data, (int)ap.has_grad(), ap.numel()); fflush(stderr);
            registerTensor("scratch_block_atom_projection", ap, ParamGroupType::SCRATCHBLOCK);
            fprintf(stderr, "[buildParameterGroups] DIAG-D2: registered OK\n"); fflush(stderr);

            // text_feature_projection ELIMINATED — text features merged into atom embeddings (dims 48-63)
        } catch (const std::exception& ex) {
            fprintf(stderr, "[buildParameterGroups] DIAG-D-EXCEPTION in scratchblock: %s\n", ex.what()); fflush(stderr);
            throw;
        }
        fprintf(stderr, "[buildParameterGroups] DIAG-D4: scratchblock done\n"); fflush(stderr);
    } else {
        fprintf(stderr, "[buildParameterGroups] DIAG-D-SKIP: scratchblock not enabled\n"); fflush(stderr);
    }

    // ReasoningHead parameters
    if (reasoning_head_layer_) {
        registerTensor("reasoning_head_w_op", reasoning_head_layer_->W_op(), ParamGroupType::REASONING_HEAD);
        registerNonDecayTensor("reasoning_head_b_op", reasoning_head_layer_->b_op(), ParamGroupType::REASONING_HEAD);
        registerTensor("reasoning_head_w_arg1", reasoning_head_layer_->w_arg1(), ParamGroupType::REASONING_HEAD);
        registerTensor("reasoning_head_w_arg2", reasoning_head_layer_->w_arg2(), ParamGroupType::REASONING_HEAD);
        fprintf(stderr, "[buildParameterGroups] DIAG-D4c: reasoning head registered\n"); fflush(stderr);
    }

    // ExecutionBlock parameters (v2: differentiable rewrite)
    if (execution_block_layer_) {
        registerTensor("exec_block_w_decode_1", execution_block_layer_->w_decode_1(), ParamGroupType::EXECUTION_BLOCK);
        registerNonDecayTensor("exec_block_b_decode_1", execution_block_layer_->b_decode_1(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_w_decode_2", execution_block_layer_->w_decode_2(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_w_arg1_select", execution_block_layer_->w_arg1_select(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_w_arg2_select", execution_block_layer_->w_arg2_select(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_op_select", execution_block_layer_->W_op_select(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_key_proj", execution_block_layer_->W_key_proj(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_write_query", execution_block_layer_->W_write_query(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_write_key", execution_block_layer_->W_write_key(), ParamGroupType::EXECUTION_BLOCK);
        registerNonDecayTensor("exec_block_alpha", execution_block_layer_->alpha(), ParamGroupType::EXECUTION_BLOCK);
        registerNonDecayTensor("exec_block_beta", execution_block_layer_->beta(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_step_embeddings", execution_block_layer_->step_embeddings(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_type_num_embed", execution_block_layer_->type_num_embed(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_value_to_emb", execution_block_layer_->W_value_to_emb(), ParamGroupType::EXECUTION_BLOCK);
        registerNonDecayTensor("exec_block_b_value_to_emb", execution_block_layer_->b_value_to_emb(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_w_inject_gate", execution_block_layer_->w_inject_gate(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_Q_read", execution_block_layer_->W_Q_read(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_K_read", execution_block_layer_->W_K_read(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_V_read", execution_block_layer_->W_V_read(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_O_read", execution_block_layer_->W_O_read(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_gate_read", execution_block_layer_->W_gate_read(), ParamGroupType::EXECUTION_BLOCK);
        registerNonDecayTensor("exec_block_tau", execution_block_layer_->tau(), ParamGroupType::EXECUTION_BLOCK);
        // Trace encoding weights
        registerTensor("exec_block_E_slot", execution_block_layer_->E_slot(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_E_op", execution_block_layer_->E_op(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_scal", execution_block_layer_->W_scal(), ParamGroupType::EXECUTION_BLOCK);
        registerNonDecayTensor("exec_block_b_scal", execution_block_layer_->b_scal(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_trace", execution_block_layer_->W_trace(), ParamGroupType::EXECUTION_BLOCK);
        registerNonDecayTensor("exec_block_b_trace", execution_block_layer_->b_trace(), ParamGroupType::EXECUTION_BLOCK);
        // Reasoning state update gate
        registerTensor("exec_block_W_reason_gate", execution_block_layer_->W_reason_gate(), ParamGroupType::EXECUTION_BLOCK);
        registerTensor("exec_block_W_trace_gate", execution_block_layer_->W_trace_gate(), ParamGroupType::EXECUTION_BLOCK);
        fprintf(stderr, "[buildParameterGroups] DIAG-D4d: execution block v2 registered\n"); fflush(stderr);
    }

    // Decode-time slot selector (gated by selector_enabled inside execution_block_enabled)
    if (config_.selector_enabled && decode_time_slot_selector_layer_) {
        auto& sel = *decode_time_slot_selector_layer_;
        registerTensor("selector_W_q_select", sel.W_q_select(), ParamGroupType::SLOT_SELECTOR);
        registerTensor("selector_W_k_select", sel.W_k_select(), ParamGroupType::SLOT_SELECTOR);
        registerTensor("selector_null_key_select", sel.null_key_select(), ParamGroupType::SLOT_SELECTOR);
        registerNonDecayTensor("selector_null_logit_bias", sel.null_logit_bias(), ParamGroupType::SLOT_SELECTOR);
        fprintf(stderr, "[buildParameterGroups] DIAG-D4e: slot selector registered (4 tensors)\n"); fflush(stderr);
    }

    // Multi-token prediction (MTP) auxiliary heads
    const int mtp_k = (config_.mtp_enabled ? config_.mtp_k : 0);
    for (int k = 0; k < mtp_k; ++k) {
        MTPHead* head = getMtpHead(k);
        if (!head || !head->weight.data || !head->weight.has_grad()) continue;
        registerTensor("mtp_head_" + std::to_string(k) + "_weight", head->weight, ParamGroupType::MTP);
        tryRegisterBias("mtp_head_" + std::to_string(k) + "_bias", head->bias, ParamGroupType::MTP);
    }

    // Final RMSNorm (between encoder output and LM head)
    // γ_final acts as a logit temperature — without weight decay it grows unbounded
    // (1.0→3.0) causing overconfident predictions → mode collapse. Weight decay
    // provides the restoring force that per-layer gammas get from mixed gradient signals.
    // lr_mult=0.1 prevents the initial spike: γ_final has a monotonic "scale up" gradient
    // bias in early training (scaling logits reduces CE faster than learning representations),
    // so it needs a slower learning rate to prevent overshooting before representations form.
    //
    // EMPIRICAL UPDATE (Apr 2026, 20k-step run): wd_mult=1.0 + lr_mult=0.1 + ADAMW_WEIGHT_DECAY=0.01
    // produces effective decay ~6e-7/step, ~25× too weak vs the ~31% inflation observed by step
    // 20k. Setting `lm_head_freeze_final_rms_gamma=true` in ai_config.json disables \u03b3 entirely
    // (held at 1.0). When frozen, has_grad()==false and we skip registration here.
    if (lm_head_layer_->finalRmsGamma().data && lm_head_layer_->finalRmsGamma().has_grad()) {
        fprintf(stderr, "[buildParameterGroups] DIAG-D5: about to register final_rms_gamma data=%p grad=%d numel=%zu\n",
                (void*)lm_head_layer_->finalRmsGamma().data, (int)lm_head_layer_->finalRmsGamma().has_grad(),
                lm_head_layer_->finalRmsGamma().numel()); fflush(stderr);
        registerTensor("final_rms_gamma",
                       lm_head_layer_->finalRmsGammaMutable_UnfrozenOnly("buildParameterGroups"),
                       ParamGroupType::RMSNORM, -1, 1.0f, 0.1f);
        fprintf(stderr, "[buildParameterGroups] DIAG-D5: registered OK\n"); fflush(stderr);
    } else {
        fprintf(stderr, "[buildParameterGroups] final_rms_gamma FROZEN (no grad, no param group) — held at 1.0\n");
        fflush(stderr);
    }

    fprintf(stderr, "[buildParameterGroups] DIAG-E: %zu groups registered, allocating optimizer states\n", parameter_groups_.size()); fflush(stderr);
    // Collect sizes for centralized optimizer state allocation
    std::vector<size_t> sizes;
    sizes.reserve(parameter_groups_.size());
    for (const auto& group : parameter_groups_) {
        sizes.push_back(group.size());
    }
    
    // Allocate optimizer states via TrainingState (centralized ownership)
    training_state_.allocateOptimizerStates(sizes);
    
    fprintf(stderr, "[buildParameterGroups] DIAG-F: optimizer states allocated (m=%zu v=%zu), binding tensors\n",
            training_state_.optimizer_m_states.size(), training_state_.optimizer_v_states.size()); fflush(stderr);
    // Bind optimizer Tensors to parameter groups (groups hold pointers, NOT ownership)
    for (size_t i = 0; i < parameter_groups_.size(); ++i) {
        parameter_groups_[i].m_tensor = &training_state_.optimizer_m_states[i];
        parameter_groups_[i].v_tensor = &training_state_.optimizer_v_states[i];
    }
    fprintf(stderr, "[buildParameterGroups] DIAG-G: all bindings done\n"); fflush(stderr);
    
    // Note: Gradient norm measurement uses free functions in GradNormGPU.{cu,hpp}
    // Phase2_TrainingLoop.cu calls measureGradientNorms() directly via TrainingState::grad_norm_scratch

    BWD_INFO("[buildParameterGroups] Built " << parameter_groups_.size() << " parameter groups");
    
    // ISSUE #110 DIAGNOSTIC: Count parameter groups by type to understand what was registered
    int emb_count = 0, attn_count = 0, ffn_count = 0, rms_count = 0, other_count = 0;
    for (const auto& pg : parameter_groups_) {
        switch (pg.type) {
            case ParamGroupType::EMBEDDING: emb_count++; break;
            case ParamGroupType::ATTENTION: attn_count++; break;
            case ParamGroupType::FFN: ffn_count++; break;
            case ParamGroupType::RMSNORM: rms_count++; break;
            default: other_count++; break;
        }
    }
    fprintf(stderr, "[buildParameterGroups] TOTAL: %zu groups (emb=%d, attn=%d, ffn=%d, rms=%d, other=%d)\n",
            parameter_groups_.size(), emb_count, attn_count, ffn_count, rms_count, other_count);
}

//======================================================
//  dumpGradientValues - Debug: Dump gradient RMS stats for each parameter group
//======================================================//

void LanguageModel::dumpGradientValues(int step, const std::string& filepath) {
    if (parameter_groups_.empty()) {
        buildParameterGroups();
    }
    
    cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();
    cudaStreamSynchronize(stream);  // Ensure all gradients computed
    
    std::ofstream file(filepath, std::ios::app);
    file << "\n========== STEP " << step << " GRADIENT VALUES (GRIM-text) ==========\n";
    
    for (const auto& group : parameter_groups_) {
        if (!group.grads() || group.size() == 0) continue;
        
        std::vector<float> full_buffer(group.size());
        cudaMemcpy(full_buffer.data(), group.grads(), group.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double sum_sq = 0.0;
        float min_val = full_buffer[0], max_val = full_buffer[0];
        for (size_t i = 0; i < group.size(); ++i) {
            sum_sq += (double)full_buffer[i] * full_buffer[i];
            if (full_buffer[i] < min_val) min_val = full_buffer[i];
            if (full_buffer[i] > max_val) max_val = full_buffer[i];
        }
        const float rms = std::sqrt(static_cast<float>(sum_sq / group.size()));
        
        file << "\n[" << group.name << "] size=" << group.size() 
             << " rms=" << std::scientific << std::setprecision(6) << rms
             << " min=" << min_val << " max=" << max_val << "\n";
    }
    
    file.close();
}

#endif // USE_CUDA

} // namespace GRIM

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
//  - configureUpdateProbe() / disableUpdateProbe() - Debug probes
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
#include "../../../control/ai_config_paths.hpp"
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
                              float wd_mult = 1.0f) {
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
        
        // FFN weights/biases
        registerTensor(prefix + "_ffn_w1", enc->ffnW1(), ParamGroupType::FFN, layer);
        tryRegisterBias(prefix + "_ffn_b1", enc->ffnB1(), ParamGroupType::FFN, layer);
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

            auto& tfp = scratch_block_layer_->textFeatureProjection();
            fprintf(stderr, "[buildParameterGroups] DIAG-D3: text_feature_projection data=%p grad=%d numel=%zu\n",
                    (void*)tfp.data, (int)tfp.has_grad(), tfp.numel()); fflush(stderr);
            registerTensor("scratch_block_text_feature_projection", tfp, ParamGroupType::SCRATCHBLOCK);
            fprintf(stderr, "[buildParameterGroups] DIAG-D3: registered OK\n"); fflush(stderr);
        } catch (const std::exception& ex) {
            fprintf(stderr, "[buildParameterGroups] DIAG-D-EXCEPTION in scratchblock: %s\n", ex.what()); fflush(stderr);
            throw;
        }
        fprintf(stderr, "[buildParameterGroups] DIAG-D4: scratchblock done\n"); fflush(stderr);
    } else {
        fprintf(stderr, "[buildParameterGroups] DIAG-D-SKIP: scratchblock not enabled\n"); fflush(stderr);
    }

    // Final RMSNorm (between encoder output and LM head) — no weight decay
    fprintf(stderr, "[buildParameterGroups] DIAG-D5: about to register final_rms_gamma data=%p grad=%d numel=%zu\n",
            (void*)lm_head_layer_->finalRmsGamma().data, (int)lm_head_layer_->finalRmsGamma().has_grad(),
            lm_head_layer_->finalRmsGamma().numel()); fflush(stderr);
    registerNonDecayTensor("final_rms_gamma", lm_head_layer_->finalRmsGamma(), ParamGroupType::RMSNORM);
    fprintf(stderr, "[buildParameterGroups] DIAG-D5: registered OK\n"); fflush(stderr);

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

//======================================================//
//  configureUpdateProbe / disableUpdateProbe
//======================================================//

void LanguageModel::configureUpdateProbe(const std::string& group_name, size_t sample_elems) {
    fprintf(stderr, "[configureUpdateProbe] ENTER group='%s' sample_elems=%zu\n", group_name.c_str(), sample_elems); fflush(stderr);
    update_probe_group_name_ = group_name;
    update_probe_sample_elems_ = sample_elems;
    update_probe_ready_ = false;
    
    // Ensure parameter groups are built before configuring probe
    if (parameter_groups_.empty()) {
        fprintf(stderr, "[configureUpdateProbe] parameter_groups_ is empty, calling buildParameterGroups()\n"); fflush(stderr);
        buildParameterGroups();
        fprintf(stderr, "[configureUpdateProbe] buildParameterGroups() returned, groups=%zu\n", parameter_groups_.size()); fflush(stderr);
    }
    
    if (parameter_groups_.empty()) {
        update_probe_group_index_ = static_cast<size_t>(-1);
        update_probe_weights_before_.clear();
        update_probe_weights_after_.clear();
        update_probe_grad_sample_.clear();
        return;
    }

    update_probe_group_index_ = static_cast<size_t>(-1);
    for (size_t i = 0; i < parameter_groups_.size(); ++i) {
        if (parameter_groups_[i].name == update_probe_group_name_) {
            update_probe_group_index_ = i;
            break;
        }
    }

    if (update_probe_group_index_ == static_cast<size_t>(-1)) {
        BWD_WARN("[configureUpdateProbe] target group '" << update_probe_group_name_
                 << "' not found");
        return;
    }

    const auto& probe_group = parameter_groups_[update_probe_group_index_];
    if (update_probe_sample_elems_ == 0 || update_probe_sample_elems_ > probe_group.size()) {
        update_probe_sample_elems_ = std::min<size_t>(probe_group.size(), 2048);
    }
    update_probe_weights_before_.assign(update_probe_sample_elems_, 0.0f);
    update_probe_weights_after_.assign(update_probe_sample_elems_, 0.0f);
    update_probe_grad_sample_.assign(update_probe_sample_elems_, 0.0f);
    fprintf(stderr, "[configureUpdateProbe] EXIT: group_index=%zu sample_elems=%zu probe_group_size=%zu\n",
            update_probe_group_index_, update_probe_sample_elems_,
            update_probe_group_index_ != static_cast<size_t>(-1) ? parameter_groups_[update_probe_group_index_].size() : 0);
    fflush(stderr);
}

void LanguageModel::disableUpdateProbe() {
    update_probe_group_name_.clear();
    update_probe_group_index_ = static_cast<size_t>(-1);
    update_probe_sample_elems_ = 0;
    update_probe_weights_before_.clear();
    update_probe_weights_after_.clear();
    update_probe_grad_sample_.clear();
    update_probe_ready_ = false;
}

// recordGradientClip() DELETED (Rule 26). Phase2 no longer uses GradComponentMetrics.

// updateWeights(), resetOptimizerMoments(), scaleOptimizerMoments() MOVED to
// AdamW_Kernal_GPU.{hpp,cu} as free functions: launchAdamWStep(), resetAdamWMoments(),
// scaleAdamWMoments(). AdamW stepping is training infrastructure, not model logic.

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

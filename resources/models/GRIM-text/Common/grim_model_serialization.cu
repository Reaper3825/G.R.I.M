#ifndef USE_CUDA
#define USE_CUDA
#endif

#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <cstring>
#include <sstream>
#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"
#include "../training/schemas/grim_transformer_model_generated.h"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/Serialization/Serialization_GPU.hpp"
#include "grim_model_serialization_version.hpp"


namespace GRIM {

//======================================================//
//  Helper: Flatten Matrix to Vector
//======================================================//

static std::vector<float> flattenMatrix(const Matrix& mat) {
    std::vector<float> result;
    result.reserve(mat.num_rows * mat.num_cols);
    
    for (int i = 0; i < mat.num_rows; ++i) {
        const auto& row = mat.rows[i];
        result.insert(result.end(), row.data.begin(), row.data.end());
    }
    
    return result;
}

//======================================================//
//  Helper: Unflatten Vector to Matrix
//======================================================//

static void unflattenMatrix(const std::vector<float>& data, Matrix& mat, int rows, int cols) {
    mat.num_rows = rows;
    mat.num_cols = cols;
    mat.rows.clear();
    mat.rows.reserve(rows);
    
    for (int i = 0; i < rows; ++i) {
        Vector row(cols);
        std::copy(data.begin() + i * cols, 
                 data.begin() + (i + 1) * cols,
                 row.data.begin());
        mat.rows.push_back(std::move(row));
    }
}

//======================================================//
//  Save/Load Model to FlatBuffer via Serialization Layer
//======================================================//

namespace {

SerializationModelConfigView makeConfigView(const LanguageModelConfig& cfg) {
    SerializationModelConfigView view{};
    view.vocab_size = cfg.vocab_size;
    view.d_model = cfg.d_model;
    view.num_layers = cfg.num_layers;
    view.num_heads = cfg.num_heads;
    view.num_kv_heads = cfg.num_kv_heads;  // GQA: must propagate for correct W_qkv sizing
    view.d_ff = cfg.d_ff;
    view.max_seq_len = cfg.max_seq_len;
    view.dropout_rate = cfg.dropout_rate;
    view.positional_encoding = cfg.positional_encoding;
    view.tie_embeddings = cfg.tie_embeddings;
    view.use_gpu = cfg.use_gpu;
    view.use_bias = cfg.use_bias;
    return view;
}

std::size_t embeddingElementCount(const LanguageModelConfig& cfg) {
    return static_cast<std::size_t>(cfg.vocab_size) * static_cast<std::size_t>(cfg.d_model);
}

SerializationCpuEmbeddingReadData snapshotCpuEmbedding(const GrimEmbeddingStack* embedder) {
    SerializationCpuEmbeddingReadData data{};
    if (!embedder) {
        return data;
    }
    data.num_rows = embedder->token_embed.num_rows;
    data.num_cols = embedder->token_embed.num_cols;
    data.token_data = flattenMatrix(embedder->token_embed);
    if (!embedder->rms_gamma.data.empty()) {
        data.rms_gamma = embedder->rms_gamma.data;
        data.has_rms_norm = true;
    }
    return data;
}

void assignRead(DeviceReadView& view, const float* ptr, std::size_t count) {
    view.ptr = ptr;
    view.count = ptr ? count : 0;  // Set count=0 if ptr is null to prevent download failures
}

void assignWrite(DeviceWriteView& view, float* ptr, std::size_t count) {
    view.ptr = ptr;
    view.count = count;
}

} // namespace (anonymous)

bool LanguageModel::save(const std::string& path) {
    using namespace GRIM::Logging;
    EmitModuleInfo(ModuleId::Checkpoint, "save() called for path: " + path);
    
    static std::mutex save_mutex;
    std::lock_guard<std::mutex> lock(save_mutex);
    EmitModuleInfo(ModuleId::Checkpoint, "Acquired save mutex");

    cudaError_t pre_sync_err = cudaDeviceSynchronize();
    if (pre_sync_err != cudaSuccess) {
        EmitModuleError(ModuleId::Checkpoint, std::string("Pre-save cudaDeviceSynchronize() failed: ") + cudaGetErrorString(pre_sync_err));
        std::cerr << "[LanguageModel::save] Pre-save cudaDeviceSynchronize() failed: "
                  << cudaGetErrorString(pre_sync_err) << std::endl;
        return false;
    }
    EmitModuleInfo(ModuleId::Checkpoint, "CUDA device synchronized");

    EmitModuleInfo(ModuleId::Checkpoint, "Creating SerializationLayer");
    SerializationLayer layer(SerializationConfig{});
    SerializationSaveRequest request{};
    request.path = path;
    request.model_version = GRIM_MODEL_VERSION;
    request.sources.config = makeConfigView(config_);
    EmitModuleInfo(ModuleId::Checkpoint, "Request initialized with version " + std::to_string(GRIM_MODEL_VERSION));

    EmitModuleInfo(ModuleId::Checkpoint, "Processing embeddings");
    if (config_.use_gpu && training_state_.embedding_weights.data) {
        EmitModuleInfo(ModuleId::Checkpoint, "Using training_state_ embedding weights (vocab=" + std::to_string(config_.vocab_size) + ", d_model=" + std::to_string(config_.d_model) + ")");
        assignRead(request.sources.gpu_embedding.token_embeddings,
                   training_state_.embedding_weights.data,
                   embeddingElementCount(config_));
        // final_rms_gamma is already serialized separately (line ~260), but legacy
        // checkpoint format also stores it under gpu_embedding.rms_gamma for compat.
        if (training_state_.final_rms_gamma.data) {
            EmitModuleInfo(ModuleId::Checkpoint, "Including embedding RMSNorm gamma");
            assignRead(request.sources.gpu_embedding.rms_gamma,
                       training_state_.final_rms_gamma.data,
                       static_cast<std::size_t>(config_.d_model));
            request.sources.gpu_embedding.has_rms_norm = true;
        }
    } else {
        EmitModuleInfo(ModuleId::Checkpoint, "Using CPU embedder snapshot");
        request.sources.cpu_embedding = snapshotCpuEmbedding(getEmbedderPtr());
        if (request.sources.cpu_embedding.token_data.empty()) {
            EmitModuleError(ModuleId::Checkpoint, "CPU embedder snapshot unavailable");
            std::cerr << "[LanguageModel::save] Error: CPU embedder snapshot unavailable" << std::endl;
            return false;
        }
    }

    EmitModuleInfo(ModuleId::Checkpoint, "Processing encoder layers (" + std::to_string(config_.num_layers) + " layers)");
    { std::ostringstream oss; oss << "gpu_encoder_ ptr = " << (void*)gpu_encoder_.get() << " this = " << (void*)this; EmitModuleInfo(ModuleId::Checkpoint, oss.str()); }
    auto* gpu_encoder = &getGpuEncoder();
    { std::ostringstream oss; oss << "getGpuEncoder() returned " << (void*)gpu_encoder; EmitModuleInfo(ModuleId::Checkpoint, oss.str()); }
    if (!gpu_encoder) {
        EmitModuleError(ModuleId::Checkpoint, "GPU encoder not initialized");
        std::cerr << "[LanguageModel::save] Error: GPU encoder not initialized" << std::endl;
        return false;
    }
    request.sources.encoder_layers.resize(config_.num_layers);
    const std::size_t d_model = static_cast<std::size_t>(config_.d_model);
    const std::size_t d_ff = static_cast<std::size_t>(config_.d_ff);
    
    // GQA dimensions for W_qkv sizing (no fallback - num_kv_heads must be properly set)
    const int head_dim = config_.head_dim;  // Use pre-computed value from config
    const int kv_dim = config_.num_kv_heads * head_dim;
    const int total_qkv_dim = config_.d_model + 2 * kv_dim;  // Q + K + V with GQA
    const std::size_t qkv_weight_size = static_cast<std::size_t>(total_qkv_dim) * d_model;
    
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        if (layer_idx % 2 == 0) {
            EmitModuleInfo(ModuleId::Checkpoint, "Processing layer " + std::to_string(layer_idx) + "/" + std::to_string(config_.num_layers));
        }
        auto* enc = gpu_encoder->getLayer(layer_idx);
        if (!enc) {
            EmitModuleError(ModuleId::Checkpoint, "GPU layer " + std::to_string(layer_idx) + " is null");
            std::cerr << "[LanguageModel::save] Error: GPU layer " << layer_idx << " is null" << std::endl;
            return false;
        }
        auto& view = request.sources.encoder_layers[layer_idx];
        assignRead(view.attn_w_qkv, enc->attnWqkv().data, qkv_weight_size);
        assignRead(view.attn_b_qkv, enc->attnBqkv().data, total_qkv_dim);  // GQA-aware bias size
        assignRead(view.attn_w_o, enc->attnWo().data, d_model * d_model);
        assignRead(view.attn_b_o, enc->attnBo().data, d_model);
        assignRead(view.ffn_w1, enc->ffnW1().data, d_model * d_ff);
        assignRead(view.ffn_b1, enc->ffnB1().data, d_ff);
        assignRead(view.ffn_w2, enc->ffnW2().data, d_ff * d_model);
        assignRead(view.ffn_b2, enc->ffnB2().data, d_model);
        assignRead(view.rms1_gamma, enc->rms1Gamma().data, d_model);
        assignRead(view.rms2_gamma, enc->rms2Gamma().data, d_model);
        assignRead(view.rms_post_attn_gamma, enc->rmsPostAttnGamma().data, d_model);
        assignRead(view.rms_post_ffn_gamma, enc->rmsPostFfnGamma().data, d_model);
    }

    EmitModuleInfo(ModuleId::Checkpoint, "Processing LM head (projection=" + std::string(training_state_.lm_head_weights.data ? "yes" : "no") + ", bias=" + std::string(training_state_.lm_head_bias.data ? "yes" : "no") + ")");
    request.sources.lm_head.has_projection = (training_state_.lm_head_weights.data != nullptr);
    request.sources.lm_head.projection.ptr = training_state_.lm_head_weights.data;
    request.sources.lm_head.projection.count = training_state_.lm_head_weights.data ? embeddingElementCount(config_) : 0;
    request.sources.lm_head.has_bias = (training_state_.lm_head_bias.data != nullptr);
    request.sources.lm_head.bias.ptr = training_state_.lm_head_bias.data;
    request.sources.lm_head.bias.count = training_state_.lm_head_bias.data ? static_cast<std::size_t>(config_.vocab_size) : 0;

    EmitModuleInfo(ModuleId::Checkpoint, "Processing numeric head (enabled=" +
                   std::string(config_.numeric_head_enabled ? "yes" : "no") + ")");
    request.sources.numeric_head.enabled = config_.numeric_head_enabled;
    request.sources.numeric_head.has_projection = (training_state_.numeric_head_weights.data != nullptr);
    request.sources.numeric_head.projection.ptr = training_state_.numeric_head_weights.data;
    request.sources.numeric_head.projection.count = training_state_.numeric_head_weights.data
        ? static_cast<std::size_t>(config_.d_model)
        : 0;
    request.sources.numeric_head.has_bias = (training_state_.numeric_head_bias.data != nullptr);
    request.sources.numeric_head.bias.ptr = training_state_.numeric_head_bias.data;
    request.sources.numeric_head.bias.count = training_state_.numeric_head_bias.data ? 1u : 0u;

    // Process ScratchBlock weights (if enabled)
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        constexpr int kNumAtomTypes = 16;  // AtomType enum size
        const int atom_emb_dim = config_.scratch_block_atom_embedding_dim;
        
        request.sources.scratch_block.enabled = true;
        request.sources.scratch_block.num_atom_types = kNumAtomTypes;
        request.sources.scratch_block.atom_embedding_dim = atom_emb_dim;
        request.sources.scratch_block.d_model = config_.d_model;
        request.sources.scratch_block.atom_scale = config_.scratch_block_atom_scale;
        
        if (scratch_block_layer_->atomTypeEmbeddings().data) {
            request.sources.scratch_block.atom_type_embeddings.ptr = scratch_block_layer_->atomTypeEmbeddings().data;
            request.sources.scratch_block.atom_type_embeddings.count = 
                static_cast<std::size_t>(kNumAtomTypes * atom_emb_dim);
        }
        
        if (scratch_block_layer_->atomProjection().data) {
            request.sources.scratch_block.atom_projection.ptr = scratch_block_layer_->atomProjection().data;
            request.sources.scratch_block.atom_projection.count = 
                static_cast<std::size_t>(atom_emb_dim * config_.d_model);
        }
        
        // Text feature projection [16 x d_model] - VALUE encoding path
        constexpr int kTextFeatureDim = 16;
        if (scratch_block_layer_->textFeatureProjection().data) {
            request.sources.scratch_block.text_feature_projection.ptr = scratch_block_layer_->textFeatureProjection().data;
            request.sources.scratch_block.text_feature_projection.count = 
                static_cast<std::size_t>(kTextFeatureDim * config_.d_model);
        }
        
        EmitModuleInfo(ModuleId::Checkpoint, "Processing ScratchBlock (atom_emb=" + 
                       std::to_string(request.sources.scratch_block.atom_type_embeddings.count) +
                       ", atom_proj=" + std::to_string(request.sources.scratch_block.atom_projection.count) +
                       ", text_proj=" + std::to_string(request.sources.scratch_block.text_feature_projection.count) + ")");
    }

    // Issue #33: Final RMSNorm gamma (normalizes encoder output before LM head)
    if (training_state_.final_rms_gamma.data) {
        request.sources.final_rms_gamma.ptr = training_state_.final_rms_gamma.data;
        request.sources.final_rms_gamma.count = static_cast<std::size_t>(config_.d_model);
        EmitModuleInfo(ModuleId::Checkpoint, "Processing final_rms_gamma (size=" + 
                       std::to_string(config_.d_model) + ")");
    }

    EmitModuleInfo(ModuleId::Checkpoint, "Calling SerializationLayer::save()");
    bool result = layer.save(request);
    EmitModuleInfo(ModuleId::Checkpoint, std::string("SerializationLayer::save() returned ") + (result ? "true" : "false"));
    return result;
}

bool LanguageModel::load(const std::string& path) {
    SerializationLayer layer(SerializationConfig{});
    SerializationLoadRequest request{};
    request.path = path;
    request.config = makeConfigView(config_);

    if (config_.use_gpu) {
        if (!training_state_.embedding_weights.data) {
            std::cerr << "[LanguageModel::load] Error: training_state_.embedding_weights not initialized" << std::endl;
            return false;
        }
        assignWrite(request.gpu_embedding.token_embeddings,
                    training_state_.embedding_weights.data,
                    embeddingElementCount(config_));
        if (training_state_.final_rms_gamma.data) {
            assignWrite(request.gpu_embedding.rms_gamma,
                        training_state_.final_rms_gamma.data,
                        static_cast<std::size_t>(config_.d_model));
            request.gpu_embedding.has_rms_norm = true;
        }
    }

    if (auto* embedder = getEmbedderPtr()) {
        request.cpu_embedding.set_tokens = [embedder](const std::vector<float>& data, int rows, int cols) {
            unflattenMatrix(data, embedder->token_embed, rows, cols);
        };
        request.cpu_embedding.set_rms_gamma = [embedder](const std::vector<float>& gamma) {
            embedder->rms_gamma.data = gamma;
        };
    }

    auto* gpu_encoder = &getGpuEncoder();
    if (!gpu_encoder) {
        std::cerr << "[LanguageModel::load] Error: GPU encoder not initialized" << std::endl;
        return false;
    }
    request.encoder_layers.resize(config_.num_layers);
    const std::size_t d_model = static_cast<std::size_t>(config_.d_model);
    const std::size_t d_ff = static_cast<std::size_t>(config_.d_ff);
    
    // GQA dimensions for W_qkv sizing - MUST match save() calculation!
    // BUG FIX Issue #24: load() was using MHA formula (d_model * 3) but save() uses GQA formula
    const int head_dim = config_.head_dim;  // Use pre-computed value from config
    const int kv_dim = config_.num_kv_heads * head_dim;
    const int total_qkv_dim = config_.d_model + 2 * kv_dim;  // Q + K + V with GQA
    const std::size_t qkv_weight_size = static_cast<std::size_t>(total_qkv_dim) * d_model;
    
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        auto* enc = gpu_encoder->getLayer(layer_idx);
        if (!enc) {
            std::cerr << "[LanguageModel::load] Error: GPU layer " << layer_idx << " is null" << std::endl;
            return false;
        }
        auto& view = request.encoder_layers[layer_idx];
        assignWrite(view.attn_w_qkv, enc->attnWqkv().data, qkv_weight_size);
        assignWrite(view.attn_b_qkv, enc->attnBqkv().data, total_qkv_dim);  // GQA-aware bias size
        assignWrite(view.attn_w_o, enc->attnWo().data, d_model * d_model);
        assignWrite(view.attn_b_o, enc->attnBo().data, d_model);
        assignWrite(view.ffn_w1, enc->ffnW1().data, d_model * d_ff);
        assignWrite(view.ffn_b1, enc->ffnB1().data, d_ff);
        assignWrite(view.ffn_w2, enc->ffnW2().data, d_ff * d_model);
        assignWrite(view.ffn_b2, enc->ffnB2().data, d_model);
        assignWrite(view.rms1_gamma, enc->rms1Gamma().data, d_model);
        assignWrite(view.rms2_gamma, enc->rms2Gamma().data, d_model);
        assignWrite(view.rms_post_attn_gamma, enc->rmsPostAttnGamma().data, d_model);
        assignWrite(view.rms_post_ffn_gamma, enc->rmsPostFfnGamma().data, d_model);
    }

    if (!training_state_.initialized) {
        // Initialize state based on execution mode
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
    }
    if (training_state_.lm_head_weights.data) {
        assignWrite(request.lm_head.projection,
                    training_state_.lm_head_weights.data,
                    embeddingElementCount(config_));
    }
    if (training_state_.lm_head_bias.data) {
        assignWrite(request.lm_head.bias,
                    training_state_.lm_head_bias.data,
                    static_cast<std::size_t>(config_.vocab_size));
    }
    request.lm_head.expect_bias = config_.use_bias;

    request.numeric_head.enabled = config_.numeric_head_enabled;
    if (training_state_.numeric_head_weights.data) {
        assignWrite(request.numeric_head.projection,
                    training_state_.numeric_head_weights.data,
                    static_cast<std::size_t>(config_.d_model));
    }
    if (training_state_.numeric_head_bias.data) {
        assignWrite(request.numeric_head.bias,
                    training_state_.numeric_head_bias.data,
                    1u);
    }
    request.numeric_head.expect_bias = config_.use_bias;

    // Set up ScratchBlock weight destinations (if enabled)
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        constexpr int kNumAtomTypes = 16;
        const int atom_emb_dim = config_.scratch_block_atom_embedding_dim;
        
        if (scratch_block_layer_->atomTypeEmbeddings().data) {
            assignWrite(request.scratch_block.atom_type_embeddings,
                        scratch_block_layer_->atomTypeEmbeddings().data,
                        static_cast<std::size_t>(kNumAtomTypes * atom_emb_dim));
        }
        
        if (scratch_block_layer_->atomProjection().data) {
            assignWrite(request.scratch_block.atom_projection,
                        scratch_block_layer_->atomProjection().data,
                        static_cast<std::size_t>(atom_emb_dim * config_.d_model));
        }
        
        // Text feature projection [16 x d_model] - VALUE encoding path
        constexpr int kTextFeatureDim = 16;
        if (scratch_block_layer_->textFeatureProjection().data) {
            assignWrite(request.scratch_block.text_feature_projection,
                        scratch_block_layer_->textFeatureProjection().data,
                        static_cast<std::size_t>(kTextFeatureDim * config_.d_model));
        }
        
        request.scratch_block.num_atom_types = kNumAtomTypes;
        request.scratch_block.atom_embedding_dim = atom_emb_dim;
    }

    // Issue #33: Final RMSNorm gamma destination
    if (training_state_.final_rms_gamma.data) {
        assignWrite(request.final_rms_gamma,
                    training_state_.final_rms_gamma.data,
                    static_cast<std::size_t>(config_.d_model));
    }

    return layer.load(request);
}

} // namespace GRIM

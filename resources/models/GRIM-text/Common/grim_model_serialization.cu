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
#include "../Shared/UnigramByte/Unigram.hpp"
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
    if (config_.use_gpu && embedding_layer_ && embedding_layer_->tokenWeights().data) {
        EmitModuleInfo(ModuleId::Checkpoint, "Embedding source: GPU");
        EmitModuleInfo(ModuleId::Checkpoint, "Using EmbeddingLayer token weights (vocab=" + std::to_string(config_.vocab_size) + ", d_model=" + std::to_string(config_.d_model) + ")");
        assignRead(request.sources.gpu_embedding.token_embeddings,
                   embedding_layer_->tokenWeights().data,
                   embeddingElementCount(config_));
        // final_rms_gamma is owned by LMHeadLayer (Pattern B), serialized separately (~line 260)
        // but legacy checkpoint format also stores it under gpu_embedding.rms_gamma.
        if (lm_head_layer_ && lm_head_layer_->finalRmsGamma().data) {
            EmitModuleInfo(ModuleId::Checkpoint, "Including embedding RMSNorm gamma");
            assignRead(request.sources.gpu_embedding.rms_gamma,
                       lm_head_layer_->finalRmsGamma().data,
                       static_cast<std::size_t>(config_.d_model));
            request.sources.gpu_embedding.has_rms_norm = true;
        }
    } else {
        EmitModuleInfo(ModuleId::Checkpoint, "Embedding source: CPU");
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
        assignRead(view.ffn_w_gate, enc->ffnWGate().data, d_model * d_ff);
        assignRead(view.ffn_w1, enc->ffnW1().data, d_model * d_ff);
        assignRead(view.ffn_w2, enc->ffnW2().data, d_ff * d_model);
        assignRead(view.ffn_b2, enc->ffnB2().data, d_model);
        assignRead(view.rms1_gamma, enc->rms1Gamma().data, d_model);
        assignRead(view.rms2_gamma, enc->rms2Gamma().data, d_model);
        // Issue #148: Sandwich norm gammas REMOVED — not saved to checkpoint
        // Old checkpoints may contain rms_post_attn/rms_post_ffn but they're ignored on load
        // LayerScale (Issue #109) — single scalar per sublayer
        if (enc->layerScale1().data) assignRead(view.layer_scale1, enc->layerScale1().data, 1);
        if (enc->layerScale2().data) assignRead(view.layer_scale2, enc->layerScale2().data, 1);
    }

    if (!lm_head_layer_) throw std::runtime_error("LMHeadLayer is NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    EmitModuleInfo(ModuleId::Checkpoint, "Processing LM head (projection=" + std::string(lm_head_layer_->weights().data ? "yes" : "no") + ", bias=" + std::string(lm_head_layer_->bias().data ? "yes" : "no") + ")");
    request.sources.lm_head.has_projection = (lm_head_layer_->weights().data != nullptr);
    request.sources.lm_head.projection.ptr = lm_head_layer_->weights().data;
    request.sources.lm_head.projection.count = lm_head_layer_->weights().data ? embeddingElementCount(config_) : 0;
    request.sources.lm_head.has_bias = (lm_head_layer_->bias().data != nullptr);
    request.sources.lm_head.bias.ptr = lm_head_layer_->bias().data;
    request.sources.lm_head.bias.count = lm_head_layer_->bias().data ? static_cast<std::size_t>(config_.vocab_size) : 0;

    // Process ScratchBlock weights (if enabled)
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        const int kNumAtomTypes = Tokenizer::kAtomTypeCount;
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
        
        // text_feature_projection ELIMINATED — text features merged into atom embeddings (dims 48-63)
        
        EmitModuleInfo(ModuleId::Checkpoint, "Processing ScratchBlock (atom_emb=" + 
                       std::to_string(request.sources.scratch_block.atom_type_embeddings.count) +
                       ", atom_proj=" + std::to_string(request.sources.scratch_block.atom_projection.count) + ")");
    }

    // NumericHead weights
    if (numeric_head_layer_) {
        request.sources.numeric_head.enabled = true;
        request.sources.numeric_head.d_model = config_.d_model;
        request.sources.numeric_head.weights.ptr = numeric_head_layer_->weights().data;
        request.sources.numeric_head.weights.count = static_cast<std::size_t>(2 * config_.d_model);
        request.sources.numeric_head.bias.ptr = numeric_head_layer_->bias().data;
        request.sources.numeric_head.bias.count = 2;
        EmitModuleInfo(ModuleId::Checkpoint, "Processing NumericHead weights (d_model=" +
                       std::to_string(config_.d_model) + ")");
    }

    // Issue #33: Final RMSNorm gamma (normalizes encoder output before LM head) — owned by LMHeadLayer
    if (lm_head_layer_ && lm_head_layer_->finalRmsGamma().data) {
        request.sources.final_rms_gamma.ptr = lm_head_layer_->finalRmsGamma().data;
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
        if (!embedding_layer_ || !embedding_layer_->tokenWeights().data) {
            std::cerr << "[LanguageModel::load] Error: EmbeddingLayer token weights not initialized" << std::endl;
            return false;
        }
        assignWrite(request.gpu_embedding.token_embeddings,
                    embedding_layer_->tokenWeights().data,
                    embeddingElementCount(config_));
        if (lm_head_layer_ && lm_head_layer_->finalRmsGamma().data) {
            assignWrite(request.gpu_embedding.rms_gamma,
                        lm_head_layer_->finalRmsGamma().data,
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
        assignWrite(view.ffn_w_gate, enc->ffnWGate().data, d_model * d_ff);
        assignWrite(view.ffn_w1, enc->ffnW1().data, d_model * d_ff);
        assignWrite(view.ffn_w2, enc->ffnW2().data, d_ff * d_model);
        assignWrite(view.ffn_b2, enc->ffnB2().data, d_model);
        assignWrite(view.rms1_gamma, enc->rms1Gamma().data, d_model);
        assignWrite(view.rms2_gamma, enc->rms2Gamma().data, d_model);
        // Issue #148: Sandwich norm gammas REMOVED — not loaded from checkpoint
        // LayerScale (Issue #109) — single scalar per sublayer
        if (enc->layerScale1().data) assignWrite(view.layer_scale1, enc->layerScale1().data, 1);
        if (enc->layerScale2().data) assignWrite(view.layer_scale2, enc->layerScale2().data, 1);
    }

    if (!training_state_.initialized) {
        // Initialize state based on execution mode
        if (config_.execution_mode == ModelExecutionMode::TRAINING) {
            initTrainingState();
        } else {
            initInferenceState();
        }
    }
    if (lm_head_layer_ && lm_head_layer_->weights().data) {
        assignWrite(request.lm_head.projection,
                    lm_head_layer_->weights().data,
                    embeddingElementCount(config_));
    }
    if (lm_head_layer_ && lm_head_layer_->bias().data) {
        assignWrite(request.lm_head.bias,
                    lm_head_layer_->bias().data,
                    static_cast<std::size_t>(config_.vocab_size));
    }
    request.lm_head.expect_bias = config_.use_bias;

    // Set up ScratchBlock weight destinations (if enabled)
    if (scratch_block_layer_ && scratch_block_layer_->isEnabled()) {
        const int kNumAtomTypes = Tokenizer::kAtomTypeCount;
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
        
        // text_feature_projection ELIMINATED — text features merged into atom embeddings (dims 48-63)
        // Old checkpoints may contain text_feature_projection — silently ignored on load.
        
        request.scratch_block.num_atom_types = kNumAtomTypes;
        request.scratch_block.atom_embedding_dim = atom_emb_dim;
    }

    // NumericHead weight destinations
    if (numeric_head_layer_) {
        assignWrite(request.numeric_head.weights,
                    numeric_head_layer_->weights().data,
                    static_cast<std::size_t>(2 * config_.d_model));
        assignWrite(request.numeric_head.bias,
                    numeric_head_layer_->bias().data,
                    2);
        request.numeric_head.d_model = config_.d_model;
    }

    // Issue #33: Final RMSNorm gamma destination — owned by LMHeadLayer
    if (lm_head_layer_ && lm_head_layer_->finalRmsGamma().data) {
        assignWrite(request.final_rms_gamma,
                    lm_head_layer_->finalRmsGamma().data,
                    static_cast<std::size_t>(config_.d_model));
    }

    return layer.load(request);
}

} // namespace GRIM

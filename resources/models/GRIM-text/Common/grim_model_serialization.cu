#ifndef USE_CUDA
#ifndef USE_CUDA
#define USE_CUDA
#endif
#endif

#include <fstream>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <sstream>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include "grim_model_serialization.hpp"
#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../Shared/LogRecorder/LogRecorder.hpp"
#include "../training/schemas/grim_transformer_model_generated.h"
#include "../Layers/Encoding/Encoding_GPU.hpp"
#include "../Layers/Serialization/Serialization_GPU.hpp"
#include "../Shared/UnigramByte/Unigram.hpp"
#include "grim_model_serialization_version.hpp"


namespace GRIM {

//======================================================//
//  Save/Load Model to FlatBuffer via Serialization Layer
//======================================================//

namespace {

SerializationModelConfigView makeConfigView(const Config::AiConfigSnapshot& cfg) {
    SerializationModelConfigView view{};
    view.vocab_size = HyperParameters::snapshotTrainingConfigField<int>(cfg, "vocab_size");
    view.d_model = HyperParameters::snapshotTrainingConfigField<int>(cfg, "d_model");
    view.num_layers = HyperParameters::snapshotTrainingConfigField<int>(cfg, "num_layers");
    view.num_heads = HyperParameters::snapshotTrainingConfigField<int>(cfg, "num_heads");
    view.num_kv_heads = HyperParameters::snapshotTrainingConfigField<int>(cfg, "num_kv_heads");
    view.d_ff = HyperParameters::snapshotTrainingConfigField<int>(cfg, "d_ff");
    view.max_seq_len = HyperParameters::snapshotTrainingConfigField<int>(cfg, "max_seq_len");
    view.dropout_rate = HyperParameters::snapshotTrainingConfigField<float>(cfg, "dropout_rate");
    view.positional_encoding = HyperParameters::snapshotTrainingConfigField<HyperParameters::PositionalEncodingType>(cfg, "positional_encoding");
    view.tie_embeddings = HyperParameters::snapshotTrainingConfigField<bool>(cfg, "tie_embeddings");
    view.use_gpu = HyperParameters::snapshotTrainingConfigField<bool>(cfg, "use_gpu");
    view.use_bias = HyperParameters::snapshotTrainingConfigField<bool>(cfg, "use_bias");
    return view;
}

std::size_t embeddingElementCount(const Config::AiConfigSnapshot& cfg) {
    return static_cast<std::size_t>(HyperParameters::snapshotTrainingConfigField<int>(cfg, "vocab_size")) *
           static_cast<std::size_t>(HyperParameters::snapshotTrainingConfigField<int>(cfg, "d_model"));
}

void assignRead(DeviceReadView& view, const float* ptr, std::size_t count) {
    view.ptr = ptr;
    view.count = ptr ? count : 0;  // Set count=0 if ptr is null to prevent download failures
}

void assignWrite(DeviceWriteView& view, float* ptr, std::size_t count) {
    view.ptr = ptr;
    view.count = count;
}

void requireLayerScaleVector(const Tensor& gamma,
                             std::size_t d_model,
                             const char* field_name,
                             int layer_idx) {
    if (!gamma.data) return;
    const std::string context = std::string("LanguageModel checkpoint ") + field_name;
    gamma.shape.require(context.c_str());
    if (!gamma.shape.is_2d_layout()) {
        throw std::runtime_error("LanguageModel checkpoint: " + std::string(field_name) +
                                 " in layer " + std::to_string(layer_idx) +
                                 " must be a 2D [1,d_model] gamma vector");
    }
    const auto dims = gamma.shape.as_2d();
    if (dims.rows != 1 || static_cast<std::size_t>(dims.cols) != d_model) {
        throw std::runtime_error("LanguageModel checkpoint: " + std::string(field_name) +
                                 " in layer " + std::to_string(layer_idx) +
                                 " must have shape [1,d_model]. expected=[1," +
                                 std::to_string(d_model) + "] got=[" +
                                 std::to_string(dims.rows) + "," +
                                 std::to_string(dims.cols) + "]");
    }
}

} // namespace (anonymous)

bool saveLanguageModelCheckpoint(
    LanguageModel& model,
    const GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    const ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const std::string& path) {
    using namespace GRIM::Logging;
    EmitModuleInfo(ModuleId::Checkpoint, "save() called for path: " + path);
    
    static std::mutex save_mutex;
    std::lock_guard<std::mutex> lock(save_mutex);
    EmitModuleInfo(ModuleId::Checkpoint, "Acquired save mutex");

    cudaError_t pre_sync_err = cudaDeviceSynchronize();
    if (pre_sync_err != cudaSuccess) {
        EmitModuleError(ModuleId::Checkpoint, std::string("Pre-save cudaDeviceSynchronize() failed: ") + cudaGetErrorString(pre_sync_err));
        std::cerr << "[saveLanguageModelCheckpoint] Pre-save cudaDeviceSynchronize() failed: "
                  << cudaGetErrorString(pre_sync_err) << std::endl;
        return false;
    }
    EmitModuleInfo(ModuleId::Checkpoint, "CUDA device synchronized");

    EmitModuleInfo(ModuleId::Checkpoint, "Creating SerializationLayer");
    SerializationLayer layer(SerializationConfig{});
    SerializationSaveRequest request{};
    const auto& config = model.getConfig();
    request.path = path;
    request.model_version = GRIM_MODEL_VERSION;
    request.sources.config = makeConfigView(config);
    const int vocab_size = HyperParameters::snapshotTrainingConfigField<int>(config, "vocab_size");
    const int d_model_i = HyperParameters::snapshotTrainingConfigField<int>(config, "d_model");
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(config, "num_layers");
    const int num_kv_heads = HyperParameters::snapshotTrainingConfigField<int>(config, "num_kv_heads");
    const int d_ff_i = HyperParameters::snapshotTrainingConfigField<int>(config, "d_ff");
    const int head_dim = HyperParameters::snapshotTrainingConfigField<int>(config, "head_dim");
    const float scratch_block_atom_scale = HyperParameters::snapshotTrainingConfigField<float>(config, "scratch_block_atom_scale");
    const bool mtp_enabled = HyperParameters::snapshotTrainingConfigField<bool>(config, "mtp_enabled");
    const int mtp_k = mtp_enabled
        ? HyperParameters::snapshotTrainingConfigField<int>(config, "mtp_k")
        : 0;
    auto* embedding_parameters = parameter_registry.getEmbeddingParameters();
    auto* lm_head_parameters = parameter_registry.getLmHeadParameters();
    auto* scratch_block_layer = model.getScratchBlockLayer();
    auto* scratch_block_parameters = parameter_registry.getScratchBlockParameters();
    auto* execution_block_parameters = parameter_registry.getExecutionBlockParameters();
    auto* decode_time_slot_selector = parameter_registry.getDecodeTimeSlotSelector();
    auto* gpu_encoder_owner = gpu_model_state.gpu_encoder.get();
    EmitModuleInfo(ModuleId::Checkpoint, "Request initialized with version " + std::to_string(GRIM_MODEL_VERSION));

    EmitModuleInfo(ModuleId::Checkpoint, "Processing embeddings");
    if (!embedding_parameters || !embedding_parameters->token_weights.data) {
        EmitModuleError(ModuleId::Checkpoint, "registry embedding token weights unavailable during save()");
        std::cerr << "[saveLanguageModelCheckpoint] Error: registry embedding token weights unavailable" << std::endl;
        return false;
    }
    EmitModuleInfo(ModuleId::Checkpoint, "Embedding source: GPU");
    EmitModuleInfo(ModuleId::Checkpoint, "Using registry embedding token weights (vocab=" + std::to_string(vocab_size) + ", d_model=" + std::to_string(d_model_i) + ")");
    assignRead(request.sources.gpu_embedding.token_embeddings,
               embedding_parameters->token_weights.data,
               embeddingElementCount(config));

    EmitModuleInfo(ModuleId::Checkpoint, "Processing encoder layers (" + std::to_string(num_layers) + " layers)");
    { std::ostringstream oss; oss << "gpu_model_state.gpu_encoder ptr = " << (void*)gpu_encoder_owner << " model = " << (void*)&model; EmitModuleInfo(ModuleId::Checkpoint, oss.str()); }
    auto* gpu_encoder = gpu_encoder_owner;
    { std::ostringstream oss; oss << "gpu_encoder ptr = " << (void*)gpu_encoder; EmitModuleInfo(ModuleId::Checkpoint, oss.str()); }
    if (!gpu_encoder) {
        EmitModuleError(ModuleId::Checkpoint, "GPU encoder not initialized");
        std::cerr << "[saveLanguageModelCheckpoint] Error: GPU encoder not initialized" << std::endl;
        return false;
    }
    request.sources.encoder_layers.resize(num_layers);
    const std::size_t d_model = static_cast<std::size_t>(d_model_i);
    const std::size_t d_ff = static_cast<std::size_t>(d_ff_i);
    
    // GQA dimensions for W_qkv sizing (no fallback - num_kv_heads must be properly set)
    const int kv_dim = num_kv_heads * head_dim;
    const int total_qkv_dim = d_model_i + 2 * kv_dim;  // Q + K + V with GQA
    const std::size_t qkv_weight_size = static_cast<std::size_t>(total_qkv_dim) * d_model;
    
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        if (layer_idx % 2 == 0) {
            EmitModuleInfo(ModuleId::Checkpoint, "Processing layer " + std::to_string(layer_idx) + "/" + std::to_string(num_layers));
        }
        auto* enc = gpu_encoder->getLayer(layer_idx);
        if (!enc) {
            EmitModuleError(ModuleId::Checkpoint, "GPU layer " + std::to_string(layer_idx) + " is null");
            std::cerr << "[saveLanguageModelCheckpoint] Error: GPU layer " << layer_idx << " is null" << std::endl;
            return false;
        }
        auto& view = request.sources.encoder_layers[layer_idx];
        assignRead(view.attn_w_qkv, enc->attnWqkv().data, qkv_weight_size);
        assignRead(view.attn_b_qkv, enc->attnBqkv().data, total_qkv_dim);  // GQA-aware bias size
        assignRead(view.attn_w_o, enc->attnWo().data, d_model * d_model);
        assignRead(view.attn_b_o, enc->attnBo().data, d_model);
        const auto& ffn_parameters = parameter_registry.requireFeedForwardParameters(layer_idx, "saveLanguageModelCheckpoint");
        assignRead(view.ffn_w_gate, ffn_parameters.W_gate.data, d_model * d_ff);
        assignRead(view.ffn_w1, ffn_parameters.W1.data, d_model * d_ff);
        assignRead(view.ffn_w2, ffn_parameters.W2.data, d_ff * d_model);
        assignRead(view.ffn_b2, ffn_parameters.b2.data, d_model);
        assignRead(view.rms1_gamma, enc->rms1Gamma().data, d_model);
        assignRead(view.rms2_gamma, enc->rms2Gamma().data, d_model);
        // Issue #148: Sandwich norm gammas REMOVED — not saved to checkpoint
        // Old checkpoints may contain rms_post_attn/rms_post_ffn but they're ignored on load
        // LayerScale (Issue #109) — per-channel gamma vector per sublayer
        requireLayerScaleVector(enc->layerScale1(), d_model, "layer_scale1", layer_idx);
        requireLayerScaleVector(enc->layerScale2(), d_model, "layer_scale2", layer_idx);
        if (enc->layerScale1().data) assignRead(view.layer_scale1, enc->layerScale1().data, d_model);
        if (enc->layerScale2().data) assignRead(view.layer_scale2, enc->layerScale2().data, d_model);
    }

    if (!lm_head_parameters) throw std::runtime_error("LM-head parameters are NULL at " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
    EmitModuleInfo(ModuleId::Checkpoint, "Processing LM head (projection=" + std::string(lm_head_parameters->weights.data ? "yes" : "no") + ", bias=" + std::string(lm_head_parameters->bias.data ? "yes" : "no") + ")");
    request.sources.lm_head.has_projection = (lm_head_parameters->weights.data != nullptr);
    request.sources.lm_head.projection.ptr = lm_head_parameters->weights.data;
    request.sources.lm_head.projection.count = lm_head_parameters->weights.data ? embeddingElementCount(config) : 0;
    request.sources.lm_head.has_bias = (lm_head_parameters->bias.data != nullptr);
    request.sources.lm_head.bias.ptr = lm_head_parameters->bias.data;
    request.sources.lm_head.bias.count = lm_head_parameters->bias.data ? static_cast<std::size_t>(vocab_size) : 0;

    const auto scratch_hp = HyperParameters::scratchBlockConstructionHP(config);

    // Process ScratchBlock weights (if enabled by authored architecture)
    // Use the layer's actual tensor sizes so copy count never exceeds allocation.
    // ScratchBlock allocates with Tokenizer::kAtomTypeCount (single source of truth).
    if (scratch_hp.enabled) {
        if (!scratch_block_layer) {
            throw std::runtime_error("saveLanguageModelCheckpoint: ScratchBlockConstructionHP.enabled=true but scratch_block_layer is NULL");
        }
        if (!scratch_block_parameters) {
            throw std::runtime_error("saveLanguageModelCheckpoint: ScratchBlockConstructionHP.enabled=true but registry scratch_block_parameters is NULL");
        }
        const Tensor& ate = scratch_block_parameters->atom_type_embeddings;
        const Tensor& ap = scratch_block_parameters->atom_projection;
        
        request.sources.scratch_block.enabled = true;
        request.sources.scratch_block.d_model = d_model_i;
        request.sources.scratch_block.atom_scale = scratch_block_atom_scale;
        
        if (ate.data && ate.shape.is_2d_layout()) {
            const size_t ate_numel = ate.numel();
            const int num_atom_types = ate.shape.as_2d().rows;
            const int atom_emb_dim = ate.shape.as_2d().cols;
            request.sources.scratch_block.atom_type_embeddings.ptr = ate.data;
            request.sources.scratch_block.atom_type_embeddings.count = ate_numel;
            request.sources.scratch_block.num_atom_types = num_atom_types;
            request.sources.scratch_block.atom_embedding_dim = atom_emb_dim;
        }
        
        if (ap.data && ap.shape.is_2d_layout()) {
            request.sources.scratch_block.atom_projection.ptr = ap.data;
            request.sources.scratch_block.atom_projection.count = ap.numel();
        }
        
        EmitModuleInfo(ModuleId::Checkpoint, "Processing ScratchBlock (atom_emb=" + 
                       std::to_string(request.sources.scratch_block.atom_type_embeddings.count) +
                       ", atom_proj=" + std::to_string(request.sources.scratch_block.atom_projection.count) + ")");
    }

    // ExecutionBlock v2 weights — serialized via FlatBuffer
    if (execution_block_parameters) {
        auto assignRead = [](DeviceReadView& v, const Tensor& t) {
            v.ptr = t.data;
            v.count = static_cast<std::size_t>(t.numel());
        };
        request.sources.execution_block.enabled = true;
        assignRead(request.sources.execution_block.w_decode_1, execution_block_parameters->w_decode_1);
        assignRead(request.sources.execution_block.b_decode_1, execution_block_parameters->b_decode_1);
        assignRead(request.sources.execution_block.w_decode_2, execution_block_parameters->w_decode_2);
        assignRead(request.sources.execution_block.w_arg1_select, execution_block_parameters->w_arg1_select);
        assignRead(request.sources.execution_block.w_arg2_select, execution_block_parameters->w_arg2_select);
        assignRead(request.sources.execution_block.W_op_select, execution_block_parameters->W_op_select);
        assignRead(request.sources.execution_block.W_key_proj, execution_block_parameters->W_key_proj);
        assignRead(request.sources.execution_block.W_write_query, execution_block_parameters->W_write_query);
        assignRead(request.sources.execution_block.W_write_key, execution_block_parameters->W_write_key);
        assignRead(request.sources.execution_block.alpha, execution_block_parameters->alpha);
        assignRead(request.sources.execution_block.beta, execution_block_parameters->beta);
        assignRead(request.sources.execution_block.step_embeddings, execution_block_parameters->step_embeddings);
        assignRead(request.sources.execution_block.type_num_embed, execution_block_parameters->type_num_embed);
        assignRead(request.sources.execution_block.W_value_to_emb, execution_block_parameters->W_value_to_emb);
        assignRead(request.sources.execution_block.b_value_to_emb, execution_block_parameters->b_value_to_emb);
        assignRead(request.sources.execution_block.w_inject_gate, execution_block_parameters->w_inject_gate);
        assignRead(request.sources.execution_block.W_Q_read, execution_block_parameters->W_Q_read);
        assignRead(request.sources.execution_block.W_K_read, execution_block_parameters->W_K_read);
        assignRead(request.sources.execution_block.W_V_read, execution_block_parameters->W_V_read);
        assignRead(request.sources.execution_block.W_O_read, execution_block_parameters->W_O_read);
        assignRead(request.sources.execution_block.W_gate_read, execution_block_parameters->W_gate_read);
        assignRead(request.sources.execution_block.tau, execution_block_parameters->tau);
        assignRead(request.sources.execution_block.E_slot, execution_block_parameters->E_slot);
        assignRead(request.sources.execution_block.E_op, execution_block_parameters->E_op);
        assignRead(request.sources.execution_block.W_scal, execution_block_parameters->W_scal);
        assignRead(request.sources.execution_block.b_scal, execution_block_parameters->b_scal);
        assignRead(request.sources.execution_block.W_trace, execution_block_parameters->W_trace);
        assignRead(request.sources.execution_block.b_trace, execution_block_parameters->b_trace);
        assignRead(request.sources.execution_block.W_reason_gate, execution_block_parameters->W_reason_gate);
        assignRead(request.sources.execution_block.W_trace_gate, execution_block_parameters->W_trace_gate);
        EmitModuleInfo(ModuleId::Checkpoint, "Processing ExecutionBlock v2 weights for FlatBuffer serialization");
    }

    // DecodeTimeSlotSelector weights — serialized via FlatBuffer
    if (decode_time_slot_selector) {
        auto assignRead = [](DeviceReadView& v, const Tensor& t) {
            v.ptr = t.data;
            v.count = static_cast<std::size_t>(t.numel());
        };
        request.sources.slot_selector.enabled = true;
        assignRead(request.sources.slot_selector.w_q_select, decode_time_slot_selector->W_q_select);
        assignRead(request.sources.slot_selector.w_k_select, decode_time_slot_selector->W_k_select);
        assignRead(request.sources.slot_selector.null_key_select, decode_time_slot_selector->null_key_select);
        assignRead(request.sources.slot_selector.null_logit_bias, decode_time_slot_selector->null_logit_bias);
        EmitModuleInfo(ModuleId::Checkpoint, "Processing SlotSelector weights for FlatBuffer serialization");
    }

    // Issue #33: Final RMSNorm gamma (normalizes encoder output before LM head)
    if (lm_head_parameters && lm_head_parameters->final_rms_gamma.data) {
        request.sources.final_rms_gamma.ptr = lm_head_parameters->final_rms_gamma.data;
        request.sources.final_rms_gamma.count = static_cast<std::size_t>(d_model_i);
        EmitModuleInfo(ModuleId::Checkpoint, "Processing final_rms_gamma (size=" + 
                   std::to_string(d_model_i) + ")");
    }

    EmitModuleInfo(ModuleId::Checkpoint, "Calling SerializationLayer::save()");
    bool result = layer.save(request);
    EmitModuleInfo(ModuleId::Checkpoint, std::string("SerializationLayer::save() returned ") + (result ? "true" : "false"));
    if (!result) return false;

    // MTP heads: save to sidecar <path>.mtp (checkpoint has no MTP in FlatBuffer schema)
    if (mtp_enabled && mtp_k > 0) {
        const std::string mtp_path = path + ".mtp";
        std::ofstream ofs(mtp_path, std::ios::binary);
        if (!ofs) {
            EmitModuleError(ModuleId::Checkpoint, "[save] Failed to open MTP sidecar: " + mtp_path);
            return false;
        }
        const uint32_t K = static_cast<uint32_t>(mtp_k);
        ofs.write(reinterpret_cast<const char*>(&K), sizeof(K));
        const std::size_t weight_elems = static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(d_model_i);
        const std::size_t bias_elems = static_cast<std::size_t>(vocab_size);
        std::vector<float> h_buf(std::max(weight_elems, bias_elems));
        const auto& mtp_heads = parameter_registry.mtpHeadParameterTensors();
        for (uint32_t k = 0; k < K; ++k) {
            GRIM::MtpHeadParameterTensors* head =
                (k < mtp_heads.size())
                    ? const_cast<GRIM::MtpHeadParameterTensors*>(&mtp_heads[k])
                    : nullptr;
            if (!head || !head->weight.data || !head->bias.data) continue;
            if (cudaMemcpy(h_buf.data(), head->weight.data, weight_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                EmitModuleError(ModuleId::Checkpoint, "[save] MTP head " + std::to_string(k) + " weight D2H failed");
                return false;
            }
            ofs.write(reinterpret_cast<const char*>(h_buf.data()), weight_elems * sizeof(float));
            if (cudaMemcpy(h_buf.data(), head->bias.data, bias_elems * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                EmitModuleError(ModuleId::Checkpoint, "[save] MTP head " + std::to_string(k) + " bias D2H failed");
                return false;
            }
            ofs.write(reinterpret_cast<const char*>(h_buf.data()), bias_elems * sizeof(float));
        }
        EmitModuleInfo(ModuleId::Checkpoint, "[save] MTP sidecar written: " + mtp_path + " (K=" + std::to_string(K) + ")");
    }
    return true;
}

bool loadLanguageModelCheckpoint(
    LanguageModel& model,
    const TrainingState& training_state,
    const GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const std::string& path) {
    using namespace GRIM::Logging;
    if (path.empty()) {
        EmitModuleError(ModuleId::Checkpoint, "[load] Requested checkpoint path is empty");
        std::cerr << "[loadLanguageModelCheckpoint] Error: requested checkpoint path is empty" << std::endl;
        return false;
    }
    {
        const std::filesystem::path checkpoint_path(path);
        std::error_code ec;
        const bool exists = std::filesystem::exists(checkpoint_path, ec);
        if (ec) {
            EmitModuleError(ModuleId::Checkpoint,
                            "[load] Failed to query checkpoint path '" + path + "': " + ec.message());
            std::cerr << "[loadLanguageModelCheckpoint] Error: failed to query checkpoint path '"
                      << path << "': " << ec.message() << std::endl;
            return false;
        }
        if (!exists) {
            EmitModuleError(ModuleId::Checkpoint,
                            "[load] Requested checkpoint does not exist: " + path);
            std::cerr << "[loadLanguageModelCheckpoint] Error: requested checkpoint does not exist: "
                      << path << std::endl;
            return false;
        }
        const bool regular = std::filesystem::is_regular_file(checkpoint_path, ec);
        if (ec) {
            EmitModuleError(ModuleId::Checkpoint,
                            "[load] Failed to inspect checkpoint path '" + path + "': " + ec.message());
            std::cerr << "[loadLanguageModelCheckpoint] Error: failed to inspect checkpoint path '"
                      << path << "': " << ec.message() << std::endl;
            return false;
        }
        if (!regular) {
            EmitModuleError(ModuleId::Checkpoint,
                            "[load] Requested checkpoint is not a regular file: " + path);
            std::cerr << "[loadLanguageModelCheckpoint] Error: requested checkpoint is not a regular file: "
                      << path << std::endl;
            return false;
        }
    }
    SerializationLayer layer(SerializationConfig{});
    SerializationLoadRequest request{};
    const auto& config = model.getConfig();
    request.path = path;
    request.config = makeConfigView(config);
    const int vocab_size = HyperParameters::snapshotTrainingConfigField<int>(config, "vocab_size");
    const int d_model_i = HyperParameters::snapshotTrainingConfigField<int>(config, "d_model");
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(config, "num_layers");
    const int num_heads = HyperParameters::snapshotTrainingConfigField<int>(config, "num_heads");
    const int num_kv_heads = HyperParameters::snapshotTrainingConfigField<int>(config, "num_kv_heads");
    const int d_ff_i = HyperParameters::snapshotTrainingConfigField<int>(config, "d_ff");
    const int max_seq_len = HyperParameters::snapshotTrainingConfigField<int>(config, "max_seq_len");
    const int head_dim = HyperParameters::snapshotTrainingConfigField<int>(config, "head_dim");
    const bool freeze_learned_rms_gammas = HyperParameters::snapshotTrainingConfigField<bool>(config, "freeze_learned_rms_gammas");
    const bool use_bias = HyperParameters::snapshotTrainingConfigField<bool>(config, "use_bias");
    const bool use_gpu = HyperParameters::snapshotTrainingConfigField<bool>(config, "use_gpu");
    const bool tie_embeddings = HyperParameters::snapshotTrainingConfigField<bool>(config, "tie_embeddings");
    const bool mtp_enabled = HyperParameters::snapshotTrainingConfigField<bool>(config, "mtp_enabled");
    const int mtp_k = mtp_enabled
        ? HyperParameters::snapshotTrainingConfigField<int>(config, "mtp_k")
        : 0;
    auto* embedding_parameters = parameter_registry.getEmbeddingParameters();
    auto* lm_head_parameters = parameter_registry.getLmHeadParameters();
    auto* scratch_block_layer = model.getScratchBlockLayer();
    auto* scratch_block_parameters = parameter_registry.getScratchBlockParameters();
    auto* execution_block_parameters = parameter_registry.getExecutionBlockParameters();
    auto* decode_time_slot_selector = parameter_registry.getDecodeTimeSlotSelector();

    if (!training_state.initialized) {
        EmitModuleError(ModuleId::Checkpoint,
                        "[load] Runtime state is not initialized. Caller must complete explicit startup before loadLanguageModelCheckpoint().");
        std::cerr << "[loadLanguageModelCheckpoint] Error: runtime state is not initialized. "
              << "Caller must complete explicit startup before checkpoint load." << std::endl;
        return false;
    }

    if (!gpu_model_state.gpu_encoder) {
        EmitModuleError(ModuleId::Checkpoint,
                        "[load] GPU encoder is not initialized. Caller must complete Startup::assembleGpuModel(config, training_state, gpu_model_state, parameter_registry, weight_init_seed) before loadLanguageModelCheckpoint().");
        std::cerr << "[loadLanguageModelCheckpoint] Error: GPU encoder is not initialized. "
              << "Caller must complete Startup::assembleGpuModel(config, training_state, gpu_model_state, parameter_registry, weight_init_seed) before checkpoint load." << std::endl;
        return false;
    }

    const auto scratch_hp = HyperParameters::scratchBlockConstructionHP(config);
    if (scratch_hp.enabled && !scratch_block_layer) {
        EmitModuleError(ModuleId::Checkpoint,
                        "[load] ScratchBlockConstructionHP.enabled=true but scratch_block_layer is NULL.");
        std::cerr << "[loadLanguageModelCheckpoint] Error: ScratchBlockConstructionHP.enabled=true but scratch_block_layer is NULL" << std::endl;
        return false;
    }
    if (scratch_hp.enabled && !scratch_block_parameters) {
        EmitModuleError(ModuleId::Checkpoint,
                        "[load] ScratchBlockConstructionHP.enabled=true but registry scratch_block_parameters is NULL.");
        std::cerr << "[loadLanguageModelCheckpoint] Error: ScratchBlockConstructionHP.enabled=true but registry scratch_block_parameters is NULL" << std::endl;
        return false;
    }
    if (!scratch_hp.enabled && scratch_block_layer) {
        EmitModuleError(ModuleId::Checkpoint,
                        "[load] scratch_block_layer exists while ScratchBlockConstructionHP.enabled=false.");
        std::cerr << "[loadLanguageModelCheckpoint] Error: scratch_block_layer exists while ScratchBlockConstructionHP.enabled=false" << std::endl;
        return false;
    }

    // Pattern B: call site is the sole authority for what the model requires.
    request.capabilities.requires_execution_block = (execution_block_parameters != nullptr);
    request.capabilities.requires_slot_selector     = (decode_time_slot_selector != nullptr);
    request.capabilities.requires_scratch_block   = scratch_hp.enabled;
    request.capabilities.requires_final_rms_gamma = (lm_head_parameters != nullptr
                                                      && lm_head_parameters->final_rms_gamma.data != nullptr
                                                      && !freeze_learned_rms_gammas);

    if (!embedding_parameters || !embedding_parameters->token_weights.data) {
        std::cerr << "[loadLanguageModelCheckpoint] Error: registry embedding token weights not initialized" << std::endl;
        return false;
    }
    assignWrite(request.gpu_embedding.token_embeddings,
                embedding_parameters->token_weights.data,
                embeddingElementCount(config));

    auto* gpu_encoder = gpu_model_state.gpu_encoder.get();
    request.encoder_layers.resize(num_layers);
    const std::size_t d_model = static_cast<std::size_t>(d_model_i);
    const std::size_t d_ff = static_cast<std::size_t>(d_ff_i);
    
    // GQA dimensions for W_qkv sizing - MUST match save() calculation!
    // BUG FIX Issue #24: load() was using MHA formula (d_model * 3) but save() uses GQA formula
    const int kv_dim = num_kv_heads * head_dim;
    const int total_qkv_dim = d_model_i + 2 * kv_dim;  // Q + K + V with GQA
    const std::size_t qkv_weight_size = static_cast<std::size_t>(total_qkv_dim) * d_model;
    
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        auto* enc = gpu_encoder->getLayer(layer_idx);
        if (!enc) {
            std::cerr << "[loadLanguageModelCheckpoint] Error: GPU layer " << layer_idx << " is null" << std::endl;
            return false;
        }
        auto& view = request.encoder_layers[layer_idx];
        assignWrite(view.attn_w_qkv, enc->attnWqkv().data, qkv_weight_size);
        assignWrite(view.attn_b_qkv, enc->attnBqkv().data, total_qkv_dim);  // GQA-aware bias size
        assignWrite(view.attn_w_o, enc->attnWo().data, d_model * d_model);
        assignWrite(view.attn_b_o, enc->attnBo().data, d_model);
        auto& ffn_parameters = parameter_registry.requireFeedForwardParameters(layer_idx, "loadLanguageModelCheckpoint");
        assignWrite(view.ffn_w_gate, ffn_parameters.W_gate.data, d_model * d_ff);
        assignWrite(view.ffn_w1, ffn_parameters.W1.data, d_model * d_ff);
        assignWrite(view.ffn_w2, ffn_parameters.W2.data, d_ff * d_model);
        assignWrite(view.ffn_b2, ffn_parameters.b2.data, d_model);
        if (!freeze_learned_rms_gammas) {
            assignWrite(view.rms1_gamma, enc->rms1Gamma().data, d_model);
            assignWrite(view.rms2_gamma, enc->rms2Gamma().data, d_model);
        }
        // Issue #148: Sandwich norm gammas REMOVED — not loaded from checkpoint
        // LayerScale (Issue #109) — per-channel gamma vector per sublayer
        requireLayerScaleVector(enc->layerScale1(), d_model, "layer_scale1", layer_idx);
        requireLayerScaleVector(enc->layerScale2(), d_model, "layer_scale2", layer_idx);
        if (enc->layerScale1().data) assignWrite(view.layer_scale1, enc->layerScale1().data, d_model);
        if (enc->layerScale2().data) assignWrite(view.layer_scale2, enc->layerScale2().data, d_model);
    }

    if (lm_head_parameters && lm_head_parameters->weights.data) {
        assignWrite(request.lm_head.projection,
                    lm_head_parameters->weights.data,
                    embeddingElementCount(config));
    }
    if (lm_head_parameters && lm_head_parameters->bias.data) {
        assignWrite(request.lm_head.bias,
                    lm_head_parameters->bias.data,
                    static_cast<std::size_t>(vocab_size));
    }
    request.lm_head.expect_bias = use_bias;

    // Set up ScratchBlock weight destinations (if enabled by authored architecture)
    // Use the layer's actual tensor sizes so load size matches saved checkpoint (same as save path).
    if (scratch_hp.enabled) {
        Tensor& ate = scratch_block_parameters->atom_type_embeddings;
        Tensor& ap = scratch_block_parameters->atom_projection;
        
        if (ate.data) {
            assignWrite(request.scratch_block.atom_type_embeddings, ate.data, ate.numel());
            if (ate.shape.is_2d_layout()) {
                request.scratch_block.num_atom_types = ate.shape.as_2d().rows;
                request.scratch_block.atom_embedding_dim = ate.shape.as_2d().cols;
            }
        }
        if (ap.data) {
            assignWrite(request.scratch_block.atom_projection, ap.data, ap.numel());
        }
    }

    // ExecutionBlock v2 weight destinations — loaded via FlatBuffer
    if (execution_block_parameters) {
        assignWrite(request.execution_block.w_decode_1, execution_block_parameters->w_decode_1.data, static_cast<std::size_t>(execution_block_parameters->w_decode_1.numel()));
        assignWrite(request.execution_block.b_decode_1, execution_block_parameters->b_decode_1.data, static_cast<std::size_t>(execution_block_parameters->b_decode_1.numel()));
        assignWrite(request.execution_block.w_decode_2, execution_block_parameters->w_decode_2.data, static_cast<std::size_t>(execution_block_parameters->w_decode_2.numel()));
        assignWrite(request.execution_block.w_arg1_select, execution_block_parameters->w_arg1_select.data, static_cast<std::size_t>(execution_block_parameters->w_arg1_select.numel()));
        assignWrite(request.execution_block.w_arg2_select, execution_block_parameters->w_arg2_select.data, static_cast<std::size_t>(execution_block_parameters->w_arg2_select.numel()));
        assignWrite(request.execution_block.W_op_select, execution_block_parameters->W_op_select.data, static_cast<std::size_t>(execution_block_parameters->W_op_select.numel()));
        assignWrite(request.execution_block.W_key_proj, execution_block_parameters->W_key_proj.data, static_cast<std::size_t>(execution_block_parameters->W_key_proj.numel()));
        assignWrite(request.execution_block.W_write_query, execution_block_parameters->W_write_query.data, static_cast<std::size_t>(execution_block_parameters->W_write_query.numel()));
        assignWrite(request.execution_block.W_write_key, execution_block_parameters->W_write_key.data, static_cast<std::size_t>(execution_block_parameters->W_write_key.numel()));
        assignWrite(request.execution_block.alpha, execution_block_parameters->alpha.data, static_cast<std::size_t>(execution_block_parameters->alpha.numel()));
        assignWrite(request.execution_block.beta, execution_block_parameters->beta.data, static_cast<std::size_t>(execution_block_parameters->beta.numel()));
        assignWrite(request.execution_block.step_embeddings, execution_block_parameters->step_embeddings.data, static_cast<std::size_t>(execution_block_parameters->step_embeddings.numel()));
        assignWrite(request.execution_block.type_num_embed, execution_block_parameters->type_num_embed.data, static_cast<std::size_t>(execution_block_parameters->type_num_embed.numel()));
        assignWrite(request.execution_block.W_value_to_emb, execution_block_parameters->W_value_to_emb.data, static_cast<std::size_t>(execution_block_parameters->W_value_to_emb.numel()));
        assignWrite(request.execution_block.b_value_to_emb, execution_block_parameters->b_value_to_emb.data, static_cast<std::size_t>(execution_block_parameters->b_value_to_emb.numel()));
        assignWrite(request.execution_block.w_inject_gate, execution_block_parameters->w_inject_gate.data, static_cast<std::size_t>(execution_block_parameters->w_inject_gate.numel()));
        assignWrite(request.execution_block.W_Q_read, execution_block_parameters->W_Q_read.data, static_cast<std::size_t>(execution_block_parameters->W_Q_read.numel()));
        assignWrite(request.execution_block.W_K_read, execution_block_parameters->W_K_read.data, static_cast<std::size_t>(execution_block_parameters->W_K_read.numel()));
        assignWrite(request.execution_block.W_V_read, execution_block_parameters->W_V_read.data, static_cast<std::size_t>(execution_block_parameters->W_V_read.numel()));
        assignWrite(request.execution_block.W_O_read, execution_block_parameters->W_O_read.data, static_cast<std::size_t>(execution_block_parameters->W_O_read.numel()));
        assignWrite(request.execution_block.W_gate_read, execution_block_parameters->W_gate_read.data, static_cast<std::size_t>(execution_block_parameters->W_gate_read.numel()));
        assignWrite(request.execution_block.tau, execution_block_parameters->tau.data, static_cast<std::size_t>(execution_block_parameters->tau.numel()));
        assignWrite(request.execution_block.E_slot, execution_block_parameters->E_slot.data, static_cast<std::size_t>(execution_block_parameters->E_slot.numel()));
        assignWrite(request.execution_block.E_op, execution_block_parameters->E_op.data, static_cast<std::size_t>(execution_block_parameters->E_op.numel()));
        assignWrite(request.execution_block.W_scal, execution_block_parameters->W_scal.data, static_cast<std::size_t>(execution_block_parameters->W_scal.numel()));
        assignWrite(request.execution_block.b_scal, execution_block_parameters->b_scal.data, static_cast<std::size_t>(execution_block_parameters->b_scal.numel()));
        assignWrite(request.execution_block.W_trace, execution_block_parameters->W_trace.data, static_cast<std::size_t>(execution_block_parameters->W_trace.numel()));
        assignWrite(request.execution_block.b_trace, execution_block_parameters->b_trace.data, static_cast<std::size_t>(execution_block_parameters->b_trace.numel()));
        assignWrite(request.execution_block.W_reason_gate, execution_block_parameters->W_reason_gate.data, static_cast<std::size_t>(execution_block_parameters->W_reason_gate.numel()));
        assignWrite(request.execution_block.W_trace_gate, execution_block_parameters->W_trace_gate.data, static_cast<std::size_t>(execution_block_parameters->W_trace_gate.numel()));
    }

    // DecodeTimeSlotSelector weight destinations — loaded via FlatBuffer
    if (decode_time_slot_selector) {
        assignWrite(request.slot_selector.w_q_select, decode_time_slot_selector->W_q_select.data, static_cast<std::size_t>(decode_time_slot_selector->W_q_select.numel()));
        assignWrite(request.slot_selector.w_k_select, decode_time_slot_selector->W_k_select.data, static_cast<std::size_t>(decode_time_slot_selector->W_k_select.numel()));
        assignWrite(request.slot_selector.null_key_select, decode_time_slot_selector->null_key_select.data, static_cast<std::size_t>(decode_time_slot_selector->null_key_select.numel()));
        assignWrite(request.slot_selector.null_logit_bias, decode_time_slot_selector->null_logit_bias.data, static_cast<std::size_t>(decode_time_slot_selector->null_logit_bias.numel()));
    }

    // Issue #33: Final RMSNorm gamma destination
    // When frozen, γ_final stays at 1.0 — do NOT overwrite from checkpoint.
    if (lm_head_parameters && lm_head_parameters->final_rms_gamma.data
        && !freeze_learned_rms_gammas) {
        assignWrite(request.final_rms_gamma,
                    lm_head_parameters->final_rms_gamma.data,
                    static_cast<std::size_t>(d_model_i));
    }

    bool result = layer.load(request);
    if (!result) {
        EmitModuleError(ModuleId::Checkpoint, "[load] FAILURE — dumping model & request state for diagnostics");
        EmitModuleError(ModuleId::Checkpoint, "[load]   path=" + path);
        {
            std::ostringstream oss;
            oss << "[load]   model config: vocab=" << vocab_size
                << " d_model=" << d_model_i << " num_layers=" << num_layers
                << " num_heads=" << num_heads << " num_kv_heads=" << num_kv_heads
                << " d_ff=" << d_ff_i << " max_seq_len=" << max_seq_len
                << " head_dim=" << head_dim;
            EmitModuleError(ModuleId::Checkpoint, oss.str());
        }
        {
            std::ostringstream oss;
            oss << "[load]   tie_embeddings=" << tie_embeddings
                << " use_bias=" << use_bias << " use_gpu=" << use_gpu
                << " mtp_enabled=" << mtp_enabled;
            EmitModuleError(ModuleId::Checkpoint, oss.str());
        }
        {
            std::ostringstream oss;
            oss << "[load]   capabilities: exec_block=" << request.capabilities.requires_execution_block
                << " slot_selector=" << request.capabilities.requires_slot_selector
                << " scratch=" << request.capabilities.requires_scratch_block
                << " final_rms=" << request.capabilities.requires_final_rms_gamma;
            EmitModuleError(ModuleId::Checkpoint, oss.str());
        }
        {
            std::ostringstream oss;
                oss << "[load]   registry pointers: embedding=" << (embedding_parameters ? "OK" : "NULL")
                << " lm_head_params=" << (lm_head_parameters ? "OK" : "NULL")
                << " scratch_block=" << (scratch_block_layer ? "OK" : "NULL")
                << " scratch_block_params=" << (scratch_block_parameters ? "OK" : "NULL")
                << " exec_block=" << (execution_block_parameters ? "OK" : "NULL")
                << " slot_selector=" << (decode_time_slot_selector ? "OK" : "NULL");
            EmitModuleError(ModuleId::Checkpoint, oss.str());
        }
        {
            std::ostringstream oss;
            oss << "[load]   GPU destinations: token_emb=" << (request.gpu_embedding.token_embeddings.ptr ? "set" : "NULL")
                << "(" << request.gpu_embedding.token_embeddings.count << ")"
                << " final_rms_gamma=" << (request.final_rms_gamma.ptr ? "set" : "SKIP(frozen)")
                << " lm_proj=" << (request.lm_head.projection.ptr ? "set" : "NULL")
                << "(" << request.lm_head.projection.count << ")"
                << " lm_bias=" << (request.lm_head.bias.ptr ? "set" : "NULL")
                << "(" << request.lm_head.bias.count << ")";
            EmitModuleError(ModuleId::Checkpoint, oss.str());
        }
        {
            std::ostringstream oss;
            oss << "[load]   encoder_layers=" << request.encoder_layers.size();
            if (!request.encoder_layers.empty()) {
                const auto& first = request.encoder_layers[0];
                oss << " layer0: w_qkv=" << (first.attn_w_qkv.ptr ? "set" : "NULL")
                    << "(" << first.attn_w_qkv.count << ")"
                    << " w_o=" << (first.attn_w_o.ptr ? "set" : "NULL")
                    << "(" << first.attn_w_o.count << ")"
                    << " w1=" << (first.ffn_w1.ptr ? "set" : "NULL")
                    << "(" << first.ffn_w1.count << ")";
            }
            EmitModuleError(ModuleId::Checkpoint, oss.str());
        }
        // CUDA memory state at time of failure
        {
            std::size_t free_mem = 0, total_mem = 0;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                std::ostringstream oss;
                oss << "[load]   CUDA memory: free=" << free_mem / (1024*1024)
                    << "MB total=" << total_mem / (1024*1024)
                    << "MB used=" << (total_mem - free_mem) / (1024*1024) << "MB";
                EmitModuleError(ModuleId::Checkpoint, oss.str());
            }
        }
        return false;
    }

    // MTP heads: load from sidecar <path>.mtp if present and config has MTP enabled
    if (mtp_enabled && mtp_k > 0) {
        const std::string mtp_path = path + ".mtp";
        std::ifstream ifs(mtp_path, std::ios::binary);
        if (ifs) {
            uint32_t file_k = 0;
            if (!ifs.read(reinterpret_cast<char*>(&file_k), sizeof(file_k)) || file_k != static_cast<uint32_t>(mtp_k)) {
                EmitModuleInfo(ModuleId::Checkpoint, "[load] MTP sidecar K mismatch or read failed — using freshly initialized MTP heads");
            } else {
                const std::size_t weight_elems = static_cast<std::size_t>(vocab_size) * static_cast<std::size_t>(d_model_i);
                const std::size_t bias_elems = static_cast<std::size_t>(vocab_size);
                std::vector<float> h_buf(std::max(weight_elems, bias_elems));
                auto& mtp_heads = parameter_registry.mtpHeadParameterTensors();
                bool ok = true;
                for (uint32_t k = 0; k < file_k && ok; ++k) {
                    GRIM::MtpHeadParameterTensors* head =
                        (k < mtp_heads.size())
                            ? &mtp_heads[k]
                            : nullptr;
                    if (!head || !head->weight.data || !head->bias.data) { ok = false; break; }
                    if (!ifs.read(reinterpret_cast<char*>(h_buf.data()), weight_elems * sizeof(float))) { ok = false; break; }
                    if (cudaMemcpy(head->weight.data, h_buf.data(), weight_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { ok = false; break; }
                    if (!ifs.read(reinterpret_cast<char*>(h_buf.data()), bias_elems * sizeof(float))) { ok = false; break; }
                    if (cudaMemcpy(head->bias.data, h_buf.data(), bias_elems * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) { ok = false; break; }
                }
                if (ok) {
                    EmitModuleInfo(ModuleId::Checkpoint, "[load] MTP sidecar loaded: " + mtp_path + " (K=" + std::to_string(file_k) + ")");
                } else {
                    EmitModuleInfo(ModuleId::Checkpoint, "[load] MTP sidecar read/copy failed — using freshly initialized MTP heads");
                }
            }
        } else {
            EmitModuleInfo(ModuleId::Checkpoint, "[load] No MTP sidecar — using freshly initialized MTP heads");
        }
    }
    return true;
}

} // namespace GRIM

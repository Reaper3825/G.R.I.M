#include "ParameterGroupRegistration.hpp"

#include "../../../../GRIM/grim_language_model_cuda.hpp"
#include "../../../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../../../Shared/Optimizers/OptimizerState_GPU.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIMText::Training::Startup::ModelRegistration {

#ifdef USE_CUDA

namespace {

constexpr auto kRegistrationModule = GRIM::Logging::ModuleId::Training;

using GRIM::LanguageModel;
using GRIM::ParameterGroup;
using GRIM::ParamStatsBucket;
using GRIM::ParamGroupType;
using GRIM::Tensor;
using GRIM::HyperParameters::ParameterGroupPrecision;
using GRIM::HyperParameters::LanguageModelConfig;
using GRIM::HyperParameters::scratchBlockConstructionHP;

std::string tensorDebugSummary(const Tensor& tensor) {
    std::ostringstream oss;
    oss << "data=" << tensor.data
        << " grad=" << tensor.grad_data()
        << " has_grad=" << static_cast<int>(tensor.has_grad())
        << " numel=" << tensor.numel();
    return oss.str();
}

void emitInfo(const std::string& message) {
    GRIM::Logging::EmitModuleInfo(kRegistrationModule, message);
}

size_t paramGroupTypeIndex(ParamGroupType type) {
    switch (type) {
        case ParamGroupType::EMBEDDING:       return 0;
        case ParamGroupType::LM_HEAD:         return 1;
        case ParamGroupType::ATTENTION:       return 2;
        case ParamGroupType::FFN:             return 3;
        case ParamGroupType::RMSNORM:         return 4;
        case ParamGroupType::SCRATCHBLOCK:    return 5;
        case ParamGroupType::MTP:             return 6;
        case ParamGroupType::REASONING_HEAD:  return 7;
        case ParamGroupType::EXECUTION_BLOCK: return 8;
        case ParamGroupType::SLOT_SELECTOR:   return 9;
        case ParamGroupType::COUNT: break;
    }
    throw std::runtime_error("[buildParameterGroups] invalid ParamGroupType::COUNT in registered group summary");
}

const char* paramGroupTypeSummaryName(ParamGroupType type) {
    switch (type) {
        case ParamGroupType::EMBEDDING:       return "embedding";
        case ParamGroupType::LM_HEAD:         return "lm_head";
        case ParamGroupType::ATTENTION:       return "attention";
        case ParamGroupType::FFN:             return "ffn";
        case ParamGroupType::RMSNORM:         return "rmsnorm";
        case ParamGroupType::SCRATCHBLOCK:    return "scratchblock";
        case ParamGroupType::MTP:             return "mtp";
        case ParamGroupType::REASONING_HEAD:  return "reasoning_head";
        case ParamGroupType::EXECUTION_BLOCK: return "execution_block";
        case ParamGroupType::SLOT_SELECTOR:   return "slot_selector";
        case ParamGroupType::COUNT: break;
    }
    throw std::runtime_error("[buildParameterGroups] invalid ParamGroupType::COUNT in registered group summary");
}

size_t parameterPrecisionIndex(ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::FP32:         return 0;
        case ParameterGroupPrecision::BF16_COMPUTE: return 1;
        case ParameterGroupPrecision::UNSPECIFIED: break;
    }
    throw std::runtime_error("[buildParameterGroups] invalid ParameterGroupPrecision in registered group summary");
}

const char* parameterPrecisionSummaryName(ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::FP32:         return "FP32";
        case ParameterGroupPrecision::BF16_COMPUTE: return "BF16_COMPUTE";
        case ParameterGroupPrecision::UNSPECIFIED: break;
    }
    throw std::runtime_error("[buildParameterGroups] invalid ParameterGroupPrecision in registered group summary");
}

void validateRegisteredPrecisionSupport(const std::string& name,
                                        ParamGroupType type,
                                        ParameterGroupPrecision precision) {
    switch (precision) {
        case ParameterGroupPrecision::FP32:
            return;
        case ParameterGroupPrecision::BF16_COMPUTE:
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " requests BF16_COMPUTE for parameter group type " +
                                     paramGroupTypeSummaryName(type) +
                                     ", but TensorContract implicit BF16 compute consumers are not wired for optimizer-visible parameter tensors in this build; set training.config.parameter_precision_" +
                                     paramGroupTypeSummaryName(type) + " to fp32");
        case ParameterGroupPrecision::UNSPECIFIED:
            break;
    }
    throw std::runtime_error("[buildParameterGroups] " + name +
                             " has UNSPECIFIED or unknown ParameterGroupPrecision");
}

template <typename LayerT>
LayerT& requireLayer(LayerT* layer, const char* layer_name, const char* caller) {
    if (!layer) {
        throw std::runtime_error(std::string(caller) + ": " + layer_name +
                                 " is NULL - startup initialization order is broken");
    }
    return *layer;
}

void throwUntrainableTensor(const std::string& name, const Tensor& tensor, int layer) {
    throw std::runtime_error("[buildParameterGroups] " + name +
                             " is not a trainable parameter group tensor: " +
                             tensorDebugSummary(tensor) + " layer=" + std::to_string(layer));
}

class Registrar {
public:
    Registrar(std::vector<ParameterGroup>& groups,
              const LanguageModelConfig& config)
        : groups_(groups), config_(config) {}

    void addTensor(const std::string& name,
                   Tensor& tensor,
                   ParamGroupType type,
                   ParamStatsBucket stats_bucket,
                   int layer = -1,
                   float wd_mult = 1.0f,
                   float lr_mult = 1.0f) {
        if (!tensor.data || !tensor.has_grad() || tensor.numel() == 0) {
            throwUntrainableTensor(name, tensor, layer);
        }
        if (stats_bucket == ParamStatsBucket::COUNT) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " has invalid ParamStatsBucket::COUNT");
        }
        const void* data_ptr = static_cast<const void*>(tensor.data);
        for (const void* registered_ptr : registered_data_) {
            if (registered_ptr == data_ptr) {
                throw std::runtime_error("[buildParameterGroups] duplicate tensor.data registration for " + name +
                                         " would double-step the same memory: " +
                                         tensorDebugSummary(tensor) + " layer=" + std::to_string(layer));
            }
        }
        registered_data_.push_back(data_ptr);

        const ParameterGroupPrecision precision = precisionForType(type);
        validateRegisteredPrecisionSupport(name, type, precision);

        const TensorContract::PrecisionType tensor_precision =
            TensorContract::precision_from_parameter_group_precision(precision);
        if (tensor.precision() != TensorContract::PrecisionType::FP32 &&
            tensor.precision() != tensor_precision) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " tensor precision metadata already equals " +
                                     TensorContract::precision_name(tensor.precision()) +
                                     " but registration requires " +
                                     TensorContract::precision_name(tensor_precision));
        }
        tensor.set_compute_precision(tensor_precision, "ParameterGroupRegistration::Registrar::addTensor");

        ParameterGroup group{};
        group.name = name;
        group.tensor = &tensor;
        group.m_tensor = nullptr;
        group.v_tensor = nullptr;
        group.type = type;
        group.parameter_precision = precision;
        group.stats_bucket = stats_bucket;
        group.layer_index = layer;
        group.weight_decay_multiplier = wd_mult;
        group.lr_multiplier = lr_mult;
        if (layer >= 0) {
            group.upsilon = GRIM::HyperParameters::UPSILON_BASE *
                std::sqrt(static_cast<float>(GRIM::HyperParameters::UPSILON_REFERENCE_LAYERS) /
                          static_cast<float>(layer + 1));
        }
        groups_.push_back(group);
    }

    void addConfigGatedTensor(const std::string& name,
                              Tensor& tensor,
                              ParamGroupType type,
                              ParamStatsBucket stats_bucket,
                              int layer,
                              bool enabled,
                              const char* disabled_reason) {
        if (enabled) {
            addTensor(name, tensor, type, stats_bucket, layer);
            return;
        }

        if (tensor.data || tensor.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " exists while disabled (" + disabled_reason + "): " +
                                     tensorDebugSummary(tensor));
        }
    }

private:
    ParameterGroupPrecision precisionForType(ParamGroupType type) const {
        switch (type) {
            case ParamGroupType::EMBEDDING:       return config_.parameter_precision_embedding;
            case ParamGroupType::LM_HEAD:         return config_.parameter_precision_lm_head;
            case ParamGroupType::ATTENTION:       return config_.parameter_precision_attention;
            case ParamGroupType::FFN:             return config_.parameter_precision_ffn;
            case ParamGroupType::RMSNORM:         return config_.parameter_precision_rmsnorm;
            case ParamGroupType::SCRATCHBLOCK:    return config_.parameter_precision_scratchblock;
            case ParamGroupType::MTP:             return config_.parameter_precision_mtp;
            case ParamGroupType::REASONING_HEAD:  return config_.parameter_precision_reasoning_head;
            case ParamGroupType::EXECUTION_BLOCK: return config_.parameter_precision_execution_block;
            case ParamGroupType::SLOT_SELECTOR:   return config_.parameter_precision_slot_selector;
            case ParamGroupType::COUNT: break;
        }
        throw std::runtime_error("[buildParameterGroups] invalid ParamGroupType::COUNT for parameter precision lookup");
    }

    std::vector<ParameterGroup>& groups_;
    const LanguageModelConfig& config_;
    std::vector<const void*> registered_data_;
};

void validateEmbeddingLmHeadAliasing(const Tensor& embedding_weights,
                                     const Tensor& lm_head_weights,
                                     const LanguageModelConfig& config) {
    if (!embedding_weights.data || !lm_head_weights.data) {
        throw std::runtime_error("[buildParameterGroups] cannot validate embedding/LM-head aliasing with NULL data: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }

    const bool tied = config.tie_embeddings;
    const bool same_data = embedding_weights.data == lm_head_weights.data;
    if (tied && !same_data) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=true but embedding and LM-head data pointers differ: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }
    if (!tied && same_data) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=false but embedding and LM-head data pointers are identical: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }
    if (tied && config.parameter_precision_embedding != config.parameter_precision_lm_head) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=true requires parameter_precision_embedding and parameter_precision_lm_head to match; embedding=" +
                                 std::string(parameterPrecisionSummaryName(config.parameter_precision_embedding)) +
                                 " lm_head=" +
                                 std::string(parameterPrecisionSummaryName(config.parameter_precision_lm_head)));
    }
}

void registerTopLevelParameters(LanguageModel& model,
                                Registrar& registrar,
                                const LanguageModelConfig& config) {
    auto& embedding = requireLayer(model.getEmbeddingLayer(), "EmbeddingLayer", "registerTopLevelParameters");
    auto& lm_head = requireLayer(model.getLmHeadLayer(), "LMHeadLayer", "registerTopLevelParameters");
    validateEmbeddingLmHeadAliasing(embedding.tokenWeights(), lm_head.weights(), config);

    if (!config.tie_embeddings) {
        registrar.addTensor("embedding",
                            embedding.tokenWeights(),
                            ParamGroupType::EMBEDDING,
                            ParamStatsBucket::EMBEDDING);

        registrar.addTensor("lm_head_weight",
                            lm_head.weights(),
                            ParamGroupType::LM_HEAD,
                            ParamStatsBucket::LM_HEAD);
    } else {
        registrar.addTensor("embedding_lm_head_tied",
                            lm_head.weights(),
                            ParamGroupType::LM_HEAD,
                            ParamStatsBucket::EMBEDDING);
    }

    registrar.addConfigGatedTensor("lm_head_bias",
                                   lm_head.bias(),
                                   ParamGroupType::LM_HEAD,
                                   ParamStatsBucket::LM_HEAD,
                                   -1,
                                   config.use_bias,
                                   "config.use_bias=false");

    const Tensor& final_gamma = lm_head.finalRmsGamma();
    if (config.freeze_learned_rms_gammas) {
        if (final_gamma.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] final_rms_gamma is frozen by config but still has a grad buffer: " +
                                     tensorDebugSummary(final_gamma));
        }
    } else {
        registrar.addTensor("final_rms_gamma",
                            lm_head.finalRmsGammaMutable_UnfrozenOnly("ParameterGroupRegistration::buildParameterGroups"),
                            ParamGroupType::RMSNORM,
                            ParamStatsBucket::LM_HEAD);
    }
}

void registerEncoderParameters(LanguageModel& model,
                               Registrar& registrar,
                               const LanguageModelConfig& config) {
    GRIM::GPUGrimEncoder& gpu_encoder = model.getGpuEncoder();

    for (int layer = 0; layer < config.num_layers; ++layer) {
        GRIM::GPUEncoderLayer* enc = gpu_encoder.getLayer(layer);
        if (!enc) {
            throw std::runtime_error("[buildParameterGroups] Encoder layer " + std::to_string(layer) +
                                     " is NULL - initGPU() did not build the configured topology");
        }

        const std::string prefix = "layer" + std::to_string(layer);

        registrar.addTensor(prefix + "_qkv_weight",
                    enc->attnWqkv(),
                    ParamGroupType::ATTENTION,
                    ParamStatsBucket::ENCODER,
                    layer);
        registrar.addConfigGatedTensor(prefix + "_qkv_bias",
                           enc->attnBqkv(),
                           ParamGroupType::ATTENTION,
                           ParamStatsBucket::ENCODER,
                           layer,
                           config.use_bias,
                           "config.use_bias=false");
        registrar.addTensor(prefix + "_wo_weight",
                    enc->attnWo(),
                    ParamGroupType::ATTENTION,
                    ParamStatsBucket::ENCODER,
                    layer);
        registrar.addConfigGatedTensor(prefix + "_wo_bias",
                           enc->attnBo(),
                           ParamGroupType::ATTENTION,
                           ParamStatsBucket::ENCODER,
                           layer,
                           config.use_bias,
                           "config.use_bias=false");

        registrar.addTensor(prefix + "_ffn_w_gate",
                    enc->ffnWGate(),
                    ParamGroupType::FFN,
                    ParamStatsBucket::ENCODER,
                    layer);
        registrar.addTensor(prefix + "_ffn_w1",
                    enc->ffnW1(),
                    ParamGroupType::FFN,
                    ParamStatsBucket::ENCODER,
                    layer);
        registrar.addTensor(prefix + "_ffn_w2",
                    enc->ffnW2(),
                    ParamGroupType::FFN,
                    ParamStatsBucket::ENCODER,
                    layer);
        registrar.addConfigGatedTensor(prefix + "_ffn_b2",
                           enc->ffnB2(),
                           ParamGroupType::FFN,
                           ParamStatsBucket::ENCODER,
                           layer,
                           config.use_bias,
                           "config.use_bias=false");

        if (config.freeze_learned_rms_gammas) {
            if (enc->rms1Gamma().has_grad()) {
                throw std::runtime_error("[buildParameterGroups] " + prefix +
                                         "_rms1_gamma is frozen by config but still has a grad buffer: " +
                                         tensorDebugSummary(enc->rms1Gamma()));
            }
            if (enc->rms2Gamma().has_grad()) {
                throw std::runtime_error("[buildParameterGroups] " + prefix +
                                         "_rms2_gamma is frozen by config but still has a grad buffer: " +
                                         tensorDebugSummary(enc->rms2Gamma()));
            }
        } else {
            registrar.addTensor(prefix + "_rms1_gamma",
                                enc->rms1Gamma(),
                                ParamGroupType::RMSNORM,
                                ParamStatsBucket::ENCODER,
                                layer);
            registrar.addTensor(prefix + "_rms2_gamma",
                                enc->rms2Gamma(),
                                ParamGroupType::RMSNORM,
                                ParamStatsBucket::ENCODER,
                                layer);
        }

        registrar.addConfigGatedTensor(prefix + "_layer_scale1",
                                       enc->layerScale1(),
                                       ParamGroupType::RMSNORM,
                                       ParamStatsBucket::ENCODER,
                                       layer,
                                       config.use_layer_scale,
                                       "config.use_layer_scale=false");
        registrar.addConfigGatedTensor(prefix + "_layer_scale2",
                                       enc->layerScale2(),
                                       ParamGroupType::RMSNORM,
                                       ParamStatsBucket::ENCODER,
                                       layer,
                                       config.use_layer_scale,
                                       "config.use_layer_scale=false");
    }
}

void registerScratchBlockParameters(LanguageModel& model,
                                    Registrar& registrar,
                                    const LanguageModelConfig& config) {
    auto* scratch_block = model.getScratchBlockLayer();
    const auto scratch_hp = scratchBlockConstructionHP(config);

    if (!scratch_hp.enabled) {
        if (scratch_block) {
            throw std::runtime_error("[buildParameterGroups] ScratchBlock layer exists while ScratchBlockConstructionHP.enabled=false");
        }
        return;
    }

    if (!scratch_block) {
        throw std::runtime_error("[buildParameterGroups] ScratchBlockConstructionHP.enabled=true but ScratchBlock layer is NULL");
    }

    auto& atom_type_embeddings = scratch_block->atomTypeEmbeddings();
    registrar.addTensor("scratch_block_atom_type_embeddings",
                        atom_type_embeddings,
                        ParamGroupType::SCRATCHBLOCK,
                        ParamStatsBucket::ENCODER);

    auto& atom_projection = scratch_block->atomProjection();
    registrar.addTensor("scratch_block_atom_projection",
                        atom_projection,
                        ParamGroupType::SCRATCHBLOCK,
                        ParamStatsBucket::ENCODER);

    auto& structured_gate_weight = scratch_block->structuredGateWeight();
    registrar.addTensor("scratch_block_structured_gate_weight",
                        structured_gate_weight,
                        ParamGroupType::SCRATCHBLOCK,
                        ParamStatsBucket::ENCODER);
}

void registerExecutionBlockParameters(LanguageModel& model,
                                      Registrar& registrar,
                                      const LanguageModelConfig& config) {
    auto* execution_block = model.getExecutionBlockLayer();
    auto* slot_selector = model.getDecodeTimeSlotSelectorLayer();

    if (!config.execution_block_enabled) {
        if (execution_block) {
            throw std::runtime_error("[buildParameterGroups] ExecutionBlock layer exists while config.execution_block_enabled=false");
        }
        if (config.selector_enabled || slot_selector) {
            throw std::runtime_error("[buildParameterGroups] Slot selector requires config.execution_block_enabled=true");
        }
        return;
    }

    auto& block = requireLayer(execution_block, "ExecutionBlockLayer", "registerExecutionBlockParameters");
    registrar.addTensor("exec_block_w_decode_1", block.w_decode_1(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_b_decode_1", block.b_decode_1(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_w_decode_2", block.w_decode_2(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_w_arg1_select", block.w_arg1_select(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_w_arg2_select", block.w_arg2_select(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_op_select", block.W_op_select(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_key_proj", block.W_key_proj(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_write_query", block.W_write_query(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_write_key", block.W_write_key(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_alpha", block.alpha(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_beta", block.beta(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_step_embeddings", block.step_embeddings(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_type_num_embed", block.type_num_embed(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_value_to_emb", block.W_value_to_emb(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_b_value_to_emb", block.b_value_to_emb(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_w_inject_gate", block.w_inject_gate(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_Q_read", block.W_Q_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_K_read", block.W_K_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_V_read", block.W_V_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_O_read", block.W_O_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_gate_read", block.W_gate_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_tau", block.tau(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_E_slot", block.E_slot(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_E_op", block.E_op(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_scal", block.W_scal(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_b_scal", block.b_scal(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_trace", block.W_trace(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_b_trace", block.b_trace(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_reason_gate", block.W_reason_gate(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_trace_gate", block.W_trace_gate(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);

    if (!config.selector_enabled) {
        if (slot_selector) {
            throw std::runtime_error("[buildParameterGroups] DecodeTimeSlotSelector exists while config.selector_enabled=false");
        }
        return;
    }

    auto& selector = requireLayer(slot_selector, "DecodeTimeSlotSelectorLayer", "registerExecutionBlockParameters");
    registrar.addTensor("selector_W_q_select", selector.W_q_select(), ParamGroupType::SLOT_SELECTOR, ParamStatsBucket::ENCODER);
    registrar.addTensor("selector_W_k_select", selector.W_k_select(), ParamGroupType::SLOT_SELECTOR, ParamStatsBucket::ENCODER);
    registrar.addTensor("selector_null_key_select", selector.null_key_select(), ParamGroupType::SLOT_SELECTOR, ParamStatsBucket::ENCODER);
    registrar.addTensor("selector_null_logit_bias", selector.null_logit_bias(), ParamGroupType::SLOT_SELECTOR, ParamStatsBucket::ENCODER);
}

void registerMtpParameters(LanguageModel& model,
                           Registrar& registrar,
                           const LanguageModelConfig& config) {
    if (!config.mtp_enabled) {
        if (model.getMtpHead(0)) {
            throw std::runtime_error("[buildParameterGroups] MTP heads exist while config.mtp_enabled=false");
        }
        return;
    }

    if (config.mtp_k <= 0) {
        throw std::runtime_error("[buildParameterGroups] config.mtp_enabled=true but config.mtp_k <= 0");
    }

    for (int k = 0; k < config.mtp_k; ++k) {
        auto* head = model.getMtpHead(k);
        if (!head) {
            throw std::runtime_error("[buildParameterGroups] Missing MTP head " + std::to_string(k) +
                                     " for configured mtp_k=" + std::to_string(config.mtp_k));
        }
        registrar.addTensor("mtp_head_" + std::to_string(k) + "_weight",
                            head->weight,
                    ParamGroupType::MTP,
                    ParamStatsBucket::ENCODER);
        registrar.addTensor("mtp_head_" + std::to_string(k) + "_bias",
                    head->bias,
                    ParamGroupType::MTP,
                    ParamStatsBucket::ENCODER);
    }

    if (model.getMtpHead(config.mtp_k)) {
        throw std::runtime_error("[buildParameterGroups] MTP head vector contains more entries than config.mtp_k");
    }
}

void clearOptimizerBindings(std::vector<ParameterGroup>& groups) {
    for (auto& group : groups) {
        group.m_tensor = nullptr;
        group.v_tensor = nullptr;
    }
}

void validateOptimizerStateAllocation(const std::vector<ParameterGroup>& groups,
                                      const GRIM::OptimizerState& optimizer_state) {
    if (optimizer_state.m_states.size() != groups.size()) {
        throw std::runtime_error("[bindOptimizerState] optimizer m_states size mismatch: states=" +
                                 std::to_string(optimizer_state.m_states.size()) +
                                 " groups=" + std::to_string(groups.size()));
    }
    if (optimizer_state.v_states.size() != groups.size()) {
        throw std::runtime_error("[bindOptimizerState] optimizer v_states size mismatch: states=" +
                                 std::to_string(optimizer_state.v_states.size()) +
                                 " groups=" + std::to_string(groups.size()));
    }

    for (size_t i = 0; i < groups.size(); ++i) {
        const size_t group_size = groups[i].size();
        const size_t m_size = optimizer_state.m_states[i].numel();
        const size_t v_size = optimizer_state.v_states[i].numel();
        if (m_size != group_size) {
            throw std::runtime_error("[bindOptimizerState] optimizer m_state size mismatch for group " +
                                     std::to_string(i) + " (" + groups[i].name + "): state=" +
                                     std::to_string(m_size) + " group=" + std::to_string(group_size));
        }
        if (v_size != group_size) {
            throw std::runtime_error("[bindOptimizerState] optimizer v_state size mismatch for group " +
                                     std::to_string(i) + " (" + groups[i].name + "): state=" +
                                     std::to_string(v_size) + " group=" + std::to_string(group_size));
        }
    }
}

void validateRegisteredTensorPrecisionMetadata(const std::vector<ParameterGroup>& groups) {
    for (const auto& group : groups) {
        if (!group.tensor) {
            throw std::runtime_error("[buildParameterGroups] group " + group.name +
                                     " has NULL tensor while validating precision metadata");
        }
        validateRegisteredPrecisionSupport(group.name, group.type, group.parameter_precision);
        const TensorContract::PrecisionType expected =
            TensorContract::precision_from_parameter_group_precision(group.parameter_precision);
        const TensorContract::PrecisionType actual = group.tensor->precision();
        if (actual != expected) {
            throw std::runtime_error("[buildParameterGroups] group " + group.name +
                                     " parameter_precision=" +
                                     parameterPrecisionSummaryName(group.parameter_precision) +
                                     " but TensorContract tensor metadata is " +
                                     TensorContract::precision_name(actual));
        }
    }
}

void emitGroupSummary(const std::vector<ParameterGroup>& groups) {
    constexpr size_t kParamGroupTypeCount = static_cast<size_t>(ParamGroupType::COUNT);
    constexpr size_t kPrecisionCount = 2;
    static_assert(kParamGroupTypeCount == 10,
                  "Registered group precision summary must list every ParamGroupType");

    const std::array<ParamGroupType, kParamGroupTypeCount> group_types = {
        ParamGroupType::EMBEDDING,
        ParamGroupType::LM_HEAD,
        ParamGroupType::ATTENTION,
        ParamGroupType::FFN,
        ParamGroupType::RMSNORM,
        ParamGroupType::SCRATCHBLOCK,
        ParamGroupType::MTP,
        ParamGroupType::REASONING_HEAD,
        ParamGroupType::EXECUTION_BLOCK,
        ParamGroupType::SLOT_SELECTOR
    };
    const std::array<ParameterGroupPrecision, kPrecisionCount> precisions = {
        ParameterGroupPrecision::FP32,
        ParameterGroupPrecision::BF16_COMPUTE
    };
    std::array<std::array<int, kPrecisionCount>, kParamGroupTypeCount> precision_counts{};

    int emb_count = 0;
    int attn_count = 0;
    int ffn_count = 0;
    int rms_count = 0;
    int other_count = 0;

    for (const auto& group : groups) {
        switch (group.type) {
            case ParamGroupType::EMBEDDING: ++emb_count; break;
            case ParamGroupType::LM_HEAD: ++other_count; break;
            case ParamGroupType::ATTENTION: ++attn_count; break;
            case ParamGroupType::FFN: ++ffn_count; break;
            case ParamGroupType::RMSNORM: ++rms_count; break;
            case ParamGroupType::SCRATCHBLOCK: ++other_count; break;
            case ParamGroupType::MTP: ++other_count; break;
            case ParamGroupType::REASONING_HEAD: ++other_count; break;
            case ParamGroupType::EXECUTION_BLOCK: ++other_count; break;
            case ParamGroupType::SLOT_SELECTOR: ++other_count; break;
            case ParamGroupType::COUNT:
                throw std::runtime_error("[buildParameterGroups] group " + group.name +
                                         " has invalid ParamGroupType::COUNT");
        }

        if (group.parameter_precision == ParameterGroupPrecision::UNSPECIFIED) {
            throw std::runtime_error("[buildParameterGroups] group " + group.name +
                                     " has UNSPECIFIED parameter_precision");
        }

        ++precision_counts[paramGroupTypeIndex(group.type)][parameterPrecisionIndex(group.parameter_precision)];
    }

    emitInfo("[buildParameterGroups] Parameter group summary: total=" + std::to_string(groups.size()) +
             " emb=" + std::to_string(emb_count) +
             " attn=" + std::to_string(attn_count) +
             " ffn=" + std::to_string(ffn_count) +
             " rms=" + std::to_string(rms_count) +
             " other=" + std::to_string(other_count));

    std::ostringstream precision_summary;
    precision_summary << "[buildParameterGroups] Registered group precision summary:";
    for (const ParamGroupType type : group_types) {
        const size_t type_index = paramGroupTypeIndex(type);
        int type_total = 0;
        for (const ParameterGroupPrecision precision : precisions) {
            type_total += precision_counts[type_index][parameterPrecisionIndex(precision)];
        }
        if (type_total == 0) {
            continue;
        }

        precision_summary << ' ' << paramGroupTypeSummaryName(type) << "{";
        for (size_t precision_index = 0; precision_index < precisions.size(); ++precision_index) {
            if (precision_index != 0) {
                precision_summary << ',';
            }
            const ParameterGroupPrecision precision = precisions[precision_index];
            precision_summary << parameterPrecisionSummaryName(precision) << '='
                              << precision_counts[type_index][precision_index];
        }
        precision_summary << '}';
    }
    emitInfo(precision_summary.str());
}

void validateParameterRegistrationConfig(const LanguageModelConfig& config) {
    GRIM::HyperParameters::validateRootConfigDocument(
        config, "buildParameterGroups");
    if (config.num_layers <= 0) {
        throw std::runtime_error("buildParameterGroups: num_layers must be > 0, got " +
                                 std::to_string(config.num_layers));
    }
    if (config.vocab_size <= 0) {
        throw std::runtime_error("buildParameterGroups: vocab_size must be > 0, got " +
                                 std::to_string(config.vocab_size));
    }
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_embedding, "parameter_precision_embedding", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_lm_head, "parameter_precision_lm_head", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_attention, "parameter_precision_attention", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_ffn, "parameter_precision_ffn", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_rmsnorm, "parameter_precision_rmsnorm", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_scratchblock, "parameter_precision_scratchblock", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_mtp, "parameter_precision_mtp", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_reasoning_head, "parameter_precision_reasoning_head", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_execution_block, "parameter_precision_execution_block", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(config.parameter_precision_slot_selector, "parameter_precision_slot_selector", "buildParameterGroups");
}

} // namespace

void buildParameterGroups(LanguageModel& model) {
    const LanguageModelConfig& config = model.getConfig();
    validateParameterRegistrationConfig(config);

    auto& model_groups = model.parameterGroups();
    std::vector<ParameterGroup> rebuilt_groups;
    rebuilt_groups.reserve(model_groups.size());

    Registrar registrar(rebuilt_groups, config);
    registerTopLevelParameters(model, registrar, config);
    registerEncoderParameters(model, registrar, config);

    registerScratchBlockParameters(model, registrar, config);
    registerExecutionBlockParameters(model, registrar, config);
    registerMtpParameters(model, registrar, config);

    validateRegisteredTensorPrecisionMetadata(rebuilt_groups);
    clearOptimizerBindings(rebuilt_groups);

    // Transaction boundary: model.parameterGroups() is replaced only after the
    // complete configured inventory has been discovered and validated. A thrown
    // registration check must never leave LanguageModel with a half-built group
    // vector that downstream optimizer/checkpoint code could observe.
    model_groups.swap(rebuilt_groups);

    emitInfo("[buildParameterGroups] Built " + std::to_string(model_groups.size()) + " parameter groups");
    emitGroupSummary(model_groups);
}

void bindOptimizerState(LanguageModel& model,
                        GRIM::OptimizerState& optimizer_state,
                        cudaStream_t stream) {
    auto& groups = model.parameterGroups();
    if (groups.empty()) {
        throw std::runtime_error("[bindOptimizerState] parameter groups are empty - caller MUST run startup parameter registration first");
    }
    if (stream == nullptr) {
        throw std::runtime_error("[bindOptimizerState] stream is NULL - caller MUST provide valid CUDA stream");
    }

    std::vector<size_t> sizes;
    sizes.reserve(groups.size());
    for (const auto& group : groups) {
        sizes.push_back(group.size());
    }

    optimizer_state.allocate(sizes, stream);
    validateOptimizerStateAllocation(groups, optimizer_state);

    emitInfo("[bindOptimizerState] Allocated optimizer state tensors for " +
             std::to_string(groups.size()) + " parameter groups");
    for (size_t i = 0; i < groups.size(); ++i) {
        groups[i].m_tensor = &optimizer_state.m_states[i];
        groups[i].v_tensor = &optimizer_state.v_states[i];
    }
    emitInfo("[bindOptimizerState] Bound optimizer state tensors to parameter groups");
}

#endif // USE_CUDA

} // namespace GRIMText::Training::Startup::ModelRegistration


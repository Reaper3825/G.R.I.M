#include "ParameterGroupRegistration.hpp"
#include "ParameterRegistry.hpp"

#include "ModelGpuState.hpp"

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
using GRIM::HyperParameters::OptimizerUpdateHP;
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
        case ParamGroupType::EXECUTION_BLOCK: return 7;
        case ParamGroupType::SLOT_SELECTOR:   return 8;
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
              const GRIM::Config::AiConfigSnapshot& config)
    : groups_(groups), config_(config), optimizer_hp_(GRIM::HyperParameters::optimizerUpdateHP(config)) {}

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
        if (optimizer_hp_.use_depth_aware_upsilon && layer >= 0) {
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
            case ParamGroupType::EMBEDDING:       return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_embedding");
            case ParamGroupType::LM_HEAD:         return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_lm_head");
            case ParamGroupType::ATTENTION:       return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_attention");
            case ParamGroupType::FFN:             return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_ffn");
            case ParamGroupType::RMSNORM:         return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_rmsnorm");
            case ParamGroupType::SCRATCHBLOCK:    return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_scratchblock");
            case ParamGroupType::MTP:             return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_mtp");
            case ParamGroupType::EXECUTION_BLOCK: return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_execution_block");
            case ParamGroupType::SLOT_SELECTOR:   return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_slot_selector");
            case ParamGroupType::COUNT: break;
        }
        throw std::runtime_error("[buildParameterGroups] invalid ParamGroupType::COUNT for parameter precision lookup");
    }

    std::vector<ParameterGroup>& groups_;
    const GRIM::Config::AiConfigSnapshot& config_;
    const OptimizerUpdateHP optimizer_hp_;
    std::vector<const void*> registered_data_;
};

void validateEmbeddingLmHeadAliasing(const Tensor& embedding_weights,
                                     const Tensor& lm_head_weights,
                                     const GRIM::Config::AiConfigSnapshot& config) {
    if (!embedding_weights.data || !lm_head_weights.data) {
        throw std::runtime_error("[buildParameterGroups] cannot validate embedding/LM-head aliasing with NULL data: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }

    const bool tied = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "tie_embeddings");
    const bool same_data = embedding_weights.data == lm_head_weights.data;
    if (tied && !same_data) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=true but embedding and LM-head data pointers differ: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }
    if (!tied && same_data) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=false but embedding and LM-head data pointers are identical: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }
    const auto embedding_precision = GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_embedding");
    const auto lm_head_precision = GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_lm_head");
    if (tied && embedding_precision != lm_head_precision) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=true requires parameter_precision_embedding and parameter_precision_lm_head to match; embedding=" +
                                 std::string(parameterPrecisionSummaryName(embedding_precision)) +
                                 " lm_head=" +
                                 std::string(parameterPrecisionSummaryName(lm_head_precision)));
    }
}

void registerTopLevelParameters(LanguageModel& model,
                                ParameterRegistry::StartupParameterRegistry& parameter_registry,
                                Registrar& registrar,
                                const GRIM::Config::AiConfigSnapshot& config) {
    auto& embedding = requireLayer(model.getEmbeddingLayer(), "EmbeddingLayer", "registerTopLevelParameters");
    requireLayer(model.getLmHeadLayer(), "LMHeadLayer", "registerTopLevelParameters");
    auto& lm_head_parameters = parameter_registry.requireLmHeadParameters("registerTopLevelParameters");
    validateEmbeddingLmHeadAliasing(embedding.tokenWeights(), lm_head_parameters.weights, config);

    const bool tie_embeddings = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "tie_embeddings");
    const bool use_bias = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_bias");
    const bool freeze_learned_rms_gammas = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "freeze_learned_rms_gammas");

    if (!tie_embeddings) {
        registrar.addTensor("embedding",
                            embedding.tokenWeights(),
                            ParamGroupType::EMBEDDING,
                            ParamStatsBucket::EMBEDDING);

        registrar.addTensor("lm_head_weight",
                            lm_head_parameters.weights,
                            ParamGroupType::LM_HEAD,
                            ParamStatsBucket::LM_HEAD);
    } else {
        registrar.addTensor("embedding_lm_head_tied",
                            lm_head_parameters.weights,
                            ParamGroupType::LM_HEAD,
                            ParamStatsBucket::EMBEDDING);
    }

    registrar.addConfigGatedTensor("lm_head_bias",
                                   lm_head_parameters.bias,
                                   ParamGroupType::LM_HEAD,
                                   ParamStatsBucket::LM_HEAD,
                                   -1,
                                   use_bias,
                                   "config.use_bias=false");

    const Tensor& final_gamma = lm_head_parameters.final_rms_gamma;
    if (freeze_learned_rms_gammas) {
        if (final_gamma.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] final_rms_gamma is frozen by config but still has a grad buffer: " +
                                     tensorDebugSummary(final_gamma));
        }
    } else {
        registrar.addTensor("final_rms_gamma",
                            lm_head_parameters.final_rms_gamma,
                            ParamGroupType::RMSNORM,
                            ParamStatsBucket::LM_HEAD);
    }
}

void registerEncoderParameters(Startup::GpuModelState& gpu_model_state,
                               ParameterRegistry::StartupParameterRegistry& parameter_registry,
                               Registrar& registrar,
                               const GRIM::Config::AiConfigSnapshot& config) {
    auto* gpu_encoder = gpu_model_state.gpu_encoder.get();
    if (!gpu_encoder) {
        throw std::runtime_error(
            "[buildParameterGroups] gpu_model_state.gpu_encoder is NULL - "
            "Startup::assembleGpuModel(config, training_state, gpu_model_state, parameter_registry, weight_init_seed) must complete before parameter registration");
    }
    const int num_layers = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "num_layers");
    const bool use_bias = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_bias");
    const bool freeze_learned_rms_gammas = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "freeze_learned_rms_gammas");
    const bool use_layer_scale = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_layer_scale");
    if (static_cast<int>(parameter_registry.feedForwardParameterTensors().size()) != num_layers) {
        throw std::runtime_error("[buildParameterGroups] feed_forward_parameter_tensors size must equal config.num_layers. size=" +
                                 std::to_string(parameter_registry.feedForwardParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }

    for (int layer = 0; layer < num_layers; ++layer) {
        GRIM::EncodingLayer* enc = gpu_encoder->getLayer(layer);
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
                           use_bias,
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
                           use_bias,
                           "config.use_bias=false");

        auto& ffn_parameters = parameter_registry.requireFeedForwardParameters(layer, "registerEncoderParameters");
        ParameterRegistry::registerFeedForwardParameters(ffn_parameters, layer, use_bias, registrar);

        if (freeze_learned_rms_gammas) {
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
                                       use_layer_scale,
                                       "config.use_layer_scale=false");
        registrar.addConfigGatedTensor(prefix + "_layer_scale2",
                                       enc->layerScale2(),
                                       ParamGroupType::RMSNORM,
                                       ParamStatsBucket::ENCODER,
                                       layer,
                                                    use_layer_scale,
                                       "config.use_layer_scale=false");
    }
}

void registerScratchBlockParameters(LanguageModel& model,
                                    Registrar& registrar,
                                                const GRIM::Config::AiConfigSnapshot& config) {
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
                                      ParameterRegistry::StartupParameterRegistry& parameter_registry,
                                      Registrar& registrar,
                                      const GRIM::Config::AiConfigSnapshot& config) {
    auto* execution_block = model.getExecutionBlockLayer();
    auto* execution_block_parameters = parameter_registry.getExecutionBlockParameters();
    auto* slot_selector = parameter_registry.getDecodeTimeSlotSelector();
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(config);
    const auto selector_hp = GRIM::HyperParameters::decodeTimeSelectorConstructionHP(config);

    if (!execution_hp.enabled) {
        if (execution_block) {
            throw std::runtime_error("[buildParameterGroups] ExecutionBlock layer exists while config.execution_block_enabled=false");
        }
        if (execution_block_parameters) {
            throw std::runtime_error("[buildParameterGroups] ExecutionBlock parameter owner exists while config.execution_block_enabled=false");
        }
        if (selector_hp.enabled || slot_selector) {
            throw std::runtime_error("[buildParameterGroups] Slot selector requires config.execution_block_enabled=true");
        }
        return;
    }

    requireLayer(execution_block, "ExecutionBlockLayer", "registerExecutionBlockParameters");
    auto& execution_block_tensor_owner = requireLayer(
        execution_block_parameters,
        "ExecutionBlockParameterTensors",
        "registerExecutionBlockParameters");
    ParameterRegistry::registerExecutionBlockParameters(execution_block_tensor_owner, registrar);

    if (!selector_hp.enabled) {
        if (slot_selector) {
            throw std::runtime_error("[buildParameterGroups] DecodeTimeSlotSelector exists while config.selector_enabled=false");
        }
        return;
    }

    auto& selector = requireLayer(slot_selector, "DecodeTimeSlotSelector", "registerExecutionBlockParameters");
    ParameterRegistry::registerDecodeTimeSlotSelectorParameters(selector, registrar);
}

void registerMtpParameters(ParameterRegistry::StartupParameterRegistry& parameter_registry,
                           Registrar& registrar,
                           const GRIM::Config::AiConfigSnapshot& config) {
    const auto mtp_hp = GRIM::HyperParameters::mtpFeatureHP(config);
    auto& mtp_heads = parameter_registry.mtpHeadParameterTensors();
    if (!mtp_hp.enabled) {
        if (!mtp_heads.empty()) {
            throw std::runtime_error("[buildParameterGroups] MTP heads exist while config.mtp_enabled=false");
        }
        return;
    }

    if (mtp_hp.k <= 0) {
        throw std::runtime_error("[buildParameterGroups] config.mtp_enabled=true but config.mtp_k <= 0");
    }

    for (int k = 0; k < mtp_hp.k; ++k) {
        if (k < 0 || k >= static_cast<int>(mtp_heads.size())) {
            throw std::runtime_error("[buildParameterGroups] Missing MTP head " + std::to_string(k) +
                                     " for configured mtp_k=" + std::to_string(mtp_hp.k));
        }
        auto& mtp_head_parameters = mtp_heads[static_cast<std::size_t>(k)];
        ParameterRegistry::registerMtpHeadParameters(mtp_head_parameters, k, registrar);
    }

    if (static_cast<int>(mtp_heads.size()) > mtp_hp.k) {
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
    static_assert(kParamGroupTypeCount == 9,
                  "Registered group precision summary must list every ParamGroupType");

    const std::array<ParamGroupType, kParamGroupTypeCount> group_types = {
        ParamGroupType::EMBEDDING,
        ParamGroupType::LM_HEAD,
        ParamGroupType::ATTENTION,
        ParamGroupType::FFN,
        ParamGroupType::RMSNORM,
        ParamGroupType::SCRATCHBLOCK,
        ParamGroupType::MTP,
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

void validateParameterRegistrationConfig(const GRIM::Config::AiConfigSnapshot& config) {
    const int num_layers = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "num_layers");
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "vocab_size");
    if (num_layers <= 0) {
        throw std::runtime_error("buildParameterGroups: num_layers must be > 0, got " +
                                 std::to_string(num_layers));
    }
    if (vocab_size <= 0) {
        throw std::runtime_error("buildParameterGroups: vocab_size must be > 0, got " +
                                 std::to_string(vocab_size));
    }
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_embedding"), "parameter_precision_embedding", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_lm_head"), "parameter_precision_lm_head", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_attention"), "parameter_precision_attention", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_ffn"), "parameter_precision_ffn", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_rmsnorm"), "parameter_precision_rmsnorm", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_scratchblock"), "parameter_precision_scratchblock", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_mtp"), "parameter_precision_mtp", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_execution_block"), "parameter_precision_execution_block", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_slot_selector"), "parameter_precision_slot_selector", "buildParameterGroups");
}

} // namespace

void initializeFeedForwardParameterTensors(
    std::vector<GRIM::FeedForwardParameterTensors>& feed_forward_parameter_tensors,
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!init_stream) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: init_stream is NULL");
    }
    if (encoder_hp.num_layers <= 0) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: encoder_hp.num_layers must be > 0, got " +
                                 std::to_string(encoder_hp.num_layers));
    }
    if (encoder_hp.d_model <= 0) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: encoder_hp.d_model must be > 0, got " +
                                 std::to_string(encoder_hp.d_model));
    }
    if (encoder_hp.d_ff <= 0) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: encoder_hp.d_ff must be > 0, got " +
                                 std::to_string(encoder_hp.d_ff));
    }
    if (!feed_forward_parameter_tensors.empty()) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: registry FFN tensor vector is already initialized; refusing to allocate twice. size=" +
                                 std::to_string(feed_forward_parameter_tensors.size()));
    }

    const auto ffn_hp = GRIM::HyperParameters::feedForwardLayerConstructionHP(encoder_hp);
    const float residual_projection_init_gain = ffn_hp.residual_projection_init_gain;
    if (!std::isfinite(residual_projection_init_gain) || residual_projection_init_gain <= 0.0f) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: residual_projection_init_gain must be positive finite");
    }
    if (!std::isfinite(ffn_hp.dropout_rate) || ffn_hp.dropout_rate < 0.0f || ffn_hp.dropout_rate >= 1.0f) {
        throw std::runtime_error("initializeFeedForwardParameterTensors: dropout_rate must be finite and in [0,1), got " +
                                 std::to_string(ffn_hp.dropout_rate));
    }

    feed_forward_parameter_tensors.resize(static_cast<std::size_t>(encoder_hp.num_layers));
    for (int layer = 0; layer < encoder_hp.num_layers; ++layer) {
        auto& tensors = feed_forward_parameter_tensors[static_cast<std::size_t>(layer)];
        const std::uint64_t ffn_seed = weight_init_seed + 4 + static_cast<std::uint64_t>(layer) * 10ULL;

        tensors.W_gate = GRIM::Tensor::zeros({ffn_hp.d_model, ffn_hp.d_ff}, init_stream, "ffn_w_gate");
        tensors.W_gate.requires_grad_();
        tensors.W_gate.ensure_grad();
        GRIM::Tensor::xavier_uniform_(tensors.W_gate, ffn_seed, init_stream);

        tensors.W1 = GRIM::Tensor::zeros({ffn_hp.d_model, ffn_hp.d_ff}, init_stream, "ffn_w1");
        tensors.W1.requires_grad_();
        tensors.W1.ensure_grad();
        GRIM::Tensor::xavier_uniform_(tensors.W1, ffn_seed + 1, init_stream);

        tensors.W2 = GRIM::Tensor::zeros({ffn_hp.d_ff, ffn_hp.d_model}, init_stream, "ffn_w2");
        tensors.W2.requires_grad_();
        tensors.W2.ensure_grad();
        GRIM::Tensor::xavier_uniform_with_gain_(tensors.W2, ffn_seed + 2, residual_projection_init_gain, init_stream);

        if (ffn_hp.use_bias) {
            tensors.b2 = GRIM::Tensor::zeros({1, ffn_hp.d_model}, init_stream, "ffn_b2");
            tensors.b2.requires_grad_();
            tensors.b2.ensure_grad();
        }
    }

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(std::string("initializeFeedForwardParameterTensors: cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(sync_err));
    }

    emitInfo("[initializeFeedForwardParameterTensors] Initialized registry-owned FFN tensors for " +
             std::to_string(encoder_hp.num_layers) + " layers");
}

void buildParameterGroups(LanguageModel& model,
                          Startup::GpuModelState& gpu_model_state,
                          ParameterRegistry::StartupParameterRegistry& parameter_registry) {
    const GRIM::Config::AiConfigSnapshot& config = model.getConfig();
    validateParameterRegistrationConfig(config);

    auto& model_groups = model.parameterGroups();
    std::vector<ParameterGroup> rebuilt_groups;
    rebuilt_groups.reserve(model_groups.size());

    Registrar registrar(rebuilt_groups, config);
    registerTopLevelParameters(model, parameter_registry, registrar, config);
    registerEncoderParameters(gpu_model_state, parameter_registry, registrar, config);

    registerScratchBlockParameters(model, registrar, config);
    registerExecutionBlockParameters(model, parameter_registry, registrar, config);
    registerMtpParameters(parameter_registry, registrar, config);

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


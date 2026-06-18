#include "ParameterGroupRegistration.hpp"
#include "ParameterRegistry.hpp"

#include "ModelGpuState.hpp"

#include "../../../../GRIM/grim_language_model_cuda.hpp"
#include "../../../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../../../Shared/Batching/BatchPayload.hpp"
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
#include <utility>
#include <vector>

namespace GRIMText::Training::Startup::ModelRegistration {

#ifdef USE_CUDA

namespace {

constexpr auto kRegistrationModule = GRIM::Logging::ModuleId::Training;

using GRIM::ParameterGroup;
using GRIM::ParamStatsBucket;
using GRIM::ParamGroupType;
using GRIM::Tensor;
using GRIM::HyperParameters::ExecutionBlockConstructionHP;
using GRIM::HyperParameters::ParameterGroupPrecision;
using GRIM::HyperParameters::OptimizerUpdateHP;

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
        case ParamGroupType::MTP:             return 5;
        case ParamGroupType::EXECUTION_BLOCK: return 6;
        case ParamGroupType::NUMBER_ENCODER:  return 7;
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
        case ParamGroupType::MTP:             return "mtp";
        case ParamGroupType::EXECUTION_BLOCK: return "execution_block";
        case ParamGroupType::NUMBER_ENCODER:  return "number_encoder";
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
            case ParamGroupType::MTP:             return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_mtp");
            case ParamGroupType::EXECUTION_BLOCK: return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_execution_block");
            case ParamGroupType::NUMBER_ENCODER:  return GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config_, "parameter_precision_number_encoder");
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

void registerTopLevelParameters(Startup::GpuModelState& gpu_model_state,
                                ParameterRegistry::StartupParameterRegistry& parameter_registry,
                                Registrar& registrar,
                                const GRIM::Config::AiConfigSnapshot& config) {
    (void)gpu_model_state;
    auto& embedding_parameters = parameter_registry.requireEmbeddingParameters("registerTopLevelParameters");
    auto& lm_head_parameters = parameter_registry.requireLmHeadParameters("registerTopLevelParameters");
    validateEmbeddingLmHeadAliasing(embedding_parameters.token_weights, lm_head_parameters.weights, config);

    const bool tie_embeddings = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "tie_embeddings");
    const bool use_bias = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_bias");
    const bool freeze_learned_rms_gammas = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "freeze_learned_rms_gammas");

    if (!tie_embeddings) {
        ParameterRegistry::registerEmbeddingParameters(embedding_parameters, registrar);

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
    if (static_cast<int>(parameter_registry.encodingLayerParameterTensors().size()) != num_layers) {
        throw std::runtime_error("[buildParameterGroups] encoding_layer_parameter_tensors size must equal config.num_layers. size=" +
                                 std::to_string(parameter_registry.encodingLayerParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }
    if (static_cast<int>(parameter_registry.feedForwardParameterTensors().size()) != num_layers) {
        throw std::runtime_error("[buildParameterGroups] feed_forward_parameter_tensors size must equal config.num_layers. size=" +
                                 std::to_string(parameter_registry.feedForwardParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }

    for (int layer = 0; layer < num_layers; ++layer) {
        if (!gpu_encoder->getLayer(layer)) {
            throw std::runtime_error("[buildParameterGroups] Encoder layer " + std::to_string(layer) +
                                     " is NULL - initGPU() did not build the configured topology");
        }

        const std::string prefix = "layer" + std::to_string(layer);
        auto& encoding_parameters = parameter_registry.requireEncodingLayerParameters(layer, "registerEncoderParameters");
        ParameterRegistry::registerEncodingLayerParameters(
            encoding_parameters,
            layer,
            use_bias,
            freeze_learned_rms_gammas,
            use_layer_scale,
            registrar);

        auto& ffn_parameters = parameter_registry.requireFeedForwardParameters(layer, "registerEncoderParameters");
        ParameterRegistry::registerFeedForwardParameters(ffn_parameters, layer, use_bias, registrar);

        if (freeze_learned_rms_gammas) {
            if (encoding_parameters.rms1_gamma.has_grad()) {
                throw std::runtime_error("[buildParameterGroups] " + prefix +
                                         "_rms1_gamma is frozen by config but still has a grad buffer: " +
                                         tensorDebugSummary(encoding_parameters.rms1_gamma));
            }
            if (encoding_parameters.rms2_gamma.has_grad()) {
                throw std::runtime_error("[buildParameterGroups] " + prefix +
                                         "_rms2_gamma is frozen by config but still has a grad buffer: " +
                                         tensorDebugSummary(encoding_parameters.rms2_gamma));
            }
        }
    }
}

void registerExecutionBlockParameters(Startup::GpuModelState& gpu_model_state,
                                      ParameterRegistry::StartupParameterRegistry& parameter_registry,
                                      Registrar& registrar,
                                      const GRIM::Config::AiConfigSnapshot& config) {
    (void)gpu_model_state;
    auto* execution_block_parameters = parameter_registry.getExecutionBlockParameters();
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(config);

    if (!execution_hp.enabled) {
        if (execution_block_parameters) {
            throw std::runtime_error("[buildParameterGroups] ExecutionBlock parameter owner exists while config.execution_block_enabled=false");
        }
        return;
    }

    auto& execution_block_tensor_owner = requireLayer(
        execution_block_parameters,
        "ExecutionBlockParameterTensors",
        "registerExecutionBlockParameters");
    ParameterRegistry::registerExecutionBlockParameters(execution_block_tensor_owner, registrar);
}

void registerNumberEncoderParameters(ParameterRegistry::StartupParameterRegistry& parameter_registry,
                                     Registrar& registrar,
                                     const GRIM::Config::AiConfigSnapshot& config) {
    auto* number_encoder_parameters = parameter_registry.getNumberEncoderParameters();
    const auto number_encoder_hp = GRIM::HyperParameters::numberEncoderConstructionHP(config);

    if (!number_encoder_hp.enabled) {
        if (number_encoder_parameters) {
            throw std::runtime_error("[buildParameterGroups] NumberEncoder parameter owner exists while config.number_encoder_enabled=false");
        }
        return;
    }

    auto& number_encoder_tensor_owner = requireLayer(
        number_encoder_parameters,
        "NumberEncoderParameterTensors",
        "registerNumberEncoderParameters");
    ParameterRegistry::registerNumberEncoderParameters(number_encoder_tensor_owner, registrar);
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
    static_assert(kParamGroupTypeCount == 8,
                  "Registered group precision summary must list every ParamGroupType");

    const std::array<ParamGroupType, kParamGroupTypeCount> group_types = {
        ParamGroupType::EMBEDDING,
        ParamGroupType::LM_HEAD,
        ParamGroupType::ATTENTION,
        ParamGroupType::FFN,
        ParamGroupType::RMSNORM,
        ParamGroupType::MTP,
        ParamGroupType::EXECUTION_BLOCK,
        ParamGroupType::NUMBER_ENCODER
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
            case ParamGroupType::MTP: ++other_count; break;
            case ParamGroupType::EXECUTION_BLOCK: ++other_count; break;
            case ParamGroupType::NUMBER_ENCODER: ++other_count; break;
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
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_mtp"), "parameter_precision_mtp", "buildParameterGroups");
    GRIM::HyperParameters::validateParameterGroupPrecision(GRIM::HyperParameters::snapshotTrainingConfigField<ParameterGroupPrecision>(config, "parameter_precision_execution_block"), "parameter_precision_execution_block", "buildParameterGroups");
}

void validateExecutionBlockConstructionHP(
    const ExecutionBlockConstructionHP& hp,
    const char* caller) {
    if (!hp.enabled) {
        return;
    }
    if (hp.d_model <= 0) {
        throw std::runtime_error(std::string(caller) + ": d_model must be positive");
    }
    if (hp.atom_embedding_dim <= 0) {
        throw std::runtime_error(std::string(caller) + ": atom_embedding_dim must be positive");
    }
    if (hp.num_ops <= 0) {
        throw std::runtime_error(std::string(caller) + ": num_ops must be positive");
    }
    if (hp.num_slots <= 0) {
        throw std::runtime_error(std::string(caller) + ": num_slots must be positive");
    }
    if (hp.num_exec_steps <= 0) {
        throw std::runtime_error(std::string(caller) + ": num_exec_steps must be positive");
    }
    if (hp.d_key <= 0) {
        throw std::runtime_error(std::string(caller) + ": d_key must be positive");
    }
    if (hp.d_key > 64) {
        throw std::runtime_error(std::string(caller) + ": d_key must be <= 64");
    }
    if (hp.d_type <= 0) {
        throw std::runtime_error(std::string(caller) + ": d_type must be positive");
    }
    if (hp.cross_attn_head_dim <= 0) {
        throw std::runtime_error(std::string(caller) + ": cross_attn_head_dim must be positive");
    }
    if (hp.value_decode_input_dim <= 0) {
        throw std::runtime_error(std::string(caller) + ": value_decode_input_dim must be positive");
    }
    if (hp.value_decode_hidden_dim <= 0) {
        throw std::runtime_error(std::string(caller) + ": value_decode_hidden_dim must be positive");
    }
    if (hp.value_decode_input_dim + 16 > hp.atom_embedding_dim) {
        throw std::runtime_error(std::string(caller) + ": value_decode_input_dim + 16 must fit within atom_embedding_dim");
    }
    if (hp.num_scratch_slots < 0) {
        throw std::runtime_error(std::string(caller) + ": num_scratch_slots must be non-negative");
    }
    if (hp.num_scratch_slots >= hp.num_slots) {
        throw std::runtime_error(std::string(caller) + ": num_scratch_slots must be < num_slots");
    }
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
        tensors.W_gate.alloc_grad();
        GRIM::Tensor::xavier_uniform_(tensors.W_gate, ffn_seed, init_stream);

        tensors.W1 = GRIM::Tensor::zeros({ffn_hp.d_model, ffn_hp.d_ff}, init_stream, "ffn_w1");
        tensors.W1.requires_grad_();
        tensors.W1.alloc_grad();
        GRIM::Tensor::xavier_uniform_(tensors.W1, ffn_seed + 1, init_stream);

        tensors.W2 = GRIM::Tensor::zeros({ffn_hp.d_ff, ffn_hp.d_model}, init_stream, "ffn_w2");
        tensors.W2.requires_grad_();
        tensors.W2.alloc_grad();
        GRIM::Tensor::xavier_uniform_with_gain_(tensors.W2, ffn_seed + 2, residual_projection_init_gain, init_stream);

        if (ffn_hp.use_bias) {
            tensors.b2 = GRIM::Tensor::zeros({1, ffn_hp.d_model}, init_stream, "ffn_b2");
            tensors.b2.requires_grad_();
            tensors.b2.alloc_grad();
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

void initializeEncodingLayerParameterTensors(
    std::vector<GRIM::EncodingLayerParameterTensors>& encoding_layer_parameter_tensors,
    const GRIM::HyperParameters::EncoderLayerConstructionHP& encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!init_stream) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: init_stream is NULL");
    }
    if (encoder_hp.num_layers <= 0) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: encoder_hp.num_layers must be > 0, got " +
                                 std::to_string(encoder_hp.num_layers));
    }
    if (encoder_hp.d_model <= 0) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: encoder_hp.d_model must be > 0, got " +
                                 std::to_string(encoder_hp.d_model));
    }
    if (encoder_hp.qkv_dim <= 0) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: encoder_hp.qkv_dim must be > 0, got " +
                                 std::to_string(encoder_hp.qkv_dim));
    }
    if (!encoding_layer_parameter_tensors.empty()) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: registry encoder tensor vector is already initialized; refusing to allocate twice. size=" +
                                 std::to_string(encoding_layer_parameter_tensors.size()));
    }

    const float residual_projection_init_gain = encoder_hp.residual_projection_init_gain;
    if (!std::isfinite(residual_projection_init_gain) || residual_projection_init_gain <= 0.0f) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: residual_projection_init_gain must be positive finite");
    }
    if (!std::isfinite(encoder_hp.layer_scale_init)) {
        throw std::runtime_error("initializeEncodingLayerParameterTensors: layer_scale_init must be finite");
    }

    encoding_layer_parameter_tensors.resize(static_cast<std::size_t>(encoder_hp.num_layers));
    std::vector<float> ones(static_cast<std::size_t>(encoder_hp.d_model), 1.0f);
    std::vector<float> layer_scale_init(static_cast<std::size_t>(encoder_hp.d_model), encoder_hp.layer_scale_init);

    for (int layer = 0; layer < encoder_hp.num_layers; ++layer) {
        auto& tensors = encoding_layer_parameter_tensors[static_cast<std::size_t>(layer)];
        const std::uint64_t layer_seed = weight_init_seed + 2 + static_cast<std::uint64_t>(layer) * 10ULL;

        tensors.rms1_gamma = GRIM::Tensor::zeros({encoder_hp.d_model}, init_stream, "enc_rms1_gamma");
        if (!encoder_hp.freeze_learned_rms_gammas) {
            tensors.rms1_gamma.requires_grad_();
            tensors.rms1_gamma.alloc_grad();
        }
        cudaError_t copy_err = cudaMemcpyAsync(
            tensors.rms1_gamma.data,
            ones.data(),
            static_cast<std::size_t>(encoder_hp.d_model) * sizeof(float),
            cudaMemcpyHostToDevice,
            init_stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaMemcpyAsync failed for rms1_gamma: ") +
                                     cudaGetErrorString(copy_err));
        }

        tensors.rms2_gamma = GRIM::Tensor::zeros({encoder_hp.d_model}, init_stream, "enc_rms2_gamma");
        if (!encoder_hp.freeze_learned_rms_gammas) {
            tensors.rms2_gamma.requires_grad_();
            tensors.rms2_gamma.alloc_grad();
        }
        copy_err = cudaMemcpyAsync(
            tensors.rms2_gamma.data,
            ones.data(),
            static_cast<std::size_t>(encoder_hp.d_model) * sizeof(float),
            cudaMemcpyHostToDevice,
            init_stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaMemcpyAsync failed for rms2_gamma: ") +
                                     cudaGetErrorString(copy_err));
        }

        tensors.W_qkv = GRIM::Tensor::zeros({encoder_hp.qkv_dim, encoder_hp.d_model}, init_stream, "enc_W_qkv");
        tensors.W_qkv.requires_grad_();
        tensors.W_qkv.alloc_grad();
        GRIM::Tensor::xavier_uniform_(tensors.W_qkv, layer_seed + 0, init_stream);

        tensors.W_o = GRIM::Tensor::zeros({encoder_hp.d_model, encoder_hp.d_model}, init_stream, "enc_W_o");
        tensors.W_o.requires_grad_();
        tensors.W_o.alloc_grad();
        GRIM::Tensor::xavier_uniform_with_gain_(tensors.W_o, layer_seed + 1, residual_projection_init_gain, init_stream);

        if (encoder_hp.use_bias) {
            tensors.b_qkv = GRIM::Tensor::zeros({encoder_hp.qkv_dim}, init_stream, "enc_b_qkv");
            tensors.b_qkv.requires_grad_();
            tensors.b_qkv.alloc_grad();

            tensors.b_o = GRIM::Tensor::zeros({encoder_hp.d_model}, init_stream, "enc_b_o");
            tensors.b_o.requires_grad_();
            tensors.b_o.alloc_grad();
        }

        if (encoder_hp.use_layer_scale) {
            tensors.layer_scale1 = GRIM::Tensor::zeros({1, encoder_hp.d_model}, init_stream, "enc_layer_scale1");
            tensors.layer_scale1.requires_grad_();
            tensors.layer_scale1.alloc_grad();
            copy_err = cudaMemcpyAsync(
                tensors.layer_scale1.data,
                layer_scale_init.data(),
                static_cast<std::size_t>(encoder_hp.d_model) * sizeof(float),
                cudaMemcpyHostToDevice,
                init_stream);
            if (copy_err != cudaSuccess) {
                throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaMemcpyAsync failed for layer_scale1: ") +
                                         cudaGetErrorString(copy_err));
            }

            tensors.layer_scale2 = GRIM::Tensor::zeros({1, encoder_hp.d_model}, init_stream, "enc_layer_scale2");
            tensors.layer_scale2.requires_grad_();
            tensors.layer_scale2.alloc_grad();
            copy_err = cudaMemcpyAsync(
                tensors.layer_scale2.data,
                layer_scale_init.data(),
                static_cast<std::size_t>(encoder_hp.d_model) * sizeof(float),
                cudaMemcpyHostToDevice,
                init_stream);
            if (copy_err != cudaSuccess) {
                throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaMemcpyAsync failed for layer_scale2: ") +
                                         cudaGetErrorString(copy_err));
            }
        }
    }

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(std::string("initializeEncodingLayerParameterTensors: cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(sync_err));
    }

    emitInfo("[initializeEncodingLayerParameterTensors] Initialized registry-owned encoder tensors for " +
             std::to_string(encoder_hp.num_layers) + " layers");
}

void initializeEmbeddingParameterTensors(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::EmbeddingLayerConstructionHP& embedding_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream,
    bool requires_grad) {
    if (!init_stream) {
        throw std::runtime_error("initializeEmbeddingParameterTensors: init_stream is NULL");
    }
    if (embedding_hp.vocab_size <= 0) {
        throw std::runtime_error("initializeEmbeddingParameterTensors: vocab_size must be positive, got " +
                                 std::to_string(embedding_hp.vocab_size));
    }
    if (embedding_hp.d_model <= 0) {
        throw std::runtime_error("initializeEmbeddingParameterTensors: d_model must be positive, got " +
                                 std::to_string(embedding_hp.d_model));
    }
    if (parameter_registry.getEmbeddingParameters()) {
        throw std::runtime_error("initializeEmbeddingParameterTensors: registry embedding tensor owner is already initialized");
    }

    parameter_registry.embedding_parameters = std::make_unique<GRIM::EmbeddingParameterTensors>();
    auto& embedding_parameters = *parameter_registry.embedding_parameters;
    embedding_parameters.token_weights = Tensor::zeros(
        {embedding_hp.vocab_size, embedding_hp.d_model},
        init_stream,
        "embedding.token_weights");
    if (requires_grad) {
        embedding_parameters.token_weights.requires_grad_();
        embedding_parameters.token_weights.alloc_grad();
    }
    Tensor::xavier_uniform_(embedding_parameters.token_weights, weight_init_seed, init_stream);

    emitInfo("[initializeEmbeddingParameterTensors] Initialized registry-owned embedding tensor [" +
             std::to_string(embedding_hp.vocab_size) + ", " +
             std::to_string(embedding_hp.d_model) + "]");
}

void initializeLmHeadParameterTensors(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::LMHeadLayerConstructionHP& lm_head_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream,
    GRIM::Tensor* tied_embedding_weights) {
    if (!init_stream) {
        throw std::runtime_error("initializeLmHeadParameterTensors: init_stream is NULL");
    }
    if (lm_head_hp.d_model <= 0) {
        throw std::runtime_error("initializeLmHeadParameterTensors: d_model must be positive, got " +
                                 std::to_string(lm_head_hp.d_model));
    }
    if (lm_head_hp.vocab_size <= 0) {
        throw std::runtime_error("initializeLmHeadParameterTensors: vocab_size must be positive, got " +
                                 std::to_string(lm_head_hp.vocab_size));
    }
    if (parameter_registry.getLmHeadParameters()) {
        throw std::runtime_error("initializeLmHeadParameterTensors: registry LM-head tensor owner is already initialized");
    }
    if (lm_head_hp.tie_embeddings != (tied_embedding_weights != nullptr)) {
        throw std::runtime_error(
            "initializeLmHeadParameterTensors: tie_embeddings grouping/runtime ownership mismatch. tie_embeddings=" +
            std::string(lm_head_hp.tie_embeddings ? "true" : "false") +
            " tied_embedding_weights=" +
            std::string(tied_embedding_weights ? "non-null" : "NULL"));
    }

    parameter_registry.lm_head_parameters = std::make_unique<GRIM::LMHeadParameterTensors>();
    auto& parameter_tensors = *parameter_registry.lm_head_parameters;
    parameter_tensors.weights = Tensor();
    parameter_tensors.bias = Tensor();
    parameter_tensors.final_rms_gamma = Tensor();
    parameter_tensors.owns_weights = !tied_embedding_weights;

    if (tied_embedding_weights) {
        if (!tied_embedding_weights->data) {
            throw std::runtime_error(
                "initializeLmHeadParameterTensors: tied embedding weights are NULL — embedding MUST be initialized before LM head");
        }
        if (!tied_embedding_weights->shape.is_2d_layout()) {
            throw std::runtime_error(
                "initializeLmHeadParameterTensors: tied embedding weights must be 2D [vocab_size,d_model]");
        }
        const auto& embedding_shape = tied_embedding_weights->shape.as_2d();
        if (embedding_shape.rows != lm_head_hp.vocab_size || embedding_shape.cols != lm_head_hp.d_model) {
            throw std::runtime_error(
                "initializeLmHeadParameterTensors: tied embedding shape mismatch. Expected [" +
                std::to_string(lm_head_hp.vocab_size) + "," + std::to_string(lm_head_hp.d_model) +
                "], got [" + std::to_string(embedding_shape.rows) + "," +
                std::to_string(embedding_shape.cols) + "]");
        }

        parameter_tensors.weights = Tensor::from_ptr(
            tied_embedding_weights->data,
            tied_embedding_weights->shape,
            false,
            true,
            "lm_head.weights_tied");
        parameter_tensors.weights.share_grad(*tied_embedding_weights);
        parameter_tensors.weights.owns_data = false;
        parameter_tensors.weights.requires_grad = true;
    } else {
        parameter_tensors.weights = Tensor::zeros(
            {lm_head_hp.vocab_size, lm_head_hp.d_model},
            init_stream,
            "lm_head.weights");
        parameter_tensors.weights.requires_grad_();
        parameter_tensors.weights.alloc_grad();
        Tensor::xavier_uniform_(parameter_tensors.weights, weight_init_seed, init_stream);
    }

    if (lm_head_hp.use_bias) {
        parameter_tensors.bias = Tensor::zeros({lm_head_hp.vocab_size}, init_stream, "lm_head.bias");
        parameter_tensors.bias.requires_grad_();
        parameter_tensors.bias.alloc_grad();
    }

    parameter_tensors.final_rms_gamma = Tensor::zeros({lm_head_hp.d_model}, init_stream, "final_rms_gamma");
    if (!lm_head_hp.freeze_learned_rms_gammas) {
        parameter_tensors.final_rms_gamma.requires_grad_();
        parameter_tensors.final_rms_gamma.alloc_grad();
    }

    std::vector<float> ones(static_cast<std::size_t>(lm_head_hp.d_model), 1.0f);
    cudaMemcpyAsync(parameter_tensors.final_rms_gamma.data,
                    ones.data(),
                    static_cast<std::size_t>(lm_head_hp.d_model) * sizeof(float),
                    cudaMemcpyHostToDevice,
                    init_stream);

    emitInfo("[initializeLmHeadParameterTensors] Initialized registry-owned LM-head tensors");
}

void initializeExecutionBlockParameterTensors(
    ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const ExecutionBlockConstructionHP& execution_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!execution_hp.enabled) {
        if (parameter_registry.getExecutionBlockParameters()) {
            throw std::runtime_error("initializeExecutionBlockParameterTensors: ExecutionBlock disabled but registry owner already exists");
        }
        return;
    }
    if (!init_stream) {
        throw std::runtime_error("initializeExecutionBlockParameterTensors: init_stream is NULL");
    }
    if (parameter_registry.getExecutionBlockParameters()) {
        throw std::runtime_error("initializeExecutionBlockParameterTensors: registry ExecutionBlock tensor owner is already initialized");
    }

    validateExecutionBlockConstructionHP(execution_hp, "initializeExecutionBlockParameterTensors");

    parameter_registry.execution_block_parameters = std::make_unique<GRIM::ExecutionBlockParameterTensors>();
    auto& params = *parameter_registry.execution_block_parameters;

    const int dm = execution_hp.d_model;
    const int dk = execution_hp.d_key;
    const int dt = execution_hp.d_type;
    const int hd = execution_hp.cross_attn_head_dim;
    const int nop = execution_hp.num_ops;
    const int V = execution_hp.num_slots;
    const int K = execution_hp.num_exec_steps;
    const int vid = execution_hp.value_decode_input_dim;
    const int vhd = execution_hp.value_decode_hidden_dim;

    auto make_param = [&](int rows, int cols, std::uint64_t seed, const char* name) -> Tensor {
        Tensor tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(rows, cols),
            true,
            init_stream,
            name);
        tensor.requires_grad_();
        tensor.alloc_grad();
        Tensor::xavier_uniform_(tensor, seed, init_stream);
        return std::move(tensor);
    };
    auto make_bias = [&](int cols, const char* name) -> Tensor {
        Tensor tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, cols),
            true,
            init_stream,
            name);
        tensor.requires_grad_();
        tensor.alloc_grad();
        return std::move(tensor);
    };
    auto make_scalar = [&](float init_val, const char* name) -> Tensor {
        Tensor tensor = Tensor::zeros(
            TensorContract::TensorShape::make_BSM(1, 1),
            true,
            init_stream,
            name);
        tensor.requires_grad_();
        tensor.alloc_grad();
        const cudaError_t copy_err = cudaMemcpyAsync(
            tensor.data,
            &init_val,
            sizeof(float),
            cudaMemcpyHostToDevice,
            init_stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("initializeExecutionBlockParameterTensors: cudaMemcpyAsync failed for ") +
                                     name + ": " + cudaGetErrorString(copy_err));
        }
        return std::move(tensor);
    };

    params.w_decode_1 = make_param(vid, vhd, weight_init_seed, "exec_block.w_decode_1");
    params.b_decode_1 = make_bias(vhd, "exec_block.b_decode_1");
    params.w_decode_2 = make_param(vhd, 1, weight_init_seed + 1, "exec_block.w_decode_2");
    params.w_arg1_select = make_param(3 * dm, dm, weight_init_seed + 2, "exec_block.w_arg1_select");
    params.w_arg2_select = make_param(3 * dm, dm, weight_init_seed + 3, "exec_block.w_arg2_select");
    params.W_op_select = make_param(3 * dm, nop, weight_init_seed + 4, "exec_block.W_op_select");
    params.W_key_proj = make_param(dm, dk, weight_init_seed + 5, "exec_block.W_key_proj");
    params.W_write_query = make_param(4 * dm, dk, weight_init_seed + 7, "exec_block.W_write_query");
    params.W_write_key = make_param(dk, dk, weight_init_seed + 8, "exec_block.W_write_key");
    params.alpha = make_scalar(1.0f, "exec_block.alpha");
    params.beta = make_scalar(1.0f, "exec_block.beta");
    params.step_embeddings = make_param(K, dm, weight_init_seed + 9, "exec_block.step_embeddings");
    params.type_num_embed = make_param(1, dt, weight_init_seed + 10, "exec_block.type_num_embed");
    params.W_Q_read = make_param(dm, hd, weight_init_seed + 11, "exec_block.W_Q_read");
    params.W_K_read = make_param(dk, hd, weight_init_seed + 12, "exec_block.W_K_read");
    params.W_V_read = make_param(dm, hd, weight_init_seed + 13, "exec_block.W_V_read");
    params.W_O_read = make_param(hd, dm, weight_init_seed + 14, "exec_block.W_O_read");
    params.W_value_to_emb = make_param(1, dm, weight_init_seed + 15, "exec_block.W_value_to_emb");
    params.b_value_to_emb = make_bias(dm, "exec_block.b_value_to_emb");
    params.E_slot = make_param(V, dm, weight_init_seed + 16, "exec_block.E_slot");
    params.E_op = make_param(nop, dm, weight_init_seed + 17, "exec_block.E_op");
    params.W_scal = make_param(3, dm, weight_init_seed + 18, "exec_block.W_scal");
    params.b_scal = make_bias(dm, "exec_block.b_scal");
    params.W_trace = make_param(K * dm, dm, weight_init_seed + 19, "exec_block.W_trace");
    params.b_trace = make_bias(dm, "exec_block.b_trace");
    params.W_reason_gate = make_param(2 * dm, dm, weight_init_seed + 20, "exec_block.W_reason_gate");
    params.W_trace_gate = make_param(2 * dm, dm, weight_init_seed + 21, "exec_block.W_trace_gate");

    params.w_inject_gate = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(dm, 1),
        true,
        init_stream,
        "exec_block.w_inject_gate");
    params.w_inject_gate.requires_grad_();
    params.w_inject_gate.alloc_grad();
    {
        std::vector<float> neg_two(static_cast<std::size_t>(dm), -2.0f);
        const cudaError_t copy_err = cudaMemcpyAsync(
            params.w_inject_gate.data,
            neg_two.data(),
            static_cast<std::size_t>(dm) * sizeof(float),
            cudaMemcpyHostToDevice,
            init_stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("initializeExecutionBlockParameterTensors: cudaMemcpyAsync failed for exec_block.w_inject_gate: ") +
                                     cudaGetErrorString(copy_err));
        }
    }

    params.W_gate_read = Tensor::zeros(
        TensorContract::TensorShape::make_BSM(dm, 1),
        true,
        init_stream,
        "exec_block.W_gate_read");
    params.W_gate_read.requires_grad_();
    params.W_gate_read.alloc_grad();
    params.tau = make_scalar(1.0f, "exec_block.tau");

    emitInfo("[initializeExecutionBlockParameterTensors] Initialized registry-owned ExecutionBlock tensors");
}

void initializeNumberEncoderParameterTensors(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::HyperParameters::NumberEncoderConstructionHP& number_encoder_hp,
    std::uint64_t weight_init_seed,
    cudaStream_t init_stream) {
    if (!number_encoder_hp.enabled) {
        if (parameter_registry.getNumberEncoderParameters()) {
            throw std::runtime_error("initializeNumberEncoderParameterTensors: NumberEncoder disabled but registry owner already exists");
        }
        return;
    }
    if (!init_stream) {
        throw std::runtime_error("initializeNumberEncoderParameterTensors: init_stream is NULL");
    }
    if (parameter_registry.getNumberEncoderParameters()) {
        throw std::runtime_error("initializeNumberEncoderParameterTensors: registry NumberEncoder tensor owner is already initialized");
    }
    if (number_encoder_hp.d_model <= 0) {
        throw std::runtime_error("initializeNumberEncoderParameterTensors: d_model must be > 0, got " +
                                 std::to_string(number_encoder_hp.d_model));
    }
    if (number_encoder_hp.d_hidden <= 0) {
        throw std::runtime_error("initializeNumberEncoderParameterTensors: d_hidden must be > 0, got " +
                                 std::to_string(number_encoder_hp.d_hidden));
    }
    if (number_encoder_hp.max_digit_slots <= 0) {
        throw std::runtime_error("initializeNumberEncoderParameterTensors: max_digit_slots must be > 0, got " +
                                 std::to_string(number_encoder_hp.max_digit_slots));
    }
    if (number_encoder_hp.pow10_buckets != 2 * number_encoder_hp.max_abs_pow10 + 1 ||
        number_encoder_hp.pow10_buckets <= 0) {
        throw std::runtime_error("initializeNumberEncoderParameterTensors: pow10_buckets=" +
                                 std::to_string(number_encoder_hp.pow10_buckets) +
                                 " does not match 2 * max_abs_pow10 + 1 (max_abs_pow10=" +
                                 std::to_string(number_encoder_hp.max_abs_pow10) + ")");
    }

    const int d_model = number_encoder_hp.d_model;
    const int d_hidden = number_encoder_hp.d_hidden;

    auto params = std::make_unique<GRIM::NumberEncoderParameterTensors>();
    auto make_xavier = [&](int rows, int cols, std::uint64_t seed, const char* name) -> GRIM::Tensor {
        GRIM::Tensor t = GRIM::Tensor::zeros({rows, cols}, init_stream, name);
        t.requires_grad_();
        t.alloc_grad();
        GRIM::Tensor::xavier_uniform_(t, seed, init_stream);
        return t;
    };

    params->digit_emb = make_xavier(10, d_model, weight_init_seed, "number_encoder.digit_emb");
    params->pow10_emb = make_xavier(number_encoder_hp.pow10_buckets, d_model, weight_init_seed + 1, "number_encoder.pow10_emb");
    params->W_c1 = make_xavier(GRIM::Batching::BatchPayload::kNumberSlotFeatureDim, d_hidden, weight_init_seed + 2, "number_encoder.W_c1");
    params->b_c1 = GRIM::Tensor::zeros({1, d_hidden}, init_stream, "number_encoder.b_c1");
    params->b_c1.requires_grad_();
    params->b_c1.alloc_grad();
    params->W_c2 = make_xavier(d_hidden, d_model, weight_init_seed + 3, "number_encoder.W_c2");
    params->W_g1 = make_xavier(GRIM::Batching::BatchPayload::kNumberGlobalFeatureDim, d_hidden, weight_init_seed + 4, "number_encoder.W_g1");
    params->b_g1 = GRIM::Tensor::zeros({1, d_hidden}, init_stream, "number_encoder.b_g1");
    params->b_g1.requires_grad_();
    params->b_g1.alloc_grad();
    params->W_g2 = make_xavier(d_hidden, d_model, weight_init_seed + 5, "number_encoder.W_g2");

    const cudaError_t sync_err = cudaStreamSynchronize(init_stream);
    if (sync_err != cudaSuccess) {
        throw std::runtime_error(std::string("initializeNumberEncoderParameterTensors: cudaStreamSynchronize failed: ") +
                                 cudaGetErrorString(sync_err));
    }

    parameter_registry.number_encoder_parameters = std::move(params);
    emitInfo("[initializeNumberEncoderParameterTensors] Initialized registry-owned NumberEncoder tensors (d_model=" +
             std::to_string(d_model) + ", d_hidden=" + std::to_string(d_hidden) +
             ", pow10_buckets=" + std::to_string(number_encoder_hp.pow10_buckets) +
             ", max_digit_slots=" + std::to_string(number_encoder_hp.max_digit_slots) + ")");
}

void buildParameterGroups(const GRIM::Config::AiConfigSnapshot& config,
                          Startup::GpuModelState& gpu_model_state,
                          ParameterRegistry::StartupParameterRegistry& parameter_registry) {
    validateParameterRegistrationConfig(config);

    auto& registry_groups = parameter_registry.parameterGroups();
    std::vector<ParameterGroup> rebuilt_groups;
    rebuilt_groups.reserve(registry_groups.size());

    Registrar registrar(rebuilt_groups, config);
    registerTopLevelParameters(gpu_model_state, parameter_registry, registrar, config);
    registerEncoderParameters(gpu_model_state, parameter_registry, registrar, config);

    registerNumberEncoderParameters(parameter_registry, registrar, config);
    registerExecutionBlockParameters(gpu_model_state, parameter_registry, registrar, config);
    registerMtpParameters(parameter_registry, registrar, config);

    validateRegisteredTensorPrecisionMetadata(rebuilt_groups);
    clearOptimizerBindings(rebuilt_groups);

    // Transaction boundary: parameter_registry.parameterGroups() is replaced only after the
    // complete configured inventory has been discovered and validated. A thrown
    // registration check must never leave StartupParameterRegistry with a half-built group
    // vector that downstream optimizer/checkpoint code could observe.
    registry_groups.swap(rebuilt_groups);

    emitInfo("[buildParameterGroups] Built " + std::to_string(registry_groups.size()) + " parameter groups");
    emitGroupSummary(registry_groups);
}

void bindOptimizerState(ParameterRegistry::StartupParameterRegistry& parameter_registry,
                        GRIM::OptimizerState& optimizer_state,
                        cudaStream_t stream) {
    auto& groups = parameter_registry.requireParameterGroups("bindOptimizerState");
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


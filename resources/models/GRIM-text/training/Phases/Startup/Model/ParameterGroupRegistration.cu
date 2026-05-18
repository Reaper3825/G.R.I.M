#include "ParameterGroupRegistration.hpp"

#include "../../../../GRIM/grim_language_model_cuda.hpp"
#include "../../../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../../../Shared/LogRecorder/LogRecorder.hpp"
#include "../../../../Shared/Optimizers/OptimizerState_GPU.hpp"

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
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
using GRIM::HyperParameters::ParameterRegistrationHP;

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
    explicit Registrar(std::vector<ParameterGroup>& groups) : groups_(groups) {}

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
        if (!registered_data_.insert(data_ptr).second) {
            throw std::runtime_error("[buildParameterGroups] duplicate tensor.data registration for " + name +
                                     " would double-step the same memory: " +
                                     tensorDebugSummary(tensor) + " layer=" + std::to_string(layer));
        }

        ParameterGroup group{};
        group.name = name;
        group.tensor = &tensor;
        group.m_tensor = nullptr;
        group.v_tensor = nullptr;
        group.type = type;
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

    void addNonDecayTensor(const std::string& name,
                           Tensor& tensor,
                           ParamGroupType type,
                           ParamStatsBucket stats_bucket,
                           int layer = -1) {
        addTensor(name, tensor, type, stats_bucket, layer, 0.0f);
    }

    void addConfigGatedNonDecayTensor(const std::string& name,
                                      Tensor& tensor,
                                      ParamGroupType type,
                                      ParamStatsBucket stats_bucket,
                                      int layer,
                                      bool enabled,
                                      const char* disabled_reason) {
        if (enabled) {
            addNonDecayTensor(name, tensor, type, stats_bucket, layer);
            return;
        }

        if (tensor.data || tensor.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] " + name +
                                     " exists while disabled (" + disabled_reason + "): " +
                                     tensorDebugSummary(tensor));
        }
    }

private:
    std::vector<ParameterGroup>& groups_;
    std::unordered_set<const void*> registered_data_;
};

void validateEmbeddingLmHeadAliasing(const Tensor& embedding_weights,
                                     const Tensor& lm_head_weights,
                                     const ParameterRegistrationHP& hp) {
    if (!embedding_weights.data || !lm_head_weights.data) {
        throw std::runtime_error("[buildParameterGroups] cannot validate embedding/LM-head aliasing with NULL data: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }

    const bool tied = !hp.register_untied_embedding;
    const bool same_data = embedding_weights.data == lm_head_weights.data;
    if (tied && !same_data) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=true but embedding and LM-head data pointers differ: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }
    if (!tied && same_data) {
        throw std::runtime_error("[buildParameterGroups] tie_embeddings=false but embedding and LM-head data pointers are identical: embedding=" +
                                 tensorDebugSummary(embedding_weights) + " lm_head=" + tensorDebugSummary(lm_head_weights));
    }
}

void registerTopLevelParameters(LanguageModel& model,
                                Registrar& registrar,
                                const ParameterRegistrationHP& hp) {
    auto& embedding = requireLayer(model.getEmbeddingLayer(), "EmbeddingLayer", "registerTopLevelParameters");
    auto& lm_head = requireLayer(model.getLmHeadLayer(), "LMHeadLayer", "registerTopLevelParameters");
    validateEmbeddingLmHeadAliasing(embedding.tokenWeights(), lm_head.weights(), hp);

    if (hp.register_untied_embedding) {
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
    registrar.addConfigGatedNonDecayTensor("lm_head_bias",
                                           lm_head.bias(),
                                           ParamGroupType::LM_HEAD,
                                           ParamStatsBucket::LM_HEAD,
                                           -1,
                                           hp.register_lm_head_bias,
                                           "config.use_bias=false");
}

void registerEncoderParameters(LanguageModel& model,
                               Registrar& registrar,
                               const ParameterRegistrationHP& hp) {
    GRIM::GPUGrimEncoder& gpu_encoder = model.getGpuEncoder();

    for (int layer = 0; layer < hp.num_layers; ++layer) {
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
        registrar.addConfigGatedNonDecayTensor(prefix + "_qkv_bias",
                                               enc->attnBqkv(),
                                               ParamGroupType::ATTENTION,
                               ParamStatsBucket::ENCODER,
                                               layer,
                                               hp.register_encoder_biases,
                                               "config.use_bias=false");
        registrar.addTensor(prefix + "_wo_weight",
                    enc->attnWo(),
                    ParamGroupType::ATTENTION,
                    ParamStatsBucket::ENCODER,
                    layer);
        registrar.addConfigGatedNonDecayTensor(prefix + "_wo_bias",
                                               enc->attnBo(),
                                               ParamGroupType::ATTENTION,
                               ParamStatsBucket::ENCODER,
                                               layer,
                                               hp.register_encoder_biases,
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
        registrar.addConfigGatedNonDecayTensor(prefix + "_ffn_b2",
                                               enc->ffnB2(),
                                               ParamGroupType::FFN,
                               ParamStatsBucket::ENCODER,
                                               layer,
                                               hp.register_encoder_biases,
                                               "config.use_bias=false");

        registrar.addNonDecayTensor(prefix + "_rms1_gamma",
                        enc->rms1Gamma(),
                        ParamGroupType::RMSNORM,
                        ParamStatsBucket::ENCODER,
                        layer);
        registrar.addNonDecayTensor(prefix + "_rms2_gamma",
                        enc->rms2Gamma(),
                        ParamGroupType::RMSNORM,
                        ParamStatsBucket::ENCODER,
                        layer);

        registrar.addConfigGatedNonDecayTensor(prefix + "_layer_scale1",
                                               enc->layerScale1(),
                                               ParamGroupType::RMSNORM,
                               ParamStatsBucket::ENCODER,
                                               layer,
                                               hp.register_layer_scale,
                                               "config.use_layer_scale=false");
        registrar.addConfigGatedNonDecayTensor(prefix + "_layer_scale2",
                                               enc->layerScale2(),
                                               ParamGroupType::RMSNORM,
                               ParamStatsBucket::ENCODER,
                                               layer,
                                               hp.register_layer_scale,
                                               "config.use_layer_scale=false");
    }
}

void registerScratchBlockParameters(LanguageModel& model,
                                    Registrar& registrar,
                                    const ParameterRegistrationHP& hp) {
    auto* scratch_block = model.getScratchBlockLayer();

    if (!hp.register_scratch_block) {
        if (scratch_block && scratch_block->isEnabled()) {
            throw std::runtime_error("[buildParameterGroups] ScratchBlock layer exists and is enabled while config.use_scratch_block=false");
        }
        return;
    }

    if (!scratch_block) {
        throw std::runtime_error("[buildParameterGroups] config.use_scratch_block=true but ScratchBlock layer is NULL");
    }
    if (!scratch_block->isEnabled()) {
        throw std::runtime_error("[buildParameterGroups] ScratchBlock layer is present but disabled during startup registration");
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
}

void registerReasoningHeadParameters(LanguageModel& model,
                                     Registrar& registrar,
                                     const ParameterRegistrationHP& hp) {
    auto* reasoning_head = model.getReasoningHeadLayer();

    if (!hp.register_reasoning_head) {
        if (reasoning_head) {
            throw std::runtime_error("[buildParameterGroups] ReasoningHead layer exists while config.reasoning_head_enabled=false");
        }
        return;
    }

    auto& head = requireLayer(reasoning_head, "ReasoningHeadLayer", "registerReasoningHeadParameters");
    registrar.addTensor("reasoning_head_w_op", head.W_op(), ParamGroupType::REASONING_HEAD, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("reasoning_head_b_op", head.b_op(), ParamGroupType::REASONING_HEAD, ParamStatsBucket::ENCODER);
    registrar.addTensor("reasoning_head_w_arg1", head.w_arg1(), ParamGroupType::REASONING_HEAD, ParamStatsBucket::ENCODER);
    registrar.addTensor("reasoning_head_w_arg2", head.w_arg2(), ParamGroupType::REASONING_HEAD, ParamStatsBucket::ENCODER);
}

void registerExecutionBlockParameters(LanguageModel& model,
                                      Registrar& registrar,
                                      const ParameterRegistrationHP& hp) {
    auto* execution_block = model.getExecutionBlockLayer();
    auto* slot_selector = model.getDecodeTimeSlotSelectorLayer();

    if (!hp.register_execution_block) {
        if (execution_block) {
            throw std::runtime_error("[buildParameterGroups] ExecutionBlock layer exists while config.execution_block_enabled=false");
        }
        if (hp.register_slot_selector || slot_selector) {
            throw std::runtime_error("[buildParameterGroups] Slot selector requires config.execution_block_enabled=true");
        }
        return;
    }

    auto& block = requireLayer(execution_block, "ExecutionBlockLayer", "registerExecutionBlockParameters");
    registrar.addTensor("exec_block_w_decode_1", block.w_decode_1(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("exec_block_b_decode_1", block.b_decode_1(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_w_decode_2", block.w_decode_2(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_w_arg1_select", block.w_arg1_select(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_w_arg2_select", block.w_arg2_select(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_op_select", block.W_op_select(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_key_proj", block.W_key_proj(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_write_query", block.W_write_query(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_write_key", block.W_write_key(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("exec_block_alpha", block.alpha(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("exec_block_beta", block.beta(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_step_embeddings", block.step_embeddings(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_type_num_embed", block.type_num_embed(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_value_to_emb", block.W_value_to_emb(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("exec_block_b_value_to_emb", block.b_value_to_emb(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_w_inject_gate", block.w_inject_gate(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_Q_read", block.W_Q_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_K_read", block.W_K_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_V_read", block.W_V_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_O_read", block.W_O_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_gate_read", block.W_gate_read(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("exec_block_tau", block.tau(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_E_slot", block.E_slot(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_E_op", block.E_op(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_scal", block.W_scal(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("exec_block_b_scal", block.b_scal(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_trace", block.W_trace(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("exec_block_b_trace", block.b_trace(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_reason_gate", block.W_reason_gate(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);
    registrar.addTensor("exec_block_W_trace_gate", block.W_trace_gate(), ParamGroupType::EXECUTION_BLOCK, ParamStatsBucket::ENCODER);

    if (!hp.register_slot_selector) {
        if (slot_selector) {
            throw std::runtime_error("[buildParameterGroups] DecodeTimeSlotSelector exists while config.selector_enabled=false");
        }
        return;
    }

    auto& selector = requireLayer(slot_selector, "DecodeTimeSlotSelectorLayer", "registerExecutionBlockParameters");
    registrar.addTensor("selector_W_q_select", selector.W_q_select(), ParamGroupType::SLOT_SELECTOR, ParamStatsBucket::ENCODER);
    registrar.addTensor("selector_W_k_select", selector.W_k_select(), ParamGroupType::SLOT_SELECTOR, ParamStatsBucket::ENCODER);
    registrar.addTensor("selector_null_key_select", selector.null_key_select(), ParamGroupType::SLOT_SELECTOR, ParamStatsBucket::ENCODER);
    registrar.addNonDecayTensor("selector_null_logit_bias", selector.null_logit_bias(), ParamGroupType::SLOT_SELECTOR, ParamStatsBucket::ENCODER);
}

void registerMtpParameters(LanguageModel& model,
                           Registrar& registrar,
                           const ParameterRegistrationHP& hp) {
    if (!hp.register_mtp) {
        if (model.getMtpHead(0)) {
            throw std::runtime_error("[buildParameterGroups] MTP heads exist while config.mtp_enabled=false");
        }
        return;
    }

    if (hp.mtp_k <= 0) {
        throw std::runtime_error("[buildParameterGroups] config.mtp_enabled=true but config.mtp_k <= 0");
    }

    for (int k = 0; k < hp.mtp_k; ++k) {
        auto* head = model.getMtpHead(k);
        if (!head) {
            throw std::runtime_error("[buildParameterGroups] Missing MTP head " + std::to_string(k) +
                                     " for configured mtp_k=" + std::to_string(hp.mtp_k));
        }
        registrar.addTensor("mtp_head_" + std::to_string(k) + "_weight",
                            head->weight,
                    ParamGroupType::MTP,
                    ParamStatsBucket::ENCODER);
        registrar.addNonDecayTensor("mtp_head_" + std::to_string(k) + "_bias",
                                    head->bias,
                        ParamGroupType::MTP,
                        ParamStatsBucket::ENCODER);
    }

    if (model.getMtpHead(hp.mtp_k)) {
        throw std::runtime_error("[buildParameterGroups] MTP head vector contains more entries than config.mtp_k");
    }
}

void registerFinalRmsGamma(LanguageModel& model,
                           Registrar& registrar,
                           const ParameterRegistrationHP& hp) {
    auto& lm_head = requireLayer(model.getLmHeadLayer(), "LMHeadLayer", "registerFinalRmsGamma");
    const Tensor& final_gamma = lm_head.finalRmsGamma();

    if (!hp.register_final_rms_gamma) {
        if (final_gamma.has_grad()) {
            throw std::runtime_error("[buildParameterGroups] final_rms_gamma is frozen by config but still has a grad buffer: " +
                                     tensorDebugSummary(final_gamma));
        }
        return;
    }

    registrar.addTensor("final_rms_gamma",
                        lm_head.finalRmsGammaMutable_UnfrozenOnly("ParameterGroupRegistration::buildParameterGroups"),
                        ParamGroupType::RMSNORM,
                        ParamStatsBucket::LM_HEAD,
                        -1,
                        1.0f,
                        0.1f);
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

void emitGroupSummary(const std::vector<ParameterGroup>& groups) {
    int emb_count = 0;
    int attn_count = 0;
    int ffn_count = 0;
    int rms_count = 0;
    int other_count = 0;

    for (const auto& group : groups) {
        switch (group.type) {
            case ParamGroupType::EMBEDDING: ++emb_count; break;
            case ParamGroupType::ATTENTION: ++attn_count; break;
            case ParamGroupType::FFN: ++ffn_count; break;
            case ParamGroupType::RMSNORM: ++rms_count; break;
            default: ++other_count; break;
        }
    }

    emitInfo("[buildParameterGroups] Parameter group summary: total=" + std::to_string(groups.size()) +
             " emb=" + std::to_string(emb_count) +
             " attn=" + std::to_string(attn_count) +
             " ffn=" + std::to_string(ffn_count) +
             " rms=" + std::to_string(rms_count) +
             " other=" + std::to_string(other_count));
}

} // namespace

void buildParameterGroups(LanguageModel& model, const ParameterRegistrationHP& hp) {
    auto& model_groups = model.parameterGroups();
    std::vector<ParameterGroup> rebuilt_groups;
    rebuilt_groups.reserve(model_groups.size());

    Registrar registrar(rebuilt_groups);
    registerTopLevelParameters(model, registrar, hp);
    registerEncoderParameters(model, registrar, hp);

    registerScratchBlockParameters(model, registrar, hp);
    registerReasoningHeadParameters(model, registrar, hp);
    registerExecutionBlockParameters(model, registrar, hp);
    registerMtpParameters(model, registrar, hp);
    registerFinalRmsGamma(model, registrar, hp);

    clearOptimizerBindings(rebuilt_groups);

    // Transaction boundary: model.parameterGroups() is replaced only after the
    // complete configured inventory has been discovered and validated. A thrown
    // registration check must never leave LanguageModel with a half-built group
    // vector that downstream optimizer/checkpoint code could observe.
    model_groups.swap(rebuilt_groups);

    emitInfo("[buildParameterGroups] Built " + std::to_string(model_groups.size()) + " parameter groups");
    emitGroupSummary(model_groups);
}

void buildParameterGroups(LanguageModel& model) {
    model.buildParameterGroups();
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

namespace GRIM {

#ifdef USE_CUDA

void LanguageModel::buildParameterGroups() {
    const auto hp = GRIM::HyperParameters::parameterRegistrationHP(config_);
    GRIMText::Training::Startup::ModelRegistration::buildParameterGroups(*this, hp);
}

void LanguageModel::bindOptimizerState(OptimizerState& optimizer_state, cudaStream_t stream) {
    GRIMText::Training::Startup::ModelRegistration::bindOptimizerState(*this, optimizer_state, stream);
}

#endif // USE_CUDA

} // namespace GRIM

//======================================================//
//  Phase2_InferenceLoop.cu
//  Session-time inference orchestration.
//======================================================//

#include "Phase2_InferenceLoop.hpp"

#include "../../Shared/Batching/BatchDeviceUpload.hpp"
#include "Shared/Forward/GeneratedSequence.hpp"
#include "../../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../../Shared/Forward/ModelForward_GPU.hpp"
#include "../../Shared/Execution/DecodeTimeNumPolicy.hpp"
#include "../../Shared/Execution/DecodeTimeResolveResult.hpp"
#include "../../Shared/Sampling/Sampling.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace GRIMText::Training {

namespace {

using GRIM::HyperParameters::GenerationHP;
using GRIM::HyperParameters::GenerationStreamCallback;
using GRIM::HyperParameters::MTPFeatureHP;
using GRIM::HyperParameters::SamplingStrategy;

struct InferenceForwardScope {
    GRIM::Forward::ModelForwardOutputs& forward_outputs;
    ~InferenceForwardScope() {
        forward_outputs.clear();
    }
};

void validateInferenceContext(const TrainingContext& ctx) {
    if (!ctx.model) {
        throw std::runtime_error("Phase2 payload inference: Phase1 context model is NULL");
    }
    if (!ctx.tokenizer) {
        throw std::runtime_error("Phase2 payload inference: Phase1 context tokenizer is NULL");
    }
}

void validatePromptPayload(const GRIM::Batching::BatchPayload& prompt_payload) {
    prompt_payload.validate("Phase2 payload inference");
    if (!prompt_payload.isInferencePrefill()) {
        throw std::runtime_error("Phase2 payload inference: prompt_payload must be InferencePrefill");
    }
    if (prompt_payload.batch_size != 1) {
        throw std::runtime_error("Phase2 payload inference: prompt_payload.batch_size must be 1");
    }
    if (prompt_payload.seq_atom_tables.empty()) {
        throw std::runtime_error("Phase2 payload inference: prompt_payload.seq_atom_tables is empty");
    }
}

void validateInferenceForwardPayload(
    const GRIM::LanguageModel& model,
    const GRIM::Batching::BatchPayload& active_payload,
    const char* caller)
{
    active_payload.validate(caller);
    if (!active_payload.isInferencePrefill()) {
        throw std::runtime_error(std::string(caller) + ": payload must be InferencePrefill");
    }
    if (active_payload.batch_size != 1) {
        throw std::runtime_error(std::string(caller) + ": batch_size must be 1");
    }

    const auto& config = model.getConfig();
    const bool use_gpu = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_gpu");
    const int max_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "max_seq_len");
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "vocab_size");
    if (!use_gpu) {
        throw std::runtime_error(std::string(caller) + ": config.use_gpu must be true");
    }
    if (active_payload.max_seq_len <= 0 || active_payload.max_seq_len > max_seq_len) {
        throw std::runtime_error(std::string(caller) + ": payload max_seq_len=" +
                                 std::to_string(active_payload.max_seq_len) + " out of range [1, " +
                                 std::to_string(max_seq_len) + "]");
    }
    if (active_payload.total_tokens <= 0) {
        throw std::runtime_error(std::string(caller) + ": payload total_tokens must be > 0");
    }
    if (active_payload.vocab_size != vocab_size) {
        throw std::runtime_error(std::string(caller) + ": payload.vocab_size=" +
                                 std::to_string(active_payload.vocab_size) +
                                 " != config.vocab_size=" + std::to_string(vocab_size));
    }
}

std::vector<GRIM::Forward::MTPHeadForwardView> buildDetachedForwardMtpHeadViews(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const MTPFeatureHP& mtp_hp,
    cudaStream_t stream)
{
    std::vector<GRIM::Forward::MTPHeadForwardView> views;
    if (!mtp_hp.enabled || mtp_hp.k <= 0) {
        return views;
    }

    views.reserve(static_cast<size_t>(mtp_hp.k));
    auto& mtp_head_parameter_tensors = parameter_registry.mtpHeadParameterTensors();
    for (int k = 0; k < mtp_hp.k; ++k) {
        if (k < 0 || k >= static_cast<int>(mtp_head_parameter_tensors.size())) {
            throw std::runtime_error(
                "Phase2 payload inference: parameter_registry is missing MTP head " + std::to_string(k));
        }
        auto* head = &mtp_head_parameter_tensors[static_cast<std::size_t>(k)];
        if (!head->weight.data || !head->bias.data) {
            throw std::runtime_error(
                "Phase2 payload inference: MTP head " + std::to_string(k) +
                " has NULL weight or bias tensor");
        }

        GRIM::Forward::MTPHeadForwardView view{};
        view.weight = head->weight.detach(stream);
        view.bias = head->bias.detach(stream);
        views.push_back(std::move(view));
    }

    return views;
}

GRIM::GeneratedSequence generateOneSequence(
    GRIM::LanguageModel& model,
    GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::PBM::PBMState& pbm,
    GRIM::Batching::BatchPayload& prompt_payload,
    const GenerationHP& cfg,
    GenerationStreamCallback* stream_callback)
{
    validatePromptPayload(prompt_payload);

    const auto& config = model.getConfig();
    const bool use_gpu = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_gpu");
    const int max_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "max_seq_len");
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "vocab_size");
    const bool execution_block_enabled = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "execution_block_enabled");
    const auto& prompt_tokens = prompt_payload.input_ids;
    const auto& prompt_numeric_values = prompt_payload.numeric_values;
    const auto& prompt_atom_mask = prompt_payload.atom_mask;
    const auto& prompt_atom_flags = prompt_payload.atom_flags;
    const auto& prompt_token_to_slot_map = prompt_payload.token_to_slot_map;
    const auto& prompt_atom_entry_ids = prompt_payload.atom_entry_ids;
    const auto& prompt_atom_table = prompt_payload.seq_atom_tables[0];

    GRIM::GeneratedSequence sequence;
    sequence.token_ids = prompt_tokens;
    sequence.token_numeric_values = prompt_numeric_values;
    sequence.token_atom_mask = prompt_atom_mask;
    sequence.context_atom_table = prompt_atom_table;
    if (prompt_token_to_slot_map.size() != prompt_tokens.size()) {
        throw std::runtime_error("Phase2 payload inference: token_to_slot_map length mismatch");
    }
    sequence.token_to_slot_map = prompt_token_to_slot_map;
    if (prompt_atom_entry_ids.size() != prompt_tokens.size()) {
        throw std::runtime_error("Phase2 payload inference: atom_entry_ids length mismatch");
    }
    sequence.atom_entry_ids = prompt_atom_entry_ids;

    if (!use_gpu) {
        throw std::runtime_error("Phase2 payload inference requires config.use_gpu=true");
    }

    if (prompt_tokens.size() >= static_cast<size_t>(max_seq_len)) {
        throw std::runtime_error("Phase2 payload inference: prompt length " +
                                 std::to_string(prompt_tokens.size()) + " exceeds max_seq_len " +
                                 std::to_string(max_seq_len));
    }
    if (prompt_numeric_values.size() != prompt_tokens.size() ||
        prompt_atom_mask.size() != prompt_tokens.size() ||
        prompt_atom_flags.size() != prompt_tokens.size()) {
        throw std::runtime_error("Phase2 payload inference: side-channel length mismatch");
    }
    std::vector<uint32_t> sequence_atom_flags = prompt_atom_flags;

    if (cfg.max_new_tokens <= 0) {
        throw std::runtime_error("Phase2 payload inference: max_new_tokens must be > 0");
    }
    if (cfg.min_new_tokens < 0) {
        throw std::runtime_error("Phase2 payload inference: min_new_tokens must be non-negative");
    }
    if (cfg.min_new_tokens > cfg.max_new_tokens) {
        throw std::runtime_error("Phase2 payload inference: min_new_tokens exceeds max_new_tokens");
    }
    if (vocab_size <= 0) {
        throw std::runtime_error("Phase2 payload inference: invalid vocab_size");
    }
    const int learned_piece_count = vocab_size - GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET;
    if (learned_piece_count < 0) {
        throw std::runtime_error("Phase2 payload inference: vocab_size=" + std::to_string(vocab_size) +
                                 " is smaller than fixed tokenizer offset=" +
                                 std::to_string(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET));
    }

    GRIM::Tokenizer::TokenLayout token_layout;
    token_layout.num_special = GRIM::Tokenizer::NUM_SPECIAL_TOKENS;
    token_layout.num_bytes = GRIM::Tokenizer::BYTE_VOCAB_SIZE;
    token_layout.num_atoms = GRIM::Tokenizer::ATOM_VOCAB_SIZE;
    token_layout.num_unigram = learned_piece_count;
    if (cfg.strategy == SamplingStrategy::BEAM_SEARCH) {
        throw std::runtime_error("Phase2 payload inference: BEAM_SEARCH is not supported");
    }

    GRIM::Sampling::SamplingConfig sampling_cfg = GRIM::Sampling::buildSamplingConfigFromGenerationFields(
        static_cast<int>(cfg.strategy),
        cfg.do_sample,
        cfg.temperature,
        cfg.top_k,
        cfg.top_p,
        cfg.min_p,
        cfg.typical_p,
        cfg.repetition_penalty,
        cfg.repetition_penalty_window,
        cfg.frequency_penalty,
        cfg.presence_penalty,
        cfg.no_repeat_ngram_size,
        cfg.eos_token_id,
        cfg.bos_token_id,
        cfg.pad_token_id,
        cfg.unk_token_id,
        cfg.bad_words_ids,
        cfg.seed);

    if (execution_block_enabled) {
        std::vector<int> numeric_mask = cfg.masked_numeric_literal_ids;
        if (numeric_mask.empty()) {
            throw std::runtime_error("Phase2 payload inference: masked_numeric_literal_ids is empty while execution block is enabled");
        }
        sampling_cfg.bad_token_ids.insert(sampling_cfg.bad_token_ids.end(),
                                          numeric_mask.begin(), numeric_mask.end());
        std::sort(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end());
        sampling_cfg.bad_token_ids.erase(
            std::unique(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end()),
            sampling_cfg.bad_token_ids.end());
    }

    auto& generation_state = model.getGenerationState();
    generation_state.resetSession();

    const auto scratch_hp = GRIM::HyperParameters::scratchBlockConstructionHP(config);
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(config);
    const auto selector_hp = GRIM::HyperParameters::decodeTimeSelectorConstructionHP(config);
    const auto mtp_hp = GRIM::HyperParameters::mtpFeatureHP(config);
    auto* decode_time_slot_selector = parameter_registry.getDecodeTimeSlotSelector();
    GRIM::ScratchBlockLayer* scratch_block_layer = model.getScratchBlockLayer();
    if (scratch_hp.enabled && !scratch_block_layer) {
        throw std::runtime_error("Phase2 payload inference: ScratchBlockConstructionHP.enabled=true but ScratchBlock layer is NULL");
    }
    if (!scratch_hp.enabled && scratch_block_layer) {
        throw std::runtime_error("Phase2 payload inference: ScratchBlock layer exists while ScratchBlockConstructionHP.enabled=false");
    }

    const bool selector_active = selector_hp.enabled
        && decode_time_slot_selector != nullptr
        && execution_hp.enabled
        && scratch_hp.enabled
        && scratch_block_layer != nullptr
        && generation_state.has_exec_memory;
    if (!selector_active) {
        const bool scratchblock_generation_active = cfg.enable_scratchblock_reasoning &&
                                                    scratch_hp.enabled;
        if (scratchblock_generation_active) {
            const int int_tid = GRIM::Tokenizer::atomTypeToTokenId(GRIM::Tokenizer::AtomType::ATOM_INT);
            const int float_tid = GRIM::Tokenizer::atomTypeToTokenId(GRIM::Tokenizer::AtomType::ATOM_FLOAT);
            sampling_cfg.bad_token_ids.push_back(int_tid);
            sampling_cfg.bad_token_ids.push_back(float_tid);
            std::sort(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end());
            sampling_cfg.bad_token_ids.erase(
                std::unique(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end()),
                sampling_cfg.bad_token_ids.end());
        }
    }

    GRIM::Sampling::SamplingPipeline pipeline(sampling_cfg);

    const bool scratchblock_active = cfg.enable_scratchblock_reasoning &&
                                     scratch_hp.enabled;

    auto runSharedForwardForCurrentSequence = [&](GRIM::Batching::BatchPayload& active_payload)
        -> std::vector<float> {
        validateInferenceForwardPayload(model, active_payload, "generateOneSequence");

        auto& training_state = model.getTrainingState();
        if (!training_state.initialized) {
            throw std::runtime_error("generateOneSequence: training state not initialized");
        }
        cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
        const auto bindings = GRIM::Batching::uploadBatchToDevice(
            model.getConfig(),
            active_payload,
            stream);

        GRIM::Forward::ModelForwardRuntimePayload runtime_payload{};
        runtime_payload.execution_runtime = &generation_state.execution_runtime;
        runtime_payload.read_gate_accum_tensor = nullptr;

        auto* gpu_encoder = gpu_model_state.gpu_encoder.get();
        if (!gpu_encoder) {
            throw std::runtime_error(
                "Phase2::generateOneSequence: gpu_model_state.gpu_encoder is NULL");
        }

        GRIM::Forward::ModelForwardRequest request{};
        request.config = &config;
        request.gpu_encoder = gpu_encoder;
        request.parameter_registry = &parameter_registry;
        request.pbm = &pbm;
        request.embedding_layer = model.getEmbeddingLayer();
        request.lm_head = model.getLmHeadLayer();
        request.scratch_block = scratch_hp.enabled ? scratch_block_layer : nullptr;
        request.execution_block = model.getExecutionBlockLayer();
        request.cublas_handle = training_state.cublas_handle.get();
        request.stream = stream;
        request.payload = &active_payload;
        request.bindings = &bindings;
        request.batch_idx = 0;
        const bool emit_mtp_logits = mtp_hp.enabled && mtp_hp.k > 0;
        if (emit_mtp_logits) {
            request.mtp_heads = buildDetachedForwardMtpHeadViews(parameter_registry, mtp_hp, stream);
        }
        request.graph = GRIM::Forward::ModelForwardGraphPolicy{
            /*connect_parameter_graph=*/false,
            /*retain_backward_graph=*/false,
            /*enable_dropout=*/false,
            /*emit_mtp_logits=*/emit_mtp_logits};

        auto forward_outputs = GRIM::Forward::executeModelForward(request, runtime_payload);
        InferenceForwardScope inference_forward_scope{forward_outputs};

        generation_state.decode_selector.reset();
        if (selector_active) {
            const GRIM::Tensor* live_lm_head_input = forward_outputs.liveLmHeadInputOrNull();
            if (!live_lm_head_input || !live_lm_head_input->data) {
                throw std::runtime_error(
                    "generateOneSequence: selector_active but live LM-head input tensor is NULL after shared forward");
            }
            const auto& live_shape = live_lm_head_input->shape.require(
                "generateOneSequence selector live_lm_head_input").as_2d();
            if (live_shape.rows < active_payload.total_tokens) {
                throw std::runtime_error(
                    "generateOneSequence: selector live_lm_head_input rows=" +
                    std::to_string(live_shape.rows) + " < payload.total_tokens=" +
                    std::to_string(active_payload.total_tokens));
            }
            if (live_shape.cols != selector_hp.d_model) {
                throw std::runtime_error(
                    "generateOneSequence: selector live_lm_head_input cols=" +
                    std::to_string(live_shape.cols) + " != selector_hp.d_model=" +
                    std::to_string(selector_hp.d_model));
            }

            const float* d_last_hidden_state = live_lm_head_input->data +
                static_cast<size_t>(active_payload.total_tokens - 1) *
                static_cast<size_t>(selector_hp.d_model);
            generation_state.decode_selector = GRIM::resolveDecodeTimeNumSlotSelectionOrMask(
                decode_time_slot_selector,
                selector_hp,
                generation_state.execution_runtime.decode_time_selector_runtime,
                selector_hp.enabled,
                execution_hp.enabled,
                generation_state.has_exec_memory,
                generation_state.exec_memory,
                d_last_hidden_state,
                stream,
                training_state.cublas_handle.get());
        }

        const auto& live_logits = forward_outputs.logits_tensor;
        if (!live_logits.data) {
            throw std::runtime_error(
            "generateOneSequence: live logits tensor is NULL after shared forward");
        }
        if (emit_mtp_logits && forward_outputs.mtp_logits_tensors.size()
                != static_cast<size_t>(mtp_hp.k)) {
            throw std::runtime_error(
            "generateOneSequence: live mtp_logits_tensors.size()=" +
            std::to_string(forward_outputs.mtp_logits_tensors.size()) +
                " != mtp_hp.k=" + std::to_string(mtp_hp.k) +
                " after shared forward");
        }

        const size_t expected_logits =
            static_cast<size_t>(active_payload.total_tokens) * static_cast<size_t>(vocab_size);
        if (static_cast<size_t>(live_logits.numel()) < expected_logits) {
            throw std::runtime_error(
                "generateOneSequence: live logits numel=" +
                std::to_string(live_logits.numel()) +
                " < payload.total_tokens * config.vocab_size=" + std::to_string(expected_logits));
        }

        std::vector<float> logits(static_cast<size_t>(vocab_size));
        const size_t last_token_offset =
            static_cast<size_t>(active_payload.total_tokens - 1) * static_cast<size_t>(vocab_size);
        cudaError_t copy_err = cudaMemcpyAsync(
            logits.data(),
            live_logits.data + last_token_offset,
            static_cast<size_t>(vocab_size) * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error("generateOneSequence: cudaMemcpyAsync logits failed: " +
                                     std::string(cudaGetErrorString(copy_err)));
        }

        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error("generateOneSequence: cudaStreamSynchronize failed: " +
                                     std::string(cudaGetErrorString(sync_err)));
        }

        return logits;
    };

    std::vector<float> logits_vec = runSharedForwardForCurrentSequence(prompt_payload);
    if (logits_vec.empty()) {
        throw std::runtime_error("Phase2 payload inference: shared forward returned empty first-token logits");
    }

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        const int current_len = static_cast<int>(sequence.token_ids.size());
        if (current_len >= max_seq_len) {
            sequence.finished = true;
            break;
        }

        if (selector_active) {
            if (!generation_state.decode_selector.valid) {
                std::fprintf(stderr,
                    "[Selector Debug] step=%d selector_enabled=%d selectorLayer=%d "
                    "exec_block_enabled=%d scratchLayer=%d scratchConfigured=%d has_exec_mem=%d "
                    "decode_selector_valid=%d\n",
                    step,
                    static_cast<int>(selector_hp.enabled),
                    static_cast<int>(decode_time_slot_selector != nullptr),
                    static_cast<int>(execution_hp.enabled),
                    static_cast<int>(scratch_block_layer != nullptr),
                    static_cast<int>(scratch_hp.enabled),
                    static_cast<int>(generation_state.has_exec_memory),
                    static_cast<int>(generation_state.decode_selector.valid));
                throw std::runtime_error(
                    "Phase2 payload inference: selector_active but decode_selector_valid is false at step " +
                    std::to_string(step));
            }
            const int int_tid = GRIM::Tokenizer::atomTypeToTokenId(GRIM::Tokenizer::AtomType::ATOM_INT);
            const int float_tid = GRIM::Tokenizer::atomTypeToTokenId(GRIM::Tokenizer::AtomType::ATOM_FLOAT);
            if (static_cast<int>(generation_state.decode_selector.status)
                != static_cast<int>(GRIM::SlotSelectionStatus::Selected)) {
                logits_vec[int_tid] = -1e30f;
                logits_vec[float_tid] = -1e30f;
            }
        }

        GRIM::Sampling::SampleResult sample = pipeline.selectNextToken(
            logits_vec, sequence.token_ids, vocab_size);

        if (sample.token_id < 0 || sample.token_id >= vocab_size) {
            throw std::runtime_error("Phase2 payload inference: sampled token out of range (token_id=" +
                                     std::to_string(sample.token_id) + ", vocab=" +
                                     std::to_string(vocab_size) + ")");
        }

        float token_numeric_value = 0.0f;
        uint8_t token_atom_mask_val = 0;
        int32_t new_token_slot_id = -1;

        if (scratchblock_active && token_layout.isAtom(sample.token_id)) {
            token_atom_mask_val = 1;
            if (GRIM::Tokenizer::isNumericAtom(
                    GRIM::Tokenizer::tokenIdToAtomType(sample.token_id))) {
                if (!selector_active || !generation_state.decode_selector.valid
                          || static_cast<int>(generation_state.decode_selector.status)
                              != static_cast<int>(GRIM::SlotSelectionStatus::Selected)) {
                    throw std::runtime_error(
                        "Phase2 payload inference: sampled numeric atom but selector did not resolve a slot "
                        "(status=" + std::to_string(static_cast<int>(generation_state.decode_selector.status)) + ")");
                }
                new_token_slot_id = generation_state.decode_selector.selected_slot;
                token_numeric_value = generation_state.decode_selector.selected_value;
            }
        }

        if (sample.token_id == cfg.eos_token_id &&
            step + 1 >= cfg.min_new_tokens) {
            sequence.token_ids.push_back(sample.token_id);
            sequence.token_scores.push_back(sample.log_probability);
            sequence.token_numeric_values.push_back(0.0f);
            sequence.token_atom_mask.push_back(0);
            sequence.token_to_slot_map.push_back(-1);
            sequence.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
            sequence.score += sample.log_probability;
            sequence.finished = true;
            if (stream_callback) {
                (*stream_callback)(sample.token_id, sample.probability);
            }
            break;
        }

        std::vector<int> next_token_ids = sequence.token_ids;
        std::vector<float> next_numeric_values = sequence.token_numeric_values;
        std::vector<uint8_t> next_atom_mask = sequence.token_atom_mask;
        std::vector<uint32_t> next_atom_flags = sequence_atom_flags;
        std::vector<int32_t> next_token_to_slot_map = sequence.token_to_slot_map;
        std::vector<uint32_t> next_atom_entry_ids = sequence.atom_entry_ids;

        next_token_ids.push_back(sample.token_id);
        next_numeric_values.push_back(token_numeric_value);
        next_atom_mask.push_back(token_atom_mask_val);
        next_atom_flags.push_back(0);
        next_token_to_slot_map.push_back(new_token_slot_id);
        next_atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);

        GRIM::Batching::BatchPayload step_payload = GRIM::Batching::buildInferenceBatchPayload(
            next_token_ids,
            next_numeric_values,
            next_atom_mask,
            next_atom_flags,
            prompt_atom_table,
            next_atom_entry_ids,
            next_token_to_slot_map,
            vocab_size,
            /*batch_capacity=*/1,
            static_cast<size_t>(max_seq_len),
            execution_hp.num_slots);

        logits_vec = runSharedForwardForCurrentSequence(step_payload);
        if (logits_vec.empty()) {
            throw std::runtime_error("Phase2 payload inference: shared forward returned empty logits");
        }

        sequence.token_ids.push_back(sample.token_id);
        sequence.token_scores.push_back(sample.log_probability);
        sequence.token_numeric_values.push_back(token_numeric_value);
        sequence.token_atom_mask.push_back(token_atom_mask_val);
        sequence.token_to_slot_map.push_back(new_token_slot_id);
        sequence.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
        sequence_atom_flags.push_back(0);
        sequence.score += sample.log_probability;

        if (stream_callback) {
            (*stream_callback)(sample.token_id, sample.probability);
        }
    }

    if (!sequence.finished) {
        sequence.finished = true;
    }

    return sequence;
}

std::vector<GRIM::GeneratedSequence> generatePayloadSequences(
    TrainingContext& ctx,
    GRIM::Batching::BatchPayload& prompt_payload,
    const GRIM::HyperParameters::GenerationHP& generation_hp,
    GRIM::HyperParameters::GenerationStreamCallback* stream_callback)
{
    validateInferenceContext(ctx);
    validatePromptPayload(prompt_payload);
    if (generation_hp.num_return_sequences <= 0) {
        throw std::runtime_error("Phase2 payload inference: num_return_sequences must be > 0");
    }
    if (stream_callback && generation_hp.num_return_sequences != 1) {
        throw std::runtime_error("Phase2 payload inference: streaming requires num_return_sequences == 1");
    }

    std::vector<GRIM::GeneratedSequence> outputs;
    outputs.reserve(static_cast<size_t>(generation_hp.num_return_sequences));
    for (int i = 0; i < generation_hp.num_return_sequences; ++i) {
        GenerationHP sequence_hp = generation_hp;
        if (sequence_hp.seed != 0) {
            sequence_hp.seed += static_cast<unsigned int>(i);
        }
        outputs.push_back(generateOneSequence(*ctx.model, ctx.gpu_model, ctx.parameter_registry, ctx.pbm_owner.state(), prompt_payload, sequence_hp, stream_callback));

    }
    return outputs;
}

} // namespace

Phase2TextInferenceResult executePhase2TextInference(
    TrainingContext& ctx,
    const std::string& prompt,
    const GRIM::HyperParameters::GenerationHP& generation_hp)
{
    validateInferenceContext(ctx);
    if (prompt.empty()) {
        throw std::runtime_error("executePhase2TextInference: prompt is empty");
    }

    auto& tokenizer = *ctx.tokenizer;
    const auto& model_config = ctx.config;
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "vocab_size");
    const int batch_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "batch_size");
    const int max_cached_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "max_cached_seq_len");
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(model_config);

    Phase2TextInferenceResult result;

    const auto start_encode = std::chrono::high_resolution_clock::now();
    auto encoded = tokenizer.tokenizeWithMetadata(prompt);
    auto tokens = std::move(encoded.token_ids);
    auto numeric_values = std::move(encoded.token_numeric_values);
    auto atom_mask = std::move(encoded.token_atom_mask);
    auto atom_flags = std::move(encoded.token_atom_flags);
    auto prompt_atom_table = encoded.atom_table;
    auto atom_entry_ids = std::move(encoded.atom_entry_ids);
    const auto end_encode = std::chrono::high_resolution_clock::now();
    result.encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_encode - start_encode).count();

    const int eos_id = GRIM::Tokenizer::EOS_TOKEN_ID;
    if (!tokens.empty() && tokens.back() == eos_id) {
        tokens.pop_back();
        if (numeric_values.empty()) {
            throw std::runtime_error("executePhase2TextInference: numeric_values empty while removing EOS");
        }
        numeric_values.pop_back();
        if (atom_mask.empty()) {
            throw std::runtime_error("executePhase2TextInference: atom_mask empty while removing EOS");
        }
        atom_mask.pop_back();
        if (atom_flags.empty()) {
            throw std::runtime_error("executePhase2TextInference: atom_flags empty while removing EOS");
        }
        atom_flags.pop_back();
        if (atom_entry_ids.empty()) {
            throw std::runtime_error("executePhase2TextInference: atom_entry_ids empty while removing EOS");
        }
        atom_entry_ids.pop_back();
    }

    result.prompt_token_count = tokens.size();

    const std::vector<int32_t> prompt_token_to_slot_map;
    auto prompt_payload = GRIM::Batching::buildInferenceBatchPayload(
        tokens,
        numeric_values,
        atom_mask,
        atom_flags,
        prompt_atom_table,
        atom_entry_ids,
        prompt_token_to_slot_map,
        vocab_size,
        static_cast<size_t>(batch_size),
        static_cast<size_t>(max_cached_seq_len),
        execution_hp.num_slots);

    const auto start_generation = std::chrono::high_resolution_clock::now();
    auto generated = generatePayloadSequences(ctx, prompt_payload, generation_hp, nullptr);
    const auto end_generation = std::chrono::high_resolution_clock::now();
    result.generation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_generation - start_generation).count();

    if (generated.empty()) {
        throw std::runtime_error("executePhase2TextInference: payload generation returned no sequences");
    }

    const auto start_decode = std::chrono::high_resolution_clock::now();
    const auto& sequence = generated[0];
    result.text = tokenizer.decode(GRIM::Tokenizer::DecodeRequest(
        sequence.token_ids,
        sequence.atom_entry_ids,
        sequence.context_atom_table.get(),
        sequence.token_numeric_values,
        sequence.token_atom_mask));
    const auto end_decode = std::chrono::high_resolution_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_decode - start_decode).count();
    result.sequence_token_count = sequence.token_ids.size();

    return result;
}

} // namespace GRIMText::Training
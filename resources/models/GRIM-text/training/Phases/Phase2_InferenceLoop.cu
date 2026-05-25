//======================================================//
//  Phase2_InferenceLoop.cu
//  Session-time inference orchestration.
//======================================================//

#include "Phase2_InferenceLoop.hpp"

#include "../../Shared/Sampling/Sampling.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/Execution/DecodeTimeNumPolicy.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

namespace GRIMText::Training {

namespace {

using GRIM::HyperParameters::GenerationHP;
using GRIM::HyperParameters::GenerationStreamCallback;
using GRIM::HyperParameters::SamplingStrategy;

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

GRIM::GeneratedSequence generateOneSequence(
    GRIM::LanguageModel& model,
    const GRIM::Batching::BatchPayload& prompt_payload,
    const GenerationHP& cfg,
    GenerationStreamCallback* stream_callback)
{
    validatePromptPayload(prompt_payload);

    const auto& config = model.getConfig();
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

    if (!config.use_gpu) {
        throw std::runtime_error("Phase2 payload inference requires config.use_gpu=true");
    }

    if (prompt_tokens.size() >= static_cast<size_t>(config.max_seq_len)) {
        throw std::runtime_error("Phase2 payload inference: prompt length " +
                                 std::to_string(prompt_tokens.size()) + " exceeds max_seq_len " +
                                 std::to_string(config.max_seq_len));
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
    if (config.vocab_size <= 0) {
        throw std::runtime_error("Phase2 payload inference: invalid vocab_size");
    }
    const int vocab_size = config.vocab_size;
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

    if (config.execution_block_enabled) {
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
    const bool selector_active = config.selector_enabled
        && model.getDecodeTimeSlotSelectorLayer() != nullptr
        && model.getDecodeTimeNumPolicy() != nullptr
        && config.execution_block_enabled
        && model.getScratchBlockLayer() != nullptr
        && model.isScratchBlockEnabled()
        && generation_state.has_exec_memory;
    if (!selector_active) {
        const bool scratchblock_generation_active = cfg.enable_scratchblock_reasoning &&
                                                    config.use_scratch_block &&
                                                    model.isScratchBlockEnabled();
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
                                     config.use_scratch_block &&
                                     model.isScratchBlockEnabled();

    GRIM::Vector logits_vec = model.getNextTokenLogits(prompt_payload);
    if (logits_vec.data.empty()) {
        throw std::runtime_error("Phase2 payload inference: getNextTokenLogits returned empty first-token logits");
    }

    for (int step = 0; step < cfg.max_new_tokens; ++step) {
        const int current_len = static_cast<int>(sequence.token_ids.size());
        if (current_len >= config.max_seq_len) {
            sequence.finished = true;
            break;
        }

        if (selector_active) {
            if (!generation_state.decode_selector.valid) {
                std::fprintf(stderr,
                    "[Selector Debug] step=%d selector_enabled=%d selectorLayer=%d numPolicy=%d "
                    "exec_block_enabled=%d scratchLayer=%d scratchEnabled=%d has_exec_mem=%d "
                    "decode_selector_valid=%d\n",
                    step,
                    static_cast<int>(config.selector_enabled),
                    static_cast<int>(model.getDecodeTimeSlotSelectorLayer() != nullptr),
                    static_cast<int>(model.getDecodeTimeNumPolicy() != nullptr),
                    static_cast<int>(config.execution_block_enabled),
                    static_cast<int>(model.getScratchBlockLayer() != nullptr),
                    static_cast<int>(model.isScratchBlockEnabled()),
                    static_cast<int>(generation_state.has_exec_memory),
                    static_cast<int>(generation_state.decode_selector.valid));
                throw std::runtime_error(
                    "Phase2 payload inference: selector_active but decode_selector_valid is false at step " +
                    std::to_string(step));
            }
            const int int_tid = GRIM::Tokenizer::atomTypeToTokenId(GRIM::Tokenizer::AtomType::ATOM_INT);
            const int float_tid = GRIM::Tokenizer::atomTypeToTokenId(GRIM::Tokenizer::AtomType::ATOM_FLOAT);
            if (generation_state.decode_selector.status
                != static_cast<uint8_t>(GRIM::SlotSelectionStatus::Selected)) {
                logits_vec.data[int_tid] = -1e30f;
                logits_vec.data[float_tid] = -1e30f;
            }
        }

        GRIM::Sampling::SampleResult sample = pipeline.selectNextToken(
            logits_vec.data, sequence.token_ids, vocab_size);

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
                    || generation_state.decode_selector.status
                       != static_cast<uint8_t>(GRIM::SlotSelectionStatus::Selected)) {
                    throw std::runtime_error(
                        "Phase2 payload inference: sampled numeric atom but selector did not resolve a slot "
                        "(status=" + std::to_string(generation_state.decode_selector.status) + ")");
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
            config.vocab_size,
            static_cast<size_t>(config.batch_size),
            static_cast<size_t>(config.max_cached_seq_len),
            config.execution_block_num_slots);

        logits_vec = model.getNextTokenLogits(step_payload);
        if (logits_vec.data.empty()) {
            throw std::runtime_error("Phase2 payload inference: getNextTokenLogits returned empty logits");
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
    const GRIM::Batching::BatchPayload& prompt_payload,
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
        outputs.push_back(generateOneSequence(*ctx.model, prompt_payload, sequence_hp, stream_callback));
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
        model_config.vocab_size,
        static_cast<size_t>(model_config.batch_size),
        static_cast<size_t>(model_config.max_cached_seq_len),
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
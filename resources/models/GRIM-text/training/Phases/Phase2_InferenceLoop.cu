//======================================================//
//  Phase2_InferenceLoop.cu
//  Session-time inference orchestration.
//======================================================//

#include "Phase2_InferenceLoop.hpp"

#include "../../Shared/Batching/BatchDeviceUpload.hpp"
#include "Shared/Forward/GeneratedSequence.hpp"
#include "../../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../../Shared/Forward/ModelForward_GPU.hpp"
#include "../../Shared/InferenceState/KvCacheState_GPU.hpp"
#include "../../Shared/Sampling/Sampling.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/UnigramByte/TokenLayout.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace GRIMText::Training {

namespace {

using GRIM::HyperParameters::GenerationHP;
using GRIM::HyperParameters::SamplingStrategy;

struct InferenceForwardScope {
    GRIM::Forward::ModelForwardOutputs& forward_outputs;
    ~InferenceForwardScope() {
        forward_outputs.clear();
    }
};

std::shared_ptr<GRIM::Batching::BatchDeviceStorage> ensureInferenceDeviceStorage(
    GRIM::Batching::BatchPayload& payload,
    const GRIM::Config::AiConfigSnapshot& config,
    cudaStream_t stream,
    std::shared_ptr<GRIM::Batching::BatchDeviceStorage> shared_storage,
    const char* caller)
{
    if (payload.device_storage) {
        return payload.device_storage;
    }
    if (!shared_storage) {
        shared_storage = GRIM::Batching::createBatchDeviceStorage(config, stream);
    }
    GRIM::Batching::attachBatchDeviceStorage(payload, std::move(shared_storage), caller);
    return payload.device_storage;
}

void validateInferenceContext(const TrainingContext& ctx) {
    if (!ctx.model) {
        throw std::runtime_error("Phase2 payload inference: Phase1 context model is NULL");
    }
    if (!ctx.training_state) {
        throw std::runtime_error("Phase2 payload inference: Phase1 context training_state is NULL");
    }
    if (!ctx.generation_state) {
        throw std::runtime_error("Phase2 payload inference: Phase1 context generation_state is NULL");
    }
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(ctx.config, "vocab_size");
    const auto layout = GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
        static_cast<std::uint32_t>(vocab_size),
        "validateInferenceContext");
    if (layout.total_vocab() != vocab_size) {
        throw std::runtime_error(
            "Phase2 payload inference: actual_vocab_size-derived token layout=" +
            std::to_string(layout.total_vocab()) +
            " != config.vocab_size=" + std::to_string(vocab_size));
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
    if (!active_payload.isInference()) {
        throw std::runtime_error(std::string(caller) + ": payload must be an inference payload");
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

GRIM::GeneratedSequence generateOneSequence(
    GRIM::LanguageModel& model,
    GRIM::TrainingState& training_state,
    GRIM::GenerationState& generation_state,
    GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::PBM::PBMState& pbm,
    GRIM::Batching::BatchPayload& prompt_payload,
    const GenerationHP& cfg,
    GRIM::HyperParameters::GenerationStreamCallback* stream_callback)
{
    validatePromptPayload(prompt_payload);

    const auto& config = model.getConfig();
    const bool use_gpu = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_gpu");
    const int max_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "max_seq_len");
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "vocab_size");
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(config);
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
    const auto token_layout = GRIM::Tokenizer::tokenLayoutFromActualVocabOrThrow(
        static_cast<std::uint32_t>(vocab_size),
        "generateOneSequence");
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

    if (execution_hp.enabled) {
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

    generation_state.resetSession();

    const auto number_encoder_hp = GRIM::HyperParameters::numberEncoderConstructionHP(config);

    // The execution-entangled decode-time slot selector was deleted
    // (docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md, Workstream 1). Until the
    // numeric-meaning selector lands, numeric atom placeholders are NEVER
    // sampleable while the execution block is enabled: there is no owner that
    // can bind a value to a generated <INT>/<FLOAT> slot.
    if (execution_hp.enabled) {
        const int int_tid = GRIM::Tokenizer::atomTypeToTokenId(GRIM::Tokenizer::AtomType::ATOM_INT);
        const int float_tid = GRIM::Tokenizer::atomTypeToTokenId(GRIM::Tokenizer::AtomType::ATOM_FLOAT);
        sampling_cfg.bad_token_ids.push_back(int_tid);
        sampling_cfg.bad_token_ids.push_back(float_tid);
        std::sort(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end());
        sampling_cfg.bad_token_ids.erase(
            std::unique(sampling_cfg.bad_token_ids.begin(), sampling_cfg.bad_token_ids.end()),
            sampling_cfg.bad_token_ids.end());
    }

    GRIM::Sampling::SamplingPipeline pipeline(sampling_cfg);

    const bool atom_generation_active = execution_hp.enabled;
    // ── KV-cache session setup ───────────────────────────────────────────────
    // Decode runs incrementally: prefill the prompt once (q_len=prompt_len), then
    // decode one token at a time, reusing
    // the per-layer K/V cache. This replaces the previous O(n^2) full-recompute.
    if (!training_state.initialized) {
        throw std::runtime_error("generateOneSequence: training state not initialized");
    }
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    auto* gpu_encoder = gpu_model_state.gpu_encoder.get();
    if (!gpu_encoder) {
        throw std::runtime_error("Phase2::generateOneSequence: gpu_model_state.gpu_encoder is NULL");
    }
    const int num_layers = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "num_layers");
    const int num_heads = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "num_heads");
    const int num_kv_heads = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "num_kv_heads");
    const int d_model = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "d_model");
    const int rotary_dim = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "rotary_dim");
    const int max_cached_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "max_cached_seq_len");
    if (num_heads <= 0 || d_model % num_heads != 0) {
        throw std::runtime_error("generateOneSequence: invalid num_heads/d_model for head_dim derivation");
    }
    const int head_dim = d_model / num_heads;

    GRIM::KvCacheState& kv_cache = generation_state.kv_cache;
    kv_cache.ensureAllocated(num_layers, num_heads, num_kv_heads, head_dim, rotary_dim,
                             max_cached_seq_len, pbm.rope_inv_freq, stream);
    kv_cache.beginSession(stream);

    // Arg/option selector decode bridge: when enabled, a generated numeric-atom
    // placeholder (<INT>/<FLOAT>) is bound to the candidate atom-entry the trained
    // selector picks from this row's pool (prompt atoms).
    const bool use_selector =
        GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "selector_enabled");
    std::shared_ptr<GRIM::Batching::BatchDeviceStorage> inference_device_storage;

    // Host copy of the last n_tail logit rows. Rows are ordered oldest..newest.
    struct TailLogits {
        int n_rows = 0;
        int vocab = 0;
        std::vector<float> primary;            // [n_rows * vocab]
        int num_pool_atoms = 0;                // candidate count for the selector rows
        std::vector<float> selector;           // [n_rows * num_pool_atoms] (empty if no selector)
        const float* primaryRow(int i) const {
            return primary.data() + static_cast<size_t>(i) * static_cast<size_t>(vocab);
        }
        const float* selectorRow(int i) const {
            return selector.data() + static_cast<size_t>(i) * static_cast<size_t>(num_pool_atoms);
        }
    };

    // Run one cached forward over `active_payload` (q_len rows). The shared forward
    // appends this window's K/V to every layer's cache and advances cache_seqlens
    // by q_len. Returns the last n_tail rows of the primary logits.
    auto runCachedForward = [&](GRIM::Batching::BatchPayload& active_payload,
                                int n_tail, bool want_selector) -> TailLogits {
        validateInferenceForwardPayload(model, active_payload, "generateOneSequence");
        const int q_len = active_payload.total_tokens;
        if (n_tail <= 0 || n_tail > q_len) {
            throw std::runtime_error("generateOneSequence: invalid n_tail=" + std::to_string(n_tail) +
                                     " for q_len=" + std::to_string(q_len));
        }
        inference_device_storage = ensureInferenceDeviceStorage(
            active_payload, config, stream, std::move(inference_device_storage), "generateOneSequence");
        const auto bindings = GRIM::Batching::uploadBatchToDevice(model.getConfig(), active_payload, stream);

        GRIM::Forward::ModelForwardRuntimePayload runtime_payload{};
        runtime_payload.execution_runtime = &generation_state.execution_runtime;
        runtime_payload.read_gate_accum_tensor = nullptr;

        GRIM::Forward::ModelForwardRequest request{};
        request.config = &config;
        request.gpu_encoder = gpu_encoder;
        request.parameter_registry = &parameter_registry;
        request.pbm = &pbm;
        request.execution_block_enabled = execution_hp.enabled;
        request.cublas_handle = training_state.cublas_handle.get();
        request.stream = stream;
        request.payload = &active_payload;
        request.bindings = &bindings;
        request.batch_idx = 0;
        request.kv_cache = &kv_cache;
        const bool emit_selector_logits = want_selector && use_selector;
        request.graph = GRIM::Forward::ModelForwardGraphPolicy{
            /*connect_parameter_graph=*/false,
            /*retain_backward_graph=*/false,
            /*enable_dropout=*/false,
            /*emit_selector_logits=*/emit_selector_logits};

        auto forward_outputs = GRIM::Forward::executeModelForward(request, runtime_payload);
        InferenceForwardScope inference_forward_scope{forward_outputs};

        const auto& live_logits = forward_outputs.logits_tensor;
        if (!live_logits.data) {
            throw std::runtime_error("generateOneSequence: live logits tensor is NULL after cached forward");
        }
        const size_t expected_logits =
            static_cast<size_t>(q_len) * static_cast<size_t>(vocab_size);
        if (static_cast<size_t>(live_logits.numel()) < expected_logits) {
            throw std::runtime_error("generateOneSequence: live logits numel=" +
                std::to_string(live_logits.numel()) + " < q_len*vocab=" + std::to_string(expected_logits));
        }
        TailLogits tail;
        tail.n_rows = n_tail;
        tail.vocab = vocab_size;
        const size_t row_bytes = static_cast<size_t>(vocab_size) * sizeof(float);
        const size_t tail_off = static_cast<size_t>(q_len - n_tail) * static_cast<size_t>(vocab_size);
        tail.primary.resize(static_cast<size_t>(n_tail) * static_cast<size_t>(vocab_size));
        cudaError_t copy_err = cudaMemcpyAsync(
            tail.primary.data(), live_logits.data + tail_off,
            static_cast<size_t>(n_tail) * row_bytes, cudaMemcpyDeviceToHost, stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error("generateOneSequence: cudaMemcpyAsync primary logits failed: " +
                                     std::string(cudaGetErrorString(copy_err)));
        }
        if (emit_selector_logits && forward_outputs.selector_logits.data && bindings.num_pool_atoms > 0) {
            tail.num_pool_atoms = bindings.num_pool_atoms;
            const size_t sel_w = static_cast<size_t>(bindings.num_pool_atoms);
            const size_t sel_tail_off = static_cast<size_t>(q_len - n_tail) * sel_w;
            tail.selector.resize(static_cast<size_t>(n_tail) * sel_w);
            cudaError_t scopy = cudaMemcpyAsync(
                tail.selector.data(), forward_outputs.selector_logits.data + sel_tail_off,
                static_cast<size_t>(n_tail) * sel_w * sizeof(float), cudaMemcpyDeviceToHost, stream);
            if (scopy != cudaSuccess) {
                throw std::runtime_error("generateOneSequence: cudaMemcpyAsync selector logits failed: " +
                                         std::string(cudaGetErrorString(scopy)));
            }
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error("generateOneSequence: cudaStreamSynchronize failed: " +
                                     std::string(cudaGetErrorString(sync_err)));
        }
        return tail;
    };

    // Build a decode/verification payload over `feed_tokens`. Generated tokens are
    // plain text (no atoms); execution and number-encoder are disabled on this path.
    auto buildDecodePayload = [&](const std::vector<int>& feed_tokens) -> GRIM::Batching::BatchPayload {
        const int q = static_cast<int>(feed_tokens.size());
        std::vector<float> numeric(static_cast<size_t>(q), 0.0f);
        std::vector<uint8_t> amask(static_cast<size_t>(q), 0);
        std::vector<uint32_t> aflags(static_cast<size_t>(q), 0);
        std::vector<uint32_t> aentry(static_cast<size_t>(q), GRIM::Tokenizer::kAtomEntryNone);
        const std::vector<int32_t> slotmap;  // empty -> no execution-active row
        auto decode_payload = GRIM::Batching::buildInferenceBatchPayload(
            feed_tokens, numeric, amask, aflags, prompt_atom_table, aentry, slotmap,
            vocab_size, /*batch_capacity=*/1, static_cast<size_t>(q),
            execution_hp.num_slots,
            number_encoder_hp.enabled ? number_encoder_hp.max_digit_slots : 0,
            number_encoder_hp.max_abs_pow10);
        decode_payload.mode = GRIM::Batching::BatchPayloadMode::InferenceDecode;
        return decode_payload;
    };

    const int prompt_len = static_cast<int>(prompt_tokens.size());
    auto committedNewTokens = [&]() -> int {
        return static_cast<int>(sequence.token_ids.size()) - prompt_len;
    };

    // Commit one generated token to the output sequence with explicit atom binding.
    auto commitToken = [&](const GRIM::Sampling::SampleResult& s,
                           float numeric_value, uint8_t atom_mask, uint32_t atom_entry_id) {
        sequence.token_ids.push_back(s.token_id);
        sequence.token_scores.push_back(s.log_probability);
        sequence.token_numeric_values.push_back(numeric_value);
        sequence.token_atom_mask.push_back(atom_mask);
        sequence.token_to_slot_map.push_back(-1);
        sequence.atom_entry_ids.push_back(atom_entry_id);
        sequence_atom_flags.push_back(0);
        sequence.score += s.log_probability;
        if (stream_callback) {
            (*stream_callback)(s.token_id, s.probability);
        }
    };

    // Commit a sampled token, binding the arg/option selector's choice when the
    // token is a numeric-atom placeholder and the selector is active. The selector
    // picks the highest-scoring candidate entry in this row's pool; its exact value
    // and entry id are bound so decode() round-trips the chosen number. Non-atoms,
    // and numeric atoms with no candidate pool, commit as plain text (numeric 0).
    auto commitSampled = [&](const GRIM::Sampling::SampleResult& s, const TailLogits& tail, int row) {
        const bool is_numeric_atom = token_layout.isAtom(s.token_id) &&
            GRIM::Tokenizer::isNumericAtom(GRIM::Tokenizer::tokenIdToAtomType(s.token_id));
        if (use_selector && is_numeric_atom && tail.num_pool_atoms > 0 &&
            !tail.selector.empty() && prompt_atom_table) {
            const float* sel_row = tail.selectorRow(row);
            int best = 0;
            float best_score = sel_row[0];
            for (int e = 1; e < tail.num_pool_atoms; ++e) {
                if (sel_row[e] > best_score) { best_score = sel_row[e]; best = e; }
            }
            // batch_size == 1 for inference: row_atom_offset[0] == 0, so the
            // batch-global pool index equals the prompt table's row-local entry id.
            const uint32_t entry_id = static_cast<uint32_t>(best);
            const auto entry = prompt_atom_table->getAtom(entry_id);
            if (entry.has_value()) {
                commitToken(s, entry->numeric_value, /*atom_mask=*/1, entry_id);
                return;
            }
        }
        commitToken(s, 0.0f, /*atom_mask=*/0, GRIM::Tokenizer::kAtomEntryNone);
    };

    // Select the next token from a primary-logit row using the SAME pipeline +
    // pre-min_new_tokens EOS mask as the full-recompute decoder.
    auto selectFrom = [&](const float* logit_row, int committed_new_tokens) -> GRIM::Sampling::SampleResult {
        std::vector<float> row(logit_row, logit_row + vocab_size);
        if (committed_new_tokens + 1 < cfg.min_new_tokens &&
            cfg.eos_token_id >= 0 && cfg.eos_token_id < vocab_size) {
            row[static_cast<size_t>(cfg.eos_token_id)] = -1e30f;
        }
        GRIM::Sampling::SampleResult s = pipeline.selectNextToken(row, sequence.token_ids, vocab_size);
        if (s.token_id < 0 || s.token_id >= vocab_size) {
            throw std::runtime_error("Phase2 payload inference: sampled token out of range (token_id=" +
                                     std::to_string(s.token_id) + ", vocab=" + std::to_string(vocab_size) + ")");
        }
        if (atom_generation_active && token_layout.isAtom(s.token_id) &&
            GRIM::Tokenizer::isNumericAtom(GRIM::Tokenizer::tokenIdToAtomType(s.token_id))) {
            // Numeric atom ids are masked while execution is enabled; sampling one
            // is a pipeline bug (no decode-time slot selector exists).
            throw std::runtime_error("Phase2 payload inference: sampled numeric atom token_id=" +
                std::to_string(s.token_id) + " but numeric placeholders are masked while execution is enabled");
        }
        return s;
    };

    // ── Prefill: populate the cache from the prompt; read the last position. ──
    TailLogits prefill = runCachedForward(
        prompt_payload, /*n_tail=*/1, /*want_selector=*/use_selector);

    bool finished = false;

    // First generated token ("pending"), sampled from the prompt's last position.
    // Its K/V is appended by the next cached forward as window position 0.
    {
        GRIM::Sampling::SampleResult first = selectFrom(prefill.primaryRow(0), committedNewTokens());
        commitSampled(first, prefill, 0);
        if (first.token_id == cfg.eos_token_id && committedNewTokens() >= cfg.min_new_tokens) {
            finished = true;
        }
    }
    // Decode one pending token per cached forward.
    while (!finished) {
        if (committedNewTokens() >= cfg.max_new_tokens) {
            break;
        }
        if (static_cast<int>(sequence.token_ids.size()) >= max_seq_len) {
            break;
        }

        const int pending = sequence.token_ids.back();
        const int cache_base = kv_cache.currentSeqlen();

        std::vector<int> feed{pending};
        GRIM::Batching::BatchPayload decode_payload = buildDecodePayload(feed);
        TailLogits tail = runCachedForward(
            decode_payload, /*n_tail=*/1, /*want_selector=*/use_selector);

        GRIM::Sampling::SampleResult chosen =
            selectFrom(tail.primaryRow(0), committedNewTokens());
        commitSampled(chosen, tail, 0);

        // The token just fed is now cached; the newly sampled token remains
        // pending for the next iteration.
        kv_cache.setSeqlen(cache_base + 1, stream);

        if (chosen.token_id == cfg.eos_token_id &&
            committedNewTokens() >= cfg.min_new_tokens) {
            finished = true;
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
    auto& training_state = ctx.requireTrainingState("generatePayloadSequences");
    auto& generation_state = ctx.requireGenerationState("generatePayloadSequences");
    for (int i = 0; i < generation_hp.num_return_sequences; ++i) {
        GenerationHP sequence_hp = generation_hp;
        if (sequence_hp.seed != 0) {
            sequence_hp.seed += static_cast<unsigned int>(i);
        }
        outputs.push_back(generateOneSequence(*ctx.model, training_state, generation_state, ctx.gpu_model, ctx.parameter_registry, ctx.pbm_owner.state(), prompt_payload, sequence_hp, stream_callback));

    }
    return outputs;
}

} // namespace

Phase2TextInferenceResult executePhase2TextInference(
    TrainingContext& ctx,
    GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt,
    const GRIM::HyperParameters::GenerationHP& generation_hp)
{
    validateInferenceContext(ctx);
    if (prompt.empty()) {
        throw std::runtime_error("executePhase2TextInference: prompt is empty");
    }
    const auto& model_config = ctx.config;
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "vocab_size");
    const int batch_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "batch_size");
    const int max_cached_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "max_cached_seq_len");
    const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(model_config);

    if (tokenizer.vocabSize() != vocab_size) {
        throw std::runtime_error(
            "executePhase2TextInference: tokenizer.vocabSize()=" +
            std::to_string(tokenizer.vocabSize()) +
            " != config.vocab_size=" + std::to_string(vocab_size));
    }

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

    std::vector<int32_t> prompt_token_to_slot_map(tokens.size(), -1);
    if (execution_hp.enabled) {
        const int value_slots = execution_hp.num_slots - execution_hp.num_scratch_slots;
        const int int_token_id = GRIM::Tokenizer::atomTypeToTokenId(
            GRIM::Tokenizer::AtomType::ATOM_INT);
        const int float_token_id = GRIM::Tokenizer::atomTypeToTokenId(
            GRIM::Tokenizer::AtomType::ATOM_FLOAT);
        int next_slot = 0;
        for (size_t t = 0; t < tokens.size() && next_slot < value_slots; ++t) {
            if (tokens[t] == int_token_id || tokens[t] == float_token_id) {
                prompt_token_to_slot_map[t] = next_slot++;
            }
        }
    }
    const auto number_encoder_hp = GRIM::HyperParameters::numberEncoderConstructionHP(model_config);
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
        execution_hp.num_slots,
        number_encoder_hp.enabled ? number_encoder_hp.max_digit_slots : 0,
        number_encoder_hp.max_abs_pow10);

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

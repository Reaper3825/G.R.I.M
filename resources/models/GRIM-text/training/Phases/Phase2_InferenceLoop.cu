//======================================================//
//  Phase2_InferenceLoop.cu
//  Session-time inference orchestration.
//======================================================//

#include "Phase2_InferenceLoop.hpp"

#include "../../Shared/AtomInsertion/AtomInsertionData.hpp"
#include "../../Shared/AtomInsertion/AtomInsertionDecode.hpp"
#include "../../Shared/AtomInsertion/AtomInsertionDecisionLayout.hpp"
#include "../../Shared/Batching/BatchDeviceUpload.hpp"
#include "Shared/Forward/GeneratedSequence.hpp"
#include "../../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../../Shared/Forward/ModelForward_GPU.hpp"
#include "../../Shared/InferenceState/KvCacheState_GPU.hpp"
#include "../../Shared/InferenceState/LocalAtomRetrievalInferenceState.hpp"
#include "../../Shared/LocalAtomRetrieval/LocalAtomRetrievalForward.hpp"
#include "../../Shared/Sampling/Sampling.hpp"
#include "../../Shared/UnigramByte/AtomTable.hpp"
#include "../../Shared/UnigramByte/SequenceLocalAtomTable.hpp"
#include "../../Shared/UnigramByte/TokenLayout.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace GRIMText::Training {

namespace {

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

void validateInferenceContext(const TrainingContext& ctx,
                              bool require_generation_state = true) {
    if (!ctx.training_state) {
        throw std::runtime_error("Phase2 payload inference: Phase1 context training_state is NULL");
    }
    if (require_generation_state && !ctx.generation_state) {
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

void validateInferenceForwardPayload(
    const GRIM::Config::AiConfigSnapshot& config,
    const GRIM::Batching::BatchPayload& active_payload,
    const char* caller);

std::vector<float> runAtomInsertionForward(
    TrainingContext& ctx,
    GRIM::Batching::BatchPayload& payload) {
    constexpr const char* caller = "runAtomInsertionForward";
    validateInferenceForwardPayload(ctx.config, payload, caller);
    if (!payload.EnableAtomIdentification) {
        throw std::runtime_error(
            "runAtomInsertionForward: atom identification is disabled on payload");
    }

    auto& training_state = ctx.requireTrainingState(caller);
    if (!training_state.initialized) {
        throw std::runtime_error(
            "runAtomInsertionForward: training state is not initialized");
    }
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    auto* gpu_encoder = ctx.gpu_model.gpu_encoder.get();
    if (!gpu_encoder) {
        throw std::runtime_error(
            "runAtomInsertionForward: gpu_model.gpu_encoder is NULL");
    }

    (void)ensureInferenceDeviceStorage(
        payload, ctx.config, stream, nullptr, caller);
    const auto bindings =
        GRIM::Batching::uploadBatchToDevice(ctx.config, payload, stream);

    GRIM::Forward::ModelForwardRuntimePayload runtime_payload{};
    GRIM::Forward::ModelForwardRequest request{};
    request.config = &ctx.config;
    request.gpu_encoder = gpu_encoder;
    request.parameter_registry = &ctx.parameter_registry;
    request.pbm = &ctx.pbm_owner.state();
    request.cublas_handle = training_state.cublas_handle.get();
    request.stream = stream;
    request.payload = &payload;
    request.bindings = &bindings;
    request.batch_idx = 0;
    request.kv_cache = nullptr;
    request.graph = GRIM::Forward::ModelForwardGraphPolicy{
        /*connect_parameter_graph=*/false,
        /*enable_dropout=*/false,
        /*emit_local_atom_retrieval=*/false};

    auto forward_outputs =
        GRIM::Forward::executeModelForward(request, runtime_payload);
    InferenceForwardScope inference_forward_scope{forward_outputs};
    const auto& logits = forward_outputs.logits_tensor;
    if (!logits.data) {
        throw std::runtime_error(
            "runAtomInsertionForward: gap logits tensor is NULL");
    }
    const auto& shape = logits.shape.require(
        "runAtomInsertionForward logits_tensor");
    if (!shape.is_2d_layout()) {
        throw std::runtime_error(
            "runAtomInsertionForward: gap logits must be a 2D tensor");
    }
    const auto matrix = shape.as_2d();
    const int gap_rows = payload.atomInsertionGapRowCount();
    if (matrix.rows != gap_rows || matrix.cols != payload.vocab_size) {
        throw std::runtime_error(
            "runAtomInsertionForward: gap-logit shape mismatch");
    }

    constexpr std::size_t decision_count = static_cast<std::size_t>(
        GRIM::AtomInsertion::kAtomDecisionClassCount);
    const std::size_t destination_pitch = decision_count * sizeof(float);
    const std::size_t source_pitch =
        static_cast<std::size_t>(payload.vocab_size) * sizeof(float);
    std::vector<float> decision_logits(
        static_cast<std::size_t>(gap_rows) * decision_count);
    const cudaError_t copy_error = cudaMemcpy2DAsync(
        decision_logits.data(),
        destination_pitch,
        logits.data + GRIM::AtomInsertion::kAtomDecisionVocabColumnOffset,
        source_pitch,
        destination_pitch,
        static_cast<std::size_t>(gap_rows),
        cudaMemcpyDeviceToHost,
        stream);
    if (copy_error != cudaSuccess) {
        throw std::runtime_error(
            "runAtomInsertionForward: delimiter-logit copy failed: " +
            std::string(cudaGetErrorString(copy_error)));
    }
    const cudaError_t sync_error = cudaStreamSynchronize(stream);
    if (sync_error != cudaSuccess) {
        throw std::runtime_error(
            "runAtomInsertionForward: stream synchronization failed: " +
            std::string(cudaGetErrorString(sync_error)));
    }
    return decision_logits;
}

Phase2TextInferenceResult executeAtomInsertionTextInference(
    TrainingContext& ctx,
    const std::string& prompt,
    int vocab_size,
    int max_sequence_capacity) {
    constexpr float decision_logit = 0.0f;
    Phase2TextInferenceResult result;

    const auto start_encode = std::chrono::high_resolution_clock::now();
    auto payload = GRIM::AtomInsertion::buildAtomInsertionInferencePayload(
        prompt,
        vocab_size,
        max_sequence_capacity,
        "executeAtomInsertionTextInference");
    const auto end_encode = std::chrono::high_resolution_clock::now();
    result.encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_encode - start_encode).count();
    result.prompt_token_count =
        static_cast<std::size_t>(payload.total_tokens);

    const auto start_generation = std::chrono::high_resolution_clock::now();
    const std::vector<float> decision_logits =
        runAtomInsertionForward(ctx, payload);
    const auto end_generation = std::chrono::high_resolution_clock::now();
    result.generation_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end_generation - start_generation).count();

    const auto start_decode = std::chrono::high_resolution_clock::now();
    result.text = GRIM::AtomInsertion::decodeAtomDecisionPredictions(
        prompt,
        payload.atom_insertion_valid_gap_mask,
        decision_logits,
        decision_logit);
    const auto end_decode = std::chrono::high_resolution_clock::now();
    result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_decode - start_decode).count();
    result.sequence_token_count =
        static_cast<std::size_t>(payload.total_tokens);
    return result;
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
    const GRIM::Config::AiConfigSnapshot& config,
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
    const GRIM::Config::AiConfigSnapshot& config,
    GRIM::TrainingState& training_state,
    GRIM::GenerationState& generation_state,
    GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::PBM::PBMState& pbm,
    GRIM::Batching::BatchPayload& prompt_payload,
    GRIM::Tokenizer::UniByte& tokenizer,
    const GRIM::HyperParameters::GenerationHP& cfg,
    GRIM::HyperParameters::GenerationStreamCallback* stream_callback)
{
    validatePromptPayload(prompt_payload);
    const auto model_hp = GRIM::HyperParameters::modelHP(config);
    const bool local_atom_retrieval_enabled =
        model_hp.local_atom_retrieval_enabled;

    const bool use_gpu = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(config, "use_gpu");
    const int max_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "max_seq_len");
    const int vocab_size = prompt_payload.vocab_size;
    const auto& prompt_tokens = prompt_payload.input_ids;
    const auto& prompt_numeric_values = prompt_payload.numeric_values;
    const auto& prompt_atom_mask = prompt_payload.atom_mask;
    const auto& prompt_atom_flags = prompt_payload.atom_flags;
    const auto& prompt_token_to_slot_index_map = prompt_payload.token_to_slot_index_map;
    const auto& prompt_atom_entry_ids = prompt_payload.atom_entry_ids;
    const auto& prompt_local_atom_indices =
        prompt_payload.token_local_atom_indices;
    const auto authored_prompt_atom_table = prompt_payload.seq_atom_tables[0];
    auto generation_atom_table = authored_prompt_atom_table
        ? authored_prompt_atom_table->cloneHostForGeneration()
        : std::make_shared<GRIM::Tokenizer::AtomTable>();
    prompt_payload.seq_atom_tables[0] = generation_atom_table;
    const std::shared_ptr<const GRIM::Tokenizer::AtomTable> prompt_atom_table =
        generation_atom_table;
    if (prompt_payload.seq_local_atom_tables.size() != 1 ||
        !prompt_payload.seq_local_atom_tables[0]) {
        throw std::runtime_error(
            "Phase2 payload inference: prompt local atom table is missing");
    }
    auto generation_local_atom_table =
        std::make_shared<GRIM::Tokenizer::SequenceLocalAtomTable>(
            *prompt_payload.seq_local_atom_tables[0]);

    GRIM::GeneratedSequence sequence;
    sequence.token_ids = prompt_tokens;
    sequence.token_numeric_values = prompt_numeric_values;
    sequence.token_atom_mask = prompt_atom_mask;
    sequence.context_atom_table = prompt_atom_table;
    sequence.context_local_atom_table = generation_local_atom_table;
    if (prompt_token_to_slot_index_map.size() != prompt_tokens.size()) {
        throw std::runtime_error("Phase2 payload inference: token_to_slot_index_map length mismatch");
    }
    sequence.token_to_slot_index_map = prompt_token_to_slot_index_map;
    if (prompt_atom_entry_ids.size() != prompt_tokens.size()) {
        throw std::runtime_error("Phase2 payload inference: atom_entry_ids length mismatch");
    }
    sequence.atom_entry_ids = prompt_atom_entry_ids;
    if (prompt_local_atom_indices.size() != prompt_tokens.size()) {
        throw std::runtime_error(
            "Phase2 payload inference: token_local_atom_indices length mismatch");
    }
    sequence.token_local_atom_indices = prompt_local_atom_indices;

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
    if (cfg.strategy == GRIM::HyperParameters::SamplingStrategy::BEAM_SEARCH) {
        throw std::runtime_error("Phase2 payload inference: BEAM_SEARCH is not supported");
    }

    GRIM::Sampling::SamplingConfig sampling_cfg = GRIM::Sampling::buildSamplingConfigFromGenerationFields(
        cfg.strategy,
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

    generation_state.resetSession();

    GRIM::Sampling::SamplingPipeline pipeline(sampling_cfg);
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
    GRIM::LocalAtomRetrievalInferenceState& retrieval_state =
        generation_state.local_atom_retrieval;
    if (local_atom_retrieval_enabled) {
        retrieval_state.ensureAllocated(d_model, max_cached_seq_len, stream);
    }

    std::shared_ptr<GRIM::Batching::BatchDeviceStorage> inference_device_storage;

    // Host copy of the last n_tail logit rows. Rows are ordered oldest..newest.
    struct TailLogits {
        int n_rows = 0;
        int vocab = 0;
        std::vector<float> primary;  // [n_rows * vocab]
        std::vector<float> current_encoder; // Decode row [d_model].
        std::vector<float> local_retrieval; // [NO_REFERENCE, typed candidates].
        const float* primaryRow(int i) const {
            return primary.data() + static_cast<size_t>(i) * static_cast<size_t>(vocab);
        }
    };

    // Run one cached forward over `active_payload` (q_len rows). The shared forward
    // appends this window's K/V to every layer's cache and advances cache_seqlens
    // by q_len. Returns the last n_tail rows of the primary logits.
    auto runCachedForward = [&](GRIM::Batching::BatchPayload& active_payload,
                                int n_tail) -> TailLogits {
        validateInferenceForwardPayload(config, active_payload, "generateOneSequence");
        const int q_len = active_payload.total_tokens;
        if (n_tail <= 0 || n_tail > q_len) {
            throw std::runtime_error("generateOneSequence: invalid n_tail=" + std::to_string(n_tail) +
                                     " for q_len=" + std::to_string(q_len));
        }
        inference_device_storage = ensureInferenceDeviceStorage(
            active_payload, config, stream, std::move(inference_device_storage), "generateOneSequence");
        const auto bindings = GRIM::Batching::uploadBatchToDevice(config, active_payload, stream);

        GRIM::Forward::ModelForwardRuntimePayload runtime_payload{};

        GRIM::Forward::ModelForwardRequest request{};
        request.config = &config;
        request.gpu_encoder = gpu_encoder;
        request.parameter_registry = &parameter_registry;
        request.pbm = &pbm;
        request.cublas_handle = training_state.cublas_handle.get();
        request.stream = stream;
        request.payload = &active_payload;
        request.bindings = &bindings;
        request.batch_idx = 0;
        request.kv_cache = &kv_cache;
        request.graph = GRIM::Forward::ModelForwardGraphPolicy{
            /*connect_parameter_graph=*/false,
            /*enable_dropout=*/false,
            /*emit_local_atom_retrieval=*/
                local_atom_retrieval_enabled &&
                active_payload.isInferencePrefill()};

        auto forward_outputs = GRIM::Forward::executeModelForward(request, runtime_payload);
        InferenceForwardScope inference_forward_scope{forward_outputs};

        if (local_atom_retrieval_enabled &&
            active_payload.isInferencePrefill()) {
            retrieval_state.seedFromPrefill(
                forward_outputs.local_atom_candidate_embeddings,
                active_payload,
                stream);
        } else if (local_atom_retrieval_enabled && q_len == 1 &&
                   GRIM::Tokenizer::isAtomOpenTokenId(
                       active_payload.input_ids[0])) {
            GRIM::LocalAtomRetrieval::LocalAtomRetrievalDecodeForward(
                forward_outputs.encoder_output_tensor,
                parameter_registry,
                retrieval_state,
                GRIM::Tokenizer::tokenIdToAtomType(
                    active_payload.input_ids[0]),
                stream,
                forward_outputs);
        }

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
        if (active_payload.isInferenceDecode()) {
            const auto encoder_shape =
                forward_outputs.encoder_output_tensor.shape.as_2d();
            if (encoder_shape.rows != q_len || encoder_shape.cols != d_model) {
                throw std::runtime_error(
                    "generateOneSequence: decode encoder output shape mismatch");
            }
            tail.current_encoder.resize(static_cast<std::size_t>(d_model));
            copy_err = cudaMemcpyAsync(
                tail.current_encoder.data(),
                forward_outputs.encoder_output_tensor.data +
                    static_cast<std::size_t>(q_len - 1) * d_model,
                static_cast<std::size_t>(d_model) * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream);
            if (copy_err != cudaSuccess) {
                throw std::runtime_error(
                    "generateOneSequence: current encoder row copy failed: " +
                    std::string(cudaGetErrorString(copy_err)));
            }
            if (forward_outputs.local_atom_retrieval_logits.data) {
                const auto retrieval_shape =
                    forward_outputs.local_atom_retrieval_logits.shape.as_2d();
                if (retrieval_shape.rows != 1 || retrieval_shape.cols <= 0) {
                    throw std::runtime_error(
                        "generateOneSequence: decode retrieval logits shape mismatch");
                }
                tail.local_retrieval.resize(
                    static_cast<std::size_t>(retrieval_shape.cols));
                copy_err = cudaMemcpyAsync(
                    tail.local_retrieval.data(),
                    forward_outputs.local_atom_retrieval_logits.data,
                    tail.local_retrieval.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream);
                if (copy_err != cudaSuccess) {
                    throw std::runtime_error(
                        "generateOneSequence: retrieval logits copy failed: " +
                        std::string(cudaGetErrorString(copy_err)));
                }
            }
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error("generateOneSequence: cudaStreamSynchronize failed: " +
                                     std::string(cudaGetErrorString(sync_err)));
        }
        return tail;
    };

    auto buildDecodePayload = [&](const std::vector<int>& feed_tokens)
        -> GRIM::Batching::BatchPayload {
        if (feed_tokens.empty()) {
            throw std::runtime_error("generateOneSequence: decode token window is empty");
        }
        std::vector<float> numeric(feed_tokens.size(), 0.0f);
        std::vector<uint8_t> amask(feed_tokens.size(), 0);
        std::vector<uint32_t> aflags(feed_tokens.size(), 0);
        std::vector<uint32_t> aentry(
            feed_tokens.size(), GRIM::Tokenizer::kAtomEntryNone);
        std::vector<uint32_t> local_indices(
            feed_tokens.size(), GRIM::Tokenizer::kLocalAtomIndexNone);
        auto decode_payload = GRIM::Batching::buildInferenceBatchPayload(
            feed_tokens, numeric, amask, aflags, prompt_atom_table, aentry,
            generation_local_atom_table, local_indices,
            vocab_size, /*batch_capacity=*/1,
            /*max_cached_seq_len=*/feed_tokens.size(),
            local_atom_retrieval_enabled,
            GRIM::Batching::BatchPayloadMode::InferenceDecode);
        return decode_payload;
    };

    const int prompt_len = static_cast<int>(prompt_tokens.size());
    auto committedNewTokens = [&]() -> int {
        return static_cast<int>(sequence.token_ids.size()) - prompt_len;
    };

    auto commitToken = [&](const GRIM::Sampling::SampleResult& s) {
        sequence.token_ids.push_back(s.token_id);
        sequence.token_scores.push_back(s.log_probability);
        sequence.token_numeric_values.push_back(0.0f);
        sequence.token_atom_mask.push_back(0);
        sequence.token_to_slot_index_map.push_back(-1);
        sequence.atom_entry_ids.push_back(GRIM::Tokenizer::kAtomEntryNone);
        sequence.token_local_atom_indices.push_back(
            GRIM::Tokenizer::kLocalAtomIndexNone);
        sequence_atom_flags.push_back(0);
        sequence.score += s.log_probability;
        if (stream_callback) {
            (*stream_callback)(s.token_id, s.probability);
        }
    };

    auto commitForcedToken = [&](int token_id) {
        GRIM::Sampling::SampleResult forced{};
        forced.token_id = token_id;
        forced.probability = 1.0f;
        forced.log_probability = 0.0f;
        commitToken(forced);
    };

    // Select the next token from a primary-logit row using the SAME pipeline +
    // pre-min_new_tokens EOS mask as the full-recompute decoder.
    auto selectFrom = [&](const TailLogits& tail, int tail_row,
                          int committed_new_tokens) -> GRIM::Sampling::SampleResult {
        const float* logit_row = tail.primaryRow(tail_row);
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
        return s;
    };

    auto selectLocalRetrievalSlot = [&]
        (const TailLogits& tail,
         GRIM::Tokenizer::AtomType type,
         int remaining_token_capacity) -> int {
        const int candidate_count = retrieval_state.candidateCount(type);
        if (tail.local_retrieval.size() !=
            static_cast<std::size_t>(candidate_count + 1)) {
            throw std::runtime_error(
                "generateOneSequence: retrieval logit count disagrees with typed candidate bank");
        }
        if (!std::isfinite(tail.local_retrieval[0])) {
            throw std::runtime_error(
                "generateOneSequence: NO_REFERENCE logit is non-finite");
        }

        int best_slot = GRIM::Batching::kLocalAtomNoReferenceTarget;
        float best_logit = tail.local_retrieval[0];
        for (int local_index = 0; local_index < candidate_count;
             ++local_index) {
            const auto& content =
                retrieval_state.candidateContentTokenIds(type, local_index);
            const int replay_token_count =
                static_cast<int>(content.size()) + 1; // matching CLOSE
            if (replay_token_count > remaining_token_capacity) {
                continue;
            }
            const float logit = tail.local_retrieval[
                static_cast<std::size_t>(local_index + 1)];
            if (!std::isfinite(logit)) {
                throw std::runtime_error(
                    "generateOneSequence: candidate retrieval logit is non-finite");
            }
            if (logit > best_logit) {
                best_logit = logit;
                best_slot = local_index + 1;
            }
        }
        return best_slot;
    };

    auto processPendingLocalAtom = [&]
        (const TailLogits& tail, std::size_t pending_sequence_index) {
        if (!local_atom_retrieval_enabled) {
            return;
        }
        if (tail.current_encoder.size() != static_cast<std::size_t>(d_model)) {
            throw std::runtime_error(
                "generateOneSequence: pending token encoder signal is unavailable");
        }
        const int token_id = sequence.token_ids[pending_sequence_index];

        if (retrieval_state.inside_atom) {
            const int expected_close =
                GRIM::Tokenizer::atomTypeToCloseTokenId(
                    retrieval_state.active_type);
            if (retrieval_state.replaying_reference) {
                if (token_id == expected_close) {
                    retrieval_state.clearActiveSpan();
                    return;
                }
                if (GRIM::Tokenizer::isAtomOpenTokenId(token_id) ||
                    GRIM::Tokenizer::isAtomCloseTokenId(token_id)) {
                    throw std::runtime_error(
                        "generateOneSequence: replayed local atom contains an unexpected boundary");
                }
                return;
            }

            if (token_id == expected_close) {
                if (retrieval_state.active_content_count <= 0 ||
                    retrieval_state.active_content_token_ids.empty()) {
                    throw std::runtime_error(
                        "generateOneSequence: generated local atom has no content");
                }
                std::vector<float> mean_embedding(
                    static_cast<std::size_t>(d_model));
                const float inverse_count =
                    1.0f / static_cast<float>(
                        retrieval_state.active_content_count);
                for (int feature = 0; feature < d_model; ++feature) {
                    mean_embedding[static_cast<std::size_t>(feature)] =
                        retrieval_state.active_embedding_sum[
                            static_cast<std::size_t>(feature)] * inverse_count;
                }

                const std::string raw_text = tokenizer.decode(
                    GRIM::Tokenizer::DecodeRequest(
                        retrieval_state.active_content_token_ids));
                const auto address = generation_local_atom_table->ticket(
                    retrieval_state.active_type, raw_text);
                const int local_index = static_cast<int>(address.local_index);
                const int candidate_count = retrieval_state.candidateCount(
                    retrieval_state.active_type);
                if (local_index == candidate_count) {
                    retrieval_state.appendCandidate(
                        retrieval_state.active_type,
                        local_index,
                        mean_embedding,
                        retrieval_state.active_content_token_ids,
                        stream);
                } else if (local_index < 0 || local_index > candidate_count) {
                    throw std::runtime_error(
                        "generateOneSequence: generated local atom table index is not dense");
                }
                sequence.token_local_atom_indices[
                    retrieval_state.active_open_sequence_index] =
                    address.local_index;
                retrieval_state.clearActiveSpan();
                return;
            }

            if (GRIM::Tokenizer::isAtomOpenTokenId(token_id) ||
                GRIM::Tokenizer::isAtomCloseTokenId(token_id)) {
                throw std::runtime_error(
                    "generateOneSequence: generated local atom contains a nested or mismatched boundary");
            }
            if (retrieval_state.active_embedding_sum.empty()) {
                retrieval_state.active_embedding_sum.assign(
                    static_cast<std::size_t>(d_model), 0.0f);
            }
            for (int feature = 0; feature < d_model; ++feature) {
                retrieval_state.active_embedding_sum[
                    static_cast<std::size_t>(feature)] +=
                    tail.current_encoder[static_cast<std::size_t>(feature)];
            }
            retrieval_state.active_content_token_ids.push_back(token_id);
            ++retrieval_state.active_content_count;
            return;
        }

        if (!GRIM::Tokenizer::isAtomOpenTokenId(token_id)) {
            return;
        }
        const auto type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
        const int remaining_token_capacity = std::min(
            cfg.max_new_tokens - committedNewTokens(),
            max_seq_len - static_cast<int>(sequence.token_ids.size()));
        const int selected_slot = selectLocalRetrievalSlot(
            tail, type, remaining_token_capacity);

        retrieval_state.clearActiveSpan();
        retrieval_state.inside_atom = true;
        retrieval_state.active_type = type;
        retrieval_state.active_open_sequence_index = pending_sequence_index;
        if (selected_slot == GRIM::Batching::kLocalAtomNoReferenceTarget) {
            retrieval_state.active_embedding_sum.assign(
                static_cast<std::size_t>(d_model), 0.0f);
            return;
        }

        const int local_index = selected_slot - 1;
        sequence.token_local_atom_indices[pending_sequence_index] =
            static_cast<std::uint32_t>(local_index);
        retrieval_state.replaying_reference = true;
        retrieval_state.forced_token_ids =
            retrieval_state.candidateContentTokenIds(type, local_index);
        retrieval_state.forced_token_ids.push_back(
            GRIM::Tokenizer::atomTypeToCloseTokenId(type));
        retrieval_state.forced_token_cursor = 0;
    };

    // ── Prefill: populate the cache from the prompt; read the last position. ──
    TailLogits prefill = runCachedForward(prompt_payload, /*n_tail=*/1);

    bool finished = false;

    // First generated token ("pending"), sampled from the prompt's last position.
    // Its K/V is appended by the next cached forward as window position 0.
    {
        GRIM::Sampling::SampleResult first = selectFrom(prefill, 0, committedNewTokens());
        commitToken(first);
        if (first.token_id == cfg.eos_token_id && committedNewTokens() >= cfg.min_new_tokens) {
            finished = true;
        }
    }
    // Decode one pending LM token per cached forward.
    while (!finished) {
        const int pending_token = sequence.token_ids.back();
        if (committedNewTokens() >= cfg.max_new_tokens ||
            static_cast<int>(sequence.token_ids.size()) >= max_seq_len) {
            break;
        }

        const int cache_base = kv_cache.currentSeqlen();
        std::vector<int> pending_window{pending_token};
        GRIM::Batching::BatchPayload decode_payload =
            buildDecodePayload(pending_window);
        TailLogits tail = runCachedForward(decode_payload, /*n_tail=*/1);
        kv_cache.setSeqlen(cache_base + 1, stream);

        processPendingLocalAtom(tail, sequence.token_ids.size() - 1);

        GRIM::Sampling::SampleResult chosen{};
        if (local_atom_retrieval_enabled && retrieval_state.hasForcedToken()) {
            chosen.token_id = retrieval_state.popForcedToken();
            chosen.probability = 1.0f;
            chosen.log_probability = 0.0f;
            commitForcedToken(chosen.token_id);
        } else {
            chosen = selectFrom(tail, 0, committedNewTokens());
            commitToken(chosen);
        }

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
    GRIM::Tokenizer::UniByte& tokenizer,
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
        GRIM::HyperParameters::GenerationHP sequence_hp = generation_hp;
        if (sequence_hp.seed != 0) {
            sequence_hp.seed += static_cast<unsigned int>(i);
        }
        outputs.push_back(generateOneSequence(ctx.config, training_state, generation_state, ctx.gpu_model, ctx.parameter_registry, ctx.pbm_owner.state(), prompt_payload, tokenizer, sequence_hp, stream_callback));

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
    if (prompt.empty()) {
        throw std::runtime_error("executePhase2TextInference: prompt is empty");
    }
    const auto& model_config = ctx.config;
    const bool atom_insertion_enabled =
        GRIM::HyperParameters::atomInsertionBoundaryProjectionHP(model_config)
            .enabled;
    const bool local_atom_retrieval_enabled =
        GRIM::HyperParameters::modelHP(model_config)
            .local_atom_retrieval_enabled;
    validateInferenceContext(ctx, !atom_insertion_enabled);
    const int vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "vocab_size");
    const int batch_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "batch_size");
    const int max_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "max_seq_len");
    const int max_cached_seq_len = GRIM::HyperParameters::snapshotTrainingConfigField<int>(model_config, "max_cached_seq_len");

    if (tokenizer.vocabSize() != vocab_size) {
        throw std::runtime_error(
            "executePhase2TextInference: tokenizer.vocabSize()=" +
            std::to_string(tokenizer.vocabSize()) +
            " != config.vocab_size=" + std::to_string(vocab_size));
    }

    if (atom_insertion_enabled) {
        return executeAtomInsertionTextInference(
            ctx,
            prompt,
            vocab_size,
            std::min(max_seq_len, max_cached_seq_len));
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
    auto prompt_local_atom_table = encoded.local_atom_table;
    auto local_atom_indices = std::move(encoded.token_local_atom_indices);
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
        if (local_atom_indices.empty()) {
            throw std::runtime_error(
                "executePhase2TextInference: token_local_atom_indices empty while removing EOS");
        }
        local_atom_indices.pop_back();
    }

    result.prompt_token_count = tokens.size();

    auto prompt_payload = GRIM::Batching::buildInferenceBatchPayload(
        tokens,
        numeric_values,
        atom_mask,
        atom_flags,
        prompt_atom_table,
        atom_entry_ids,
        prompt_local_atom_table,
        local_atom_indices,
        vocab_size,
        static_cast<size_t>(batch_size),
        static_cast<size_t>(max_cached_seq_len),
        local_atom_retrieval_enabled);

    const auto start_generation = std::chrono::high_resolution_clock::now();
    auto generated = generatePayloadSequences(
        ctx, prompt_payload, tokenizer, generation_hp, nullptr);
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

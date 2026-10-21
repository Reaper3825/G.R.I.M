//======================================================//
//  Phase2_InferenceLoop.cu
//  Session-time inference orchestration.
//======================================================//

#include "Phase2_InferenceLoop.hpp"
#include "../../Shared/InferenceState/InferencePrefill.hpp"
#include "../../Shared/InferenceState/InferenceGenerationSession.hpp"
#include "../../Shared/Forward/InferenceResultDecode.hpp"

#include "../../Shared/AtomInsertion/AtomInsertionData.hpp"
#include "../../Shared/AtomInsertion/AtomInsertionDecode.hpp"
#include "../../Shared/AtomInsertion/AtomInsertionDecisionLayout.hpp"
#include "../../Shared/Batching/BatchDeviceUpload.hpp"
#include "Shared/Forward/GeneratedSequence.hpp"
#include "../../Shared/Forward/ModelForwardRuntimePayload.hpp"
#include "../../Shared/Forward/ModelForward_GPU.hpp"
#include "../../Shared/Forward/CachedInferenceForward.hpp"
#include "../../Shared/InferenceState/KvCacheState_GPU.hpp"
#include "../../Shared/InferenceState/LocalAtomRetrievalInferenceState.hpp"
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

std::vector<float> runAtomInsertionForward(
    TrainingContext& ctx,
    GRIM::Batching::BatchPayload& payload) {
    constexpr const char* caller = "runAtomInsertionForward";
    const auto gpu_hp = GRIM::HyperParameters::gpuModelInitializationHP(ctx.config);
    if (!gpu_hp.use_gpu) {
        throw std::runtime_error("runAtomInsertionForward: config.use_gpu must be true");
    }
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
    result.continuation_text = GRIM::AtomInsertion::decodeAtomDecisionPredictions(
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

GRIM::GeneratedSequence advanceOneSequence(
    const GRIM::Config::AiConfigSnapshot& config,
    GRIM::TrainingState& training_state,
    GRIM::GenerationState& generation_state,
    GRIMText::Training::Startup::GpuModelState& gpu_model_state,
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    const GRIM::PBM::PBMState& pbm,
    GRIM::Tokenizer::UniByte& tokenizer,
    const GRIM::Lenses::LensCaptureRequest* lens_capture,
    bool inspect_only = false)
{
    const auto model_hp = GRIM::HyperParameters::modelHP(config);
    const bool local_atom_retrieval_enabled =
        model_hp.local_atom_retrieval_enabled;

    const auto gpu_hp = GRIM::HyperParameters::gpuModelInitializationHP(config);
    const bool use_gpu = gpu_hp.use_gpu;
    if (!generation_state.active_session || !generation_state.prefill ||
        generation_state.tokenizer != &tokenizer)
        throw std::logic_error("advanceOneSequence: no matching live inference session");
    auto& prompt_payload = *generation_state.prefill;
    auto& session = *generation_state.active_session;
    const int vocab_size = prompt_payload.vocab_size;
    if (!use_gpu) {
        throw std::runtime_error("Phase2 payload inference requires config.use_gpu=true");
    }
    auto& sequence = session.sequence;
    const auto prompt_atom_table = sequence.context_atom_table;
    const auto generation_local_atom_table = sequence.context_local_atom_table;

    if (!training_state.initialized) {
        throw std::runtime_error("advanceOneSequence: training state not initialized");
    }
    cudaStream_t stream = training_state.stream_ctrl.getPrimaryStream();
    auto* gpu_encoder = gpu_model_state.gpu_encoder.get();
    if (!gpu_encoder) {
        throw std::runtime_error("advanceOneSequence: gpu_model_state.gpu_encoder is NULL");
    }
    const int num_layers = model_hp.encoder_num_layers;
    const int num_heads = model_hp.encoder_num_heads;
    const int num_kv_heads = model_hp.encoder_num_kv_heads;
    const int d_model = model_hp.encoder_d_model;
    const int rotary_dim = model_hp.encoder_rotary_dim;
    const int max_cached_seq_len = model_hp.max_cached_seq_len;
    const int head_dim = model_hp.encoder_head_dim;

    GRIM::KvCacheState& kv_cache = generation_state.kv_cache;
    const int prior_cursor = kv_cache.host_seqlen;
    kv_cache.ensureAllocated(num_layers, num_heads, num_kv_heads, head_dim, rotary_dim,
                             max_cached_seq_len, pbm.rope_inv_freq, stream);
    if (prior_cursor != kv_cache.host_seqlen)
        throw std::logic_error("advanceOneSequence: cache geometry changed during a live session");
    const bool needs_prefill = kv_cache.host_seqlen == 0;
    if (needs_prefill) kv_cache.beginSession(stream);
    GRIM::LocalAtomRetrievalInferenceState& retrieval_state =
        generation_state.local_atom_retrieval;
    if (local_atom_retrieval_enabled) {
        retrieval_state.ensureAllocated(d_model, max_cached_seq_len, stream);
    }

    auto& inference_device_storage = generation_state.inference_device_storage;

    auto runCachedForward = [&](GRIM::Batching::BatchPayload& active_payload,
                                GRIM::Forward::CachedInferenceReadback readback) {
        GRIM::Forward::ModelForwardRequest request{};
        request.config = &config;
        request.gpu_encoder = gpu_encoder;
        request.parameter_registry = &parameter_registry;
        request.pbm = &pbm;
        request.cublas_handle = training_state.cublas_handle.get();
        request.stream = stream;
        request.kv_cache = &kv_cache;
        request.lens_capture = active_payload.isInferencePrefill() ? lens_capture : nullptr;
        return GRIM::Forward::runCachedInferenceForward(
            request, active_payload, inference_device_storage,
            local_atom_retrieval_enabled ? &retrieval_state : nullptr, readback);
    };

    auto buildDecodePayload = [&](const std::vector<int>& feed_tokens)
        -> GRIM::Batching::BatchPayload {
        if (feed_tokens.empty()) {
            throw std::runtime_error("advanceOneSequence: decode token window is empty");
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

    auto selectLocalRetrievalSlot = [&]
        (const GRIM::Forward::CachedInferenceResult& tail,
         GRIM::Tokenizer::AtomType type,
         int remaining_token_capacity) -> int {
        const int candidate_count = retrieval_state.candidateCount(type);
        if (tail.retrieval_logits.size() !=
            static_cast<std::size_t>(candidate_count + 1)) {
            throw std::runtime_error(
                "advanceOneSequence: retrieval logit count disagrees with typed candidate bank");
        }
        if (!std::isfinite(tail.retrieval_logits[0])) {
            throw std::runtime_error(
                "advanceOneSequence: NO_REFERENCE logit is non-finite");
        }

        int best_slot = GRIM::Batching::kLocalAtomNoReferenceTarget;
        float best_logit = tail.retrieval_logits[0];
        for (int local_index = 0; local_index < candidate_count;
             ++local_index) {
            const auto& content =
                retrieval_state.candidateContentTokenIds(type, local_index);
            const int replay_token_count =
                static_cast<int>(content.size()) + 1; // matching CLOSE
            if (replay_token_count > remaining_token_capacity) {
                continue;
            }
            const float logit = tail.retrieval_logits[
                static_cast<std::size_t>(local_index + 1)];
            if (!std::isfinite(logit)) {
                throw std::runtime_error(
                    "advanceOneSequence: candidate retrieval logit is non-finite");
            }
            if (logit > best_logit) {
                best_logit = logit;
                best_slot = local_index + 1;
            }
        }
        return best_slot;
    };

    auto processPendingLocalAtom = [&]
        (const GRIM::Forward::CachedInferenceResult& tail, std::size_t pending_sequence_index) {
        if (!local_atom_retrieval_enabled) {
            return;
        }
        if (tail.encoder_row.size() != static_cast<std::size_t>(d_model)) {
            throw std::runtime_error(
                "advanceOneSequence: pending token encoder signal is unavailable");
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
                        "advanceOneSequence: replayed local atom contains an unexpected boundary");
                }
                return;
            }

            if (token_id == expected_close) {
                if (retrieval_state.active_content_count <= 0 ||
                    retrieval_state.active_content_token_ids.empty()) {
                    throw std::runtime_error(
                        "advanceOneSequence: generated local atom has no content");
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
                        "advanceOneSequence: generated local atom table index is not dense");
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
                    "advanceOneSequence: generated local atom contains a nested or mismatched boundary");
            }
            if (retrieval_state.active_embedding_sum.empty()) {
                retrieval_state.active_embedding_sum.assign(
                    static_cast<std::size_t>(d_model), 0.0f);
            }
            for (int feature = 0; feature < d_model; ++feature) {
                retrieval_state.active_embedding_sum[
                    static_cast<std::size_t>(feature)] +=
                    tail.encoder_row[static_cast<std::size_t>(feature)];
            }
            retrieval_state.active_content_token_ids.push_back(token_id);
            ++retrieval_state.active_content_count;
            return;
        }

        if (!GRIM::Tokenizer::isAtomOpenTokenId(token_id)) {
            return;
        }
        const auto type = GRIM::Tokenizer::tokenIdToAtomType(token_id);
        const int remaining_token_capacity = remainingContextTokenCapacity(session, model_hp);
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
    if (needs_prefill) {
        auto prefill = runCachedForward(prompt_payload, {/*last_logits=*/!inspect_only});
        sequence.prefill_lens_snapshot = prefill.lens_snapshot;
        sequence.prefill_lens_capture_result = prefill.lens_capture_result;
        if (inspect_only) {
            auto result = takeSequence(session, GRIM::Inference::InferenceReturnReason::ExplicitFinish);
            generation_state.resetSession();
            return result;
        }
        commitToken(session, selectNextToken(session, prompt_payload, prefill.last_logits.data()));
    }
    while (!shouldReturn(session, model_hp)) {
        const auto position = static_cast<std::size_t>(kv_cache.host_seqlen);
        if (position >= sequence.token_ids.size())
            throw std::logic_error("advanceOneSequence: missing unprocessed token");
        const int pending_token = sequence.token_ids[position];
        auto decode_payload = buildDecodePayload({pending_token});
        auto tail = runCachedForward(decode_payload,
            {/*last_logits=*/true,
             /*encoder_row=*/local_atom_retrieval_enabled,
             /*retrieval_logits=*/local_atom_retrieval_enabled &&
                 GRIM::Tokenizer::isAtomOpenTokenId(pending_token)});
        processPendingLocalAtom(tail, position);
        if (static_cast<std::size_t>(kv_cache.host_seqlen) != position + 1)
            throw std::logic_error("advanceOneSequence: KV cursor did not advance exactly once");
        if (static_cast<std::size_t>(kv_cache.host_seqlen) < sequence.token_ids.size())
            continue;
        if (local_atom_retrieval_enabled && retrieval_state.hasForcedToken()) {
            commitForcedToken(session, retrieval_state.popForcedToken());
        } else {
            commitToken(session, selectNextToken(session, prompt_payload, tail.last_logits.data()));
        }
    }

    const auto reason = returnReason(session, model_hp);
    if (reason == GRIM::Inference::InferenceReturnReason::SpanBoundary) {
        sequence.return_reason = reason;
        sequence.finished = false;
        return sequence; // Snapshot; GenerationState retains the live session and cache.
    }
    auto result = takeSequence(session, reason);
    generation_state.resetSession();
    return result;
}

Phase2TextInferenceResult decodeInferenceResult(
    GRIM::Tokenizer::UniByte& tokenizer, GRIM::GeneratedSequence sequence,
    const std::vector<int>& prefill_ids, std::int64_t generation_ms) {
    Phase2TextInferenceResult result;
    result.prompt_token_count = prefill_ids.size();
    result.generation_ms = generation_ms;
    const auto start_decode = std::chrono::high_resolution_clock::now();
    result.continuation_text = tokenizer.decode(
        GRIM::Forward::continuationDecodeRequest(sequence, prefill_ids));
    result.token_ids = sequence.token_ids;
    result.prefill_lens_snapshot = sequence.prefill_lens_snapshot;
    result.prefill_lens_capture_result = sequence.prefill_lens_capture_result;
    result.sequence_token_count = sequence.token_ids.size();
    result.decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start_decode).count();
    result.generated_sequence = std::move(sequence);
    return result;
}

} // namespace

Phase2TextInferenceResult executePhase2PayloadInference(
    TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer,
    const GRIM::Batching::BatchPayload& prefill,
    const GRIM::HyperParameters::GenerationHP& generation_hp,
    const GRIM::Lenses::LensCaptureRequest* lens_capture, bool inspect_only) {
    // Consume Phase1-owned state and a structurally validated batch payload.
    // Only enforce requirements specific to this inference entry point.
    if (!prefill.isInferencePrefill())
        throw std::invalid_argument("Inference generation requires an InferencePrefill payload");
    if (prefill.EnableAtomIdentification)
        throw std::invalid_argument("Token generation cannot consume atom-insertion gaps");
    if (prefill.batch_size != 1 || prefill.total_tokens != prefill.seq_lengths.at(0))
        throw std::invalid_argument("Inference generation requires one compact prefill row");
    if (generation_hp.num_return_sequences != 1)
        throw std::invalid_argument("Inference result requires num_return_sequences == 1");
    const auto prompt_token_count = static_cast<size_t>(prefill.seq_lengths.at(0));
    const std::vector<int> prefill_ids(prefill.input_ids.begin(),
        prefill.input_ids.begin() + prompt_token_count);
    const auto start_generation = std::chrono::high_resolution_clock::now();
    // Generation clones atom registries and attaches device storage locally.
    // Repeated lens/baseline runs must not mutate the caller's input payload.
    auto generation_prefill = prefill;
    auto& training_state = ctx.requireTrainingState("executePhase2PayloadInference");
    auto& generation_state = ctx.requireGenerationState("executePhase2PayloadInference");
    generation_state.resetSession();
    generation_state.prefill = std::move(generation_prefill);
    generation_state.tokenizer = &tokenizer;
    GRIM::GeneratedSequence sequence;
    try {
        generation_state.active_session.emplace(GRIM::Inference::createInferenceGenerationSession(
            *generation_state.prefill, GRIM::HyperParameters::modelHP(ctx.config),
            generation_hp, nullptr, inspect_only));
        sequence = advanceOneSequence(
            ctx.config, training_state, generation_state, ctx.gpu_model,
            ctx.parameter_registry, ctx.pbm_owner.state(), tokenizer, lens_capture, inspect_only);
    } catch (...) {
        generation_state.resetSession();
        throw;
    }
    return decodeInferenceResult(tokenizer, std::move(sequence), prefill_ids,
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start_generation).count());
}

Phase2TextInferenceResult resumePhase2PayloadInference(
    TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer) {
    auto& state = ctx.requireGenerationState("resumePhase2PayloadInference");
    if (!state.active_session || !state.prefill || state.tokenizer != &tokenizer ||
        state.active_session->sequence.return_reason != GRIM::Inference::InferenceReturnReason::SpanBoundary)
        throw std::logic_error("resumePhase2PayloadInference: no matching paused session");
    const auto prefill_ids = state.prefill->input_ids;
    const auto start = std::chrono::high_resolution_clock::now();
    GRIM::Inference::acknowledgeSpanReturn(*state.active_session);
    GRIM::GeneratedSequence sequence;
    try {
        sequence = advanceOneSequence(ctx.config,
            ctx.requireTrainingState("resumePhase2PayloadInference"), state,
            ctx.gpu_model, ctx.parameter_registry, ctx.pbm_owner.state(), tokenizer, nullptr);
    } catch (...) {
        state.resetSession();
        throw;
    }
    return decodeInferenceResult(tokenizer, std::move(sequence), prefill_ids,
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - start).count());
}

Phase2TextInferenceResult executePhase2TextInference(
    TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt,
    const GRIM::HyperParameters::GenerationHP& generation_hp,
    const GRIM::Lenses::LensCaptureRequest* lens_capture) {
    if (prompt.empty()) throw std::invalid_argument("Inference prompt is empty");
    const auto model_hp = GRIM::HyperParameters::modelHP(ctx.config);
    if (model_hp.atom_insertion_enabled) {
        if (lens_capture) throw std::invalid_argument("Lens capture requires token rows, not atom-insertion gaps");
        return executeAtomInsertionTextInference(ctx, prompt,
            model_hp.embedding_vocab_size,
            std::min(model_hp.training_max_seq_len, model_hp.max_cached_seq_len));
    }
    const auto start = std::chrono::high_resolution_clock::now();
    auto prefill = GRIM::Inference::buildInferencePrefill(ctx.config, tokenizer, prompt);
    const auto encode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count();
    auto result = executePhase2PayloadInference(ctx, tokenizer, prefill, generation_hp, lens_capture);
    result.encode_ms = encode_ms;
    return result;
}

} // namespace GRIMText::Training

#include "InferenceGenerationSession.hpp"

#include "../Batching/BatchPayload.hpp"
#include "../UnigramByte/AtomTable.hpp"
#include "../UnigramByte/SequenceLocalAtomTable.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

namespace GRIM::Inference {
namespace {
Sampling::SamplingConfig makeSamplingConfig(const HyperParameters::GenerationHP& cfg) {
    if (cfg.max_new_tokens <= 0) {
        throw std::runtime_error("InferenceGenerationSession: max_new_tokens must be > 0");
    }
    if (cfg.min_new_tokens < 0) {
        throw std::runtime_error("InferenceGenerationSession: min_new_tokens must be non-negative");
    }
    if (cfg.min_new_tokens > cfg.max_new_tokens) {
        throw std::runtime_error("InferenceGenerationSession: min_new_tokens exceeds max_new_tokens");
    }
    if (cfg.strategy == GRIM::HyperParameters::SamplingStrategy::BEAM_SEARCH) {
        throw std::runtime_error("InferenceGenerationSession: BEAM_SEARCH is not supported");
    }

    return GRIM::Sampling::buildSamplingConfigFromGenerationFields(
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
}
} // namespace

InferenceGenerationSession createInferenceGenerationSession(
    Batching::BatchPayload& prefill,
    const HyperParameters::ModelHP& model_hp,
    const HyperParameters::GenerationHP& generation_hp,
    HyperParameters::GenerationStreamCallback* stream_callback,
    bool inspect_only)
{
    InferenceGenerationSession session{
        generation_hp,
        stream_callback,
        Sampling::SamplingPipeline(makeSamplingConfig(generation_hp)),
        {},
        ConceptSpanTracker(prefill.concept_span_delimiters)};
    const auto prompt_len = static_cast<int>(prefill.input_ids.size());
    const int context_limit = std::min(model_hp.training_max_seq_len, model_hp.max_cached_seq_len);
    if (prompt_len > context_limit || (!inspect_only && prompt_len == context_limit)) {
        throw std::runtime_error("InferenceGenerationSession: prompt length " +
            std::to_string(prompt_len) + " exceeds context limit " + std::to_string(context_limit));
    }
    if (prefill.seq_local_atom_tables.size() != 1 || !prefill.seq_local_atom_tables[0]) {
        throw std::runtime_error("InferenceGenerationSession: prompt local atom table is missing");
    }
    const auto authored_atoms = prefill.seq_atom_tables.at(0);
    auto atoms = authored_atoms ? authored_atoms->cloneHostForGeneration()
                               : std::make_shared<Tokenizer::AtomTable>();
    prefill.seq_atom_tables[0] = atoms;
    session.sequence.context_atom_table = std::move(atoms);
    session.sequence.context_local_atom_table =
        std::make_shared<Tokenizer::SequenceLocalAtomTable>(*prefill.seq_local_atom_tables[0]);
    session.sequence.token_ids = prefill.input_ids;
    session.sequence.concept_span_delimiters = prefill.concept_span_delimiters;
    session.sequence.token_origins.assign(prefill.input_ids.size(), TokenOrigin::SuppliedContext);
    session.sequence.token_numeric_values = prefill.numeric_values;
    session.sequence.token_atom_mask = prefill.atom_mask;
    session.sequence.atom_entry_ids = prefill.atom_entry_ids;
    session.sequence.token_local_atom_indices = prefill.token_local_atom_indices;
    return session;
}

int sampledTokenCount(const InferenceGenerationSession& session) {
    return static_cast<int>(std::count(session.sequence.token_origins.begin(),
        session.sequence.token_origins.end(), TokenOrigin::SampledAction));
}

int remainingTokenCapacity(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp) {
    return remainingInferenceTokenCapacity(
        sampledTokenCount(session), static_cast<int>(session.sequence.token_ids.size()),
        session.config.max_new_tokens, std::min(model_hp.training_max_seq_len, model_hp.max_cached_seq_len));
}

int remainingContextTokenCapacity(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp) {
    return std::min(model_hp.training_max_seq_len, model_hp.max_cached_seq_len) -
        static_cast<int>(session.sequence.token_ids.size());
}

InferenceReturnReason returnReason(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp) {
    const auto& events = session.sequence.span_events;
    const bool span_closed = events.size() > session.acknowledged_span_events &&
        events.back().kind == GeneratedSpanEventKind::Close &&
        events.back().token_position + 1 == session.sequence.token_ids.size();
    const bool sampled_tail = !session.sequence.token_origins.empty() &&
        session.sequence.token_origins.back() == TokenOrigin::SampledAction;
    return evaluateInferenceReturn(
        session.sequence.finished, sampledTokenCount(session),
        static_cast<int>(session.sequence.token_ids.size()),
        sampled_tail ? session.sequence.token_ids.back() : -1,
        session.config.max_new_tokens, session.config.min_new_tokens,
        std::min(model_hp.training_max_seq_len, model_hp.max_cached_seq_len),
        session.config.eos_token_id, span_closed);
}

bool finished(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp) {
    return isTerminalInferenceReturn(returnReason(session, model_hp));
}

bool shouldReturn(const InferenceGenerationSession& session,
    const HyperParameters::ModelHP& model_hp) {
    return returnReason(session, model_hp) != InferenceReturnReason::Continue;
}

Sampling::SampleResult selectNextToken(InferenceGenerationSession& session,
    const Batching::BatchPayload& prefill, const float* logits) {
    std::vector<float> row(logits, logits + prefill.vocab_size);
    if (sampledTokenCount(session) + 1 < session.config.min_new_tokens &&
        session.config.eos_token_id >= 0 && session.config.eos_token_id < prefill.vocab_size) {
        row[static_cast<std::size_t>(session.config.eos_token_id)] = -1e30f;
    }
    auto sample = session.pipeline.selectNextToken(row, session.sequence.token_ids, prefill.vocab_size);
    if (sample.token_id < 0 || sample.token_id >= prefill.vocab_size) {
        throw std::runtime_error("InferenceGenerationSession: sampled token out of range (token_id=" +
            std::to_string(sample.token_id) + ", vocab=" + std::to_string(prefill.vocab_size) + ")");
    }
    return sample;
}

void commitToken(InferenceGenerationSession& session, const Sampling::SampleResult& sample,
                 TokenOrigin origin) {
    session.sequence.score_token_positions.push_back(session.sequence.token_ids.size());
    session.sequence.token_ids.push_back(sample.token_id);
    session.sequence.token_origins.push_back(origin);
    session.sequence.token_scores.push_back(sample.log_probability);
    session.sequence.token_numeric_values.push_back(0.0f);
    session.sequence.token_atom_mask.push_back(0);
    session.sequence.atom_entry_ids.push_back(Tokenizer::kAtomEntryNone);
    session.sequence.token_local_atom_indices.push_back(Tokenizer::kLocalAtomIndexNone);
    session.sequence.score += sample.log_probability;
    session.span_tracker.observe(sample.token_id, session.sequence.token_ids.size() - 1,
                                 session.sequence.span_events);
    if (session.stream_callback) {
        (*session.stream_callback)(sample.token_id, sample.probability);
    }
}

void commitForcedToken(InferenceGenerationSession& session, int token_id) {
    Sampling::SampleResult forced{};
    forced.token_id = token_id;
    forced.probability = 1.0f;
    forced.log_probability = 0.0f;
    commitToken(session, forced, TokenOrigin::ForcedToken);
}

void acknowledgeSpanReturn(InferenceGenerationSession& session) {
    if (session.sequence.return_reason != InferenceReturnReason::SpanBoundary)
        throw std::logic_error("acknowledgeSpanReturn: session is not at a span return");
    session.acknowledged_span_events = session.sequence.span_events.size();
    session.sequence.return_reason = InferenceReturnReason::Continue;
}

GeneratedSequence takeSequence(InferenceGenerationSession& session, InferenceReturnReason reason) {
    session.sequence.return_reason = reason;
    session.sequence.finished = isTerminalInferenceReturn(reason);
    return std::move(session.sequence);
}

} // namespace GRIM::Inference

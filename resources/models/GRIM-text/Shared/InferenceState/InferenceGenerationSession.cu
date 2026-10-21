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

InferenceGenerationSession::InferenceGenerationSession(
    Batching::BatchPayload& prefill,
    const HyperParameters::ModelHP& model_hp,
    const HyperParameters::GenerationHP& generation_hp,
    HyperParameters::GenerationStreamCallback* stream_callback,
    bool inspect_only)
    : cfg_(generation_hp),
      max_seq_len_(model_hp.training_max_seq_len),
      vocab_size_(prefill.vocab_size),
      prompt_len_(static_cast<int>(prefill.input_ids.size())),
      stream_callback_(stream_callback),
      pipeline_(makeSamplingConfig(generation_hp)) {
    if (prompt_len_ > max_seq_len_ || (!inspect_only && prompt_len_ == max_seq_len_)) {
        throw std::runtime_error("InferenceGenerationSession: prompt length " +
            std::to_string(prompt_len_) + " exceeds max_seq_len " + std::to_string(max_seq_len_));
    }
    if (prefill.seq_local_atom_tables.size() != 1 || !prefill.seq_local_atom_tables[0]) {
        throw std::runtime_error("InferenceGenerationSession: prompt local atom table is missing");
    }
    const auto authored_atoms = prefill.seq_atom_tables.at(0);
    auto atoms = authored_atoms ? authored_atoms->cloneHostForGeneration()
                               : std::make_shared<Tokenizer::AtomTable>();
    prefill.seq_atom_tables[0] = atoms;
    sequence_.context_atom_table = std::move(atoms);
    sequence_.context_local_atom_table =
        std::make_shared<Tokenizer::SequenceLocalAtomTable>(*prefill.seq_local_atom_tables[0]);
    sequence_.token_ids = prefill.input_ids;
    sequence_.token_origins.assign(prefill.input_ids.size(), TokenOrigin::SuppliedContext);
    sequence_.token_numeric_values = prefill.numeric_values;
    sequence_.token_atom_mask = prefill.atom_mask;
    sequence_.atom_entry_ids = prefill.atom_entry_ids;
    sequence_.token_local_atom_indices = prefill.token_local_atom_indices;
}

int InferenceGenerationSession::committedNewTokens() const {
    return static_cast<int>(sequence_.token_ids.size()) - prompt_len_;
}

int InferenceGenerationSession::remainingTokenCapacity() const {
    return std::min(cfg_.max_new_tokens - committedNewTokens(),
                    max_seq_len_ - static_cast<int>(sequence_.token_ids.size()));
}

bool InferenceGenerationSession::finished() const {
    return sequence_.finished || remainingTokenCapacity() <= 0 ||
        (committedNewTokens() >= cfg_.min_new_tokens && committedNewTokens() > 0 &&
         sequence_.token_ids.back() == cfg_.eos_token_id);
}

Sampling::SampleResult InferenceGenerationSession::selectNextToken(const float* logits) {
    std::vector<float> row(logits, logits + vocab_size_);
    if (committedNewTokens() + 1 < cfg_.min_new_tokens &&
        cfg_.eos_token_id >= 0 && cfg_.eos_token_id < vocab_size_) {
        row[static_cast<std::size_t>(cfg_.eos_token_id)] = -1e30f;
    }
    auto sample = pipeline_.selectNextToken(row, sequence_.token_ids, vocab_size_);
    if (sample.token_id < 0 || sample.token_id >= vocab_size_) {
        throw std::runtime_error("InferenceGenerationSession: sampled token out of range (token_id=" +
            std::to_string(sample.token_id) + ", vocab=" + std::to_string(vocab_size_) + ")");
    }
    return sample;
}

void InferenceGenerationSession::commitToken(const Sampling::SampleResult& sample,
                                             TokenOrigin origin) {
    sequence_.score_token_positions.push_back(sequence_.token_ids.size());
    sequence_.token_ids.push_back(sample.token_id);
    sequence_.token_origins.push_back(origin);
    sequence_.token_scores.push_back(sample.log_probability);
    sequence_.token_numeric_values.push_back(0.0f);
    sequence_.token_atom_mask.push_back(0);
    sequence_.atom_entry_ids.push_back(Tokenizer::kAtomEntryNone);
    sequence_.token_local_atom_indices.push_back(Tokenizer::kLocalAtomIndexNone);
    sequence_.score += sample.log_probability;
    if (stream_callback_) {
        (*stream_callback_)(sample.token_id, sample.probability);
    }
}

void InferenceGenerationSession::commitForcedToken(int token_id) {
    Sampling::SampleResult forced{};
    forced.token_id = token_id;
    forced.probability = 1.0f;
    forced.log_probability = 0.0f;
    commitToken(forced, TokenOrigin::ForcedToken);
}

GeneratedSequence InferenceGenerationSession::takeSequence() {
    sequence_.finished = true;
    return std::move(sequence_);
}

} // namespace GRIM::Inference

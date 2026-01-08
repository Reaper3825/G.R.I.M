//======================================================//
//  RareTokens_GPU.cu
//  Implementation of rarity scoring helpers
//======================================================//

#include "RareTokens_GPU.hpp"
#include "../UnigramByte/Unigram.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace GRIM::RareTokens {

namespace {

inline bool validToken(int token_id, uint32_t vocab_size) {
    return token_id >= 0 && static_cast<uint32_t>(token_id) < vocab_size;
}

[[noreturn]] void failConfig(const char* caller, const std::string& message) {
    throw std::runtime_error(std::string("RareTokens::") + caller + ": " + message);
}

inline void requireVocabSize(uint32_t vocab_size, const char* caller) {
    if (vocab_size == 0) {
        failConfig(caller, "vocab_size must be explicitly set by the caller");
    }
    if (vocab_size < static_cast<uint32_t>(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET)) {
        failConfig(caller,
                   "vocab_size must include byte+atom token ranges (>= " +
                       std::to_string(GRIM::Tokenizer::UNIGRAM_VOCAB_OFFSET) + ")");
    }
}

inline void requireConfig(const Config& cfg, const char* caller) {
    requireVocabSize(cfg.vocab_size, caller);
    if (!std::isfinite(cfg.smoothing) || cfg.smoothing < 0.0f) {
        failConfig(caller, "smoothing must be finite and >= 0");
    }
    if (!std::isfinite(cfg.max_boost) || cfg.max_boost <= 0.0f) {
        failConfig(caller, "max_boost must be finite and > 0");
    }
    if (!std::isfinite(cfg.rarity_exponent) || cfg.rarity_exponent <= 0.0f) {
        failConfig(caller, "rarity_exponent must be finite and > 0");
    }
}

inline void requireTableSize(size_t table_size, const Config& cfg, const char* name, const char* caller) {
    if (table_size != cfg.vocab_size) {
        failConfig(caller,
                   std::string(name) + " size (" + std::to_string(table_size) +
                       ") must match cfg.vocab_size (" + std::to_string(cfg.vocab_size) + ")");
    }
}

inline void requireValidToken(int token_id, uint32_t vocab_size, const char* caller) {
    if (!validToken(token_id, vocab_size)) {
        failConfig(caller,
                   "token_id " + std::to_string(token_id) +
                       " is out of range for vocab_size " + std::to_string(vocab_size));
    }
}

inline SequenceRarity scoreSequenceUnchecked(uint32_t seq_id,
                                             const std::vector<int>& tokens,
                                             const std::vector<float>& inv_freq,
                                             uint32_t vocab_size,
                                             const char* caller) {
    SequenceRarity result{};
    result.seq_id = seq_id;
    if (tokens.empty()) {
        return result;
    }

    float accum = 0.0f;
    uint32_t used = 0;
    for (int token_id : tokens) {
        requireValidToken(token_id, vocab_size, caller);
        const float w = inv_freq[static_cast<size_t>(token_id)];
        accum += w;
        used++;
        if (w > 1.0f) {
            result.rare_token_hits++;
        }
    }

    if (used > 0) {
        float rarity = accum / static_cast<float>(used);
        // Slight boost for sequences that actually contain rare tokens.
        if (result.rare_token_hits > 0) {
            const float bonus = 1.0f + std::min(3u, result.rare_token_hits) * 0.1f;
            rarity *= bonus;
        }
        result.rarity = rarity;
    }
    return result;
}

} // namespace

std::vector<uint64_t> computeFrequencies(const std::vector<std::vector<int>>& sequences,
                                         uint32_t vocab_size) {
    requireVocabSize(vocab_size, "computeFrequencies");
    std::vector<uint64_t> counts(vocab_size, 0);
    for (const auto& seq : sequences) {
        for (int token_id : seq) {
            requireValidToken(token_id, vocab_size, "computeFrequencies");
            counts[static_cast<size_t>(token_id)]++;
        }
    }
    return counts;
}

std::vector<uint64_t> computeFrequencies(const std::vector<const std::vector<int>*>& sequences,
                                         uint32_t vocab_size) {
    requireVocabSize(vocab_size, "computeFrequencies");
    std::vector<uint64_t> counts(vocab_size, 0);
    for (const auto* seq_ptr : sequences) {
        if (!seq_ptr) {
            failConfig("computeFrequencies", "sequence pointer is null");
        }
        for (int token_id : *seq_ptr) {
            requireValidToken(token_id, vocab_size, "computeFrequencies");
            counts[static_cast<size_t>(token_id)]++;
        }
    }
    return counts;
}

std::vector<float> buildInverseFrequencyTable(const std::vector<uint64_t>& freqs,
                                              const Config& cfg) {
    requireConfig(cfg, "buildInverseFrequencyTable");
    requireTableSize(freqs.size(), cfg, "freqs", "buildInverseFrequencyTable");
    const uint32_t vocab_size = cfg.vocab_size;
    std::vector<float> weights(vocab_size, 0.0f);

    const float base = static_cast<float>(cfg.rare_threshold) + cfg.smoothing;
    for (uint32_t i = 0; i < vocab_size; ++i) {
        const uint64_t count = freqs[i];
        const float denom = static_cast<float>(count) + cfg.smoothing;
        float w = base / std::max(denom, 1.0f);
        if (cfg.rarity_exponent != 1.0f) {
            w = std::pow(w, cfg.rarity_exponent);
        }
        w = std::min(w, cfg.max_boost);
        weights[i] = w;
    }
    return weights;
}

SequenceRarity scoreSequence(uint32_t seq_id,
                             const std::vector<int>& tokens,
                             const std::vector<float>& inv_freq,
                             const Config& cfg) {
    requireConfig(cfg, "scoreSequence");
    requireTableSize(inv_freq.size(), cfg, "inv_freq", "scoreSequence");
    return scoreSequenceUnchecked(seq_id, tokens, inv_freq, cfg.vocab_size, "scoreSequence");
}

std::vector<float> scoreSequences(const std::vector<const std::vector<int>*>& sequences,
                                  const std::vector<float>& inv_freq,
                                  const Config& cfg) {
    requireConfig(cfg, "scoreSequences");
    requireTableSize(inv_freq.size(), cfg, "inv_freq", "scoreSequences");
    std::vector<float> scores;
    scores.reserve(sequences.size());
    for (size_t i = 0; i < sequences.size(); ++i) {
        const auto* seq = sequences[i];
        if (!seq) {
            failConfig("scoreSequences", "sequence pointer is null");
        }
        auto sr = scoreSequenceUnchecked(static_cast<uint32_t>(i), *seq, inv_freq, cfg.vocab_size, "scoreSequences");
        scores.push_back(sr.rarity);
    }
    return scores;
}

} // namespace GRIM::RareTokens

//======================================================//
//  RareTokens_GPU.hpp
//  Lightweight rarity scoring utilities for batching
//======================================================//

#pragma once

#include <cstdint>
#include <vector>

namespace GRIM::RareTokens {

struct Config {
    uint32_t vocab_size = 0;          // total vocab size (bytes + atoms + unigram); must be set by caller
    uint64_t rare_threshold = 32;     // counts below this are considered rare
    float smoothing = 1.0f;           // additive smoothing for inverse frequency
    float max_boost = 8.0f;           // cap for per-token boost
    float rarity_exponent = 0.5f;     // exponent applied to inverse frequency
};

struct SequenceRarity {
    uint32_t seq_id = 0;
    float rarity = 0.0f;              // higher = rarer
    uint32_t rare_token_hits = 0;     // how many tokens fell under threshold
};

// Compute token frequency table from a vector of token vectors.
std::vector<uint64_t> computeFrequencies(const std::vector<std::vector<int>>& sequences,
                                         uint32_t vocab_size);
// Compute token frequency table from a vector of token pointers (no copies).
std::vector<uint64_t> computeFrequencies(const std::vector<const std::vector<int>*>& sequences,
                                         uint32_t vocab_size);

// Convert counts into inverse-frequency weights (higher for rare tokens).
std::vector<float> buildInverseFrequencyTable(const std::vector<uint64_t>& freqs,
                                              const Config& cfg);

// Score a single sequence using inverse-frequency weights.
SequenceRarity scoreSequence(uint32_t seq_id,
                             const std::vector<int>& tokens,
                             const std::vector<float>& inv_freq,
                             const Config& cfg);

// Batch convenience: returns rarity scores aligned with input ordering (index = seq index).
std::vector<float> scoreSequences(const std::vector<const std::vector<int>*>& sequences,
                                  const std::vector<float>& inv_freq,
                                  const Config& cfg);

} // namespace GRIM::RareTokens

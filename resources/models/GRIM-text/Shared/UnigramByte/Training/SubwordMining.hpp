//======================================================//
//  SubwordMining.hpp
//  Training-only unigram subword candidate mining
//
//  Owns deterministic sampling, atom-aware candidate mining,
//  parallel count aggregation, and overflow-checked count math.
//  UnigramTrainer calls this module once and receives the full
//  mined candidate-count result.
//
//  Author: GRIM Team
//  Date: December 2025
//======================================================//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM {
namespace Tokenizer {

struct AtomSpan;

using UnigramSubwordCount = std::uint64_t;
using UnigramSubwordCountMap = std::unordered_map<std::string, UnigramSubwordCount>;

struct UnigramSubwordMiningRequest {
    const std::vector<std::string>& training_units;
    const std::vector<std::vector<AtomSpan>>& atom_spans;
    bool enable_parallel_subword_mining = false;
    int configured_worker_count = 0;
    std::size_t configured_max_mining_bytes = 0;
    const char* log_prefix = "[SubwordMining]";
};

struct UnigramSubwordMiningResult {
    UnigramSubwordCountMap subword_counts;
    std::size_t total_training_bytes = 0;
    std::size_t max_mining_bytes = 0;
    bool used_sampling = false;
    double sampling_ratio = 1.0;
    std::size_t sampled_start_bytes = 0;
    std::size_t context_bytes = 0;
    std::size_t sampled_spans = 0;
    unsigned int worker_count = 1;
};

bool isValidUnigramVocabCharacter(const std::string& ch);
bool isUnigramRepetitionNoise(const std::string& s);
bool isValidUnigramSubword(const std::string& s);

UnigramSubwordCount addUnigramSubwordCountsForTraining(UnigramSubwordCount current,
                                                       UnigramSubwordCount delta,
                                                       const char* caller);

UnigramSubwordMiningResult mineUnigramSubwordsFromTrainingUnits(
    const UnigramSubwordMiningRequest& request);

} // namespace Tokenizer
} // namespace GRIM

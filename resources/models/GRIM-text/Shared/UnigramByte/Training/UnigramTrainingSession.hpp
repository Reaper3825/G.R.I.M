// One vocabulary-training call: host ownership and flat execution addresses.
#pragma once

#include "../Unigram.hpp"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace GRIM::Tokenizer {

struct UnigramTrainingSegment {
    size_t byte_offset = 0;
    size_t byte_length = 0;
};

struct UnigramTrainingMatch {
    size_t end_offset = 0; // Absolute offset in the packed corpus.
    int piece_index = -1;
};

enum class UnigramTrainingAddressSpace { Host, Device };

// Session-local owners. No stage keeps addresses into individual strings.
// Allocation/rebinding is performed by the session preparation functions.
struct UnigramTrainingStorage {
    std::vector<char> corpus_bytes;
    std::vector<UnigramTrainingSegment> segments;
    std::vector<UnigramTrainingMatch> matches;
    std::vector<size_t> match_offsets; // CSR: one range per corpus byte.
    std::vector<float> piece_scores;
    std::vector<double> expected_counts;
    std::vector<double> alpha;
    std::vector<double> beta;
};

// The single object passed through fitting, expectation, maximization and
// pruning. This is a non-owning struct; its storage, HP and model vocabulary
// must outlive the call. Host control pointers are NEVER device kernel inputs.
struct UnigramTrainingSession {
    const ::GRIM::HyperParameters::TokenizerHP* hp = nullptr;
    std::vector<UnigramPiece>* pieces = nullptr;
    std::unordered_map<std::string, int>* piece_to_id = nullptr;
    UnigramTrainingStorage* storage = nullptr;
    int total_piece_target = 0;
    bool enable_byte_fallback = false;

    // Flat execution views. Every view has session-owned backing storage.
    UnigramTrainingAddressSpace address_space = UnigramTrainingAddressSpace::Host;
    const char* corpus_bytes = nullptr;
    size_t corpus_byte_count = 0;
    const UnigramTrainingSegment* segments = nullptr;
    size_t segment_count = 0;
    const UnigramTrainingMatch* matches = nullptr;
    size_t match_count = 0;
    const size_t* match_offsets = nullptr;
    float* piece_scores = nullptr;
    double* expected_counts = nullptr;
    size_t piece_count = 0;
    double* alpha = nullptr;
    double* beta = nullptr;
    size_t workspace_capacity = 0;

    uint64_t vocabulary_generation = 0;
    uint64_t matches_generation = 0;
    uint64_t expectation_generation = 0;
    double log_likelihood = 0.0;
    double expected_learned_piece_tokens = 0.0;
    double expected_fixed_penalty_byte_fallback_tokens = 0.0;
};

// Called once after normalization. Packs eligible ranges without substring
// temporaries and records the existing exclusion/newline boundaries.
void prepareUnigramTrainingSessionCorpus(UnigramTrainingSession& session,
    const std::vector<std::string>& normalized_texts,
    const std::vector<std::vector<AtomSpan>>& normalized_exclusion_spans);

// Called after admission and every vocabulary rewrite. Builds matching edges
// once per vocabulary shape; EM score updates reuse these edges unchanged.
void refreshUnigramTrainingSessionVocabulary(UnigramTrainingSession& session);
void requireUnigramTrainingSessionReady(const UnigramTrainingSession& session,
    const char* caller);

} // namespace GRIM::Tokenizer

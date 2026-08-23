//======================================================//
//  grmt_vocab_metrics_test.hpp
//  Corpus + vocabulary diagnostic metrics for .grmt files
//
//  Single-pass streaming scan. Dumps actionable metrics:
//  Shannon entropy, bytes/token, fertility, vocab utilization,
//  byte-fallback rate, OOV rate, top/bottom token frequency,
//  sequence length distribution.
//======================================================//

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM {
namespace Test {

//======================================================//
//  GRMT corpus metrics (computed by streaming scan)
//======================================================//
struct GRMTCorpusMetrics {
    // Header
    uint32_t grmt_version    = 0;
    uint32_t num_sequences   = 0;
    uint32_t vocab_size      = 0;   // from GRMT header

    // Token statistics
    uint64_t total_tokens    = 0;
    uint32_t distinct_ids    = 0;   // unique IDs observed in corpus

    // Computed metrics
    double shannon_entropy   = 0.0;   // bits
    double bytes_per_token   = 0.0;   // decoded UTF-8 bytes / token count
    double fertility         = 0.0;   // tokens / whitespace-delimited word

    // Vocab utilization
    double vocab_utilization = 0.0;   // distinct_ids / vocab_size (%)
    uint32_t dead_vocab_ids  = 0;     // vocab_size - distinct_ids

    // Token class breakdown (% of total_tokens)
    uint64_t byte_fallback_count = 0;
    uint64_t numeric_count       = 0;
    uint64_t atom_count          = 0;
    uint64_t unk_count           = 0;
    uint64_t special_count       = 0;  // <pad>, <s>, </s>
    uint64_t unigram_count       = 0;  // regular vocab pieces

    // Sequence stats
    uint32_t seq_len_min     = 0;
    uint32_t seq_len_max     = 0;
    double   seq_len_mean    = 0.0;
    double   seq_len_stddev  = 0.0;

    // Token frequency histogram (sparse)
    std::unordered_map<uint32_t, uint64_t> token_hist;

    // Scan state
    uint32_t sequences_scanned = 0;  // may differ from num_sequences on read error
    bool     scan_ok           = true;
};

//======================================================//
//  Core API — scan once, report once
//======================================================//

/// Load vocab.bin → id-to-text map
std::unordered_map<int, std::string> loadVocabMap(const std::string& path);

/// Single-pass streaming scan of GRMT file
GRMTCorpusMetrics scanGRMT(const std::string& grmt_path,
                           const std::unordered_map<int, std::string>& vocab);

/// Print full metrics report to stdout
void printMetricsReport(const GRMTCorpusMetrics& m,
                        const std::unordered_map<int, std::string>& vocab,
                        const std::string& vocab_path,
                        const std::string& grmt_path);

/// Check for red flags. Returns number of warnings.
int checkSanity(const GRMTCorpusMetrics& m);

} // namespace Test
} // namespace GRIM

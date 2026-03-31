//======================================================//
//  grmt_vocab_metrics_test.hpp
//  Corpus + vocabulary diagnostic metrics for .grmt files
//
//  C++ replacement for bridges2_vocab_corpus_metrics.py.
//  Computes the same metrics (Shannon entropy, bytes/token,
//  fertility) as a streaming single-pass scan — no need
//  to hold the entire corpus in memory.
//======================================================//

#pragma once

#include "unigrambyte_self_test.hpp"   // TestResult, UnigramByteTestSuite, ASSERT_*

#include <cstdint>
#include <string>
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
    uint32_t vocab_size      = 0;

    // Token statistics
    uint64_t total_tokens    = 0;
    uint32_t distinct_ids    = 0;

    // Computed metrics
    double shannon_entropy   = 0.0;   // bits
    double bytes_per_token   = 0.0;   // decoded UTF-8 bytes / token count
    double fertility         = 0.0;   // tokens / whitespace-delimited word

    // Sequence stats
    uint32_t seq_len_min     = 0;
    uint32_t seq_len_max     = 0;
    double   seq_len_mean    = 0.0;
};

//======================================================//
//  Register all grmt-vocab-metrics tests
//======================================================//
void registerGRMTVocabMetricsTests(UnigramByteTestSuite& suite,
                                   const std::string& vocab_path,
                                   const std::string& grmt_path);

} // namespace Test
} // namespace GRIM

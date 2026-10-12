# Unigram vocabulary training stages

`UnigramTrainer.cu` coordinates the full atom-aware training entrypoint.
The public `UnigramLM::trainFromCorpus` signatures remain in `Unigram.hpp`.
Both entrypoints require `TokenizerHP`; the atom-aware entrypoint also accepts
explicit exclusion spans and manual pieces. There are no positional training
defaults. The prepass validates settings before scanning the corpus.

Training policies are authored as required `tokenizer_*` leaves under
`ai_config.json` → `training.config`. `HyperParameters_GPU.hpp` loads, validates,
and preserves them in the finalized snapshot; `tokenizerHP(snapshot)` slices
them directly into `TokenizerHP`, which travels through the training stages.
`validateTokenizerTrainingPolicies` owns the shared bounds for startup and
explicit HP callers. Missing JSON leaves and invalid numeric values fail.

| Policy | Config leaves (all prefixed `tokenizer_`) |
| --- | --- |
| EM | `em_max_iterations`, `em_convergence_tolerance`, `piece_count_smoothing`, `em_fail_on_iteration_limit` |
| Vocabulary fitting | `prune_shrink_factor`, `prune_em_iterations`, `minimum_learned_to_fallback_ratio` |
| Mining scheduling | `subword_mining_parallel_min_texts`, `subword_mining_fallback_workers`, `subword_mining_reserved_cores`, `subword_mining_chunks_per_worker`, `subword_mining_min_chunk_texts`, `subword_mining_max_chunk_texts` |
| Mining sampling/allocation | `subword_mining_sample_span_bytes`, `subword_mining_initial_count_reserve`, `subword_mining_worker_count_reserve` |
| Repetition filters | `repetition_max_alpha_run`, `repetition_max_digit_run`, `repetition_max_punctuation_run`, `repetition_max_whitespace_run`, `repetition_pattern_min_bytes`, `repetition_pattern_max_bytes`, `repetition_pattern_min_repeats`, `repetition_doubled_min_half_bytes`, `repetition_stutter_min_words` |
| Unicode admission | `exclude_unicode_controls`, `exclude_unicode_formatting`, `exclude_unicode_private_use`, `exclude_nonbreaking_space` |
| Candidate deduplication | `structural_dedup_trim_edges` |
| Character diagnostics | `diagnostic_min_char_frequency`, `diagnostic_min_char_seeds` |

JSON values retain the former policies: 50 EM iterations, relative tolerance
0.0001, smoothing 0.1, pruning keep fraction 0.75 and two refits. Hitting the EM
limit logs and continues unless `em_fail_on_iteration_limit` is enabled.
An automatic mining worker count uses hardware concurrency, falls back to
`subword_mining_fallback_workers` when unavailable, then reserves the configured
cores. An explicit `subword_mining_workers` count overrides that automatic
choice; the minimum-document threshold still applies. The former
`GRIM_SUBWORD_MINING_WORKERS` environment override has been removed.
Count-map reserve values may be zero to omit preallocation.

Unicode range endpoints, valid UTF-8, fixed newline/word-boundary handling,
token IDs, the cross-runtime piece-length contract, and the byte fallback
penalty remain structural constants. Unicode exclusion switches govern the
existing single-codepoint admission policy. Character seeds remain diagnostic.

| Module | Responsibility |
| --- | --- |
| `UnigramTrainingPrepass` | TokenizerHP entrypoint: detect atom and fixed numeric exclusions, load manual pieces, and invoke training. |
| `UnigramTrainingValidation` | Validate parameters and guard candidate counts and learned posterior mass. |
| `UnigramCorpusPreparation` | Validate original spans, normalize text and span offsets, exclude exact manual matches, and measure the longest eligible segment. |
| `SubwordMining` | Mine candidate counts from eligible normalized corpus ranges. |
| `UnigramCharacterDiagnostics` | Report character coverage and reset the initial vocabulary. Character seeds remain diagnostic only. |
| `UnigramCandidateAdmission` | Rank, filter, and deduplicate mined candidates; append manual pieces and initialize learned scores. |
| `UnigramVocabularyFitting` | Coordinate initial EM, pruning rounds with EM refits, and final EM. |
| `UnigramTrainingSession.hpp` | Plain session/storage structs, flat buffer addresses, segment and match records, and vocabulary generation checks. |
| `UnigramEMConvergence` | Run E/M iterations, check convergence, and emit iteration telemetry. |
| `UnigramExpectationStep` | Prepare persistent session corpus/matches, traverse cached segment edges, and aggregate dense posterior counts and likelihood. |
| `UnigramForwardBackward` | Compute segment-level forward/backward weights and occurrence posteriors. |
| `UnigramMaximizationStep` | Update ordinary piece scores using smoothed learned counts. |
| `UnigramLikelihoodPruning` | Estimate removal loss and select survivors while protecting coverage. |
| `UnigramTrainingFinalization` | Build/upload the fitted trie and record runtime readiness. |

The prepared corpus owns normalized text and exclusions for mining. A
`UnigramTrainingSession` is created in the training entrypoint and passed by
reference through fitting, convergence, E/M iterations, and pruning. Its
`UnigramTrainingStorage` owner lives for the same call. The session packs
eligible corpus bytes once and records exclusion/newline-separated ranges by
offset and length; E-steps do not create substring copies. Host control pointers
(`hp`, model vocabulary/index, and storage) are distinct from flat execution
addresses. Those host C++ objects must never be passed to device kernels.

Vocabulary order and its text index are changed together through
`VocabWriteOp.hpp`. The M-step changes scores only. Pruning moves survivors out;
the fitting coordinator rewrites the vocabulary before subsequent EM passes.
Matching uses a temporary trie once after candidate admission and once after
each pruning rewrite. It stores score-independent edges grouped by start byte
in a flat CSR layout (`match_offsets`, `matches`), bounded by each segment.
All EM iterations within that vocabulary shape reuse those edges. Scores and
expected counts are dense arrays indexed by piece position; learned counts no
longer travel through a token-ID hash map. Forward/backward scratch is allocated
once for the longest eligible segment and reused by serial host execution.
`runUnigramEMIteration` owns the shared E/M sequence for convergence and pruning
refits. Byte fallback remains a fixed penalty outside learned-piece probability
normalization, and manual pieces remain outside learned posterior counts.

Vocabulary refresh invalidates prior counts, increments the vocabulary
generation, rebuilds matching edges and rebinds all views. Readiness checks
reject stale generations, reallocated backing storage, and unsupported address
spaces. A no-progress prune leaves the live vocabulary untouched. The current
backend remains CPU; the flat addresses prepare the execution boundary for a
future GPU implementation, without introducing device allocation or kernels.

Focused validation (no trainer/runtime target build):

```text
python resources/models/GRIM-text/Tests/test_tokenizer_training_policy_host.py
python resources/models/GRIM-text/Tests/test_tokenizer_training_policy_host.py --session
python resources/models/GRIM-text/Tests/test_tokenizer_training_policy_host.py --syntax
```

The session tests compare cached expectations against the existing trie-based
forward/backward estimator and exercise score updates, pruning/refits, manual
exclusion, raw-byte UTF-8 fallback, and stale-address rejection. Full-corpus
timing is not established by these focused tests. Cached occurrence edges use
additional memory proportional to corpus matching density.

Stage-local preprocessing and admission temporaries expire when their stage
returns. The mining result remains owned by the trainer for the training call.
Runtime finalization is a private `UnigramLM` method because it updates the
model's runtime generation and training report.

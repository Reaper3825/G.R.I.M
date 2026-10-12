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
| `UnigramEMConvergence` | Run E/M iterations, check convergence, and emit iteration telemetry. |
| `UnigramExpectationStep` | Traverse eligible corpus segments and aggregate posterior counts and likelihood. |
| `UnigramForwardBackward` | Compute segment-level forward/backward weights and occurrence posteriors. |
| `UnigramMaximizationStep` | Update ordinary piece scores using smoothed learned counts. |
| `UnigramLikelihoodPruning` | Estimate removal loss and select survivors while protecting coverage. |
| `UnigramTrainingFinalization` | Build/upload the fitted trie and record runtime readiness. |

The prepared corpus owns normalized text and exclusions for mining and all EM
passes. Vocabulary order and its text index are changed together through
`VocabWriteOp.hpp`. The M-step changes scores only. Pruning moves survivors out;
the fitting coordinator rewrites the vocabulary before subsequent EM passes.
Forward/backward uses a fresh trie snapshot for each E-step. Byte fallback
remains a fixed penalty outside learned-piece probability normalization.

Stage-local preprocessing and admission temporaries expire when their stage
returns. The mining result remains owned by the trainer for the training call.
Runtime finalization is a private `UnigramLM` method because it updates the
model's runtime generation and training report.

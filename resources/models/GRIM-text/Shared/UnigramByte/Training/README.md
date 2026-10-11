# Unigram vocabulary training stages

`UnigramTrainer.cu` coordinates the full atom-aware training entrypoint.
The public `UnigramLM::trainFromCorpus` signatures remain in `Unigram.hpp`.

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

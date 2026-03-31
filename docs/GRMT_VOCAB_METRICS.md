# GRMT Corpus + Vocabulary Metrics Report

**Date:** March 31, 2026  
**Cluster:** Bridges-2 (PSC)  
**Vocab file:** `training/data/vocab.bin`  
**Corpus file:** `training/data/training_data.grmt` (version 11)

---

## Corpus Size

| Metric | Value |
|--------|-------|
| Sequences | 500,146 |
| Total tokens | 355,038,940 |
| Scan integrity | 500,146 / 500,146 (100%) |

## Sequence Lengths

| Metric | Value |
|--------|-------|
| Min | 27 |
| Max | 11,890 |
| Mean | 709.9 |
| Std Dev | 466.4 |

## Vocabulary Utilization

| Metric | Value |
|--------|-------|
| vocab_size (header) | 10,262 |
| Distinct IDs observed | 8,465 |
| Utilization | 82.49% |
| Dead IDs (never appear) | 1,797 |

## Token Class Breakdown

| Class | Count | % of Total |
|-------|------:|----------:|
| Unigram pieces | 178,331,492 | 50.23% |
| Byte fallback | 173,982,098 | **49.00%** |
| Atom tokens | 2,725,350 | 0.77% |
| `<unk>` | 0 | 0.00% |
| Special (s/pad) | 0 | 0.00% |

## Quality Metrics

| Metric | Value |
|--------|-------|
| Shannon entropy | 7.5108 bits |
| Theoretical max entropy | 13.05 bits |
| Bytes per token | 2.6005 |
| Fertility | 2.4198 tok/word |

## Top 30 Tokens (by frequency)

| Rank | Token | ID | Count | % |
|-----:|-------|---:|------:|--:|
| 1 | ` ` (space) | 36 | 127,456,685 | 35.90% |
| 2 | `N` | 14 | 7,889,243 | 2.22% |
| 3 | `,` | 48 | 7,638,337 | 2.15% |
| 4 | `.` | 50 | 7,026,522 | 1.98% |
| 5 | `and` | 279 | 4,379,113 | 1.23% |
| 6 | `the` | 268 | 2,836,824 | 0.80% |
| 7 | `<ATOM1>` | 261 | 2,725,350 | 0.77% |
| 8 | `:` | 62 | 2,622,944 | 0.74% |
| 9 | `ing` | 282 | 2,432,819 | 0.69% |
| 10 | `to` | 278 | 2,244,762 | 0.63% |
| 11 | `ed` | 287 | 2,210,734 | 0.62% |
| 12 | `-` | 49 | 1,996,858 | 0.56% |
| 13 | `of` | 301 | 1,813,070 | 0.51% |
| 14 | `in` | 263 | 1,788,318 | 0.50% |
| 15 | `es` | 271 | 1,567,114 | 0.44% |
| 16 | `'` | 43 | 1,368,348 | 0.39% |
| 17 | `a` | 101 | 1,252,195 | 0.35% |
| 18 | `[` | 95 | 1,144,011 | 0.32% |
| 19 | `]` | 97 | 1,143,395 | 0.32% |
| 20 | `for` | 337 | 1,118,441 | 0.32% |
| 21 | `or` | 274 | 1,101,182 | 0.31% |
| 22 | `re` | 267 | 980,768 | 0.28% |
| 23 | `The` | 421 | 952,750 | 0.27% |
| 24 | `A` | 69 | 952,504 | 0.27% |
| 25 | `s` | 119 | 903,285 | 0.25% |
| 26 | `that` | 389 | 898,401 | 0.25% |
| 27 | `with` | 414 | 808,378 | 0.23% |
| 28 | `on` | 269 | 788,549 | 0.22% |
| 29 | `"` | 38 | 751,980 | 0.21% |
| 30 | `er` | 266 | 751,097 | 0.21% |

## Bottom 20 Tokens (rarest observed)

| Rank | Token | ID | Count |
|-----:|-------|---:|------:|
| 1 | `rpr` | 6976 | 2 |
| 2 | `ocu` | 1980 | 2 |
| 3 | `sistanc` | 9739 | 2 |
| 4 | `looki` | 5790 | 2 |
| 5 | `pecially` | 9205 | 2 |
| 6 | `restau` | 6862 | 2 |
| 7 | `cisio` | 5101 | 2 |
| 8 | `ibra` | 5824 | 2 |
| 9 | `awarenes` | 9576 | 2 |
| 10 | `recen` | 7455 | 2 |
| 11 | `nfrastructur` | 9643 | 2 |
| 12 | `itatio` | 3948 | 2 |
| 13 | `pda` | 4939 | 2 |
| 14 | `nifica` | 2175 | 2 |
| 15 | `aul` | 9420 | 2 |
| 16 | `echnology` | 2720 | 2 |
| 17 | `ltimate` | 8413 | 2 |
| 18 | `servatio` | 10133 | 2 |
| 19 | `rfo` | 2471 | 2 |
| 20 | `incorpor` | 4421 | 2 |

---

## Warnings

- **Byte fallback rate is 49.00%** — nearly half of all tokens are single-byte fallbacks rather than learned unigram pieces. This indicates the vocabulary is undersized for the corpus diversity, or the unigram training data was insufficiently representative. Consider increasing `vocab_size` or retraining the unigram model on a more representative sample.

## Key Observations

1. **Space token dominates** — ID 36 (space) accounts for 35.9% of all tokens. This is expected for whitespace-delimited text but inflates the byte fallback count since space is in the byte range [0–255].
2. **`N` at rank 2 (2.22%)** — byte 0x4E appearing this frequently suggests either data with many bare newlines/numbers or an encoding artifact worth investigating.
3. **1,797 dead vocab IDs** — 17.5% of the vocabulary is never exercised. These slots waste embedding parameters. A vocab prune + retrain would reclaim capacity.
4. **Shannon entropy 7.51 / 13.05** — the distribution is heavily skewed toward a small set of frequent tokens. Entropy efficiency is 57.5% of theoretical max.
5. **Bottom tokens appear only twice each** — the long tail of near-singleton subwords suggests the unigram trainer overfit to rare patterns. These could be pruned without loss.
6. **Fertility 2.42 tok/word** — reasonable for a 10K vocab. For comparison, GPT-2 (50K vocab) achieves ~1.3, SentencePiece 32K typically ~1.5–1.8.

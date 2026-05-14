# Tokenizer — Unigram + Byte Fallback

Files: `resources/models/GRIM-text/Shared/UnigramByte/`
- `GrimTokenizer.hpp` — alias to UniByte
- `Detectors/` — raw-text detector parent class, registry, numeric detectors, whitespace/uppercase feature detectors
- `Unigram.cu` — vocab + Viterbi

## Token layout
| Range | Meaning |
|-------|---------|
| `[0, 3]` | Reserved layout special tokens (`UNK`, `PAD`, `BOS`, `EOS`) |
| `[BYTE_TOKEN_OFFSET, BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE)` | Raw bytes (100% UTF-8 coverage) |
| `[ATOM_TOKEN_OFFSET, UNIGRAM_VOCAB_OFFSET)` | Atom placeholders |
| `[UNIGRAM_VOCAB_OFFSET, …)` | Unigram vocab |

Use `TokenLayout.hpp` / `Byte.hpp` constants and layout helpers for range checks; do not duplicate numeric offsets in runtime code.

Special-token ownership is deliberately narrow:
- `TokenLayout.hpp` owns the reserved IDs and display metadata.
- `UnigramLM::save()` writes special-token records into saved vocab files so the persisted vocab advertises the full layout.
- The tokenizer does **not** inject `BOS`/`EOS`, pad batches, decide epoch/window boundaries, or mask training targets.
- `SlidingWindow.cu` owns `BOS`/`EOS` insertion and window-boundary target fixup.
- `BatchPayload.cu` owns padding and defense masking of `UNK`/`PAD`/`BOS` targets. `EOS` remains a valid target.

## Public encode/decode surface
- `UniByte::encode(text)` is the single high-level ID-only encode wrapper.
- `UniByte::tokenizeWithMetadata(text)` is the metadata tokenization path for callers that need atom side channels; it is intentionally not another `encode*` overload.
- `UniByte::decode(DecodeRequest)` is the single high-level decode wrapper. Plain `decode(ids)` calls still work through `DecodeRequest`; atom-aware decode uses `decode(UniByteResult)` so repeated same-type atoms resolve through `atom_entry_ids` and `AtomTable`.
- `ByteEncoder` and `UnigramLM` each expose exactly one `encode` and one `decode` primitive. Do not add pointer/vector/GPU overload chains back into these classes.

## Hyperparameter grouping
Config-driven tokenizer paths consume `GRIM::HyperParameters::TokenizerHP` directly from `HyperparameterGroupings.hpp`:

- `train_tokenizer.cu` creates the grouping after `loadStartupConfig()` and passes it into `PrepareTrainingDataFromCache()`.
- `DataLoader.cu` receives that grouping and constructs `UniByte` from it directly while building vocab + GRMT artifacts.
- `DataInfoReady()`, `tokenizer_runner.cu`, and `tokenizer_self_test.cu` construct `UniByte` from the same grouping for runtime validation/loading.

Do not hand-copy `TokenizerConfig` + `TrainingHyperparameters` fields into a tokenizer wrapper. `UniByte` stores `TokenizerHP` directly; isolated tokenizer tests use explicit `TokenizerHP` fixtures when they need non-config startup values.

## Trie / detector registry construction
- Trie is built from learned unigram pieces only. Layout special tokens are not trie entries.
- `UniByte::DetectorState` owns a `DetectorRegistry` built during construction, **not** lazily.
- Raw-text detection must go through `DetectorRegistry`; do not call detector implementations directly from tokenizer runtime code.
- Detectors operate on source byte offsets only. Token-ID checks stay in token-layout helpers such as `isSpecialTokenId`, `TokenLayout::isByte`, `TokenLayout::isAtom`, and `TokenLayout::isUnigram`.

## AtomTable indexing
Token IDs include the atom offset. When indexing `entries_[]`:
```cpp
uint32_t idx = id - ATOM_TOKEN_OFFSET;
```

## Training data
- HTML must be stripped before tokenization. `DataLoader.cu` handles `stripHtmlTags()` / `decodeHtmlEntities()` / `normalizeWhitespace()` automatically.
- `DataLoader.cu` tokenizes raw content only. It must not add `BOS`/`EOS`; Phase 1 startup routes sequences through `SlidingWindow.cu` for that layout work.
- `UniByte` intentionally exposes per-text encode paths only. Do not reintroduce `encodeBatch()` / vector-of-vector tokenization; corpus batching, `BOS`/`EOS`, and sequence windows belong to the `DataLoader.cu` → `SlidingWindow.cu` startup path.
- Sliding window: `overlap_len = raw_overlap - 1` when `raw_overlap > 0`, to avoid masking the same boundary target in two consecutive windows.

## Self-test
See [Build.md](Build.md) — `unigrambyte_self_test`.

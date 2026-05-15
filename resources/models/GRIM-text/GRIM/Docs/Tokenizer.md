# Tokenizer — Unigram + Byte Fallback

Files: `resources/models/GRIM-text/Shared/UnigramByte/`
- `GrimTokenizer.hpp` — alias to UniByte
- `Detectors/` — raw-text detector parent class, registry, numeric detectors, whitespace/uppercase feature detectors
- `Unigram.cu` — learned vocab, trie build, encode/decode wrappers
- `UnigramViterbi.hpp/.cu` — RAII Viterbi segmentation session and Viterbi CUDA kernels
- `UnigramGpuMemory.hpp/.cu` — `UnigramLM` CUDA buffer lifetime and GPU upload transactions

## Token layout
| Range | Meaning |
|-------|---------|
| `[0, 3]` | Reserved layout special tokens (`UNK`, `PAD`, `BOS`, `EOS`) |
| `[BYTE_TOKEN_OFFSET, BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE)` | Raw bytes (100% UTF-8 coverage) |
| `[ATOM_TOKEN_OFFSET, UNIGRAM_VOCAB_OFFSET)` | Atom placeholders |
| `[UNIGRAM_VOCAB_OFFSET, …)` | Learned unigram pieces |

Use `TokenLayout.hpp` / `Byte.hpp` constants and layout helpers for range checks; do not duplicate numeric offsets in runtime code.

## Vocab-size ownership
- `UniByte::vocabSize()` is the tokenizer's only public vocab-size API. It means the full token ID space: `UNIGRAM_VOCAB_OFFSET + UnigramLM::pieceCount()`.
- `UnigramLM::pieceCount()` is a component count for learned subword pieces only; never use it to size model embeddings or GRMT headers.
- `Shared/UnigramByte/VocabWriteOp.hpp` is the single learned-unigram vocab mutation primitive. Any learned-piece append, upsert, or compaction must call it once so `pieces_`, `piece_to_id_`, and the position-derived token ID stay synchronized.
- Direct writes to `pieces_` or `piece_to_id_` are forbidden outside `VocabWriteOp.hpp`; config-driven learned-vocab limits must be read from `GRIM::HyperParameters::TokenizerHP` in `HyperparameterGroupings.hpp`, not from raw config structs.
- `TokenizerArtifacts/VocabArtifactIO.*` stores a `serialized_record_count` in `vocab.bin` so the vocab reader knows how many records to read. That field is not a vocab size.
- `DataLoader.cu` writes `UniByte::vocabSize()` into the `.grmt` header when it encodes training data.
- Phase 1 startup reads final training vocab size from the `.grmt` header and passes that value into model allocation. It must not derive training vocab size from `ai_config.json`, `vocab.bin`, or tokenizer internals.
- `Shared/GRMT/GrmtFormat.hpp` owns `.grmt` magic, header layout, version validation, and header write helpers. Consumers must call `readHeaderOrThrow()` / `readHeaderStatus()` / `writeHeaderOrThrow()` instead of open-coding magic/version reads.
- Learned-piece pruning belongs inside tokenizer training before save. Do not add a post-load/post-save cap API; that creates a second vocab-size authority and can desynchronize `.grmt`, `vocab.bin`, tokenizer IDs, and model embeddings.

## Persistence primitives
- `Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp/.cu` is the single bundle primitive for the tokenizer cache pair: binary `vocab.bin` plus `training_data.grmt`. It loads the vocab and validates the `.grmt` header vocab against `UniByte::vocabSize()` before a cache can be accepted.
- `Shared/TokenizerArtifacts/GrmtCorpusIO.hpp/.cu` is the single GRMT row I/O primitive. It owns RAII file open/close, temp-file cleanup, header writes, row serialization/deserialization, and fail-loud validation for side-channel array alignment.
- `Shared/TokenizerArtifacts/VocabArtifactIO.hpp/.cu` is an internal read/write helper used only by `TokenizerArtifactBundle`; it is not a standalone tokenizer save/load path.
- `DataLoader.cu` treats vocab and GRMT as an inseparable cache bundle. If either artifact is missing or the pair fails validation, it retrains the tokenizer and regenerates both artifacts together.
- `training_data_loader.hpp`, `tokenizer_runner.cu`, and GRMT diagnostics must use `GrmtCorpusReader` / `loadGrmtCorpus()` instead of open-coding row seeks. Header-only checks may still call `GRMT::readHeaderOrThrow()` directly.
- Text vocab is export-only for human inspection. Runtime loading is binary KTMG through `TokenizerArtifactBundle`; do not re-add text vocab loading.

Special-token ownership is deliberately narrow:
- `TokenLayout.hpp` owns the reserved IDs and display metadata.
- `VocabArtifactIO` writes special-token records into saved vocab files so the persisted vocab advertises the full layout.
- The tokenizer does **not** inject `BOS`/`EOS`, pad batches, decide epoch/window boundaries, or mask training targets.
- `SlidingWindow.cu` owns `BOS`/`EOS` insertion and window-boundary target fixup.
- `BatchPayload.cu` owns padding and defense masking of `UNK`/`PAD`/`BOS` targets. `EOS` remains a valid target.

## Public encode/decode surface
- `UniByte::encode(text)` is the single high-level ID-only encode wrapper.
- `UniByte::tokenizeWithMetadata(text)` is the metadata tokenization path for callers that need atom side channels; it is intentionally not another `encode*` overload.
- `UniByte::decode(DecodeRequest)` is the single high-level decode wrapper. Plain `decode(ids)` calls still work through `DecodeRequest`; atom-aware decode uses `decode(UniByteResult)` so repeated same-type atoms resolve through `atom_entry_ids` and `AtomTable`.
- `ByteEncoder` and `UnigramLM` each expose exactly one `encode` and one `decode` primitive. Do not add pointer/vector/GPU overload chains back into these classes.
- `UnigramLM::decode()` is a primitive for byte fallback + learned unigram tokens only. It must reject any token outside that primitive range; layout-aware decode belongs to `UniByte::decode(DecodeRequest)`.
- Token type classification belongs to `TokenLayout`. Do not add `UniByte::isByteToken`, `UniByte::isAtomToken`, `UniByte::isUnigramToken`, or `UniByte::tokenToString` wrappers; callers that need diagnostics should use `tokenLayout()` and read pieces directly by token ID.

## Memory / tokenization boundary
- `Unigram.hpp` must not expose raw CUDA buffer layout. It forward-declares `UnigramGpuMemory` and stores it as an opaque owner.
- `Unigram.cu` owns learned vocab I/O, trie construction, encode/decode wrappers, and primitive byte/unigram decode behavior.
- `UnigramViterbi.hpp/.cu` owns per-segmentation RAII dynamic-programming state, Viterbi backtracking validation, SentencePiece-style subword path selection, and Viterbi CUDA kernels. `UnigramLM::encode()` and tokenizer-training E-steps must instantiate `UnigramViterbiSession` instead of open-coding Viterbi/backtrack loops.
- CPU and CUDA Viterbi must consume the same forward trie: from each reachable start position, walk `text[pos]`, `text[pos + 1]`, ... through `children[c]`. Do not reverse-scan pieces unless the uploaded trie is explicitly changed to a reverse trie.
- Punctuation is data-driven, not a Viterbi boundary. In SentencePiece-style mode, punctuation bytes/chars remain in the normalized stream and may be part of learned unigram pieces (`.`, `...`, `don't`, `▁hello,`) if the trie contains those pieces and their scores win.
- Byte fallback is a coverage path only. When byte fallback is enabled, fallback transitions emit `BYTE_TOKEN_OFFSET + byte`, never `UNK_TOKEN_ID`, but selected-fallback metadata must be derived during final backtrack rather than from candidate edges created during forward DP.
- `TokenLayout.hpp::UNKNOWN_SCORE` is the single source for Viterbi fallback transition cost. CPU and CUDA Viterbi must read that constant directly; do not thread caller-provided unknown-score parameters through kernel launches or helper APIs.
- CUDA Viterbi kernels report logical validation failures through the explicit `error_code` pointer and named `kUnigramViterbiCuda*` status constants. Do not use device `assert()` for correctness checks; callers/tests must synchronize, copy the status word, and fail if it is not `kUnigramViterbiCudaOk`.
- CUDA backtrack must fail with `kUnigramViterbiCudaOutputBufferTooSmall` when the selected path token count exceeds `max_tokens`; never silently truncate and never rely on build-mode-dependent asserts for this path.
- Punctuation-heavy structural spans (URLs, paths, numbers, etc.) must be handled by raw-text detectors/training skip spans, not by hard-coded punctuation splitting inside Viterbi.
- `UnigramGpuMemory.hpp/.cu` owns `UnigramLM` durable GPU buffers, `cudaMalloc`/`cudaFree`, host-to-device upload packing, and `UnigramLM::initGPU()` / `uploadTrieToGPU()` implementations.
- GPU upload is transactional: build a fresh `UnigramGpuMemory` first, then move it into `UnigramLM` only after every allocation and copy succeeds.
- Do not add raw CUDA pointer members, cleanup lambdas, or `cudaMalloc`/`cudaFree` blocks back into `Unigram.hpp` or `Unigram.cu`; extend `UnigramGpuMemory` instead.

## Hyperparameter grouping
Config-driven tokenizer paths consume `GRIM::HyperParameters::TokenizerHP` directly from `HyperparameterGroupings.hpp`:

- `train_tokenizer.cu` creates the grouping after `loadStartupConfig()` and passes it into `PrepareTrainingDataFromCache()`.
- `DataLoader.cu` receives that grouping and constructs `UniByte` from it directly while building vocab + GRMT artifacts.
- `DataInfoReady()`, `tokenizer_runner.cu`, and `tokenizer_self_test.cu` construct `UniByte` from the same grouping for runtime validation/loading.

Do not hand-copy `TokenizerConfig` + `TrainingHyperparameters` fields into a tokenizer wrapper. `UniByte` stores `TokenizerHP` directly; isolated tokenizer tests use explicit `TokenizerHP` fixtures when they need non-config startup values.

## Trie / detector registry construction
- Trie is built from learned unigram pieces only. Layout special tokens are not trie entries.
- Viterbi consumes the built trie as a read-only segmentation index; it does not create learned pieces or mutate vocab storage.
- Rebuild the trie only after the learned-piece set changes during load/train. Runtime startup must load the saved final vocab as-is, not mutate it to a new cap.
- `UniByte` owns a `DetectorRegistry` built during construction, **not** lazily.
- Raw-text detection exists to mark source-byte spans before tokenization. Atom-emitting detectors (currently integer/float) become `StructuralSpan`s and `AtomTable` entries; non-atom detectors (currently whitespace/uppercase runs) are source-text features for diagnostics and future policies, not token IDs.
- Raw-text detection must go through `DetectorRegistry`; do not call detector implementations directly from tokenizer runtime code.
- Every concrete raw-text detector must live under `Shared/UnigramByte/Detectors/` and be registered by `makeDefaultRawTextDetectorRegistry()`. Do not add detector-like kernels, local scanner functions, or hard-coded pattern checks in `UniByte.cu`.
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

# Tokenizer — Atoms, Unigram, and Byte Overflow

Files: `resources/models/GRIM-text/Shared/UnigramByte/`
- `GrimTokenizer.hpp` — alias to UniByte
- `Detectors/` — raw-text detector parent class, registry, numeric detectors, whitespace/uppercase feature detectors
- `Unigram.cu` — learned vocab, trie build, encode/decode wrappers
- `UnigramViterbi.hpp/.cu` — RAII Viterbi segmentation session and Viterbi CUDA kernels
- `Training/SubwordMining.hpp/.cu` — training-only subword candidate mining, deterministic byte-proportional sampling, atom-aware count aggregation, and overflow-checked count math
- `Training/UnigramForwardBackward.hpp/.cu` — training-only true Unigram forward-backward expected-count estimator over learned-piece paths on non-atom residual spans
- `UnigramGpuMemory.hpp/.cu` — `UnigramLM` CUDA buffer lifetime and GPU upload transactions

## Intended tokenizer order

The tokenizer architecture is deliberately staged in this order:

1. **Atoms first.** Raw-text detectors identify numeric atoms on the original source bytes, and those spans are fully handled outside the unigram model.
2. **Pure unigram second.** `UnigramTrainer` / `Training/UnigramForwardBackward.*` train the learned-piece model only on the residual non-atom text.
3. **Byte overflow last.** Byte fallback is the runtime overflow path for any residual byte sequence that the finalized learned unigram model would otherwise leave as `UNK`.

This means byte fallback is **not** part of the intended unigram EM objective, **not** a competing path inside the learned-piece distribution, and **not** allowed to steer pruning decisions. If current code threads byte fallback through training telemetry or the forward-backward lattice, treat that as architectural drift to remove, not the desired design.

## Token layout
| Range | Meaning |
|-------|---------|
| `[0, 3]` | Reserved layout special tokens (`UNK`, `PAD`, `BOS`, `EOS`) |
| `[BYTE_TOKEN_OFFSET, BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE)` | Raw bytes (100% UTF-8 coverage) |
| `[ATOM_TOKEN_OFFSET, UNIGRAM_VOCAB_OFFSET)` | Atom placeholders |
| `[UNIGRAM_VOCAB_OFFSET, …)` | Learned unigram pieces |

Use `TokenLayout.hpp` constants and layout helpers for range checks; do not duplicate numeric offsets in runtime code.

## Vocab-size ownership
- `UniByte::vocabSize()` is the tokenizer's only public vocab-size API. It means the full token ID space: `UNIGRAM_VOCAB_OFFSET + UnigramLM::pieceCount()`.
- `UnigramLM::pieceCount()` is a component count for learned subword pieces only; never use it to size model embeddings or GRMT headers.
- `Shared/UnigramByte/VocabWriteOp.hpp` is the single learned-unigram vocab mutation primitive. Any learned-piece append, upsert, or compaction must call it once so `pieces_`, `piece_to_id_`, and the position-derived token ID stay synchronized.
- `UnigramLM::writePiece()` has been removed. Do not re-add a public/manual vocab-builder wrapper; training and artifact loading must wire `VocabWriteOp.hpp` directly until the tokenizer class split is completed.
- Direct writes to `pieces_` or `piece_to_id_` are forbidden outside `VocabWriteOp.hpp`; the transitional public storage exists only to form `UnigramVocabWriteTarget` while `UnigramLM` is being dismantled. Config-driven learned-vocab limits must be read from `GRIM::HyperParameters::TokenizerHP` in `HyperparameterGroupings.hpp`, not from raw config structs.
- `TokenizerArtifacts/VocabArtifactIO.*` stores a `serialized_record_count` in `vocab.bin` so the vocab reader knows how many records to read. That field is not a vocab size.
- `DataLoader.cu` writes `UniByte::vocabSize()` into the `.grmt` header when it encodes training data.
- Phase 1 startup reads final training vocab size from the `.grmt` header and passes that value into model allocation. It must not derive training vocab size from `ai_config.json`, `vocab.bin`, or tokenizer internals.
- `TrainingContext` does not own a durable `TokenLayout` or `DataInfo` sidecar. Training carries the authoritative `.grmt` header vocab size on `SequenceData.vocab_size`, syncs `config.vocab_size` from that fact during Phase 1, and callers that need token ranges derive `TokenLayout` locally via `TokenLayout.hpp`; do not instantiate `UniByte` during `LoadTrainingData()` just to fetch token-type counts. Training/inference diagnostics that already have a payload in scope should read `BatchPayload::vocab_size`.
- Inference startup and explicit diagnostic-inference helpers may load a runtime tokenizer and must call `UniByte::initGPU()` immediately after loading the tokenizer artifact bundle. Those paths own the tokenizer explicitly at the call boundary; they do not park it on `TrainingContext`.
- `Shared/GRMT/GrmtFormat.hpp` owns `.grmt` magic, header layout, version validation, and header write helpers. Consumers must call `readHeaderOrThrow()` / `readHeaderStatus()` / `writeHeaderOrThrow()` instead of open-coding magic/version reads.
- Learned-piece pruning belongs inside tokenizer training before save. Do not add a post-load/post-save cap API; that creates a second vocab-size authority and can desynchronize `.grmt`, `vocab.bin`, tokenizer IDs, and model embeddings.

## Persistence primitives
- `Shared/TokenizerArtifacts/GrmtSequence.hpp` is the canonical GRMT row payload type: token IDs, supervision targets, atom side channels, and execution metadata. Move row-shape fields and row-level validation methods there instead of burying them inside reader/writer headers.
- `Shared/TokenizerArtifacts/TokenizerArtifactBundle.hpp/.cu` is the single function primitive for the tokenizer cache pair: binary `vocab.bin` plus `training_data.grmt`. Its load/save/exists functions consume `TokenizerHP` directly, load the vocab, and validate the `.grmt` header vocab against `UniByte::vocabSize()` before a cache can be accepted.
- `Shared/TokenizerArtifacts/GrmtCorpusIO.hpp/.cu` is the single GRMT row I/O primitive. It owns RAII file open/close, temp-file cleanup, header writes, row serialization/deserialization, and fail-loud validation for persisted row contents; the `GrmtSequence` type itself lives in `GrmtSequence.hpp`.
- `Shared/TokenizerArtifacts/VocabArtifactIO.hpp/.cu` is an internal read/write helper used only by `TokenizerArtifactBundle` functions; it is not a standalone tokenizer save/load path.
- `DataLoader.cu` treats vocab and GRMT as an inseparable cache bundle. If either artifact is missing or the pair fails validation, it retrains the tokenizer and regenerates both artifacts together.
- `Shared/DataLoader/DataLoader.cu` and GRMT diagnostics must use `GrmtCorpusReader` / `loadGrmtCorpus()` instead of open-coding row seeks. Header-only checks may still call `GRMT::readHeaderOrThrow()` directly.
- Text vocab is export-only for human inspection. Runtime loading is binary KTMG through `TokenizerArtifactBundle` functions; do not re-add text vocab loading.

Special-token ownership is deliberately narrow:
- `TokenLayout.hpp` owns the reserved IDs and display metadata.
- `VocabArtifactIO` writes special-token records into saved vocab files so the persisted vocab advertises the full layout.
- The tokenizer does **not** inject `BOS`/`EOS`, pad batches, decide epoch/window boundaries, or mask training targets.
- `SlidingWindow.cu` owns `BOS`/`EOS` insertion and window-boundary target fixup.
- `BatchPayload.cu` owns padding and defense masking of `UNK`/`PAD`/`BOS` targets. `EOS` remains a valid target.

## Public encode/decode surface
- `UniByte::encode(text)` is the single high-level ID-only encode wrapper.
- `UniByte::tokenizeWithMetadata(text)` is the metadata tokenization path for callers that need atom side channels; it is intentionally not another `encode*` overload.
- `UniByte::decode(DecodeRequest)` is the single high-level decode primitive. `DecodeRequest` may carry ID-only text, tokenizer-produced `UniByteResult` atom side channels, or Phase2-generated numeric side channels; do not add local append/decode wrappers in inference, diagnostics, or server code.
- `UniByte::decode(DecodeRequest)` must buffer contiguous byte-fallback token IDs as a raw byte run and validate that each run is well-formed UTF-8 before appending it to the returned text. If a generated byte-token run is malformed, decode must fail loudly with token/byte context; do not reinterpret fallback bytes as Latin-1/Windows-1252 and do not emit invalid UTF-8 into JSON/text APIs.
- Tokenizer artifact-preparation paths train through `UnigramLM::trainFromCorpus(texts, TokenizerHP)`. The raw-text detector prepass and atom-span selection are owned there; do not recreate a `UniByte::trainFromCorpus()` wrapper.
- Standalone `ByteEncoder` has been removed. Byte-token IDs and conversions live in `TokenLayout.hpp`; do not recreate a wrapper class around raw byte fallback.
- `UnigramLM::decode()` is a primitive for byte fallback + learned unigram tokens only. It must reject any token outside that primitive range; layout-aware decode belongs to `UniByte::decode(DecodeRequest)`.
- Token type classification belongs to `TokenLayout`. Do not add `UniByte::isByteToken`, `UniByte::isAtomToken`, `UniByte::isUnigramToken`, or `UniByte::tokenToString` wrappers; callers that need diagnostics should derive a layout with `tokenLayoutFromActualVocabOrThrow(tokenizer.vocabSize(), caller)` and read pieces directly by token ID.
- Atom placeholders are still real model tokens after tokenizer rewrite. `ATOM_INT` / `ATOM_FLOAT` keep learned embedding and LM-head rows inside the token-type gate's ATOM subspace so the model can represent and emit placeholder tokens directly, while the per-instance numeric payload remains in the side channels (`token_numeric_values`, `token_atom_mask`, `atom_entry_ids`, `AtomTable`). Do not zero atom placeholder embedding rows or skip their embedding-backward scatter-add; only `PAD` is a structural zero row in the embedding op.

## Memory / tokenization boundary
- `Unigram.hpp` must not expose raw CUDA buffer layout. It forward-declares `UnigramGpuMemory` and stores it as an opaque owner.
- `Unigram.cu` owns learned vocab I/O, trie construction, encode/decode wrappers, and primitive byte/unigram decode behavior.
- `UnigramViterbi.hpp/.cu` owns the CUDA-backed production Viterbi session, per-segmentation launch/backtrack validation, SentencePiece-style subword path selection, and Viterbi CUDA kernels. `UnigramLM::encode()` must instantiate `UnigramViterbiSession` instead of open-coding Viterbi/backtrack loops. Tokenizer-training E-steps must use `Training/UnigramForwardBackward.*` because true Unigram EM needs posterior expected counts over all learned-piece segmentations of the residual non-atom text.
- Production Viterbi consumes the uploaded forward trie: from each reachable start position, walk `text[pos]`, `text[pos + 1]`, ... through `children[c]`. Do not reverse-scan pieces unless the uploaded trie is explicitly changed to a reverse trie.
- `UnigramLM::initGPU()` is the default-capacity runtime initializer for generic/server use. Corpus/tokenizer training paths must not rely on that static default. CUDA Viterbi must fail loudly if the GPU trie is uninitialized or its uploaded generation does not match the live trie generation.
- Tokenizer training owns the final production-runtime boundary: after the final score mutation and final `buildTrie()`, `UnigramLM::trainFromCorpus()` must call `initGPUForMaxSequenceLength(longest_normalized_training_segment)`, populate `UnigramTrainingRuntimeReport`, and return only with the uploaded generation matching the live trie. `UnigramLM::trainFromCorpus(texts, TokenizerHP)` now asserts that ready state before returning; callers that need to verify the finalized runtime must do so directly through `tokenizer.unigramLM().requireRuntimeReadyForLastTraining(...)` instead of repairing training with generic `initGPU()` after training.
- Punctuation is data-driven, not a Viterbi boundary. In SentencePiece-style mode, punctuation bytes/chars remain in the normalized stream and may be part of learned unigram pieces (`.`, `...`, `don't`, `▁hello,`) if the trie contains those pieces and their scores win.
- ASCII spacing bytes are rewritten before unigram segmentation: space, tab, LF, CR, and CRLF all become the shared SentencePiece `▁` marker, with CRLF collapsed to one marker. This intentionally prevents raw `\n` / `\r` / `\t` byte fallback tokens from surviving into the runtime segmentation stream; decode maps `▁` back to plain space because the single marker does not preserve the original whitespace kind.
- Byte fallback is a runtime coverage path only. It is deliberately **raw byte-level fallback** with an **unnormalized fixed per-byte penalty**, not UTF-8-character fallback and not part of the learned-piece probability distribution. It runs only after atoms have already been removed from the text and after learned-piece segmentation has consumed the residual non-atom span. When byte fallback is enabled, each fallback transition advances exactly one byte and emits `BYTE_TOKEN_OFFSET + byte`, never `UNK_TOKEN_ID`. A multibyte UTF-8 codepoint therefore produces one fallback transition/token per byte if no learned piece covers it. CUDA Viterbi records `d_viterbi_prev_is_fallback[end]` on the selected backpointer during forward DP, and backtrack reads that flag to mark selected-fallback metadata. Never infer fallback selection from token ID range.
- `TokenLayout.hpp::UNKNOWN_SCORE` is the single source for the unnormalized per-byte fallback penalty. Runtime segmentation must read that constant directly and apply it once per raw fallback byte; do not thread caller-provided unknown-score parameters through kernel launches or helper APIs. Scores are initialized with `kViterbiUnreachableScore`, but forward-DP reachability is owned by backpointers: position `0` is reachable by definition, and every other position is reachable only when `viterbi_prev[pos] >= 0`. Training-time use of `UNKNOWN_SCORE` inside the unigram EM lattice is architectural drift, not intended ownership.
- CUDA Viterbi tie-breaking is explicit for exact score ties: prefer learned-piece transitions over fallback transitions, then prefer the longer span, then prefer the lower token ID. Change `shouldReplaceViterbiTransition()` and this doc together if the policy changes.
- CUDA Viterbi kernels report logical validation failures through the explicit `error_code` pointer and named `kUnigramViterbiCuda*` status constants. Do not use device `assert()` for correctness checks; callers/tests must synchronize, copy the status word, and fail if it is not `kUnigramViterbiCudaOk`.
- CUDA backtrack must fail with `kUnigramViterbiCudaOutputBufferTooSmall` when the selected path token count exceeds `max_tokens`; never silently truncate and never rely on build-mode-dependent asserts for this path.
- Do not add standalone greedy trie lookup kernels for production segmentation. A best local trie match is not Viterbi-compatible because it ignores prior path score, fallback cost, and tie policy; segmentation must go through `kernelViterbiForward` plus `kernelViterbiBacktrack`.
- Punctuation-heavy structural spans (URLs, paths, numbers, etc.) must be handled by raw-text detectors/training skip spans, not by hard-coded punctuation splitting inside Viterbi.
- `Training/UnigramForwardBackward.hpp/.cu` owns tokenizer-training soft EM. It builds a training-local trie from the current learned pieces, runs log-space forward-backward over every normalized non-atom residual segment, and accumulates posterior expected counts for learned pieces only. Do not replace this with `UnigramViterbiSession` inside training; Viterbi is hard-EM / single-best-path re-estimation, not true Unigram EM.
- `UnigramLM::trainFromCorpus()` must fail at entry for invalid `target_vocab_size`, `character_coverage`, `min_subword_freq`, or `subword_mining_workers` so malformed direct calls cannot reach shrink/EM logic.
- Original atom spans must be validated before logging byte totals or calling `normalizeWithSpans()`. Span size mismatches, reversed spans, out-of-bounds spans, and overlapping/unsorted spans are hard failures before any offset rewrite.
- Byte fallback must stay outside the unigram EM objective. The training lattice, posterior expected counts, shrink ranking, and dead-token cleanup are intended to be learned-piece-only decisions over the residual non-atom text. If current code surfaces byte-fallback telemetry, dominance guards, or fallback-driven posterior mass inside training, that is bolt-on debt threading through the tokenizer rather than the architectural target.
- Step-2 character seeds are a transient training bootstrap/coverage diagnostic only. They must not be emitted as learned pieces, must not pre-seed structural dedup, must not suppress mined one-character candidates from competing normally, and must not receive pruning protection. After candidate mining starts, training must behave as if those seeds never entered the learned-vocab decision path.
- `Training/SubwordMining.hpp/.cu` owns tokenizer-training subword candidate mining. `UnigramTrainer.cu` must call `mineUnigramSubwordsFromTrainingUnits()` exactly once, then consume the returned `UnigramSubwordMiningResult`; it must not rebuild mining spans, worker-local maps, or candidate counters inline.
- Subword mining must admit one-character candidates through the ordinary mined-subword path even when a segment has no atom spans. Character coverage diagnostics are not a second vocab-construction path.
- Vocab-character validation must use `utf8DecodeAt()` for the single-codepoint decode. Do not hand-decode multi-byte candidates; continuation-byte checks, overlong rejection, surrogate rejection, truncation checks, and max-codepoint validation belong to `TextUtils`.
- Structural dedup keys in tokenizer training are comparison-only. They may trim structural edge whitespace/format codepoints, but must never trim SentencePiece `▁` (U+2581): `▁word` and `word` are distinct because word-initial position is semantic. Accepted candidates must store the original mined subword text, not the edge-trimmed dedup key, so learned pieces keep their exact boundary markers and bytes.
- Initial candidate score normalization must fail loudly if candidate admission produces zero accepted learned pieces or if accepted candidate counts do not sum to a positive finite value. Do not clamp the denominator to `1.0`; that hides the empty/invalid accepted-candidate root cause instead of surfacing it before initial score assignment.
- Do not use prefix-extension/same-count dedup as a pre-EM filter. Equal counts are not proof of redundancy; longer candidates may improve compression and must be judged by forward-backward EM plus posterior-mass pruning.
- Subword mining byte caps must use deterministic byte-proportional spans distributed across each full normalized document. Do not sample only prefixes: EM trains on full documents, so prefix-only candidate mining starves late-document patterns and creates a false candidate/EM mismatch. Sampled spans carry `MAX_PIECE_LENGTH - 1` bytes of UTF-8-snapped context overlap around the intended span, and miners must only count candidate starts inside the intended span so boundary-crossing pieces can be discovered without inflating sampled-start coverage.
- Shrink pruning ranks pieces by posterior expected mass plus expected compression gain from the converged E-step: `posterior_count + posterior_count * max(piece_bytes - 1, 0)`. Do not normalize the compression gain by expected byte span, because that gives rare long pieces an almost free ratio bonus. Do not use `count * abs(score)` as a marginal-value proxy; it can overprotect rare low-probability pieces that soft segmentation barely uses.
- User-defined learned pieces are protected explicitly: insert all user-defined indices into the shrink keep set first, then fill remaining slots by posterior mass. Never rely on a max-value sort sentinel, because more protected pieces than `keep_count` must still all survive.
- Final dead-token cleanup must fail before compaction if the dead set would delete every learned piece. Do not rewrite `pieces_` to empty and let Phase-D fail later during forward-backward lattice construction; that signals a broken pure-unigram training objective or candidate set, not something byte fallback should rescue.
- Subword mining counts must be `uint64_t`/overflow-checked and exact after global merge. Large corpora can exceed signed `int`; never store candidate counts in signed 32-bit containers. `tokenizer.prune_during_mining` is accepted as a no-op compatibility flag, but mining must not erase low-frequency candidates in worker-local maps or before merge; a piece seen once per worker can still be globally frequent.
- `UnigramGpuMemory.hpp/.cu` owns `UnigramLM` tokenizer runtime state: the derived GPU mirror of the host trie/pieces plus reusable Viterbi workspace capacity. It owns `cudaMalloc`/`cudaFree`, host-to-device upload packing, and `UnigramLM::initGPU()` / `initGPUForMaxSequenceLength()` / `uploadTrieToGPU()` implementations.
- GPU upload is transactional: a file-local RAII upload transaction owns partial allocations until every allocation/copy succeeds, then commits under `UnigramGpuMemory::viterbi_workspace_mutex`. Do not move-assign `UnigramGpuMemory` itself as scratch upload state.
- `UnigramGpuMemory` owns the reusable CUDA Viterbi workspace and serializes both Viterbi execution and runtime upload replacement with its mutex; do not allocate per-call Viterbi CUDA buffers in `Unigram.cu` or callers.
- Do not add raw CUDA pointer members, cleanup lambdas, or `cudaMalloc`/`cudaFree` blocks back into `Unigram.hpp` or `Unigram.cu`; extend `UnigramGpuMemory` instead.

## Hyperparameter grouping
Config-driven tokenizer paths consume `GRIM::HyperParameters::TokenizerHP` directly from `HyperparameterGroupings.hpp`. `TokenizerHP` carries resolved `data_path`, `vocab_path`, and `force_rebuild_vocab`; tokenizer code must not carry `StartupConfig.paths`, `PathConfig`, or rebuild booleans beside it. Tokenizer leaves collapsed directly under `training.config` use the `tokenizer_*` prefix, including `tokenizer_target_vocab_size`, `tokenizer_max_vocab_size`, `tokenizer_max_length`, `tokenizer_character_coverage`, `tokenizer_min_cleaned_text_length`, `tokenizer_min_subword_freq`, `tokenizer_prune_during_mining`, `tokenizer_enable_parallel_subword_mining`, `tokenizer_subword_mining_workers`, `tokenizer_subword_mining_max_bytes`, `tokenizer_model_type`, `tokenizer_add_bos`, `tokenizer_add_eos`, `tokenizer_unk_token`, `tokenizer_pad_token`, `tokenizer_bos_token`, `tokenizer_eos_token`, `tokenizer_enable_nfkc_normalization`, `tokenizer_enable_lowercasing`, `tokenizer_enable_parallel_tokenization`, `tokenizer_parallel_threshold`, `tokenizer_enable_byte_fallback`, `tokenizer_expected_checksum`, `tokenizer_save_text_vocab`, `tokenizer_vocab_score_multiplier`, `tokenizer_special_tokens`, `tokenizer_enable_scratch_block_reasoning`, and `tokenizer_detect_numbers`; do not recreate nested `training.config.tokenizer.*` fields:

- `train_tokenizer.cu` loads `AiConfigSnapshot::document`, finalizes it through HyperParameters, slices `TokenizerHP` directly from `AiConfigSnapshot`, and passes it into `PrepareTrainingDataFromCache()`.
- `DataLoader.cu` receives that grouping, constructs `UniByte` for tokenization, and calls `tokenizer.unigramLM().trainFromCorpus(corpus, tokenizer_hp)` while building vocab + GRMT artifacts.
- `LoadInferenceTokenizer()` and `tokenizer_self_test.cu` construct `UniByte` from the same grouping for runtime validation/loading. `LoadTrainingData()` consumes the resulting `.grmt` header/runtime token-layout facts instead of constructing `UniByte`.

Do not hand-copy raw snapshot `tokenizer_*` document leaves plus `TrainingHyperparameters` into a tokenizer wrapper, `LanguageModelConfig` tokenizer grouping, or `StartupConfig::tokenizer_*` mirror. `UniByte` stores `TokenizerHP` directly; isolated tokenizer tests use explicit `TokenizerHP` fixtures when they need non-config startup values.

## Trie / detector registry construction
- Trie is built from learned unigram pieces only. Layout special tokens are not trie entries.
- Viterbi consumes the built trie as a read-only segmentation index; it does not create learned pieces or mutate vocab storage.
- Rebuild the trie only after the learned-piece set changes during load/train. Runtime startup must load the saved final vocab as-is, not mutate it to a new cap.
- `UniByte` owns a `DetectorRegistry` built during construction, **not** lazily.
- Raw-text detection exists to mark source-byte spans before tokenization. Atom-emitting detectors (currently integer/float) become `StructuralSpan`s and `AtomTable` entries; non-atom detectors (currently whitespace/uppercase runs) are source-text features for diagnostics and future policies, not token IDs.
- Raw-text detection must go through `DetectorRegistry`; do not call detector implementations directly from tokenizer runtime code.
- Every concrete raw-text detector must live under `Shared/UnigramByte/Detectors/` and be registered by `makeDefaultRawTextDetectorRegistry()`. Do not add detector-like kernels, local scanner functions, or hard-coded pattern checks in `UniByte.cu`.
- Detector-emitted atom spans are an upstream contract: if a detector marks a span as `ATOM_INT` / `ATOM_FLOAT` and `createAtomTableFromRawTextDetections()` cannot register the raw text, runtime tokenization, tokenizer training, and GRMT reconstruction must throw immediately. Do not route that span back through unigram text tokenization and do not silently skip it during training.
- `createAtomTableFromRawTextDetections()` is the AtomTable creation and atom-token payload boundary: it allocates the per-sequence `AtomTable`, converts `RawTextDetection` records into `StructuralSpan`s, registers atom-emitting spans exactly once, and returns `AtomTokenizationPayload` records containing the finalized span, placeholder token ID, `atom_entry_id`, packed numeric value, atom flags, and atom mask before any unigram segmentation of non-atom gaps.
- `createAtomTableFromRawTextDetectionsForTokenSideChannels()` is the persisted-token reconstruction boundary: it performs AtomTable creation and validates/materializes per-token `atom_entry_ids` from caller-owned token side-channel arrays in one AtomTable-owned call. GRMT readers must not duplicate token ID, mask, flag, or numeric payload validation around this op.
- `UniByte::tokenizeWithMetadata()` must call `createAtomTableFromRawTextDetections()` immediately after `DetectorRegistry::scan()`. The later merge loop consumes returned `AtomTokenizationPayload` records directly; do not fetch `AtomEntry` again, create a parallel finalized-span record, or call `registerSpan()` a second time during placeholder merge.
- Detectors operate on source byte offsets only. Token-ID checks stay in token-layout helpers such as `isSpecialTokenId`, `TokenLayout::isByte`, `TokenLayout::isAtom`, and `TokenLayout::isUnigram`.
- Whitespace boundaries are detector/data-quality ownership. `AtomTable` must not trim, normalize, canonicalize, or repair atom text; a numeric atom containing any whitespace is an upstream detector error and registration must fail.

## AtomTable indexing
Token IDs include the atom offset. When indexing `entries_[]`:
```cpp
uint32_t idx = id - ATOM_TOKEN_OFFSET;
```

AtomTable safety contracts:
- `AtomTable::getAtom()` and `getAtomsByType()` return `AtomEntry` copies. Do not return raw pointers or references into `entries_` after releasing the table mutex.
- GPU uploads are internal-owner only: callers use `uploadToGPU(cudaStream_t)` and inspect the read-only pointer from `getGPUBuffer()`. Do not add a caller-owned `uploadToGPU(GPUAtomData&)` overload.
- GPU uploads are transactional: allocate/copy into a fresh temporary `GPUAtomData`, synchronize the copy stream, then replace internal `gpu_data_` and clear `gpu_dirty_` / `pending_gpu_upload_` only after every CUDA operation succeeds.
- `uploadToGPU(cudaStream_t)` must not free existing `gpu_data_` when the table is already clean; a clean repeat upload only refreshes `num_atoms`.
- Atom dedup uses hash buckets (`hash -> vector<atom_id>`) and compares every candidate by type plus raw text so a hash collision cannot evict an older atom from dedup lookup.
- Numeric GPU side channels keep exact payload arrays (`double`, `int64_t`, and `NumericPayloadKind`) in addition to the legacy packed float used by older embedding paths. Do not round integer atoms through `float` for GPU numeric reconstruction.
- Public numeric lookup is ID-based: use `AtomTable::getNumericValue(atom_id)` and read `NumericPayload`. Do not add `getNumericValue(const AtomEntry&)`; `AtomEntry::numeric_value` is a lossy legacy packed float.
- AtomTable payloads are numeric-only: `AtomValue` may contain only `AtomInteger` or `AtomFloat`, and `parseAtom()` accepts only `ATOM_INT` / `ATOM_FLOAT`. Do not add `AtomGeneric` or any generic/unknown payload path.
- Numeric parse failures must not register as `ATOM_INT` / `ATOM_FLOAT`. Reject the registration; there is no generic fallback payload.
- Binary AtomTable load must validate every `StringRef` against `string_pool_` before exposing entries; corrupt offsets/lengths are a hard load failure.
- Atom stringification is raw round-trip only: `atomToString()` returns `raw_text_ref`. Do not add parsed/canonical string storage (`parsed_ref`) or route decode/export through canonicalized numeric text.

## Training data
- HTML must be stripped before tokenization. `DataLoader.cu` handles `stripHtmlTags()` / `decodeHtmlEntities()` / `normalizeWhitespace()` automatically.
- `DataLoader.cu` tokenizes raw content only. It must not add `BOS`/`EOS`; Phase 1 startup routes sequences through `SlidingWindow.cu` for that layout work.
- `UniByte` intentionally exposes per-text encode paths only. Do not reintroduce `encodeBatch()` / vector-of-vector tokenization; corpus batching, `BOS`/`EOS`, and sequence windows belong to the `DataLoader.cu` → `SlidingWindow.cu` startup path.
- Minimum text-length gating is config-owned: `training.config.tokenizer_min_cleaned_text_length` is loaded into `TrainingHyperparameters`, sliced directly through `TokenizerHP`, and consumed by `DataLoader.cu` before GRMT encoding. Do not hard-code this threshold in the loader.
- Sliding window: `overlap_len = raw_overlap - 1` when `raw_overlap > 0`, to avoid masking the same boundary target in two consecutive windows.

## Self-test
See [Build.md](Build.md) — `unigrambyte_self_test`.

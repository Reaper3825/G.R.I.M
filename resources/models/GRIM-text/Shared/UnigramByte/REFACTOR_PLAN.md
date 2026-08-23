# UnigramByte Tokenizer — Refactor Plan

> **Goal:** Separation of concerns, eliminate overlaps, establish clear data flow.  
> **Scope:** File-level reorganization only. No algorithm changes, no API changes.  
> **Rule:** Every file earns its existence by owning exactly one responsibility.
> **Status note:** Viterbi segmentation now lives in `UnigramViterbi.hpp/.cu` as a per-run RAII session; `Unigram.cu` owns learned-vocab/trie wrappers, not Viterbi DP state. `Byte.hpp/.cu` were deleted; byte-token constants and conversion helpers now live in `TokenLayout.hpp`.

---

## Current State — The Problem

**10 files, 8,149 lines. Three of them are monoliths:**

| File | Lines | Problem |
|------|------:|---------|
| Unigram.cu | 2,620 | Training pipeline (~1,300 lines), inference (Viterbi/trie), GPU upload, vocab I/O, EM algorithm, noise filters, UTF-8 helpers, SentencePiece normalization, character validation — all in one file |
| AtomTable.cu | 1,745 | 12 atom type parsers (only 1 is active), serialization, GPU upload, string pool, dedup, registration, metadata |
| UniByte.cu | 1,271 | Orchestration mixed with detector implementations, atom metadata encoding, GPU kernels, decode logic |
| Unigram.hpp | 329 | Owns `AtomType` enum even though it's consumed by AtomTable and UniByte — wrong home |
| AtomTable.hpp | 478 | 12 atom value structs for types that are dead code |
| UniByte.hpp | 402 | Declares `Detector::` namespace functions implemented 1,000 lines away in UniByte.cu |
| AhoCorasick.cu | 540 | Reasonably scoped ✓ |
| AhoCorasick.hpp | 204 | Reasonably scoped ✓ |

### Root Causes

1. **Unigram.cu is doing 5+ jobs:** subword mining, EM training, noise filtering, Viterbi inference, trie building, vocab I/O (text + binary KTMG), GPU trie upload, SentencePiece normalization, UTF-8 utilities, character validation
2. **Dead code dominates AtomTable:** 11 of 12 atom type parsers are unreachable (only `ATOM_NUM` detection remains active in UniByte.cu). Full parser implementations for URL, Email, Path, Date, Time, IP, StringLiteral, Identifier sit in ~600 lines of dead code
3. **AtomType enum lives in the wrong file:** Defined in Unigram.hpp, consumed by AtomTable.hpp (includes Unigram.hpp just for this) and UniByte.cu. This creates a false coupling where the atom system depends on the unigram tokenizer
4. **Detector functions are orphaned:** Declared in `UniByte.hpp` `Detector::` namespace, implemented at the bottom of UniByte.cu (~280 lines). These are pure parsing functions with zero dependency on UniByte state
5. **No training/inference boundary:** `trainFromCorpus()` (1,300+ lines) sits next to `encode()` (20 lines) in the same file. Training helpers (`mineSubwordsFromSentence`, noise filters, etc.) are visible to inference code

---

## Include Dependency Chain (Current)

```
AhoCorasick.hpp ←── Unigram.hpp (for AtomType enum only)
AtomTable.hpp ←── Unigram.hpp (for AtomType enum and token layout constants)
UniByte.hpp ←── Unigram.hpp, AtomTable.hpp, AhoCorasick.hpp, TokenLayout.hpp
```

**The problem:** `AhoCorasick.hpp` and `AtomTable.hpp` both include `Unigram.hpp` solely for `AtomType` and token layout constants. This creates a false impression that the atom system and pattern matcher depend on the unigram language model.

---

## Target Architecture

### New File Map (14 files → each under ~500 lines)

```
UnigramByte/
├── TokenLayout.hpp          [NEW]  ~80 lines   — Token ID constants, AtomType enum, layout math
├── TextUtils.hpp            [NEW]  ~60 lines   — UTF-8 helpers, SentencePiece normalization decls
├── TextUtils.cu             [NEW]  ~200 lines  — normalizeSpaces, denormalizeSpaces, utf8SequenceLength, isValidVocabCharacter, character validators
├── Detectors/
│   ├── TokenizerDetector.hpp      [NEW] — RawTextDetector parent class + result types
│   ├── DetectorRegistry.hpp/.cu   [NEW] — detector registration + longest-match raw-text scan
│   ├── AtomDelimiterDetector.hpp/.cu [NEW] — authored typed-delimiter atom-span placement
│   └── TextFeatureDetectors.hpp/.cu [NEW] — whitespace/uppercase raw-text feature detectors
├── AhoCorasick.hpp          [MOD]  ~200 lines  — Remove #include "Unigram.hpp", include "TokenLayout.hpp" instead
├── AhoCorasick.cu           [KEEP] ~540 lines  — No changes
├── AtomTable.hpp            [MOD]  ~480 lines  — Remove #include "Unigram.hpp", include "TokenLayout.hpp" instead
├── AtomTable.cu             [MOD]  ~1100 lines — Delete dead parsers (~600 lines removed)
├── Unigram.hpp              [MOD]  ~250 lines  — Remove AtomType, token layout constants (moved to TokenLayout.hpp), include TokenLayout.hpp
├── Unigram.cu               [MOD]  ~1100 lines — Learned vocab, trie, encode/decode wrappers, vocab I/O
├── UnigramViterbi.hpp/.cu   [NEW]  ~400 lines  — RAII Viterbi session, DP buffers, Viterbi kernels
├── UnigramTrainer.hpp       [NEW]  ~40 lines   — trainFromCorpus() declaration, training config struct
├── UnigramTrainer.cu         [NEW]  ~1300 lines — trainFromCorpus(), subword mining, EM, noise filters, sentence segmentation
├── UniByte.hpp              [MOD]  ~340 lines  — public tokenizer API; no public detector methods; raw-text detection is registry-owned
├── UniByte.cu               [MOD]  ~700 lines  — Orchestration only: raw detection scan, encode pipeline, decode, GPU kernels
```

**Total: 17 files, ~6,770 lines (vs current 10 files, 8,149 lines)**  
~1,380 lines deleted (dead atom parsers + dead detector code for removed atom types).

---

## Extraction Plan — Ordered Steps

### Step 1: Create `TokenLayout.hpp` (foundation — unblocks everything else)

**Move FROM `Unigram.hpp`:**
- `AtomType` enum
- All token layout constants: `SPECIAL_TOKEN_OFFSET`, `NUM_SPECIAL_TOKENS`, `UNK_TOKEN_ID`, `PAD_TOKEN_ID`, `BOS_TOKEN_ID`, `EOS_TOKEN_ID`, `BYTE_TOKEN_OFFSET`, `BYTE_VOCAB_SIZE`, `ATOM_TOKEN_OFFSET`, `UNIGRAM_VOCAB_OFFSET`, `MAX_PIECE_LENGTH`, `UNKNOWN_SCORE`
- `configureTokenLayout()` function
- `atomTypeToTokenId()` / `tokenIdToAtomType()` helpers
- `tokenIdForIndex()` / `indexForTokenId()` utility functions

**Why first:** This is the shared vocabulary that currently forces AtomTable.hpp and AhoCorasick.hpp to include Unigram.hpp. Extracting it breaks the false dependency chain.

**After this step:**
```
TokenLayout.hpp ←── (standalone, no includes beyond <cstdint>)
Unigram.hpp ←── TokenLayout.hpp (instead of defining these itself)
AtomTable.hpp ←── TokenLayout.hpp (instead of including Unigram.hpp)
AhoCorasick.hpp ←── TokenLayout.hpp (instead of including Unigram.hpp)
```

### Step 2: Create `TextUtils.hpp` / `TextUtils.cu`

**Move FROM `Unigram.cu` (anonymous namespace / file-scope helpers):**
- `utf8SequenceLength()` — used by Unigram.cu, UniByte.cu, AtomTable.cu
- `normalizeSpaces()` / `denormalizeSpaces()` — SentencePiece ▁ normalization
- `normalizeWithSpans()` — span-aware normalization (training only, but shares logic)
- `isValidVocabCharacter()` — character validation for vocab building
- `isValidSubword()` — subword validation

**Policy note:** Viterbi must not contain hard-coded punctuation isolation. GRIM-text uses SentencePiece-style subword selection: punctuation remains ordinary normalized text and is handled by learned pieces or byte fallback.

**Why second:** These are pure utility functions with no state. Currently duplicated or forward-declared across files. Extracting them gives every file a clean import.

### Step 3: Create `Detectors/` raw-text detector subsystem

**Current target shape:**
- `Detectors/TokenizerDetector.hpp` — `RawTextDetector` parent class, raw detection result, detector options.
- `Detectors/DetectorRegistry.hpp/.cu` — registration, duplicate-name validation, priority ordering, longest-match raw-text scan.
- `Detectors/AtomDelimiterDetector.hpp/.cu` — authored typed-delimiter atom-span placement.
- `Detectors/TextFeatureDetectors.hpp/.cu` — `WhitespaceDetector`, `UppercaseRunDetector` non-atom raw-text feature detectors.

**Delete dead detectors:** Cross-reference `DetectorRegistry::scan()` call sites to confirm which detectors emit atoms. Hex/binary/path/date/time/IP/string/identifier detectors are not active and must not be recreated without direct registry ownership.

### Step 4: Extract `UnigramTrainer.hpp` / `UnigramTrainer.cu`

**Move FROM `Unigram.cu` (~1,300 lines):**
- `trainFromCorpus()` — the entire training pipeline method
- `mineSubwordsFromSentence()` — subword candidate generation
- `resolveSubwordMiningWorkerCount()` / `resolveSubwordMiningChunkSize()` — parallelism helpers
- `structuralDedupKeyForCandidate()` — edge-trim dedup
- All noise filter functions: `isRepetitionNoise()`, `hasExcessiveRunLength()`, `isRepeatedPatternNoise()`, `isDoubledTokenNoise()`, `isWordLevelStutter()`
- Sentence segmentation logic (the `clipAtomSpans` / `addSentence` lambdas)
- EM convergence loop (`runEStep`, `runMStep`, `runEMToConvergence` lambdas → extract as methods)
- Dead token pruning + backfill logic

**UnigramTrainer.cu includes:** `Unigram.hpp` (for UnigramLM class access), `TextUtils.hpp`

**Result:** `Unigram.cu` shrinks from 2,620 → ~1,100 lines (inference, trie, GPU, vocab I/O only).

**Implementation approach:** `UnigramTrainer` can be:  
- **(A)** A friend function/class that operates on `UnigramLM&` internals, or  
- **(B)** `trainFromCorpus()` stays as a method on `UnigramLM` but the implementation file is `UnigramTrainer.cu` (same class, split compilation unit)

Option **(B)** is simpler — just move the method definition to a different `.cu` file. No API change.

### Step 5: Clean up `AtomTable.cu` — Delete dead parsers

**DELETE FROM `AtomTable.cu` (~600 lines):**
- `parseURL()` — detection removed, parser unreachable
- `parseEmail()` — detection removed, parser unreachable
- `parsePath()` — detection removed, parser unreachable (verify)
- `parseDate()` — detection removed, parser unreachable
- `parseTime()` — detection removed, parser unreachable
- `parseIPAddress()` — detection removed, parser unreachable
- `parseStringLiteral()` — detection removed, parser unreachable
- `parseIdentifier()` — detection removed, parser unreachable
- Corresponding serialization arms in `atomValueSerialize()` / `atomValueToString()` for these types
- Corresponding `AtomValue` variant types in `AtomTable.hpp` if no longer referenced

**KEEP:**
- `parseInteger()`, `parseFloat()`, `parseHexInteger()`, `parseBinaryInteger()` — actively used
- `parseAtom()` dispatch — simplify to only handle active types
- All GPU upload, string pool, dedup, registration, save/load logic

**Also DELETE from `AtomTable.hpp`:**
- Unused `AtomValue` struct variants: `AtomURL`, `AtomEmail`, `AtomPath`, `AtomDate`, `AtomTime`, `AtomIP`, `AtomString`, `AtomIdentifier`
- Keep: `AtomInteger`, `AtomFloat` only. Do not keep or recreate `AtomGeneric`; unsupported source features stay in detector/data-quality ownership and must not become AtomTable payloads.

**Verify:** Check if `AtomType::ATOM_URL` etc. are referenced anywhere outside the dead parsers before deleting the enum values from `TokenLayout.hpp`. They may still be needed for the token ID range even if unused. If the enum values are only used by dead parser dispatch, delete them too.

### Step 6: Slim `UniByte.cu` — Remove extracted code

After Steps 3-4, update UniByte.cu:
- Remove detector function implementations (now in `Detectors/`)
- Remove any training-related helpers that moved to `UnigramTrainer.cu`
- Add `#include "Detectors/DetectorRegistry.hpp"` where raw-text scanning is needed.
- **Result:** ~700 lines of pure orchestration

### Step 7: Update `CMakeLists.txt`

Add new `.cu` files to the build:
- `TextUtils.cu`
- `Detectors/DetectorRegistry.cu`
- `Detectors/AtomDelimiterDetector.cu`
- `Detectors/TextFeatureDetectors.cu`
- `UnigramTrainer.cu`

New `.hpp` files (`TokenLayout.hpp`, `TextUtils.hpp`, `Detectors/*.hpp`, `UnigramTrainer.hpp`) are header-only or included by their `.cu` counterparts — verify they're picked up.

---

## New Include Dependency Graph (Target)

```
TokenLayout.hpp          ←── (standalone: <cstdint>, <string>)
   ↑
   ├── TextUtils.hpp     ←── TokenLayout.hpp
   ├── AhoCorasick.hpp   ←── TokenLayout.hpp
   ├── Unigram.hpp       ←── TokenLayout.hpp
   ├── AtomTable.hpp     ←── TokenLayout.hpp
    ├── Detectors/TokenizerDetector.hpp ←── TokenLayout.hpp
    ├── Detectors/DetectorRegistry.hpp  ←── Detectors/TokenizerDetector.hpp
      └── UniByte.hpp       ←── Unigram.hpp, AtomTable.hpp, TokenLayout.hpp, Detectors/TokenizerDetector.hpp
       ↑
       UnigramTrainer.hpp ←── Unigram.hpp, TextUtils.hpp
```

**Key improvement:** No file includes another file "just for an enum." Each inclusion represents a genuine dependency on the included file's primary abstraction.

---

## Dead Code Audit

### Confirmed Dead (DELETE)

| Code | Location | Evidence |
|------|----------|----------|
| `parseURL()` | AtomTable.cu ~120 lines | `ATOM_URL` detection removed from raw detection consumers |
| `parseEmail()` | AtomTable.cu ~80 lines | `ATOM_EMAIL` detection removed |
| `parsePath()` | AtomTable.cu ~60 lines | `ATOM_PATH` detection removed from pipeline |
| `parseDate()` | AtomTable.cu ~70 lines | `ATOM_DATE` detection removed |
| `parseTime()` | AtomTable.cu ~50 lines | `ATOM_TIME` detection removed |
| `parseIPAddress()` | AtomTable.cu ~60 lines | `ATOM_IP` detection removed |
| `parseStringLiteral()` | AtomTable.cu ~80 lines | `ATOM_STRING_LITERAL` detection removed |
| `parseIdentifier()` | AtomTable.cu ~60 lines | `ATOM_IDENTIFIER` detection removed |
| retired date/time/IP/string/identifier raw detectors | legacy detector extraction scope | Not registered in `DetectorRegistry` |
| `AtomURL`, `AtomEmail`, `AtomPath`, `AtomDate`, `AtomTime`, `AtomIP`, `AtomString`, `AtomIdentifier` structs | AtomTable.hpp | Only used by dead parsers |

**Total dead code: ~750+ lines**

### Needs Verification Before Delete

| Code | Location | Check |
|------|----------|-------|
| retired path raw detector | registry setup | Only keep if a direct `DetectorRegistry` consumer is added |
| retired URL/email/path/date/time/IP/string/identifier atom enum values | TokenLayout.hpp | Check serialized checkpoint/vocab compatibility before deleting token-layout slots |

---

## Validation Checklist

After each step, verify:

- [ ] `cmake --build build --config Release --target train_gpu` compiles
- [ ] `cmake --build build --config Release --target grim_text_server` compiles
- [ ] `cmake --build build --config Release --target unigrambyte_self_test` compiles and passes
- [ ] No new compiler warnings
- [ ] No `#include` cycles (each header compiles standalone with its own includes)
- [ ] `grep -rn "Unigram.hpp" *.hpp` → only files that genuinely use `UnigramLM` class (not just AtomType/constants)
- [ ] Token IDs unchanged — encode("hello world") produces identical output pre/post refactor

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| `trainFromCorpus()` accesses private `UnigramLM` members | Medium | Use split compilation (Option B): method stays on class, definition in separate .cu file. No `friend` needed |
| Noise filter functions reference `UnigramLM` internal helpers | Low | They're pure functions of string → bool. No class state needed. Move cleanly |
| `normalizeWithSpans` used only in training but shares code with `normalizeSpaces` | Low | Put both in TextUtils.cu. Training includes TextUtils.hpp |
| Dead parser deletion breaks checkpoint loading | Medium | Verify `AtomTable::loadBinary()` deserialization — if it reads atom type tags, keep enum values but delete parser code. The loader should skip unknown types gracefully |
| GPU kernel functions in anonymous namespaces | Low | Keep CUDA kernels in the same .cu as the function that launches them. Don't split kernels from their launchers |

---

## Execution Order Summary

```
1. TokenLayout.hpp          — foundation, unblocks include fixes
2. TextUtils.hpp/.cu        — pure utilities, unblocks everything
3. Detectors/*.hpp/.cu      — raw-text detector parent class, registry, active detectors
4. UnigramTrainer.hpp/.cu   — extract from Unigram.cu (biggest win)
5. AtomTable.cu cleanup     — delete dead parsers
6. UniByte.cu slim           — remove extracted code
7. CMakeLists.txt update     — wire new files into build
```

Each step is independently testable. If any step breaks the build, it can be reverted without affecting the others.

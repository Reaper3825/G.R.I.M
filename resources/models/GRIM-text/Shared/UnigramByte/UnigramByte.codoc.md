# UnigramByte Tokenizer — Codeboard

> A readable map of the tokenizer code after the refactor.
> If you only need the public API, start with `UniByte.hpp`.

---

## What this tokenizer is

`UnigramByte` is a composed tokenizer with four jobs:

1. `UnigramLM` does normal subword tokenization.
2. `ByteEncoder` guarantees 100% UTF-8 coverage when no unigram piece matches.
3. `AhoCorasick` + `Detectors` find numeric structures in raw text.
4. `AtomTable` stores parsed numeric values so the model can keep a placeholder token and still recover the real value.

The top-level class is `UniByte`. Everything else exists to support it.

---

## Start here

Read the tokenizer in this order:

1. `UniByte.hpp` / `UniByte.cu` — top-level orchestration and public API
2. `Unigram.hpp` / `Unigram.cu` — vocab, trie, Viterbi encode/decode, vocab I/O
3. `AtomTable.hpp` / `AtomTable.cu` — parsed numeric atom registry
4. `AhoCorasick.hpp` / `AhoCorasick.cu` + `Detectors.hpp` / `Detectors.cu` — numeric detection
5. `Byte.hpp` / `Byte.cu` — byte fallback
6. `TextUtils.hpp` / `TextUtils.cu` — normalization and UTF-8 helpers
7. `UnigramTrainer.hpp` / `UnigramTrainer.cu` — training pipeline only

If you read it in a different order, the code starts to feel like a haunted house.

---

## High-level architecture

```mermaid
flowchart LR
    TL[TokenLayout
shared constants] --> BY[ByteEncoder]
    TL --> AC[AhoCorasick]
    TL --> AT[AtomTable]
    TL --> UG[UnigramLM]
    DT[Detectors] --> UB[UniByte
orchestrator]
    BY --> UB
    AC --> UB
    AT --> UB
    UG --> UB
    TU[TextUtils] --> UG
    UG --> UT[UnigramTrainer]
```

### The mental model

- `TokenLayout.hpp` is the shared foundation.
- `UniByte` composes the rest; it should not re-implement their logic.
- `UnigramTrainer.cu` is training-only; inference lives in `Unigram.cu`.
- `Detectors.cu` contains pure pattern detection; `UniByte.cu` decides when to call it.

---

## Token layout

The live layout comes from `Byte.hpp` and `TokenLayout.hpp`.

| Range | Meaning | Notes |
|---|---|---|
| `0..3` | special tokens | `UNK=0`, `PAD=1`, `BOS=2`, `EOS=3` |
| `4..259` | byte fallback tokens | one token per raw byte |
| `260..261` | atom tokens | currently `ATOM_NONE` and `ATOM_NUM` |
| `262+` | unigram vocab | learned subword pieces |

### Masking rule

- Never predict `UNK`, `PAD`, or `BOS`.
- `EOS` **is** a valid target.
- Byte, atom, and unigram tokens are content-bearing tokens.

---

## Encode flow

```mermaid
flowchart LR
    IN[Input text] --> DET[detectStructures]
    DET --> REG[registerAtom in AtomTable]
    REG --> INJ[inject NUM placeholders]
    INJ --> NORM[normalizeSpaces]
    NORM --> SEG[UnigramLM Viterbi]
    SEG --> FB{piece found?}
    FB -->|yes| TOK[token ids]
    FB -->|no| BY[Byte fallback]
    BY --> TOK
    TOK --> OUT[UniByteResult and metadata]
```

### What actually happens

1. `UniByte::detectStructures()` finds numeric spans.
2. Each span is registered in `AtomTable` and gets metadata.
3. The original text is rewritten with numeric placeholders before unigram segmentation.
4. Text normalization happens before unigram segmentation.
5. `UnigramLM` runs Viterbi over the normalized text.
6. Any miss falls back to raw bytes through `ByteEncoder`.
7. `UniByteResult` is assembled and `validate()` checks that every parallel array matches `token_ids.size()`.

### The main result object

`UniByteResult` is the handoff object used by downstream code.

| Field | Purpose |
|---|---|
| `token_ids` | final token stream |
| `atoms` | detected source spans |
| `is_byte_fallback` | marks tokens created by byte fallback |
| `token_numeric_values` | per-token numeric payload |
| `token_atom_flags` | per-token atom flags |
| `token_text_features` | per-token side-channel text features |
| `token_atom_mask` | whether a token is an atom |
| `atom_entry_ids` | per-token index into `atom_table` |
| `atom_table` | parsed atom registry shared by the sequence |

---

## Training flow

```mermaid
flowchart LR
    CORPUS[Corpus] --> DETECT[detect numeric spans]
    DETECT --> SKIP[skip atom spans during training]
    SKIP --> SEED[seed vocabulary]
    SEED --> E[E-step Viterbi]
    E --> M[M-step recount]
    M --> P[prune low-value pieces]
    P -->|repeat| E
    P --> FINAL[final scores and buildTrie]
```

### Ownership split

- `UniByte::train()` orchestrates the training workflow.
- `UnigramLM::trainFromCorpus()` is declared in `Unigram.hpp`.
- The actual training implementation lives in `UnigramTrainer.cu`.
- Atom spans are skipped during training so numeric internals do not contaminate vocab statistics.

---

## File ownership

Use this table as the “who owns what?” map.

| File | Owns | Should not own |
|---|---|---|
| `TokenLayout.hpp` | token constants, `AtomType`, token-id helpers | runtime logic, parsing, CUDA state |
| `Byte.hpp/.cu` | byte token mapping and byte fallback encode/decode | unigram logic, atom parsing |
| `TextUtils.hpp/.cu` | UTF-8 helpers, SentencePiece-style whitespace normalization | tokenizer orchestration |
| `Detectors.hpp/.cu` | pure numeric detection helpers | token assembly, atom storage |
| `AhoCorasick.hpp/.cu` | multi-pattern prefix matching DFA | full numeric parsing, tokenizer output assembly |
| `AtomTable.hpp/.cu` | parsed atom storage, dedup, GPU packing, numeric value access | subword segmentation |
| `Unigram.hpp/.cu` | vocab, trie, Viterbi, vocab load/save, unigram GPU encode/decode | detection policy, top-level orchestration |
| `UnigramTrainer.hpp/.cu` | unigram training implementation | runtime encode path |
| `UniByte.hpp/.cu` | composition layer, public API, metadata assembly | low-level detector implementations or training internals |

---

## Main types you need to know

| Type | Defined in | Why it matters |
|---|---|---|
| `UniByte` | `UniByte.hpp` | top-level tokenizer class |
| `UniByteConfig` | `UniByte.hpp` | runtime tokenizer configuration |
| `TokenLayout` | `UniByte.hpp` | runtime view of token-region boundaries |
| `StructuralSpan` | `UniByte.hpp` | raw detected structure before or during placeholder injection |
| `UniByteResult` | `UniByte.hpp` | validated encode result passed downstream |
| `AtomType` | `TokenLayout.hpp` | shared atom type enum |
| `AtomSpan` | `Unigram.hpp` | training-only “skip this byte range” marker |
| `UnigramPiece` | `Unigram.hpp` | one vocab piece with score |
| `ViterbiNode` | `Unigram.hpp` | dynamic-programming state during segmentation |
| `AtomEntry` | `AtomTable.hpp` | cache-aligned stored atom record |
| `AtomValue` | `AtomTable.hpp` | parsed value variant for atoms |
| `AhoCorasickMatch` | `AhoCorasick.hpp` | prefix-match result for detector dispatch |

---

## Detection stack

There are two layers here on purpose:

1. `AhoCorasick` finds cheap prefixes quickly.
2. `Detectors` decide whether the full structure is actually valid.

Current detector surface:

| Function | Example inputs |
|---|---|
| `detectInteger` | `42`, `-17`, `+5` |
| `detectFloat` | `3.14`, `.5`, `-2.5e10` |
| `detectHex` | `0xFF`, `0x1A2B` |
| `detectBinary` | `0b1010` |

Current parser surface in `AtomTable`:

| Function | Purpose |
|---|---|
| `parseInteger` | decimal integer parsing |
| `parseFloat` | float parsing |
| `parseHex` | hexadecimal parsing |
| `parseBinary` | binary parsing |

Right now the only active emitted atom type is numeric: `ATOM_NUM`.

---

## Invariants you must not break

1. **Token IDs for unigram pieces are positional.**
   `token_id = UNIGRAM_VOCAB_OFFSET + index_in_pieces_`.

2. **`buildTrie()` must happen before encode.**
   Load or train vocab first, then build the trie.

3. **Byte fallback is the coverage guarantee.**
   If a piece is not found, tokenization must still succeed.

4. **Atom spans are skipped during training.**
   Training should not learn internal numeric formatting as normal subwords.

5. **`UniByteResult` arrays are parallel.**
   If one per-token field has length `n`, all per-token fields must have length `n`.

6. **`AhoCorasick` is built eagerly.**
   `DetectorState` owns the DFA and prepares it up front.

7. **`UniByte` composes; it should not absorb other layers.**
   Keep detector logic in `Detectors`, training logic in `UnigramTrainer`, and subword logic in `Unigram`.

---

## If you need to change something, go here

| Change | Primary file(s) |
|---|---|
| Add or change numeric detection | `Detectors.cu`, `AhoCorasick.cu`, `UniByte.cu` |
| Add a new atom type | `TokenLayout.hpp`, `Detectors.*`, `AtomTable.*`, `UniByte.*` |
| Change token-id ranges | `Byte.hpp`, `TokenLayout.hpp`, `UniByte.hpp` |
| Change normalization behavior | `TextUtils.cu` |
| Change unigram segmentation | `Unigram.cu` |
| Change vocab training | `UnigramTrainer.cu` |
| Change encode metadata layout | `UniByte.hpp`, `UniByte.cu` |
| Change atom GPU packing | `AtomTable.cu` |

---

## Minimal verification

After tokenizer changes, verify at least:

1. the tokenizer still builds,
2. `unigrambyte_self_test` still passes,
3. encode/decode behavior did not drift unexpectedly,
4. training still feeds into `buildTrie()` and the inference path unchanged.

---

## One-line summary

`UniByte` is the orchestrator, `UnigramLM` is the subword engine, `ByteEncoder` is the safety net, `AhoCorasick` + `Detectors` find numeric structure, and `AtomTable` keeps the real values behind the placeholder tokens.

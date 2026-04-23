# Tokenizer — Unigram + Byte Fallback

Files: `resources/models/GRIM-text/Shared/UnigramByte/`
- `GrimTokenizer.hpp` — alias to UniByte
- `AhoCorasick.cu` — pattern matching
- `Unigram.cu` — vocab + Viterbi

## Token layout
| Range | Meaning |
|-------|---------|
| `[0, 255]` | Raw bytes (100% UTF-8 coverage) |
| `[256, 511]` | Atom placeholders (`ATOM_TOKEN_BASE = 256`) |
| `[512, …]` | Unigram vocab |

## Trie / DFA construction
- Trie is auto-built by the constructor with special tokens. MUST exist before encoding.
- Aho-Corasick DFA is built during `DetectorState` construction, **not** lazily.

## AtomTable indexing
Token IDs include the `ATOM_TOKEN_BASE` (256) offset. When indexing `entries_[]`:
```cpp
uint32_t idx = id - ATOM_TOKEN_BASE;
```

## Training data
- HTML must be stripped before tokenization. `DataLoader.cu` handles `stripHtmlTags()` / `decodeHtmlEntities()` / `normalizeWhitespace()` automatically.
- Sliding window: `overlap_len = raw_overlap - 1` when `raw_overlap > 0`, to avoid masking the same boundary target in two consecutive windows.

## Self-test
See [Build.md](Build.md) — `unigrambyte_self_test` (37 tests).

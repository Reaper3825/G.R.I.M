# GRIM Tokenizer Architecture

**Unified Unigram Language Model + Byte Fallback + ScratchBlock Integration**

*Version 3.0 — January 2026*

---

## Executive Summary

The GRIM tokenizer is a hybrid subword tokenization system that combines:

1. **Unigram Language Model** — Probabilistic subword segmentation using Viterbi dynamic programming
2. **Byte-Level Fallback** — Guaranteed 100% UTF-8 coverage for any input
3. **Structural Atom Detection** — O(n) pattern recognition via Aho-Corasick DFA
4. **ScratchBlock Reasoning Layer** — Semantic side-channel for model-internal reasoning

This architecture achieves both high compression efficiency and lossless encoding while preserving structural semantics for downstream reasoning.

---

## Table of Contents

1. [Token ID Layout](#1-token-id-layout)
2. [System Components](#2-system-components)
3. [Encoding Pipeline](#3-encoding-pipeline)
4. [Unigram Language Model](#4-unigram-language-model)
5. [Byte Fallback System](#5-byte-fallback-system)
6. [Structural Detection (Aho-Corasick)](#6-structural-detection-aho-corasick)
7. [AtomTable & ScratchBlock Integration](#7-atomtable--scratchblock-integration)
8. [Text Feature Encoding](#8-text-feature-encoding)
9. [Vocabulary Training](#9-vocabulary-training)
10. [GPU Acceleration](#10-gpu-acceleration)
11. [File Formats](#11-file-formats)
12. [API Reference](#12-api-reference)
13. [Performance Characteristics](#13-performance-characteristics)

---

## 1. Token ID Layout

The tokenizer uses a fixed, non-overlapping ID space:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOKEN ID LAYOUT                                       │
├─────────────────┬────────────────────────────────────────────────────────────┤
│  [0 - 255]      │  BYTE TOKENS — Raw UTF-8 bytes (fallback encoding)         │
├─────────────────┼────────────────────────────────────────────────────────────┤
│  [256 - 274]    │  ATOM TOKENS — Structural placeholders (ScratchBlock)      │
│                 │    256: ATOM_NONE                                          │
│                 │    257: ATOM_END (boundary marker)                         │
│                 │    258: ATOM_INTEGER                                       │
│                 │    259: ATOM_FLOAT                                         │
│                 │    260: ATOM_HEX                                           │
│                 │    261: ATOM_BINARY                                        │
│                 │    262: ATOM_IDENTIFIER                                    │
│                 │    263: ATOM_STRING_LITERAL                                │
│                 │    264: ATOM_REGEX                                         │
│                 │    265: ATOM_URL                                           │
│                 │    266: ATOM_EMAIL                                         │
│                 │    267: ATOM_PATH                                          │
│                 │    268: ATOM_DATE                                          │
│                 │    269: ATOM_TIME                                          │
│                 │    270: ATOM_IP_ADDRESS                                    │
│                 │    271: ATOM_EQUATION                                      │
│                 │    272: ATOM_EXPRESSION                                    │
├─────────────────┼────────────────────────────────────────────────────────────┤
│  [275+]         │  UNIGRAM TOKENS — Learned subword vocabulary               │
│                 │    275: <unk>                                              │
│                 │    276: <pad>                                              │
│                 │    277: <s>   (BOS)                                        │
│                 │    278: </s>  (EOS)                                        │
│                 │    279+: Learned subwords...                               │
└─────────────────┴────────────────────────────────────────────────────────────┘
```

**Design Rationale:**
- Byte tokens at [0-255] allow direct `char` casting for decoding
- Atom tokens provide fixed-position placeholders for structural content
- Unigram tokens start after reserved ranges for unbounded vocabulary growth

---

## 2. System Components

### 2.1 File Structure

```
UnigramByte/
├── UniByte.hpp          # Main orchestrator (public API)
├── UniByte.cu           # Implementation + CUDA kernels
├── Unigram.hpp          # Unigram LM tokenizer
├── Unigram.cu           # Viterbi algorithm + training
├── Byte.hpp             # Byte-level encoder
├── Byte.cu              # GPU byte encoding
├── AhoCorasick.hpp      # DFA pattern matcher
├── AhoCorasick.cu       # Failure-link construction
└── AtomTable.hpp        # Atom storage & parsing
```

### 2.2 Class Hierarchy

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            UniByte                                          │
│                    (Main Orchestrator Class)                                │
│                                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  UnigramLM   │  │ ByteEncoder  │  │ AhoCorasick  │  │   AtomTable    │  │
│  │              │  │              │  │              │  │                │  │
│  │ • Vocab      │  │ • ID [0-255] │  │ • URL prefix │  │ • Atom storage │  │
│  │ • Trie       │  │ • UTF-8 safe │  │ • Email @    │  │ • Type parsing │  │
│  │ • Viterbi    │  │ • GPU encode │  │ • Hex 0x/0b  │  │ • GPU-packed   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Encoding Pipeline

### 3.1 High-Level Flow

```
                            INPUT TEXT
                                │
                                ▼
                    ┌───────────────────────┐
                    │  STRUCTURAL DETECTION │  ← Aho-Corasick DFA O(n)
                    │  (URLs, emails, nums) │
                    └───────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        ┌──────────┐      ┌──────────┐      ┌──────────┐
        │ SEGMENT  │      │  ATOM    │      │ SEGMENT  │
        │ (text)   │      │ DETECTED │      │ (text)   │
        └──────────┘      └──────────┘      └──────────┘
              │                 │                 │
              ▼                 ▼                 ▼
        ┌──────────┐      ┌──────────┐      ┌──────────┐
        │ UNIGRAM  │      │ ATOM TOK │      │ UNIGRAM  │
        │ ENCODE   │      │ + PARSE  │      │ ENCODE   │
        └──────────┘      └──────────┘      └──────────┘
              │                 │                 │
              └─────────────────┼─────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   TOKEN STREAM        │
                    │ + Numeric Values      │
                    │ + Text Features (FP16)│
                    └───────────────────────┘
```

### 3.2 Detailed Steps

| Step | Operation | Complexity | Location |
|------|-----------|------------|----------|
| 1 | Input validation | O(1) | `UniByte::encode()` |
| 2 | Structural detection | O(n) | `UniByte::detectStructures()` |
| 3 | Span sorting & deduplication | O(k log k) | `detectStructures()` |
| 4 | Whitespace widening | O(k) | `detectStructures()` |
| 5 | Segment iteration | O(n) | `encodeInternal()` |
| 6 | Unigram Viterbi encode | O(n × L) | `UnigramLM::encode()` |
| 7 | Atom parsing & feature extraction | O(k) | `encodeInternal()` |
| 8 | Result assembly | O(m) | `encodeInternal()` |

Where: n = input length, L = max token length (32), k = atom count, m = output tokens

---

## 4. Unigram Language Model

### 4.1 Mathematical Foundation

The Unigram LM assigns a probability to each subword piece:

$$P(\text{token}_i) = \exp(\text{score}_i)$$

For a segmentation $S = (t_1, t_2, \ldots, t_k)$:

$$P(S) = \prod_{i=1}^{k} P(t_i) = \exp\left(\sum_{i=1}^{k} \text{score}_i\right)$$

The **Viterbi algorithm** finds the optimal segmentation:

$$S^* = \arg\max_S \sum_{i=1}^{k} \text{score}_i$$

### 4.2 Viterbi Dynamic Programming

**State Definition:**
- `nodes[i].score` = Best log-probability to reach position `i`
- `nodes[i].prev_pos` = Previous position in optimal path
- `nodes[i].token_id` = Token ending at position `i`

**Recurrence:**
```
nodes[pos + len].score = max(
    nodes[pos + len].score,
    nodes[pos].score + trie[token].score
)
```

**Implementation Location:** `Unigram.cu` lines 1100-1165

```cpp
// Forward pass - try all pieces starting at each position
for (size_t pos = 0; pos < n; ++pos) {
    int node = 0;  // Trie root
    for (size_t len = 1; len <= MAX_PIECE_LENGTH && pos + len <= n; ++len) {
        unsigned char c = text[pos + len - 1];
        if (trie_[node].children[c] < 0) break;  // No match
        node = trie_[node].children[c];
        
        if (trie_[node].token_id >= 0) {  // Valid token endpoint
            float score = nodes[pos].score + trie_[node].score;
            if (score > nodes[pos + len].score) {
                nodes[pos + len] = {score, pos, trie_[node].token_id, len};
            }
        }
    }
}
```

### 4.3 Trie Data Structure

256-way prefix trie for O(L) token lookup:

```cpp
struct TrieNode {
    int token_id;           // -1 if not end of token
    float score;            // Log probability
    int children[256];      // Child indices (-1 if none)
};
```

**Properties:**
- Space: O(V × L) where V = vocab size, L = avg token length
- Lookup: O(L) for any token
- Cache-friendly: Sequential memory access during traversal

---

## 5. Byte Fallback System

### 5.1 Purpose

Guarantees 100% coverage for any UTF-8 input, including:
- Unknown Unicode characters
- Emojis and special symbols
- Corrupted or partial UTF-8 sequences
- Binary data embedded in text

### 5.2 Mechanism

When the Unigram model cannot segment a character:

```cpp
if (enable_byte_fallback_) {
    float byte_score = nodes[pos].score + UNKNOWN_SCORE;  // -100.0
    if (byte_score > nodes[pos + 1].score) {
        unsigned char byte_val = text[pos];
        nodes[pos + 1] = {byte_score, pos, byte_val, 1};  // Byte token ID = byte value
    }
}
```

### 5.3 UTF-8 Utilities

```cpp
namespace UTF8 {
    bool isStartByte(uint8_t b);      // Not continuation (10xxxxxx)
    bool isContinuation(uint8_t b);   // Is 10xxxxxx
    int sequenceLength(uint8_t b);    // 1-4 bytes from start byte
}
```

---

## 6. Structural Detection (Aho-Corasick)

### 6.1 Algorithm Overview

The **Aho-Corasick algorithm** enables simultaneous matching of multiple patterns in O(n) time using a deterministic finite automaton (DFA) with failure links.

**Advantage over regex:** 10-100× faster, no backtracking, predictable performance.

### 6.2 Pattern Categories

| Category | Patterns | AtomType |
|----------|----------|----------|
| **URLs** | `http://`, `https://`, `ftp://`, `ws://`, `wss://`, `file://` | `ATOM_URL` |
| **Email** | `@` (trigger full validation) | `ATOM_EMAIL` |
| **Numbers** | `0x`, `0X`, `0b`, `0B` | `ATOM_HEX`, `ATOM_BINARY` |

### 6.3 DFA Construction

```cpp
struct AhoCorasickNode {
    std::unordered_map<char, uint32_t> transitions;  // char → state
    uint32_t failure_link;                            // Suffix mismatch link
    std::vector<uint32_t> outputs;                    // Patterns ending here
};
```

**Failure Link Property:** When a character doesn't match, follow failure links to find longest proper suffix that is also a prefix of some pattern.

### 6.4 Detection Phases

```
Phase 1: Aho-Corasick prefix scan (O(n))
         → URL hits, email @ markers, hex/binary prefixes

Phase 2: Extent finding (O(k × L))
         → Extend prefixes to full spans (e.g., http:// → full URL)

Phase 3: Linear scan patterns (O(n))
         → Dates (YYYY-MM-DD), times (HH:MM:SS), IP addresses, paths

Phase 4: Deduplication & whitespace widening (O(k log k))
         → Sort by position, remove overlaps, preserve boundaries
```

---

## 7. AtomTable & ScratchBlock Integration

### 7.1 Concept

The **AtomTable** stores parsed representations of detected structures, enabling the model to reason about structural content without tokenizing it literally.

**Key Insight:** Instead of tokenizing `https://example.com/path?query=1` as ~15 subword tokens, emit a single `ATOM_URL` token with parsed components stored in AtomTable.

### 7.2 Atom Entry Structure (64-byte cache-aligned)

```cpp
struct alignas(64) AtomEntry {
    // Hot data (first 32 bytes)
    uint64_t hash;              // FNV-1a deduplication hash
    uint32_t id;                // Unique ID
    AtomType type;              // Type enum
    AtomCategory category;      // NUMERIC, TEMPORAL, STRUCTURAL, etc.
    AtomOrigin origin;          // USER_INPUT, MODEL_GENERATED, etc.
    StringRef raw_text_ref;     // Reference to string pool
    float confidence;           // Model confidence score
    
    // Warm data (second 32 bytes)
    uint64_t created_at;        // Timestamp
    uint32_t source_start/end;  // Position in original text
    float numeric_value;        // For numeric types
    uint32_t flags;             // Type-specific flags
    StringRef parsed_ref;       // Parsed string data
};
```

### 7.3 Parsed Value Types

```cpp
using AtomValue = std::variant<
    AtomInteger,    // {value: int64_t, base: int, has_sign: bool}
    AtomFloat,      // {value: double, has_exponent: bool, exponent: int}
    AtomURL,        // {scheme, host, port, path, query, fragment}
    AtomEmail,      // {local, domain}
    AtomPath,       // {is_absolute, is_windows, components[], extension}
    AtomDate,       // {year, month, day, format: ISO|US|EU}
    AtomTime,       // {hour, minute, second, is_24h, is_pm}
    AtomIP,         // {octets[4], is_valid}
    AtomString,     // {value, quote_char, has_escapes}
    AtomIdentifier, // {name, style: SNAKE|CAMEL|PASCAL|SCREAMING}
    AtomGeneric     // {raw_value}
>;
```

---

## 8. Text Feature Encoding

### 8.1 16-Dimensional FP16 Feature Vector

Each atom token carries a side-channel feature vector for model input:

```
┌─────────────────────────────────────────────────────────────────┐
│                  TEXT FEATURE LAYOUT (16 × FP16)                │
├─────────────┬───────────────────────────────────────────────────┤
│ [0-3]       │ CATEGORY ONE-HOT                                  │
│             │   [0] = NUMERIC, [1] = TEMPORAL,                  │
│             │   [2] = STRUCTURAL, [3] = STRING                  │
├─────────────┼───────────────────────────────────────────────────┤
│ [4-7]       │ SUB-TYPE ENCODING                                 │
│             │   Soft encoding for specific type within category │
├─────────────┼───────────────────────────────────────────────────┤
│ [8-11]      │ MAGNITUDE FEATURES                                │
│             │   [8] = Normalized length (len/100)               │
│             │   [9] = Log length (log₂(len+1)/10)               │
│             │   [10-11] = Type-specific (magnitude, sign, etc.) │
├─────────────┼───────────────────────────────────────────────────┤
│ [12-15]     │ SEMANTIC FEATURES                                 │
│             │   [12] = Digit ratio                              │
│             │   [13] = Alpha ratio                              │
│             │   [14] = Special char ratio                       │
│             │   [15] = Separator presence (/, :, ., @)          │
└─────────────┴───────────────────────────────────────────────────┘
```

### 8.2 Implementation

```cpp
void encodeAtomTextFeatures(
    AtomType atom_type,
    const std::string_view raw_text,
    const AtomValue* parsed,
    uint16_t* out_features  // [16] FP16 values
);
```

---

## 9. Vocabulary Training

### 9.1 Training Algorithm

```
1. SENTENCE SEGMENTATION
   └─ Split documents into sentences (prevents cross-boundary garbage tokens)

2. CHARACTER FREQUENCY ANALYSIS
   └─ Count all characters, sort by frequency

3. INITIAL VOCABULARY
   └─ Add special tokens: <unk>, <pad>, <s>, </s>
   └─ Add characters meeting coverage threshold (99.95%)

4. SUBWORD MINING
   └─ Generate n-grams (1 ≤ n ≤ 32) from sentences
   └─ Filter by minimum frequency (default: 3)
   └─ Filter linguistically invalid patterns (see below)

5. EM SCORE REFINEMENT
   └─ Run Viterbi on training corpus
   └─ Count actual token usage
   └─ Update scores: score = log(count / total)
   └─ Repeat for configured iterations

6. TRIE CONSTRUCTION
   └─ Build 256-way prefix trie for fast encoding
```

### 9.2 Subword Quality Filter

Rejects linguistically nonsensical patterns:

```cpp
static bool isValidSubword(const std::string& s) {
    // REJECT:
    // 1. Pure punctuation (except "...", "--")
    // 2. Single letter + punctuation ("P.,", "A:")
    // 3. Mixed punctuation chaos
    // 4. Space + punctuation (" ,", "; ")
    // 5. Mostly punctuation with scattered letters
    // 6. Lowercase + immediate uppercase ("aB")
    
    return true;  // Accept valid subwords
}
```

---

## 10. GPU Acceleration

### 10.1 CUDA Kernels

| Kernel | Purpose | Block Size |
|--------|---------|------------|
| `kernelClassifyTokens` | Classify byte/atom/unigram | 256 |
| `kernelDetectNumbers` | Parallel number detection | 256 |
| `kernelMarkStructuralBoundaries` | Mark URL/email positions | 256 |
| `kernelViterbiForward` | Parallel Viterbi forward pass | 256 |
| `kernelViterbiBacktrack` | Sequential backtracking | 1 |
| `kernelUnigramDecode` | Token ID → text | 1 |

### 10.2 GPU Data Structures

```cpp
struct GPUData {
    // Trie on device
    int* d_trie_children;       // [num_nodes × 256]
    int* d_trie_token_ids;      // [num_nodes]
    float* d_trie_scores;       // [num_nodes]
    
    // Piece data for decoding
    char* d_piece_data;         // Concatenated strings
    int* d_piece_offsets;       // Start offsets
    int* d_piece_lengths;       // Lengths
    
    // Viterbi workspace (pre-allocated)
    float* d_viterbi_scores;
    int* d_viterbi_prev;
    int* d_viterbi_tokens;
};
```

---

## 11. File Formats

### 11.1 Binary Vocabulary Format (KTMG v3)

```
┌────────────────────────────────────────────────────────┐
│                     HEADER (27 bytes)                  │
├────────────────┬───────────────────────────────────────┤
│ Magic          │ "KTMG" (4 bytes)                      │
│ Version        │ 3 (uint16)                            │
│ Checksum       │ Reserved (uint32)                     │
│ Vocab Size     │ Number of pieces (uint32)             │
│ Max Length     │ Max token length (uint32)             │
│ Flags          │ Reserved (3 bytes)                    │
│ Total Size     │ bytes + atoms + unigram (uint32)      │
├────────────────┴───────────────────────────────────────┤
│                    PIECES (variable)                   │
├────────────────┬───────────────────────────────────────┤
│ For each piece:│                                       │
│   Length       │ Text length (uint32)                  │
│   Text         │ UTF-8 bytes (variable)                │
│   Score        │ Log probability (float32)             │
│   Token ID     │ Assigned ID (int32) ← NEW in v3       │
└────────────────┴───────────────────────────────────────┘
```

### 11.2 Version History

| Version | Changes |
|---------|---------|
| v2 | Initial binary format |
| v3 | Added token_id per piece (no runtime computation) |

---

## 12. API Reference

### 12.1 UniByte (Main Class)

```cpp
class UniByte {
public:
    // Construction
    explicit UniByte(const UniByteConfig& config = UniByteConfig());
    
    // Initialization
    bool load(const std::string& vocab_path);
    bool save(const std::string& vocab_path, bool save_text = false) const;
    bool train(const std::vector<std::string>& texts);
    bool initGPU();
    
    // Encoding
    std::vector<int> encode(const std::string& text) const;
    UniByteResult encodeWithMetadata(const std::string& text) const;
    std::vector<std::vector<int>> encodeBatch(const std::vector<std::string>& texts) const;
    
    // Decoding
    std::string decode(const std::vector<int>& token_ids) const;
    std::string decodeWithAtoms(const std::vector<int>& token_ids,
                                 const AtomResolver& resolver) const;
    
    // Configuration
    void setScratchBlockReasoning(bool enabled);
    bool isScratchBlockReasoningEnabled() const;
};
```

### 12.2 UniByteResult Structure

```cpp
struct UniByteResult {
    std::vector<int> token_ids;              // Main output
    std::vector<StructuralSpan> atoms;       // Detected structures
    std::vector<bool> is_byte_fallback;      // Per-token fallback flag
    std::vector<float> token_numeric_values; // Numeric side-channel
    std::vector<uint8_t> token_numeric_mask; // Valid numeric flag
    std::vector<uint16_t> token_text_features; // [tokens × 16] FP16
    std::vector<uint8_t> token_text_mask;    // Valid feature flag
    
    size_t unigram_tokens;  // Count of unigram tokens
    size_t byte_tokens;     // Count of byte fallback tokens
    size_t atom_tokens;     // Count of atom tokens
};
```

---

## 13. Performance Characteristics

### 13.1 Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Encode (Viterbi) | O(n × L) | n = length, L = 32 |
| Structural Detection | O(n) | Aho-Corasick DFA |
| Decode | O(m) | m = token count |
| Trie Lookup | O(L) | L = max token length |
| Training (EM) | O(N × V × I) | N = corpus, V = vocab, I = iterations |

### 13.2 Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Vocabulary | O(V × L) | V = vocab size, L = avg length |
| Trie (CPU) | O(V × L × 256) | 256-way nodes |
| Trie (GPU) | O(V × L × 256 × 4) | int32 children |
| AtomEntry | 64 bytes | Cache-aligned |
| Text Features | 32 bytes/atom | 16 × FP16 |

### 13.3 Typical Performance

| Metric | Value | Configuration |
|--------|-------|---------------|
| Encode Speed (CPU) | ~2M chars/sec | Single thread |
| Encode Speed (GPU) | ~50M chars/sec | RTX 3080 |
| Compression Ratio | 3.5-4.5× | vs byte-level |
| Vocab Size | 37K-50K | Typical range |
| Byte Fallback Rate | <0.1% | On English text |

---

## Appendix A: Quick Reference

### Token Type Checking

```cpp
bool isByteToken(int id)    { return id >= 0 && id < 256; }
bool isAtomToken(int id)    { return id >= 256 && id < UNIGRAM_VOCAB_OFFSET; }
bool isUnigramToken(int id) { return id >= UNIGRAM_VOCAB_OFFSET; }
```

### Atom Type Conversion

```cpp
int atomTypeToTokenId(AtomType type) {
    return ATOM_TOKEN_OFFSET + static_cast<int>(type);
}

AtomType tokenIdToAtomType(int id) {
    return static_cast<AtomType>(id - ATOM_TOKEN_OFFSET);
}
```

### Configuration Example

```cpp
UniByteConfig config;
config.target_vocab_size = 50000;
config.character_coverage = 0.9995f;
config.min_subword_freq = 3;
config.enable_scratch_block_reasoning = true;
config.detect_numbers = true;
config.enable_byte_fallback = true;
config.prefer_gpu = true;

UniByte tokenizer(config);
tokenizer.load("vocab.bin");
```

---

## Appendix B: Design Decisions

### Why Unigram over BPE?

1. **Probabilistic Foundation** — Log-probabilities enable principled segmentation scoring
2. **Multiple Segmentations** — Can sample alternative tokenizations for data augmentation
3. **Better OOV Handling** — Graceful degradation with byte fallback integration
4. **Training Objective** — Optimizes actual language model likelihood, not merge frequency

### Why Aho-Corasick over Regex?

1. **Deterministic O(n)** — No backtracking, predictable performance
2. **Multi-Pattern** — Matches all patterns in single pass
3. **GPU-Friendly** — DFA representation maps well to parallel execution
4. **10-100× Faster** — Measured vs `std::regex` on typical inputs

### Why 64-byte AtomEntry?

1. **Cache Line Alignment** — One entry = one cache line fetch
2. **Hot/Warm Separation** — Frequently accessed fields in first 32 bytes
3. **Fixed Size** — Enables array-based storage, no pointer chasing
4. **GPU Transfer** — Coalesced memory access patterns

---

*Document generated for GRIM-text tokenizer review*  


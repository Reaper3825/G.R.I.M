# GRIM-text ScratchBlock Architecture

**Version:** 1.0  
**Last Updated:** December 2025  
**Module:** Internal Reasoning Layer with Atom Detection

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Philosophy](#2-design-philosophy)
3. [AtomTable Registry System](#3-atomtable-registry-system)
   - 3.1 [AtomEntry Structure](#31-atomentry-structure)
   - 3.2 [Atom Types and Categories](#32-atom-types-and-categories)
   - 3.3 [String Pool Interning](#33-string-pool-interning)
   - 3.4 [Hash-Based Deduplication](#34-hash-based-deduplication)
4. [Atom Parsing Pipeline](#4-atom-parsing-pipeline)
   - 4.1 [Type-Specific Parsers](#41-type-specific-parsers)
   - 4.2 [Parsed Value Types](#42-parsed-value-types)
   - 4.3 [Numeric Value Packing](#43-numeric-value-packing)
5. [ScratchBlock Neural Layer](#5-scratchblock-neural-layer)
   - 5.1 [Architecture Overview](#51-architecture-overview)
   - 5.2 [Token ID Range](#52-token-id-range)
   - 5.3 [Forward Pass](#53-forward-pass)
   - 5.4 [Backward Pass](#54-backward-pass)
6. [Value Encoding System](#6-value-encoding-system)
   - 6.1 [Atom Type Embeddings](#61-atom-type-embeddings)
   - 6.2 [Numeric VALUE Encoding](#62-numeric-value-encoding)
   - 6.3 [Text Feature Encoding](#63-text-feature-encoding)
7. [GPU Implementation](#7-gpu-implementation)
   - 7.1 [CUDA Kernels](#71-cuda-kernels)
   - 7.2 [Memory Layout](#72-memory-layout)
   - 7.3 [Weight Initialization](#73-weight-initialization)
8. [Serialization Format](#8-serialization-format)
9. [Configuration](#9-configuration)
10. [Integration Points](#10-integration-points)
11. [Performance Characteristics](#11-performance-characteristics)
12. [Mathematical Foundations](#12-mathematical-foundations)

---

## 1. Executive Summary

The ScratchBlock system is an **internal reasoning layer** that enables GRIM-text to understand and reason about **structural elements** in text before generating responses. Unlike traditional language models that treat numbers, URLs, and dates as arbitrary character sequences, ScratchBlock:

1. **Detects** structural atoms during tokenization (via Aho-Corasick pattern matching)
2. **Parses** atoms into typed, semantic representations (integers, floats, URLs, dates, etc.)
3. **Encodes** both the TYPE and the actual VALUE into learnable embeddings
4. **Injects** these semantic embeddings into the transformer's hidden states

This architecture allows the model to "understand" that `192.168.1.1` is an IP address with specific octets, or that `2024-12-25` is a date in December, rather than treating them as opaque token sequences.

### Key Innovation: VALUE Encoding

Traditional transformers use **one-hot** atom type embeddings (e.g., "this is an integer"). GRIM-text extends this with **VALUE encoding**—the embedding captures both the type AND the numeric magnitude:

```
Embedding = TypeOneHot[0:15] + LogMagnitude[16:31] + Sign[32:47]
```

This enables mathematical reasoning: the model knows `1000000` is a larger integer than `42`.

---

## 2. Design Philosophy

### 2.1 Offline-First Atom Detection

Atom detection happens at **tokenization time** (CPU-side), not during inference. The tokenizer uses Aho-Corasick DFA for O(n) pattern detection, identifying structural elements before they enter the model.

### 2.2 Zero-Copy String Handling

The AtomTable uses **string pool interning** with `std::string_view` references. No string copies occur during tokenization—only pointers to the original text buffer.

### 2.3 Passthrough Mode

When disabled, ScratchBlock is a pure **identity function** with zero computational overhead. This allows A/B testing and fallback behavior.

### 2.4 Cache-Line Optimization

AtomEntry is exactly **64 bytes** (one cache line) with carefully ordered fields to minimize cache misses during sequential access.

---

## 3. AtomTable Registry System

The AtomTable is a **CPU-side registry** that stores parsed structural atoms during tokenization. Each unique atom gets a dedicated token ID in the range `[256, 274]`.

### 3.1 AtomEntry Structure

```cpp
// 64 bytes exactly - one cache line
struct alignas(64) AtomEntry {
    // Hot data - accessed every lookup (32 bytes)
    uint64_t hash;           // FNV-1a hash for deduplication
    uint32_t id;             // Token ID (ATOM_TOKEN_BASE + index)
    AtomType type;           // 4 bytes (enum)
    AtomCategory category;   // 4 bytes (enum)
    AtomOrigin origin;       // 4 bytes (enum)
    double numeric_value;    // 8 bytes - parsed numeric value
    
    // Warm data - accessed during processing (24 bytes)
    StringRef raw_text_ref;  // 8 bytes - offset + length into string pool
    StringRef parsed_ref;    // 8 bytes - canonical representation
    uint32_t flags;          // Bit flags (sign, has_exponent, etc.)
    float confidence;        // Parse confidence [0, 1]
    
    // Cold data - metadata (8 bytes)
    uint32_t created_at;     // Timestamp
    uint32_t source_start;   // Position in source text
    uint32_t source_end;
};

static_assert(sizeof(AtomEntry) == 64, "AtomEntry must be cache-line aligned");
```

### 3.2 Atom Types and Categories

```cpp
enum class AtomType : uint32_t {
    // NUMERIC category (0)
    ATOM_INTEGER = 0,      // Decimal integers: 42, -1000, +5
    ATOM_FLOAT = 1,        // Floating point: 3.14, -0.001, 1e-5
    ATOM_HEX = 2,          // Hexadecimal: 0xDEADBEEF, 0x1A
    ATOM_BINARY = 3,       // Binary: 0b1010, 0b11110000
    
    // TEMPORAL category (1)
    ATOM_DATE = 4,         // Dates: 2024-12-25, 12/25/2024
    ATOM_TIME = 5,         // Times: 14:30:00, 2:30 PM
    
    // STRUCTURAL category (2)
    ATOM_URL = 6,          // URLs: https://example.com/path?q=1
    ATOM_EMAIL = 7,        // Emails: user@domain.com
    ATOM_PATH = 8,         // File paths: /usr/bin, C:\Windows
    ATOM_IP_ADDRESS = 9,   // IP addresses: 192.168.1.1
    
    // STRING category (3)
    ATOM_STRING_LITERAL = 10,  // Quoted strings: "hello", 'world'
    ATOM_IDENTIFIER = 11,      // Identifiers: camelCase, snake_case
    ATOM_REGEX = 12,           // Regular expressions: /[a-z]+/gi
    ATOM_EQUATION = 13,        // Math: E=mc², x²+y²=r²
    ATOM_EXPRESSION = 14,      // Code: func(a, b), obj.method()
    
    // Extended types
    ATOM_GENERIC = 15,
    ATOM_COUNT = 16
};

enum class AtomCategory : uint32_t {
    NUMERIC = 0,      // Numbers in any base
    TEMPORAL = 1,     // Dates and times
    STRUCTURAL = 2,   // URIs, paths, network addresses
    STRING = 3        // Textual patterns
};
```

### 3.3 String Pool Interning

Strings are stored in a contiguous `string_pool_` buffer. AtomEntry stores only `StringRef` (offset + length):

```cpp
struct StringRef {
    uint32_t offset;   // Byte offset into string_pool_
    uint32_t length;   // String length (not null-terminated)
};

// Interning: returns StringRef, copies string to pool only if new
StringRef AtomTable::internString(std::string_view text) {
    // Check if already in pool (O(n) scan, acceptable for small pools)
    // If not found, append to pool and return new ref
    uint32_t offset = string_pool_.size();
    string_pool_.insert(string_pool_.end(), text.begin(), text.end());
    return {offset, static_cast<uint32_t>(text.length())};
}

// Zero-copy variant for tokenizer (span points directly to input buffer)
StringRef AtomTable::internString(const char* ptr, size_t len);
```

### 3.4 Hash-Based Deduplication

Atoms are deduplicated using **FNV-1a hash** of (type + raw_text):

```cpp
uint64_t AtomTable::computeHash(AtomType type, std::string_view text) {
    // FNV-1a 64-bit
    uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
    
    // Mix in atom type
    hash ^= static_cast<uint64_t>(type);
    hash *= 1099511628211ULL;  // FNV prime
    
    // Mix in text bytes
    for (char c : text) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    
    return hash;
}

// O(1) lookup via hash map
uint32_t findExisting(uint64_t hash, std::string_view text) {
    auto it = hash_to_id_.find(hash);
    if (it != hash_to_id_.end()) {
        // Verify text matches (hash collision check)
        const AtomEntry& entry = entries_[it->second - ATOM_TOKEN_BASE];
        if (getString(entry.raw_text_ref) == text) {
            dedup_hits_++;
            return it->second;
        }
    }
    return UINT32_MAX;  // Not found
}
```

---

## 4. Atom Parsing Pipeline

### 4.1 Type-Specific Parsers

Each atom type has a dedicated parser that extracts semantic information:

```cpp
struct ParseResult {
    bool success;
    AtomValue value;         // Variant of parsed types
    std::string error_message;
};

ParseResult parseAtom(AtomType type, const std::string& text) {
    switch (type) {
        case ATOM_INTEGER:  return parseInteger(text);
        case ATOM_FLOAT:    return parseFloat(text);
        case ATOM_HEX:      return parseHex(text);
        case ATOM_BINARY:   return parseBinary(text);
        case ATOM_URL:      return parseURL(text);
        case ATOM_EMAIL:    return parseEmail(text);
        case ATOM_PATH:     return parsePath(text);
        case ATOM_DATE:     return parseDate(text);
        case ATOM_TIME:     return parseTime(text);
        case ATOM_IP_ADDRESS: return parseIP(text);
        // ...
    }
}
```

### 4.2 Parsed Value Types

The `AtomValue` variant holds type-specific parsed data:

```cpp
using AtomValue = std::variant<
    AtomInteger,       // int64_t value, int base, bool has_sign
    AtomFloat,         // double value, int exponent, bool has_exponent
    AtomURL,           // scheme, host, port, path, query, fragment
    AtomEmail,         // local_part, domain
    AtomPath,          // components[], is_absolute, is_windows
    AtomDate,          // year, month, day
    AtomTime,          // hour, minute, second, millisecond
    AtomIP,            // octets[4], is_valid
    AtomIdentifier,    // parts[], IdentifierStyle (camel/snake/pascal/kebab)
    AtomGeneric        // raw text fallback
>;

// Example: AtomURL structure
struct AtomURL {
    std::string scheme;      // "https"
    std::string host;        // "example.com"
    int port = -1;           // 443 or -1 if not specified
    std::string path;        // "/api/v1/users"
    std::string query;       // "id=123&name=test"
    std::string fragment;    // "section2"
};

// Example: AtomDate structure
struct AtomDate {
    int year;    // 2024
    int month;   // 12
    int day;     // 25
};
```

### 4.3 Numeric Value Packing

Parsed values are packed into a `double` for GPU-side VALUE encoding:

```cpp
void packNumericValue(AtomEntry& entry, const AtomValue& parsed) {
    entry.numeric_value = 0.0;
    
    if (auto* iv = std::get_if<AtomInteger>(&parsed)) {
        entry.numeric_value = static_cast<double>(iv->value);
    }
    else if (auto* fv = std::get_if<AtomFloat>(&parsed)) {
        entry.numeric_value = fv->value;
    }
    else if (auto* dv = std::get_if<AtomDate>(&parsed)) {
        // Pack as YYYYMMDD integer
        entry.numeric_value = dv->year * 10000.0 + dv->month * 100.0 + dv->day;
    }
    else if (auto* tv = std::get_if<AtomTime>(&parsed)) {
        // Pack as fractional day
        entry.numeric_value = tv->hour / 24.0 + tv->minute / 1440.0 + tv->second / 86400.0;
    }
    else if (auto* ip = std::get_if<AtomIP>(&parsed)) {
        // Pack as 32-bit integer
        entry.numeric_value = (ip->octets[0] << 24) | (ip->octets[1] << 16) |
                              (ip->octets[2] << 8) | ip->octets[3];
    }
    // ... other types
}
```

---

## 5. ScratchBlock Neural Layer

### 5.1 Architecture Overview

ScratchBlock is an **additive injection layer** that modifies hidden states based on detected atoms:

```
hidden' = hidden + scale * (AtomTypeEmbed @ AtomProjection) + scale * (TextFeatures @ TextProjection)
                   \_______________ TYPE path ________________/   \________ VALUE path _________/
```

**Learnable Parameters:**
| Parameter | Shape | Description |
|-----------|-------|-------------|
| `atom_type_embeddings` | `[19, 64]` | Per-type embedding vectors |
| `atom_projection` | `[64, 768]` | Projects atom embedding to d_model |
| `text_feature_projection` | `[16, 768]` | Projects FP16 text features to d_model |

### 5.2 Token ID Range

Atom tokens occupy a dedicated ID range:

```cpp
constexpr int ATOM_TOKEN_BASE = 256;    // After byte fallback [0-255]
constexpr int ATOM_TOKEN_START = 256;   // First atom token
constexpr int ATOM_TOKEN_END = 275;     // One past last (19 types)
constexpr int NUM_ATOM_TYPES = 19;

// Token ID layout:
// [0-255]     = Byte fallback (raw UTF-8)
// [256-274]   = Atom placeholder tokens
// [275+]      = Unigram vocabulary
```

### 5.3 Forward Pass

```cuda
void ScratchBlockLayer::forward(const ScratchBlockForwardArgs& args) {
    // DISABLED: Pure passthrough (zero overhead)
    if (!config_.enabled) {
        forwardPassthrough(args);  // Just copies input to output
        return;
    }
    
    // Step 1: Copy input to output (we modify output in-place)
    cudaMemcpyAsync(output, input, bytes, cudaMemcpyDeviceToDevice, stream);
    
    // Step 2: Detect atom tokens (scan for IDs in [256, 274])
    kernelDetectAtomTokens<<<grid, block>>>(
        token_ids, total_tokens, atom_positions, num_atoms, max_atoms,
        ATOM_TOKEN_START, ATOM_TOKEN_END);
    
    // Step 3: Lookup atom embeddings with VALUE encoding
    kernelLookupAtomEmbeddingsWithValue<<<atoms, atom_dim>>>(
        token_ids, atom_positions, num_atoms,
        atom_type_embeddings,       // Learnable type vectors
        token_numeric_values,       // Side-channel: parsed numeric values
        token_numeric_mask,         // Which tokens have numeric data
        atom_embeddings,            // Output: [num_atoms, atom_dim]
        atom_embedding_dim);
    
    // Step 4: Project and inject into hidden states
    kernelInjectAtomEmbeddings<<<atoms, d_model>>>(
        hidden_states,
        atom_positions, num_atoms,
        atom_embeddings, atom_projection,
        atom_dim, d_model, scale);
    
    // Step 5: Inject text features (FP16 semantic vectors)
    kernelInjectTextFeatures<<<total_tokens, d_model>>>(
        hidden_states,
        text_features,              // [total_tokens * 16] FP16
        text_mask,                  // Which tokens have features
        text_projection,            // [16, d_model]
        total_tokens, d_model, scale);
}
```

### 5.4 Backward Pass

Gradients flow through additive injection:

```cuda
void ScratchBlockLayer::backward(...) {
    // Gradients pass through unchanged (additive injection)
    cudaMemcpyAsync(grad_input, grad_output, bytes, stream);
    
    // Step 1: Compute projection gradient
    // grad_projection[k, d] = sum_atoms(atom_emb[atom, k] * grad_h[pos, d])
    kernelBackwardAtomEmbeddings<<<atoms, block>>>(
        grad_output,
        cached_atom_positions, cached_num_atoms,
        cached_atom_embeddings,
        atom_projection, grad_atom_projection,
        grad_atom_embeddings,
        atom_dim, d_model, scale);
    
    // Step 2: Route atom embedding gradients back to type embeddings
    // grad_type_emb[type, k] += sum_atoms_of_type(grad_atom_emb[atom, k])
    kernelAccumulateAtomTypeGradients<<<atoms, block>>>(
        grad_atom_embeddings,
        cached_atom_types,        // Which type each atom is
        cached_num_atoms,
        grad_atom_type_embeddings,
        atom_dim);
    
    // Step 3: Backward through text feature projection
    kernelBackwardTextFeatures<<<total_tokens, kTextFeatureDim>>>(
        grad_output,
        text_features, text_mask,
        grad_text_projection,
        total_tokens, d_model, scale);
}
```

---

## 6. Value Encoding System

### 6.1 Atom Type Embeddings

Each atom type has a learnable embedding vector of dimension `atom_embedding_dim` (default 64):

```cuda
// Type embedding lookup (traditional one-hot style)
float type_embed[ATOM_DIM];
int atom_type = (token_id - ATOM_TOKEN_START) % NUM_ATOM_TYPES;

for (int k = 0; k < ATOM_DIM; k++) {
    type_embed[k] = atom_type_embeddings[atom_type * ATOM_DIM + k];
}
```

### 6.2 Numeric VALUE Encoding

The key innovation: we **encode the actual numeric value** into the embedding, not just the type:

```cuda
__global__ void kernelLookupAtomEmbeddingsWithValue(
    const int* token_ids,
    const int* atom_positions,
    const int* num_atoms,
    int max_atoms,
    const float* atom_type_embeddings,    // [NUM_TYPES, atom_dim]
    const float* token_numeric_values,    // [total_tokens] - side channel
    const uint8_t* token_numeric_mask,    // [total_tokens] - valid flag
    float* atom_embeddings,               // [num_atoms, atom_dim] output
    int atom_embedding_dim
) {
    const int atom_idx = blockIdx.x;
    const int k_idx = threadIdx.x;
    
    int token_pos = atom_positions[atom_idx];
    int token_id = token_ids[token_pos];
    int atom_type = (token_id - ATOM_TOKEN_START) % NUM_ATOM_TYPES;
    
    // Start with type embedding
    float value = atom_type_embeddings[atom_type * atom_embedding_dim + k_idx];
    
    // VALUE encoding: modify embedding based on numeric value
    if (token_numeric_mask && token_numeric_mask[token_pos]) {
        float num_val = token_numeric_values[token_pos];
        
        // Dimensions [0-15]: Type one-hot (already in type embedding)
        
        // Dimensions [16-31]: Log-scaled magnitude
        if (k_idx >= 16 && k_idx < 32 && num_val != 0.0f) {
            float log_mag = log2f(fabsf(num_val) + 1.0f);
            int bit = k_idx - 16;
            value += fmodf(log_mag / (float)(bit + 1), 1.0f);
        }
        
        // Dimensions [32-47]: Sign and fractional info
        if (k_idx >= 32 && k_idx < 48) {
            if (k_idx == 32) {
                value += (num_val < 0.0f) ? 1.0f : 0.0f;  // Sign bit
            } else if (k_idx == 33) {
                // Fractional part indicator
                float frac = num_val - floorf(num_val);
                value += (frac > 0.001f) ? 1.0f : 0.0f;
            }
        }
    }
    
    atom_embeddings[atom_idx * atom_embedding_dim + k_idx] = value;
}
```

**Embedding Layout:**
```
Dimension Range    Content
[0-15]            Type one-hot encoding (learned)
[16-31]           Log-scaled magnitude encoding
[32-47]           Sign and fractional indicators
[48-63]           Reserved / learned features
```

### 6.3 Text Feature Encoding

Per-token text features provide additional semantic information in **16-dimensional FP16** vectors:

```cpp
constexpr int kTextFeatureDim = 16;

void encodeAtomTextFeatures(
    AtomType type,
    std::string_view raw_text,
    const AtomValue* parsed,
    uint16_t* out_features  // [16] FP16
) {
    // Dims [0-3]: Category one-hot
    int category = getCategoryForType(type);  // 0=NUMERIC, 1=TEMPORAL, etc.
    out_features[category] = floatToFp16(1.0f);
    
    // Dims [4-7]: Subtype encoding
    int subtype = getSubtypeForType(type);
    out_features[4 + subtype % 4] = floatToFp16(0.8f);
    
    // Dims [8-11]: Length/magnitude features
    float len = raw_text.size();
    out_features[8] = floatToFp16(min(len / 100.0f, 1.0f));  // Normalized length
    out_features[9] = floatToFp16(log2f(len + 1) / 10.0f);   // Log length
    
    // Dims [10-11]: Type-specific semantic features
    if (auto* iv = std::get_if<AtomInteger>(parsed)) {
        out_features[10] = floatToFp16(min(log2f(abs(iv->value) + 1) / 32.0f, 1.0f));
        out_features[11] = floatToFp16(iv->value < 0 ? 1.0f : 0.0f);
    }
    else if (auto* url = std::get_if<AtomURL>(parsed)) {
        out_features[10] = floatToFp16(url->scheme == "https" ? 1.0f : 0.5f);
        out_features[11] = floatToFp16(!url->query.empty() ? 1.0f : 0.0f);
    }
    else if (auto* date = std::get_if<AtomDate>(parsed)) {
        out_features[10] = floatToFp16((date->year - 1900) / 200.0f);
        out_features[11] = floatToFp16(date->month / 12.0f);
    }
    // ... other types
    
    // Dims [12-15]: Reserved for model learning
}
```

---

## 7. GPU Implementation

### 7.1 CUDA Kernels

**Atom Detection Kernel:**
```cuda
__global__ void kernelDetectAtomTokens(
    const int* __restrict__ token_ids,
    int total_tokens,
    int* __restrict__ atom_positions,   // [max_atoms]
    int* __restrict__ num_atoms,        // scalar
    int max_atoms,
    int atom_token_start,
    int atom_token_end
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_tokens) return;
    
    int token = token_ids[tid];
    if (token >= atom_token_start && token < atom_token_end) {
        // Atomic increment to get slot
        int slot = atomicAdd(num_atoms, 1);
        if (slot < max_atoms) {
            atom_positions[slot] = tid;
        }
    }
}
```

**Atom Injection Kernel:**
```cuda
__global__ void kernelInjectAtomEmbeddings(
    float* __restrict__ hidden_states,   // [total_tokens, d_model]
    const int* __restrict__ atom_positions,
    const int* __restrict__ num_atoms,
    int max_atoms,
    const float* __restrict__ atom_embeddings,  // [num_atoms, atom_dim]
    const float* __restrict__ projection,       // [atom_dim, d_model]
    int atom_dim,
    int d_model,
    float scale
) {
    const int atom_idx = blockIdx.x;
    const int d_idx = threadIdx.x;
    
    int num_atoms_val = ClampNumAtoms(num_atoms, max_atoms);
    if (atom_idx >= num_atoms_val || d_idx >= d_model) return;
    
    int token_pos = atom_positions[atom_idx];
    
    // Compute projected embedding: embed @ projection
    float sum = 0.0f;
    for (int k = 0; k < atom_dim; k++) {
        sum += atom_embeddings[atom_idx * atom_dim + k] * projection[k * d_model + d_idx];
    }
    
    // Additive injection
    hidden_states[token_pos * d_model + d_idx] += scale * sum;
}
```

**Text Feature Injection Kernel:**
```cuda
__global__ void kernelInjectTextFeatures(
    float* __restrict__ hidden_states,
    const uint16_t* __restrict__ text_features,  // [total_tokens * 16] FP16
    const uint8_t* __restrict__ text_mask,
    const float* __restrict__ text_projection,   // [16, d_model]
    int total_tokens,
    int d_model,
    float scale
) {
    const int token_idx = blockIdx.x;
    const int d_idx = threadIdx.x;
    
    if (!text_mask || text_mask[token_idx] == 0) return;
    
    const uint16_t* features = text_features + token_idx * kTextFeatureDim;
    
    // Project: features @ projection
    float sum = 0.0f;
    for (int k = 0; k < kTextFeatureDim; k++) {
        float feat = __half2float(*reinterpret_cast<const __half*>(&features[k]));
        sum += feat * text_projection[k * d_model + d_idx];
    }
    
    hidden_states[token_idx * d_model + d_idx] += scale * sum;
}
```

### 7.2 Memory Layout

**GPU Allocations:**
```cpp
struct ScratchBlockGPUBuffers {
    // Learnable weights
    float* d_atom_type_embeddings;     // [NUM_TYPES * atom_dim]
    float* d_atom_projection;          // [atom_dim * d_model]
    float* d_text_feature_projection;  // [kTextFeatureDim * d_model]
    
    // Gradient buffers
    float* d_atom_type_embeddings_grad;
    float* d_atom_projection_grad;
    float* d_text_feature_projection_grad;
    
    // Working memory
    int* d_atom_positions;             // [max_atoms]
    int* d_num_atoms;                  // scalar
    float* d_atom_embeddings;          // [max_atoms * atom_dim]
    float* d_grad_atom_embeddings;     // [max_atoms * atom_dim]
};
```

**Memory Footprint (default config):**
```
Component                    Size
atom_type_embeddings         19 × 64 × 4B = 4,864 B
atom_projection              64 × 768 × 4B = 196,608 B
text_feature_projection      16 × 768 × 4B = 49,152 B
Gradient buffers             3× above = 750,624 B
Working memory               max_atoms × 64 × 4B × 2 ≈ 512 KB

Total: ~1.5 MB (negligible)
```

### 7.3 Weight Initialization

All weights use **Xavier initialization** for stable gradients:

```cuda
__global__ void kernelXavierInit(float* weights, int size, float stddev, unsigned int seed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // LCG random number generator
    unsigned int state = seed + idx * 1099087573u;
    state = state * 1103515245u + 12345u;
    float u1 = (state & 0x7FFFFFFF) / float(0x7FFFFFFF);
    state = state * 1103515245u + 12345u;
    float u2 = (state & 0x7FFFFFFF) / float(0x7FFFFFFF);
    
    // Box-Muller transform
    float z = sqrtf(-2.0f * logf(u1 + 1e-10f)) * cosf(2.0f * 3.14159265f * u2);
    weights[idx] = z * stddev;
}

// Xavier standard deviations
float atom_emb_std = sqrt(2.0f / (NUM_TYPES + atom_dim));        // ≈ 0.155
float projection_std = sqrt(2.0f / (atom_dim + d_model));        // ≈ 0.049
float text_proj_std = sqrt(2.0f / (kTextFeatureDim + d_model));  // ≈ 0.051
```

---

## 8. Serialization Format

### 8.1 AtomTable Binary Format (ATMB v1)

```
┌────────────────────────────────────────────┐
│  Header (16 bytes)                         │
│  ├─ Magic: "ATMB" (4 bytes)                │
│  ├─ Version: uint32_t = 1                  │
│  ├─ Entry Count: uint32_t                  │
│  └─ Pool Size: uint32_t                    │
├────────────────────────────────────────────┤
│  Entries (entry_count × 64 bytes each)     │
│  └─ AtomEntry[] - cache-line aligned       │
├────────────────────────────────────────────┤
│  String Pool (pool_size bytes)             │
│  └─ Raw character data (no null terms)     │
└────────────────────────────────────────────┘
```

**Save/Load:**
```cpp
bool AtomTable::saveToFile(const std::string& path) const {
    const char magic[4] = {'A', 'T', 'M', 'B'};
    const uint32_t version = 1;
    const uint32_t entry_count = entries_.size();
    const uint32_t pool_size = string_pool_.size();
    
    file.write(magic, 4);
    file.write(&version, 4);
    file.write(&entry_count, 4);
    file.write(&pool_size, 4);
    file.write(entries_.data(), entry_count * sizeof(AtomEntry));
    file.write(string_pool_.data(), pool_size);
    return file.good();
}

bool AtomTable::loadFromFile(const std::string& path) {
    char magic[4];
    file.read(magic, 4);
    if (memcmp(magic, "ATMB", 4) != 0) return false;
    
    uint32_t version, entry_count, pool_size;
    file.read(&version, 4);
    if (version != 1) return false;
    
    file.read(&entry_count, 4);
    file.read(&pool_size, 4);
    
    entries_.resize(entry_count);
    file.read(entries_.data(), entry_count * sizeof(AtomEntry));
    
    string_pool_.resize(pool_size);
    file.read(string_pool_.data(), pool_size);
    
    // Rebuild indices
    rebuildIndices();
    return true;
}
```

### 8.2 Text Export Format (TSV)

For debugging and analysis:
```
id      type            category    raw_text        numeric_value   hash
256     ATOM_INTEGER    NUMERIC     42              42.0            0x1a2b3c4d...
257     ATOM_URL        STRUCTURAL  https://x.com   0.0             0x5e6f7a8b...
258     ATOM_DATE       TEMPORAL    2024-12-25      20241225.0      0x9c0d1e2f...
```

---

## 9. Configuration

### 9.1 ScratchBlock Config

```cpp
struct ScratchBlockConfig {
    int d_model = 768;              // Model hidden dimension
    int max_atoms = 1024;           // Max atoms per batch
    int atom_embedding_dim = 64;    // Atom type embedding size
    float atom_scale = 1.0f;        // Unit injection scale (embeddings scaled to match)
    bool enabled = true;            // Master enable/disable
    bool inject_atom_embeddings = true;  // Inject type embeddings
    int atom_token_start = 256;     // First atom token ID
    int atom_token_end = 275;       // One past last atom token
    cudaStream_t stream = nullptr;  // CUDA stream
};
```

### 9.2 ai_config.json

```json
{
    "tokenizer": {
        "scratch_block_reasoning": true,
        "atom_embedding_dim": 64,
        "atom_scale": 1.0
    },
    "model": {
        "scratchblock": {
            "enabled": true,
            "max_atoms": 1024,
            "inject_atom_embeddings": true
        }
    }
}
```

---

## 10. Integration Points

### 10.1 Tokenizer Integration

The tokenizer calls AtomTable during encoding:

```cpp
TokenizeResult UniByte::encode(std::string_view text, bool include_atoms) {
    TokenizeResult result;
    
    // Step 1: Run Aho-Corasick DFA to detect structural spans
    std::vector<StructuralSpan> spans = detector_.detectSpans(text);
    
    // Step 2: Register atoms and replace with placeholder tokens
    for (const auto& span : spans) {
        uint32_t atom_id;
        if (atom_table_.tryRegisterSpan(span, atom_id)) {
            // Replace span with atom placeholder token
            result.tokens.push_back(atom_id);
            
            // Generate text features for this atom
            uint16_t features[kTextFeatureDim];
            encodeAtomTextFeatures(span.atom_type, span.view(), 
                                   atom_table_.getParsedValue(atom_id), features);
            for (int i = 0; i < kTextFeatureDim; i++) {
                result.token_text_features.push_back(features[i]);
            }
        }
    }
    
    return result;
}
```

### 10.2 Model Forward Pass

ScratchBlock runs after embedding lookup:

```cpp
// In LanguageModel::forward()
// 1. Embed tokens
float* embeddings = embedding_layer_->forward(token_ids, positions);

// 2. ScratchBlock injection (modifies embeddings in-place)
if (scratch_block_ && scratch_block_->isEnabled()) {
    ScratchBlockForwardArgs args;
    args.input = embeddings;
    args.output = embeddings;  // In-place
    args.token_ids = token_ids;
    args.token_numeric_values = numeric_values;
    args.token_text_features = text_features;
    args.stream = stream;
    
    scratch_block_->forward(args);
}

// 3. Run encoder layers
for (auto& layer : encoder_layers_) {
    embeddings = layer->forward(embeddings);
}
```

### 10.3 Training Data Pipeline

GRMT binary format includes atom data:

```cpp
struct SequenceData {
    std::vector<int32_t> tokens;
    std::vector<float> numeric_values;       // Per-token numeric side-channel
    std::vector<uint8_t> numeric_mask;       // Valid flags
    std::vector<uint16_t> token_text_features;  // [tokens × 16] FP16
};
```

---

## 11. Performance Characteristics

### 11.1 Computational Cost

| Operation | Complexity | Typical Time (RTX 3080) |
|-----------|------------|-------------------------|
| Atom detection | O(N) | ~0.05 ms per 1K tokens |
| Embedding lookup | O(A × D) | ~0.02 ms per 100 atoms |
| Injection | O(A × d_model) | ~0.1 ms per 100 atoms |
| Text features | O(N × 16 × d_model) | ~0.3 ms per 1K tokens |
| **Total forward** | | **~0.5 ms typical** |
| Passthrough (disabled) | O(N) | ~0.01 ms (memcpy) |

### 11.2 Memory Efficiency

- AtomEntry is **exactly 64 bytes** (one cache line)
- String pool uses contiguous allocation
- FP16 text features save 50% memory vs FP32
- Max atoms configurable (default 1024)

### 11.3 Deduplication Statistics

Typical dedup hit rates by corpus type:
- Code: 40-60% (common literals: 0, 1, -1, "")
- Technical docs: 20-30% (URLs, paths repeat)
- Natural text: 5-10% (few structural elements)

---

## 12. Mathematical Foundations

### 12.1 Additive Injection

ScratchBlock uses **additive injection** rather than multiplicative gating:

$$h' = h + \alpha \cdot \text{Proj}(\text{Embed}(a))$$

Where:
- $h \in \mathbb{R}^{d}$ is the hidden state
- $a \in \{1, \ldots, 19\}$ is the atom type
- $\text{Embed}: \{1, \ldots, 19\} \to \mathbb{R}^{k}$ is the type embedding
- $\text{Proj}: \mathbb{R}^{k} \to \mathbb{R}^{d}$ is the learned projection
- $\alpha$ is the injection scale (default 0.1)

**Gradient Flow:**
$$\frac{\partial \mathcal{L}}{\partial h} = \frac{\partial \mathcal{L}}{\partial h'}$$

Gradients flow unchanged through the additive skip connection.

### 12.2 VALUE Encoding

For numeric atoms, we encode both type and value:

$$\text{Embed}(a, v) = E_a + \text{ValueEncode}(v)$$

Where:
- $E_a \in \mathbb{R}^{k}$ is the learnable type embedding
- $v \in \mathbb{R}$ is the parsed numeric value

Value encoding:
$$\text{ValueEncode}(v)_i = \begin{cases}
\text{sign}(v) & i = 32 \\
\log_2(|v| + 1) \mod (i - 15) & 16 \leq i < 32 \\
0 & \text{otherwise}
\end{cases}$$

### 12.3 Text Feature Projection

Text features provide a second VALUE path:

$$h' = h + \alpha \cdot \text{TextProj}(\text{Features}(a))$$

Where $\text{Features}: A \to \mathbb{R}^{16}$ extracts semantic features and $\text{TextProj}: \mathbb{R}^{16} \to \mathbb{R}^{d}$.

The combined injection:
$$h' = h + \alpha \cdot (\text{Proj}(\text{Embed}(a, v)) + \text{TextProj}(\text{Features}(a)))$$

---

## File References

| Component | File Location |
|-----------|---------------|
| AtomTable header | `Shared/UnigramByte/AtomTable.hpp` |
| AtomTable implementation | `Shared/UnigramByte/Atomtable.cu` |
| ScratchBlock layer | `Layers/ScratchBlock/ScratchBlock_GPU.cu` |
| ScratchBlock header | `Layers/ScratchBlock/ScratchBlock_GPU.hpp` |
| Text feature encoding | `Shared/UnigramByte/UniByte.cu` |
| Pinned memory pool | `Shared/ScratchBlock/ScratchBlock_GPU.hpp` |
| Token ID constants | `Shared/UnigramByte/AtomTable.hpp` |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 2025 | Initial documentation |

---

*This document describes the ScratchBlock architecture as implemented in GRIM-text. For questions or contributions, see the project repository.*

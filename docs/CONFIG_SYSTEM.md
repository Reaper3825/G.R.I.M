# GRIM Configuration System Documentation

## Overview

The GRIM project uses a three-tier configuration system:

1. **Compile-time constants** - `HyperParameters_GPU.hpp`
2. **Runtime configuration** - `ai_config.json`
3. **Configuration parser** - `ai_config_paths.hpp`

## File Organization

### 1. HyperParameters_GPU.hpp
**Location:** `resources/models/GRIM-text/Shared/HyperParameters/`

**Purpose:** Single source of truth for compile-time constants

**Contains:**
- CUDA kernel configuration (block sizes, warp size, grid limits)
- Numerical stability constants (epsilons, clamp values)
- Model architecture defaults (d_model, num_layers, num_heads, etc.)
- Optimizer defaults (AdamW beta values, weight decay)
- Tokenizer constants (byte/atom token ranges)
- Flash Attention configuration
- Loss function defaults
- Derived constants (explicitly show relationships)

**Rule:** ALL compile-time constants must be defined here. Other files include this header and use its constants.

### 2. ai_config.json
**Location:** `d:\G.R.I.M\ai_config.json`

**Purpose:** Runtime configuration that can be changed without recompilation

**Structure:**
```json
{
  "api_keys": {...},              // External service keys
  "backend": "grim_native",       // Which AI backend to use
  "paths": {                      // File/directory paths
    "grim_text": {...}            // GRIM-text specific paths
  },
  "data_collection": {...},       // Data pipeline config
  "tokenizer": {...},             // Tokenizer runtime config
  "training": {                   // Training hyperparameters
    "config": {
      "batch_size": 4,
      "learning_rate": 0.0001,
      "loss": {...},              // Loss function config
      "scratch_blocks": {...},    // Memory buffer config
      "dynamic_lr": {...},        // Learning rate schedule
      // ... etc
    }
  },
  "voice": {...},                 // Voice I/O config
  "whisper": {...}                // STT config
}
```

**JSON → C++ Owner Mapping:**
- `paths.grim_text` → `AiConfigSnapshot` `grim_text_*` fields
- `training.config` → `TrainingHyperparameters`
- `tokenizer` → `AiConfigSnapshot` `tokenizer_*` fields
- `data_collection` → `AiConfigSnapshot` `data_collection_*` fields

### 3. ai_config_paths.hpp
**Location:** `d:\G.R.I.M\control\ai_config_paths.hpp`

**Purpose:** C++ structs and parsers for `ai_config.json`

**Contains:**
- Struct definitions (`AiConfigSnapshot`, `TrainingHyperparameters`, etc.)
- One raw validator (`validateAiConfigDocument`) and one raw loader (`loadAiConfigSnapshot`)
- Raw JSON parsing/assignment helpers for the collapsed snapshot surface

**Access rule:** Consumers should load **one** `AiConfigSnapshot` and then read direct snapshot fields from that object. Do not add sidecar config wrappers or path-based leaf loaders that reparse `ai_config.json` for one subsection.

**Rule:** `ai_config_paths.hpp` is the raw authored-config layer only. It must not own runtime policy defaults or formula-derived values.

## Configuration Flow

```
Compile Time:
  HyperParameters_GPU.hpp
    ↓ (defines compile-time constants)
  Training code includes header
    ↓ (uses constants directly)
  Compiled binary

Runtime:
  ai_config.json
    ↓ (parsed by)
  ai_config_paths.hpp
    ↓ (loads into C++ structs)
  TrainingHyperparameters
    ↓ (overrides compile-time defaults)
  Training execution
```

## Value Priority (highest to lowest)

1. **Runtime JSON override** - Values in `ai_config.json` take precedence
2. **HyperParameters derivation** - Pure formulas computed from authored values after parsing
3. **Compile-time constant** - Only for static math/kernel capabilities, never runtime policy fallback

## Example: Learning Rate Configuration

```cpp
// 1. Compile-time default (HyperParameters_GPU.hpp)
// Not defined here - learning rate is runtime-only

// 2. Runtime authored value (ai_config.json)
{
  "training": {
    "config": {
      "learning_rate": 0.0001  // ACTIVE VALUE
    }
  }
}
```

## Example: Epsilon Configuration

```cpp
// 1. Compile-time constant (HyperParameters_GPU.hpp)
constexpr float EPSILON_RMSNORM = 1e-5f;

// 2. Training code uses constant directly
float rms = std::sqrt(variance + EPSILON_RMSNORM);

// No runtime override - epsilon is compile-time only for performance
```

## Adding New Configuration Values

### For Compile-Time Constants (CUDA, Math, etc.):

1. Add to `HyperParameters_GPU.hpp`:
   ```cpp
   namespace GRIM::HyperParameters {
       constexpr float MY_NEW_CONSTANT = 0.5f;
   }
   ```

2. Use in code:
   ```cpp
   #include "HyperParameters/HyperParameters_GPU.hpp"
   float value = GRIM::HyperParameters::MY_NEW_CONSTANT;
   ```

### For Runtime Configuration:

1. Add field to `ai_config_paths.hpp`:
   ```cpp
   struct TrainingHyperparameters {
       bool my_new_feature = true;  // C++ default
       // ...
   };
   ```

2. Add raw assignment in `applyTrainingConfigObject()` and require it in `validateAiConfigDocument()`:
   ```cpp
   assignTrainingField(params.my_new_feature, trainConfig, "my_new_feature");
   ```

3. Add to `ai_config.json`:
   ```json
   {
     "training": {
       "config": {
         "my_new_feature": false
       }
     }
   }
   ```

4. Use in code:
   ```cpp
   auto snapshot = loadAiConfigSnapshot("ai_config.json");
   if (snapshot->hyperparameters.my_new_feature) {
       // ...
   }
   ```

## Common Pitfalls

### ❌ Wrong: Hardcoding values in training code
```cpp
const int block_size = 256;  // BAD: Should use HyperParameters
```

### ✅ Correct: Using centralized constant
```cpp
const int block_size = GRIM::HyperParameters::CUDA_BLOCK_SIZE_STANDARD;
```

### ❌ Wrong: Mismatched defaults
```cpp
// HyperParameters_GPU.hpp
constexpr float DEFAULT_LOSS_FOCAL_GAMMA = 2.0f;

// ai_config_paths.hpp
float loss_focal_gamma = 1.5f;  // MISMATCH!
```

### ✅ Correct: Matching defaults
```cpp
// HyperParameters_GPU.hpp
constexpr float DEFAULT_LOSS_FOCAL_GAMMA = 2.0f;

// ai_config_paths.hpp
float loss_focal_gamma = 2.0f;  // Match HyperParameters::DEFAULT_LOSS_FOCAL_GAMMA
```

### ❌ Wrong: Duplicate configuration sections
```json
{
  "prediction_comparison": {...},  // Root level
  "training": {
    "config": {
      "prediction_comparison": {...}  // DUPLICATE!
    }
  }
}
```

### ✅ Correct: Single source per configuration
```json
{
  "training": {
    "config": {
      "prediction_comparison": {...}  // Only here
    }
  }
}
```

## Troubleshooting

### "Value not loading from JSON"
1. Check JSON syntax is valid (use JSON validator)
2. Verify key name matches exactly in `assignTrainingField()` call
3. Ensure struct field type matches JSON value type
4. Check `loadAiConfigSnapshot()` returns non-null

### "Compile error: undefined constant"
1. Verify `#include "HyperParameters/HyperParameters_GPU.hpp"` present
2. Check constant is in correct namespace: `GRIM::HyperParameters::CONSTANT_NAME`
3. For CUDA files: Ensure include path configured in CMakeLists.txt

### "Training uses wrong config value"
1. Verify the authored JSON key is present and spelled correctly
2. Check `validateAiConfigDocument()` requires the field at the correct path
3. Look for typos in `assignTrainingField()` or the relevant raw snapshot assignment

## Recent Changes (Dec 2024)

### Consolidation Updates
- **Removed duplicates:** `prediction_comparison`, `attention_diagnostics` now only in `training.config`
- **Removed defaults:** Authored runtime fields must be present in `ai_config.json`; pure formulas are derived in `HyperParameters_GPU.hpp`
- **Fixed cache_limits:** Removed duplicate parsing block in `ai_config_paths.hpp`
- **Organized constants:** All hyperparameters now follow Rule 20 (single source of truth)

### Derived Constants
Made 13 constants explicitly derived from base values in `HyperParameters_GPU.hpp`:
- Block sizes: `CUDA_QUANTIZATION_THREADS`, `CUDA_REDUCTION_MAX_BLOCKS` from `CUDA_BLOCK_SIZE_STANDARD`
- Tile dimensions: `CUDA_TILE_DIM_TRANSPOSE` from `CUDA_WARP_SIZE`
- Static tokenizer workspace: `UNIGRAM_MAX_SEQUENCE_LENGTH` documents implementation capacity, not model sequence policy
- Model arch: runtime `d_ff` is derived from authored `d_model * D_FF_MULTIPLIER` in `HyperParameters_GPU.hpp`
- Flash Attention: Block sizes from `CUDA_WARP_SIZE` and `CUDA_BLOCK_SIZE_STANDARD`
- Tokenizer: `ATOM_TOKEN_START`, `ATOM_TOKEN_END` from `BYTE_TOKEN_END` and derived formulas

## Related Documentation

- [INTEGRATION_PLAN.md](../INTEGRATION_PLAN.md) - GRIM-text backend integration
- [README.md](../README.md) - Quick start and dependencies
- [copilot-instructions.md](../.github/copilot-instructions.md) - Development conventions

## Maintenance Rules

### Rule 20: No File Owns Hyperparameters
Every configuration value must have ONE authoritative source:
- Compile-time → `HyperParameters_GPU.hpp`
- Runtime → `ai_config.json` + `ai_config_paths.hpp`
- Never hardcode values in .cu/.cpp files

### Backwards Compatibility: Removed
When removing configuration options, DELETE all compatibility shims, fallbacks, and legacy code paths. Let it fail loud to expose misconnects. See copilot-instructions.md pitfall #20.

### Derived Constants: Explicit
When one constant depends on another, make the relationship explicit:
```cpp
constexpr int DERIVED = BASE_CONSTANT * 4;  // GOOD: Shows relationship
constexpr int DERIVED = 1024;               // BAD: Hides dependency on BASE=256
```

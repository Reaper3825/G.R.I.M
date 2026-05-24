# GRIM Configuration System Documentation

## Overview

The GRIM project uses a three-tier configuration system:

1. **Compile-time constants and typed runtime handoff** - `HyperParameters_GPU.hpp`
2. **Runtime configuration** - `ai_config.json`
3. **Raw configuration loader** - `ai_config_paths.hpp`

## File Organization

### 1. HyperParameters_GPU.hpp
**Location:** `resources/models/GRIM-text/Shared/HyperParameters/`

**Purpose:** Single source of truth for compile-time constants, typed runtime config, validation, and derivation

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
  "paths": {                      // Broader GRIM runtime paths
    "grim_text": {...}            // Transitional duplicate; GRIM-text startup reads training.config
  },
  "data_collection": {...},       // GRIM process config, not GRIM-text startup
  "training": {                   // Training hyperparameters
    "config": {
      "grim_text_vocab": "resources/models/GRIM-text/training/data/vocab.bin",
      "grim_text_training_data": "resources/models/GRIM-text/training/data/training_data.grmt",
      "batch_size": 4,
      "learning_rate": 0.0001,
      "current_curriculum": "Pre-Trainingv1",
      "current_model_training": "",
      "clear_merged_cache_on_merge": false,
      "loss_label_smoothing_enabled": true,
      "scratch_blocks_enabled": true,
      "subprocess_tokenizer_only_mode": false,
      // ... etc
    }
  },
  "voice": {...},                 // Voice I/O config
  "whisper": {...}                // STT config
}
```

**JSON → C++ Owner Mapping:**
- `ai_config.json` → `AiConfigSnapshot::document`
- `training.config.grim_text_*` → `StartupConfig::paths` via `HyperParameters_GPU.hpp`
- `training.config` model/training/tokenizer/subprocess leaves → `TrainingHyperparameters`
- `training.config.generation_*` leaves → `GenerationConfig` through `loadGenerationConfig()`
- `training.config.clear_merged_cache_on_merge` → `TrainingHyperparameters::clear_merged_cache_on_merge`
- `data_collection` → GRIM process consumers; do not add typed GRIM-text snapshot leaves for these fields

### 3. ai_config_paths.hpp
**Location:** `d:\G.R.I.M\control\ai_config_paths.hpp`

**Purpose:** Raw C++ snapshot loader for `ai_config.json`

**Contains:**
- Struct definition (`AiConfigSnapshot`) with only `config_path` and `document`
- One raw loader (`loadAiConfigSnapshot`)
- No typed leaf mirrors, schema tables, or config subsection accessors

**Access rule:** Consumers should load **one** `AiConfigSnapshot`; GRIM-text typed reads happen in `HyperParameters_GPU.hpp`, and immutable objective-specific views are produced by `HyperparameterGroupings.hpp`.

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
    ↓ (loads raw AiConfigSnapshot document)
  HyperParameters_GPU.hpp
    ↓ (reads authored leaves, validates, derives formulas)
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

1. Add the authored leaf to `ai_config.json` under `training.config`:
  ```json
  {
    "training": {
     "config": {
      "my_new_feature": false
     }
    }
  }
  ```

2. Add the typed owner field to `TrainingHyperparameters` in `HyperParameters_GPU.hpp`:
   ```cpp
  bool my_new_feature = false;
   ```

3. Read the authored value in `loadTrainingHyperparameters()`:
   ```cpp
  assign(params.my_new_feature, "my_new_feature");
  ```

4. Use typed config after the HyperParameters handoff:
   ```cpp
  auto snapshot = GRIM::Config::loadAiConfigSnapshot();
  TrainingHyperparameters hp;
  GRIM::HyperParameters::loadTrainingHyperparameters(snapshot, hp);
  if (hp.my_new_feature) {
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

// ai_config.json / HyperParameters typed load
// Missing or mismatched authored value fails loud during load/validation.
```

### ✅ Correct: Matching defaults
```cpp
// HyperParameters_GPU.hpp
constexpr float DEFAULT_LOSS_FOCAL_GAMMA = 2.0f;

// Pure formulas stay in HyperParameters; runtime policy is authored once in ai_config.json.
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
2. Verify key name matches exactly in the `loadTrainingHyperparameters()` assignment
3. Ensure struct field type matches JSON value type
4. Check `loadAiConfigSnapshot()` loaded the canonical document

### "Compile error: undefined constant"
1. Verify `#include "HyperParameters/HyperParameters_GPU.hpp"` present
2. Check constant is in correct namespace: `GRIM::HyperParameters::CONSTANT_NAME`
3. For CUDA files: Ensure include path configured in CMakeLists.txt

### "Training uses wrong config value"
1. Verify the authored JSON key is present and spelled correctly
2. Check `loadTrainingHyperparameters()` requires the field at the correct path
3. Look for typos in `HyperParameters_GPU.hpp` assignments or grouped view slicing

## Recent Changes (Dec 2024)

### Consolidation Updates
- **Removed duplicates:** `prediction_comparison`, `attention_diagnostics` now only in `training.config`
- **Removed defaults:** Authored runtime fields must be present in `ai_config.json`; pure formulas are derived in `HyperParameters_GPU.hpp`
- **Collapsed raw config:** `ai_config_paths.hpp` now returns only `AiConfigSnapshot::{config_path, document}`; typed parsing moved to `HyperParameters_GPU.hpp`
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
- Runtime → `ai_config.json` + `HyperParameters_GPU.hpp` typed load from the raw `AiConfigSnapshot` document
- Never hardcode values in .cu/.cpp files

### Legacy Shims: Removed
When removing configuration options, DELETE all compatibility shims, fallbacks, and old code paths. Let it fail loud to expose misconnects. See copilot-instructions.md pitfall #20.

### Derived Constants: Explicit
When one constant depends on another, make the relationship explicit:
```cpp
constexpr int DERIVED = BASE_CONSTANT * 4;  // GOOD: Shows relationship
constexpr int DERIVED = 1024;               // BAD: Hides dependency on BASE=256
```

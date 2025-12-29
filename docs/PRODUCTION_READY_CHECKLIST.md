# GRIM-text Production Ready Checklist

## Overview
This document tracks issues that must be fixed before GRIM-text is considered production-ready.
Each item includes the file location, severity, and fix description.

---

## 🔴 Critical: TensorContract Integration

TensorContract provides type-safe tensor dimension validation. Several places compute GQA dimensions
manually using potentially incorrect formulas. These must be migrated to use TensorContract.

### GQA Dimension Formula (Correct)
```cpp
const int head_dim = d_model / num_heads;
const int kv_dim = num_kv_heads * head_dim;
const int total_qkv_dim = d_model + 2 * kv_dim;  // Q + K + V with GQA
```

### MHA Dimension Formula (Legacy - WRONG for GQA)
```cpp
const int qkv_size = 3 * d_model * d_model;  // ❌ Only correct when num_kv_heads == num_heads
```

---

### Issue #1: TrainingOps.cu - GPU Weight Initialization
- **File**: `resources/models/GRIM-text/training/TrainingOps.cu`
- **Line**: ~480
- **Severity**: 🔴 CRITICAL
- **Status**: ✅ FIXED (2025-12-17)
- **Problem**: Uses hardcoded `3 * cfg.d_model * cfg.d_model` for QKV weight size
- **Impact**: Buffer overrun/underrun when `num_kv_heads != num_heads`
- **Fix Applied**: Uses `TensorContract::compute_total_qkv_dim()` for size calculation

```cpp
// BEFORE (wrong)
size_t qkv_size = 3 * cfg.d_model * cfg.d_model;

// AFTER (correct)
TensorContract::GQADims gqa_dims{cfg.num_heads, cfg.num_kv_heads, head_dim, cfg.d_model};
size_t qkv_size = TensorContract::compute_total_qkv_dim(gqa_dims) * cfg.d_model;
```

---

### Issue #2: Phase1_Startup.cu - Weight Initialization
- **File**: `resources/models/GRIM-text/training/Phases/Phase1_Startup.cu`
- **Lines**: 627-640
- **Severity**: 🔴 CRITICAL
- **Status**: ✅ FIXED (2025-12-17)
- **Problem**: Uses `3 * cfg.d_model * cfg.d_model` in `launchXavierInit` calls
- **Impact**: Incorrect weight initialization size for GQA models
- **Fix Applied**: Computes GQA-aware dimensions using TensorContract

---

### Issue #3: LanguageModel_Training.cu - Gradient Buffer Allocations
- **File**: `resources/models/GRIM-text/training/LanguageModel_Training.cu`
- **Lines**: 80-100 (zeroGrad), 237-260 (buildParameterGroups)
- **Severity**: 🟡 MEDIUM
- **Status**: ✅ FIXED (2025-12-17)
- **Problem**: Computes `total_qkv_dim` manually without TensorContract validation
- **Fix Applied**: Added `TensorContract::validate_gqa_dims()` and dimension mismatch check in `buildParameterGroups()`

---

### Issue #4: BackwardPhase2_Encoder.cu - QKV Gradient Dimensions
- **File**: `resources/models/GRIM-text/Layers/BackwardOps/BackwardPhase2_Encoder.cu`
- **Line**: 640-660
- **Severity**: 🟡 MEDIUM
- **Status**: ✅ FIXED (2025-12-17)
- **Problem**: Computes `total_qkv_dim` inline without validation
- **Fix Applied**: Added `TensorContract::validate_gqa_dims()` and uses `TensorContract::compute_total_qkv_dim()`

---

### Issue #5: grim_transformer_gpu.hpp - QKV Workspace Offsets
- **File**: `resources/models/GRIM-text/Common/grim_transformer_gpu.hpp`
- **Lines**: 100-130
- **Severity**: 🟢 LOW
- **Status**: ✅ Already Correct
- **Note**: Uses `HyperParameters::isValidGQAConfig()` for validation and correct GQA-aware calculations. TensorContract integration not needed here.

---

### Issue #6: train_gpu_old.cu - Legacy Code (Deprecated)
- **File**: `resources/models/GRIM-text/training/train_gpu_old.cu`
- **Lines**: 1630-1631, 2554
- **Severity**: 🟢 LOW (deprecated file)
- **Status**: 📋 Not Applicable
- **Note**: This file is legacy backup. Do not fix - mark for removal.

---

## 🟡 Medium: Parameter Count Validation

### Issue #7: getModelStats Drift Detection
- **File**: `resources/models/GRIM-text/training/TrainingOps.cu`
- **Lines**: 163-270
- **Severity**: ✅ FIXED
- **Status**: ✅ Complete
- **Fix Applied**: 
  - Added TensorContract validation in debug path
  - Formula now uses GQA-aware dimensions
  - Drift assertion catches formula mismatches > 0.1%

---

## 🟡 Medium: LogRecorder Layer Configuration

### Issue #8: Layer Logging Not Configurable
- **File**: Multiple files
- **Severity**: ✅ FIXED
- **Status**: ✅ Complete
- **Fix Applied**:
  - Added `layers` section to `LogRecorderConfig` in `ai_config_paths.hpp`
  - Added `ConfigureLayerLogging()` API in `LogRecorder.cu`
  - Config parsing added in `Phase1_Startup.cu`
  - Silent drop in `WriteEntryToDisk()` if layer disabled

---

## 🟢 Low: Code Quality

### Issue #9: Hardcoded Log Paths
- **File**: `resources/models/GRIM-text/Shared/LogRecorder/LogRecorder.cu`
- **Line**: 22
- **Severity**: 🟢 LOW
- **Status**: ❌ Not Fixed
- **Problem**: `kDefaultLogsPath` hardcoded to absolute path
- **Fix**: Use `ai_config.json` paths instead

---

## Progress Summary

| Category | Total | Fixed | Remaining |
|----------|-------|-------|-----------|
| 🔴 Critical (TensorContract) | 2 | 2 | 0 |
| 🟡 Medium | 4 | 4 | 0 |
| 🟢 Low | 2 | 1 | 1 |
| **Total** | **8** | **7** | **1** |

---

## Next Steps

1. ~~**Priority 1**: Fix Issues #1 and #2 (Critical GQA dimension bugs)~~ ✅ DONE
2. ~~**Priority 2**: Add validation to Issues #3, #4, #5~~ ✅ DONE
3. **Priority 3**: Clean up hardcoded paths (Issue #9)
4. **Priority 4**: Remove deprecated train_gpu_old.cu (Issue #6)

---

## Changelog

| Date | Change |
|------|--------|
| 2025-12-17 | Created checklist |
| 2025-12-17 | Fixed getModelStats drift detection (Issue #7) |
| 2025-12-17 | Fixed LogRecorder layer config (Issue #8) |
| 2025-12-17 | Fixed TrainingOps.cu initGPU() GQA dimensions (Issue #1) |
| 2025-12-17 | Fixed Phase1_Startup.cu Xavier init GQA dimensions (Issue #2) |
| 2025-12-17 | Added TensorContract validation to LanguageModel_Training.cu (Issue #3) |
| 2025-12-17 | Added TensorContract validation to BackwardPhase2_Encoder.cu (Issue #4) |
| 2025-12-17 | Verified grim_transformer_gpu.hpp already correct (Issue #5) |
| 2025-12-17 | Fixed TensorContract::GQADims API usage (3 fields, member functions) |
| 2025-12-17 | Added per-layer logging in BackwardPhase2_Encoder (RMSNorm, FFN, Attention) |
| 2025-12-17 | Added step tracking to backward pass for layer logging |

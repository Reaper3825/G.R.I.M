# PyTorch Baseline Version Compatibility Fixes

## Issues Fixed (January 9, 2026)

### 1. Vocab Version 3 Support
**Problem:** PyTorch baseline only supported vocab version 2, but GRIM-text uses version 3.

**Changes Made:**
- Updated `GRIMVocab::load()` to accept versions 2-3
- Added token_id field reading for version 3 format (4 bytes after score)
- Version 3 format: `length + text + score + token_id` (vs v2: `length + text + score`)

**File:** `main.cpp` lines 328-380

### 2. GRMT Version 5 Support  
**Problem:** PyTorch baseline only supported GRMT version 4, but current training data is version 5.

**Root Cause:** Version mismatch caused immediate crash with `STATUS_STACK_BUFFER_OVERRUN` (exit code -1073740791) because exception thrown during load.

**Changes Made:**
- Updated `load_grmt()` to accept versions 4-5
- Both versions use same binary format (no structural changes)

**File:** `main.cpp` line 247

### 3. Improved Config Output
**Problem:** Config summary always showed JSONL path even when using GRMT mode, causing confusion.

**Changes Made:**
- Conditional config display based on `use_grmt` flag
- Shows relevant paths only (vocab + GRMT files when in GRMT mode)

**File:** `main.cpp` lines 1127-1140

### 4. Better Diagnostics
**Added:**
- Early startup logging to isolate crash location
- Flush calls after cout statements for real-time debugging
- Step-by-step validation logging in load_grmt()

## Version Compatibility Matrix

| Component | Old Version | New Version | Status |
|-----------|-------------|-------------|--------|
| Vocab Format | 2 only | 2-3 | ✅ Fixed |
| GRMT Format | 4 only | 4-5 | ✅ Fixed |
| Config Display | Always JSONL | Conditional | ✅ Fixed |

## Testing

```powershell
# Test GRMT mode (should work now)
cd D:\G.R.I.M
.\Tools\libtorch_baseline\run_baseline_grmt.ps1

# Or manual test
.\Tools\libtorch_baseline\build\Release\grim_libtorch_baseline.exe `
    --use_grmt 1 `
    --vocab_path resources\models\GRIM-text\training\data\vocab.bin `
    --grmt_path resources\models\GRIM-text\training\data\training_data.grmt `
    --max_tokens 100000 `
    --max_steps 10
```

## Format Details

### Vocab Binary Format v3
```
Header:
  - Magic: 'KTMG' (4 bytes)
  - Version: 3 (2 bytes)
  - Checksum: (4 bytes)
  - Config vocab size: (4 bytes)
  - Max length: (4 bytes)
  - Flags: (3 bytes)
  - Total vocab size: (4 bytes)

Pieces (repeated):
  - Length: (4 bytes)
  - Text: (variable)
  - Score: (4 bytes float)
  - Token ID: (4 bytes int32) ← NEW in v3
```

### GRMT Binary Format v5
```
Header:
  - Magic: 'GRMT' (0x474D5254)
  - Version: 5 (4 bytes)
  - Num sequences: (4 bytes)
  - Vocab size: (4 bytes)

Per Sequence:
  - Seq length: (4 bytes)
  - Tokens: (seq_len * 4 bytes)
  - Numeric values: (seq_len * 4 bytes float)
  - Numeric mask: (seq_len * 1 byte)
  - Text features: (seq_len * 16 * 2 bytes uint16)
  - Text mask: (seq_len * 1 byte)
```

## Next Steps

Both implementations now support the same data formats:
- ✅ Vocab v3 with token IDs
- ✅ GRMT v5 with text features
- ✅ 50,376 token vocabulary
- ✅ GQA (12:4 ratio)
- ✅ RMSNorm
- ✅ Weight tying
- ✅ Greedy sampling

Ready for fair comparison testing.

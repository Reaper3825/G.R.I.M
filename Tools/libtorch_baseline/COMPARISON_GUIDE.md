# PyTorch Baseline vs GRIM-text Comparison

## Quick Start

### 1. Build PyTorch Baseline
```powershell
cd Tools\libtorch_baseline
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### 2. Run with GRMT Data (matches GRIM-text exactly)
```powershell
cd Tools\libtorch_baseline
.\run_baseline_grmt.ps1
```

## Configuration Comparison

| Setting | GRIM-text | PyTorch Baseline (GRMT mode) |
|---------|-----------|------------------------------|
| **Vocab Size** | 50,376 tokens | 50,376 tokens ✅ |
| **Training Data** | 21,161 sequences | 21,161 sequences ✅ |
| **Sequence Length** | 1024 tokens | 1024 tokens ✅ |
| **Batch Size** | 6 | 6 ✅ |
| **Architecture** | 12 layers, 768 hidden, 12 heads | 12 layers, 768 hidden, 12 heads ✅ |
| **Attention** | GQA (12:4) | GQA (12:4) ✅ |
| **Normalization** | RMSNorm | RMSNorm ✅ |
| **Weight Tying** | Enabled | Enabled ✅ |
| **Learning Rate** | 3e-4 | 3e-4 ✅ |
| **Seed** | 1 | 1 ✅ |

## Expected Behavior

### Initial Loss (Step 1)
Both should start around:
- **GRIM-text**: loss ≈ 6.6 (from log)
- **PyTorch**: loss ≈ ln(50376) ≈ 10.8

Actually, GRIM-text's initial loss of 6.6 is **already trained** (not random init). For truly untrained models:
- Random baseline = ln(vocab_size) = ln(50376) ≈ **10.83**

### Sampling Output (Step 1)

**PyTorch (greedy, untrained):**
```
[Sample] "the the the the the the the the..."
```
- Picks highest probability token repeatedly

**GRIM-text (top-p, untrained):**
```
[Sample] "ature.ing cae 3cs and is dilicity..."
```
- Samples diverse random tokens from distribution

**GRIM-text (greedy, untrained) - after your change:**
```
[Sample] "the the the the the the the the..."
```
- Should match PyTorch pattern

## Sampling Strategy Impact

| Strategy | Untrained Output | Trained Output |
|----------|------------------|----------------|
| **Greedy** | Repetitive (mode collapse) | Coherent but conservative |
| **Top-p (0.9)** | Random gibberish | Creative, diverse text |
| **Top-k (40)** | Medium randomness | Balanced creativity |
| **Temperature 0.1** | Near-greedy | Focused, deterministic |
| **Temperature 2.0** | Very random | Wild, unpredictable |

## Running Comparisons

### Test 1: Same Initialization (both untrained)
```powershell
# GRIM-text (delete checkpoint first)
cd resources\models\GRIM-text\training
Remove-Item checkpoints\* -Force
cd TrainingLoop\build\Release
.\train_gpu.exe

# PyTorch baseline
cd Tools\libtorch_baseline
.\run_baseline_grmt.ps1
```

### Test 2: Same Sampling Strategy
Change GRIM-text to greedy (already done):
```cpp
// Phase2_TrainingLoop.cu line 316
cfg.strategy = GRIM::SamplingStrategy::GREEDY;
cfg.do_sample = false;
```

Rebuild GRIM-text training:
```powershell
cd resources\models\GRIM-text\training\TrainingLoop
cmake --build build --config Release
```

## Why Initial Loss Differs

If you see different initial losses, check:
1. **Checkpoint loaded?** GRIM-text may be loading a checkpoint
2. **Initialization seed?** Both use seed=1 but different RNG libraries
3. **Loss computation?** GRIM-text uses focal loss + label smoothing
4. **Vocab overlap?** Ensure both loaded same vocab.bin

## Debugging Checklist

- [ ] Both using `use_grmt=true` (50k vocab)
- [ ] Both using same `vocab.bin` file
- [ ] Both using same `training_data.grmt` file
- [ ] Both using `seq_len=1024, batch_size=6`
- [ ] Both using same seed=1
- [ ] GRIM-text: no checkpoint loaded (fresh start)
- [ ] PyTorch: executable built successfully
- [ ] Same sampling strategy (greedy vs greedy, or sampling vs sampling)

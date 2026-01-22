# Hardcoded Hidden States Diagnostic (Issue #42)

## Purpose

This diagnostic replaces encoder output with synthetic patterns to isolate whether mode collapse originates from:
1. **Encoder** producing W[277]-aligned hidden states, OR
2. **LM head/gradient system** having fundamental bugs

## Quick Start

### 1. Enable a Test Pattern

```powershell
# Test if centering fix (Issue #37) works
python test_hardcoded_states.py --pattern random_centered --status

# Test if LM head computes logits correctly
python test_hardcoded_states.py --pattern orthogonal_w277 --status
```

### 2. Rebuild Training Executable

```powershell
cd resources/models/GRIM-text/training/TrainingLoop
cmake --build build --config Release --target train_gpu
```

### 3. Run Training

```powershell
cd ../..
python test_hardcoded_states.py --pattern random_centered --run --batches 50
```

### 4. Analyze Logs

Look for diagnostic output in training log:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║ HARDCODED HIDDEN STATE DIAGNOSTIC (Issue #42)                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Batch: 1 | Pattern: random_centered                                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Hidden State Statistics (first token):                                    ║
║   Mean:     +0.000000 (should be ~0 for centered patterns)                ║
║   Variance: 0.001302  (should be ~1/768 = 0.001302)                      ║
║   Norm:     1.005123                                                      ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Alignment with W[277]:                                                    ║
║   h·W[277]:     +0.000123                                                 ║
║   ||W[277]||:   0.165421                                                  ║
║   cosine(h,W):  +0.000901 (orthogonal≈0, aligned≈1)                      ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Resulting Logits:                                                         ║
║   logit[277]:   +0.000456 (SPACE token)                                  ║
║   Top-5 predictions:                                                      ║
║     1. Token  5123: +0.001234                                             ║
║     2. Token  9876: +0.000987                                             ║
║     3. Token   277: +0.000456 <-- SPACE!                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

## Test Patterns

### 1. `random_centered` - Tests Issue #37 Fix

**What it does:**
- Generates random normal distribution with mean=0, stddev=1/√768
- Explicitly centers to ensure exact zero mean

**Expected behavior:**
- Mean = 0.0 (exact)
- Variance ≈ 1/768 = 0.001302
- Logits should be roughly uniform (no strong prediction)
- cosine(h, W[277]) ≈ 0.0 (small random value)

**If collapse happens:**
→ Gradient/optimizer bug (encoder is NOT the problem)

---

### 2. `orthogonal_w277` - Tests LM Head Computation

**What it does:**
- Generates random vector, then orthogonalizes against W[277]
- Uses Gram-Schmidt: h' = h - (h·W)/(W·W) × W

**Expected behavior:**
- cosine(h, W[277]) ≈ 0.0 (orthogonal)
- logit[277] = h·W[277] ≈ 0.0 (dot product = 0)
- Token 277 should NOT be top-5 prediction

**If logit[277] is high:**
→ LM head computation bug (matrix multiply or centering issue)

---

### 3. `aligned_w277` - Simulates Encoder Alignment

**What it does:**
- Sets h = W[277] (perfect alignment)

**Expected behavior:**
- cosine(h, W[277]) ≈ 1.0 (aligned)
- logit[277] should be HIGHEST (strong SPACE prediction)
- Token 277 should be #1 in top-5

**Purpose:**
- This simulates what happens when encoder learns to output W[277]-aligned states
- If gradient flows correctly, optimizer should DECREASE W[277] norm over batches
- If collapse happens even with aligned h → gradient sign flip (Issue #42)

---

### 4. `constant_uniform` - Tests Issue #40 Row Sum Bias

**What it does:**
- Sets all dimensions to constant: h[i] = 1/√768

**Expected behavior:**
- logit[v] = Σ(h[i] × W[v,i]) = (1/√768) × Σ(W[v,i])
- logit[v] = (row sum of W[v]) / √768
- If W[277] has systematically higher row sum → Issue #40 bias!

**If logit[277] is consistently top-1:**
→ FP32 GEMM accumulation error (Issue #40) not fully fixed

---

### 5. `zero_mean_sine` - Tests Centering Robustness

**What it does:**
- Generates sine wave: h[i] = (1/√768) × sin(2π × i × f)
- Frequency f based on dimension index
- Naturally has zero mean (sine is symmetric)

**Expected behavior:**
- Mean ≈ 0.0 (sine symmetry)
- Logits should have structured patterns (not random)
- Tests centering with non-random data

**Purpose:**
- Ensures centering works on structured data, not just random noise

---

## Interpreting Results

### Scenario 1: Collapse with Random Centered

```
Pattern: random_centered
Mean: 0.000000
logit[277] top-1 in 10/10 batches after batch 50
```

**Diagnosis:** Gradient/optimizer bug (NOT encoder)
- Hidden states are properly centered (mean=0)
- Encoder is NOT producing aligned states
- Problem is in gradient computation or optimizer updates

**Next investigation:**
- Check gradient direction for W[277] (Issue #42 logging)
- Verify optimizer step computation (AdamW formula)
- Test with different learning rates

---

### Scenario 2: High logit[277] with Orthogonal

```
Pattern: orthogonal_w277
cosine(h, W[277]): 0.000012 (orthogonal!)
logit[277]: +1.234567 (HIGHEST!)
```

**Diagnosis:** LM head computation bug
- Hidden states are orthogonal to W[277]
- Logit should be ≈0, but it's high
- Matrix multiply or centering is broken

**Next investigation:**
- Check cuBLAS GEMM call in lm_head_GPU.cu
- Verify centering is NOT applied to orthogonal pattern (should skip)
- Check for FP32 precision issues

---

### Scenario 3: No Collapse with Aligned

```
Pattern: aligned_w277
cosine(h, W[277]): 1.000000 (aligned!)
logit[277]: HIGHEST initially
After 50 batches: logit[277] decreases, W[277] norm decreases
```

**Diagnosis:** Gradient system is CORRECT!
- Optimizer correctly pushes W[277] down when h is aligned
- No gradient sign flip
- Issue #42 is FIXED

**Conclusion:** Encoder must be learning to output aligned states naturally

---

### Scenario 4: Logit[277] High with Constant Uniform

```
Pattern: constant_uniform
logit[277]: +0.005678 (rank #3)
logit[mean]: +0.004123
logit[277] - logit[mean]: +0.001555 (systematic bias!)
```

**Diagnosis:** Issue #40 row sum bias
- W[277] has higher row sum than average
- FP32 GEMM introduces systematic positive bias
- Recenter gradients fix NOT fully working

**Next investigation:**
- Verify recenter_gradients is enabled
- Check if recenterGradientRowsKernel is called
- Test with FP64 precision to confirm FP32 error

---

## Configuration

### Manual Config (ai_config.json)

```json
{
  "training": {
    "config": {
      "hardcoded_hidden_states": {
        "enabled": true,
        "pattern": "random_centered",
        "log_every_n_batches": 1
      }
    }
  }
}
```

### Disable Diagnostic

```powershell
python test_hardcoded_states.py --disable
```

## Build Requirements

The diagnostic requires rebuilding the training executable after changes:

```powershell
cd resources/models/GRIM-text/training/TrainingLoop
cmake --build build --config Release --target train_gpu
```

## Files Modified

1. **Config:**
   - `ai_config.json` - Added hardcoded_hidden_states section
   - `control/ai_config_paths.hpp` - Config parsing

2. **Core Implementation:**
   - `Layers/HardcodedStates/HardcodedStates_GPU.{hpp,cu}` - Pattern generation kernels
   - `Layers/ForwardOps/ForwardPhase1_OutputLayer.cu` - Integration point
   - `GRIM/grim_language_model_cuda.hpp` - Config structs

3. **Build System:**
   - `training/TrainingLoop/CMakeLists.txt` - Added HardcodedStates_GPU.cu

4. **Testing:**
   - `test_hardcoded_states.py` - Python helper script

## Troubleshooting

### Pattern Not Applied

Check training log for:
```
[ForwardPhase1] HARDCODED HIDDEN STATES: Replacing encoder output with pattern 1
```

If missing:
1. Verify `"enabled": true` in ai_config.json
2. Rebuild training executable
3. Check config parsing logs

### Diagnostic Box Not Showing

Increase logging frequency:
```powershell
python test_hardcoded_states.py --pattern random_centered --log-every 1
```

### Compilation Errors

Ensure CUDA toolkit is properly installed:
```powershell
nvcc --version  # Should show CUDA 12.5+
```

Check CMake cache:
```powershell
cd resources/models/GRIM-text/training/TrainingLoop
Remove-Item -Recurse -Force build/CMakeFiles
cmake --build build --config Release --target train_gpu
```

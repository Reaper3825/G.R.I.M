# Token Generation Fix - Invalid Token IDs Issue

## Problem Summary
The model was generating valid words followed by random garbage tokens like `rhrhrhrhrhcqcqozleleoz...`

**Example Output:**
```
Q: What is the capital of France?
A: whatisthecapitaloffrance?rhrhrhrhrhcqcqozleleozrhrhrhrhrhrhrhrhledrhcqcq...
```

## Root Causes Identified

### 1. **Missing Vocabulary Size Validation in Sampling**
- **Location:** `sampleFromLogits()` in `grim_language_model_gpu.cu`
- **Issue:** The discrete distribution sampler could return token IDs >= vocab_size
- **Why it happens:** 
  - Floating point errors in softmax
  - Numerical instability in probability distribution
  - No bounds checking after sampling

### 2. **Tokenizer Mismatch During Decode**
- **Location:** `TokenizerGPU::decode()` in `Tokenizer_GPU.cu`
- **Issue:** Invalid tokens (>= vocab_size) are silently skipped:
  ```cpp
  if (id < 0 || id >= vocab_size_val) {
      continue;  // Skips invalid token, no error!
  }
  ```
- **Result:** If training used one vocab but inference uses another, token meanings are scrambled

### 3. **No Validation in Generation Loop**
- **Location:** `generateSequenceGPU()` in `grim_language_model_gpu.cu`
- **Issue:** Generated tokens weren't validated before being added to sequence
- **Impact:** Invalid tokens propagate through the entire generation

## Fixes Applied

### Fix 1: Add Vocab Size Validation to sampleFromLogits
```cpp
SampleResult sampleFromLogits(const std::vector<float>& logits,
                              const GenerationConfig& cfg,
                              bool allow_sampling,
                              std::mt19937& rng,
                              int vocab_size) {  // NEW PARAMETER
    // ... existing code ...
    
    // Validate vocab_size matches logits size
    if (vocab_size > 0 && static_cast<int>(logits.size()) != vocab_size) {
        std::cerr << "[ERROR] Logits size != vocab_size" << std::endl;
        return result;
    }
    const int valid_vocab_size = vocab_size > 0 ? vocab_size : static_cast<int>(logits.size());
```

### Fix 2: Bounds Checking After Sampling
```cpp
int sampled = dist(rng);

// CRITICAL: Validate sampled token is within vocab bounds
if (sampled < 0 || sampled >= valid_vocab_size) {
    std::cerr << "[ERROR] Invalid token sampled: " << sampled 
              << " (vocab_size: " << valid_vocab_size << ")" << std::endl;
    // Fallback to argmax
    sampled = static_cast<int>(std::distance(
        probabilities.begin(),
        std::max_element(probabilities.begin(), probabilities.end())));
    if (sampled < 0 || sampled >= valid_vocab_size) {
        std::cerr << "[CRITICAL] Even argmax invalid. Setting to 0." << std::endl;
        sampled = 0;
    }
}
```

### Fix 3: Validation in All Sampling Paths
Applied the same bounds checking to:
- **Greedy sampling** (argmax path)
- **Fallback path** (when probability sum is zero)
- **Top-k/Top-p sampling** (discrete distribution path)

### Fix 4: Early Detection in Generation Loop
```cpp
const int vocab_size = config_.vocab_size;
SampleResult sample = sampleFromLogits(logits, cfg, allow_sampling, rng, vocab_size);

// Validate sampled token
if (sample.token_id < 0 || sample.token_id >= vocab_size) {
    std::cerr << "[ERROR] Invalid token generated: " << sample.token_id 
              << " at step " << step << " (vocab_size: " << vocab_size << ")" << std::endl;
    std::cerr << "         Context size: " << context.size() << std::endl;
    sequence.finished = true;
    break;  // Stop generation immediately
}
```

### Fix 5: Diagnostic Logging for NaN/Inf
```cpp
// Debug: Check for any NaN or Inf in logits
bool has_invalid = false;
for (size_t i = 0; i < logits.size(); ++i) {
    if (std::isnan(logits[i]) || std::isinf(logits[i])) {
        has_invalid = true;
        std::cerr << "[WARNING] Invalid logit at position " << i 
                  << ": " << logits[i] << std::endl;
    }
}
if (has_invalid) {
    std::cerr << "[WARNING] Step " << step 
              << " has invalid logits, this may cause sampling issues" << std::endl;
}
```

## How to Verify the Fix

### Step 1: Rebuild the Server
```powershell
cd D:\G.R.I.M\resources\models\GRIM-text\training
cmake --build build_vs_cuda --config Release
```

### Step 2: Test with Your Question
```powershell
python test_grim_model.py
```

### Step 3: Check for Error Messages
Look for these diagnostic messages in the output:
- `[ERROR] Logits size != vocab_size` - Indicates model/tokenizer mismatch
- `[ERROR] Invalid token sampled` - Sampling produced out-of-bounds token
- `[WARNING] Invalid logit at position` - NaN or Inf in logits (gradient issues)

## Additional Diagnostics to Check

### 1. Verify Vocab Consistency
```powershell
# Check vocab size in training config
cat D:\G.R.I.M\ai_config.json | Select-String "vocab_size"

# Check actual vocab file
# Should match config
```

### 2. Inspect Checkpoint
The checkpoint should match the vocab size:
- **Model:** `D:\G.R.I.M\resources\models\GRIM-text\checkpoints\checkpoint_epoch_5.bin`
- **Vocab:** `D:\G.R.I.M\resources\models\GRIM-text\training\models\vocab.bin`

### 3. Check for Gradient Issues
If you see NaN/Inf warnings, the problem is in training:
- Gradient explosion
- Numerical instability in softmax
- Learning rate too high

## What to Expect After Fix

✅ **Before Fix:**
```
A: whatisthecapitaloffrance?rhrhrhrhrhcqcqozleleozrhrhrhrhrhrhrhrhledrhcq
```

✅ **After Fix:**
```
A: Paris
```

OR if the underlying issue is gradient/logit problems:
```
A: Paris[ERROR] Invalid token sampled: 1500 (vocab_size: 1337)
[ERROR] Invalid token generated: 1500 at step 6 (vocab_size: 1337)
         Context size: 12
```

This tells you the **real** problem is in the model's forward pass or training, not just sampling.

## Possible Remaining Issues

If you still see errors after this fix, investigate:

### Issue 1: Logits Buffer Size Mismatch
Check `getNextTokenLogitsGPU()`:
```cpp
cudaMemcpy(logits.data.data(),
           training_state_.cached_logits + column_offset,
           static_cast<size_t>(cfg.vocab_size) * sizeof(float),  // Must match actual buffer
           cudaMemcpyDeviceToHost);
```

### Issue 2: LM Head Output Size
The language model head must output exactly `vocab_size` logits:
```cpp
// In LM head forward pass
// output should be [batch_size, vocab_size]
```

### Issue 3: Embedding Matrix Size Mismatch
Check that:
```cpp
config.vocab_size == embedder->token_embed.num_rows
```

## Testing Checklist

- [ ] Code compiles without errors
- [ ] Server starts successfully
- [ ] Test questions generate valid responses
- [ ] No `[ERROR] Invalid token` messages
- [ ] Output contains recognizable words
- [ ] No repeated garbage patterns
- [ ] Generation stops properly at EOS token

## Files Modified

1. `d:\G.R.I.M\resources\models\GRIM-text\Common\grim_language_model_gpu.cu`
   - Added vocab_size parameter to `sampleFromLogits()`
   - Added bounds checking in all sampling paths
   - Added token validation in `generateSequenceGPU()`
   - Added NaN/Inf diagnostic logging

## Next Steps if Issue Persists

1. **Enable verbose logging** in generation loop to see exact token IDs
2. **Compare vocab files** used in training vs inference
3. **Check model checkpoint** for corruption
4. **Verify LM head dimensions** match vocab_size
5. **Inspect training logs** for gradient anomalies

## References

- Issue: "Invalid token IDs after sampling"
- Date: December 7, 2025
- Models affected: GRIM-text checkpoint_epoch_5.bin

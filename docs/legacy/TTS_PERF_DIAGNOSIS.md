# TTS Performance Analysis - Current State

## ?? Current Problem

**Synthesis is taking 17.68 seconds** even with cached embeddings, which should be 1-3 seconds.

### Log Evidence:
```
[Coqui XTTS] ? Fast synthesis complete | Inference: 17.68s | Save: 0.09s | Total: 17.77s
```

### Missing Log Entries:
- ? No "Starting model warm-up..." message
- ? No "Model warmed up successfully" message
- ? No "GPU optimizations enabled" message

**Conclusion:** The optimization code we added is **not being executed**.

## ?? Root Cause Analysis

### Hypothesis 1: Python Process Restarting
The coqui_bridge.py process might be restarting between synthesis calls, losing the warm-up benefits.

**Check:**
```
grep "Launching Coqui XTTS" grim.log
```

If you see multiple instances, the process is restarting.

### Hypothesis 2: Warm-up Code Not Reached
The warm-up code might be failing silently before logging.

**Check:** Look for Python errors in stderr.

### Hypothesis 3: Model Re-initialization
The model might be reloading weights on each synthesis.

**Symptom:** First synthesis should be ~5-10s, subsequent ones ~1-3s. If all are ~17s, something's wrong.

## ?? Diagnostic Steps

### 1. Check if process is persistent:
```bash
# In PowerShell
Get-Process | Where-Object {$_.ProcessName -like "*python*"}
```

### 2. Check stderr for Python errors:
Look at the console/terminal where GRIM was launched for Python errors.

### 3. Force a restart and check logs:
1. Close GRIM
2. Delete all `.wav` files in `resources/tts_out/temp/`
3. Start GRIM fresh
4. Trigger TTS
5. Check logs for warm-up messages

## ?? Expected vs. Actual

### Expected Logs (with optimizations):
```
[Coqui XTTS] Model loaded successfully
[Coqui XTTS] GPU optimizations enabled (TF32, cudnn benchmark)
[Coqui XTTS] Starting model warm-up...
[Coqui XTTS] Dummy embedding loaded, running warm-up inference...
[Coqui XTTS] ? Model warmed up successfully (CUDA kernels initialized)
[Coqui XTTS] Model: tts_models/multilingual/multi-dataset/xtts_v2 | Device: cuda | XTTS v2: Yes
```

### Actual Logs (missing optimizations):
```
[Coqui XTTS] Model loaded successfully
[Coqui XTTS] Voice reference validated: ...
[Coqui XTTS] Using voice reference: ...
[Coqui XTTS] Loaded embedding from cache: default
[Coqui XTTS] Using cached embedding for: default (FAST MODE)
? Fast synthesis | Inference: 17.68s | ...
```

## ?? Quick Fix Test

Try running the Python bridge manually:

```bash
cd D:\G.R.I.M\resources\python
python coqui_bridge.py --persistent --model tts_models/multilingual/multi-dataset/xtts_v2 --gpu
```

Look for the warm-up messages. If they appear, the code is fine but GRIM isn't using the new version.

## ?? Most Likely Issue

**The coqui_bridge.py file wasn't reloaded.**

### Solution:
1. Close GRIM completely
2. Kill any lingering Python processes:
   ```powershell
   Get-Process python | Stop-Process -Force
   ```
3. Rebuild/restart GRIM
4. Check logs again

## ?? Performance Targets

### With Optimizations Working:
- **First synthesis:** 3-6s (warm-up + synthesis)
- **Subsequent syntheses:** 1-3s (warm model + cached embeddings)
- **Common phrases:** ~0s (from cache)

### Current (Without Optimizations):
- **All syntheses:** ~17s ?
- **Every synthesis is "cold"** ?

## ?? Next Steps

1. **Verify Python file updated:**
   ```powershell
   Get-Content resources\python\coqui_bridge.py | Select-String "Starting model warm-up"
   ```
   Should return the line if file is updated.

2. **Check if process is using old code:**
   - Kill Python processes
   - Restart GRIM
   - Check for warm-up logs

3. **If still slow after restart:**
   - GPU might not be available
   - Check: `torch.cuda.is_available()` in Python
   - Model might be on CPU (very slow)

4. **If warm-up runs but still slow:**
   - CUDA might be broken
   - Try CPU-only to compare
   - Check GPU memory usage

---

**Status:** Awaiting restart + log verification

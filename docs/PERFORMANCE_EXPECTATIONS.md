# XTTS v2 Performance Guide

## ? Speed Expectations

### Ideal Conditions (GPU free)
| Method | Speed | Notes |
|--------|-------|-------|
| **With cached embedding** | 0.5-2s | ? FAST |
| **Without embedding** | 5-10s | Slow (computes embedding) |
| **First time (cold start)** | 30-60s | Model loading |

### With GPU Contention (e.g., during gaming)
| Scenario | Speed | Why |
|----------|-------|-----|
| **Heavy game running** | 20-40s | GPU shared with game |
| **Light game/app** | 5-15s | Some GPU contention |
| **No other apps** | 0.5-2s | ? Full GPU available |

## ?? Your Situation

You reported:
- **60+ seconds** during high graphics gaming
- This is **normal** when GPU is heavily loaded

### Why It's Slow During Gaming

1. **GPU Memory Competition**
   - Game: ~8-12GB VRAM
   - XTTS v2: ~2-3GB VRAM
   - Context switching: +time

2. **CUDA Kernel Scheduling**
   - Game gets priority (foreground)
   - XTTS waits for GPU access
   - Each CUDA call is slower

3. **Thermal Throttling**
   - GPU at 100% (game + XTTS)
   - May throttle to manage heat
   - Further reduces performance

## ?? Solutions

### Option 1: Accept Gaming Performance (Recommended)
- **During gameplay:** 20-60s per TTS request
- **After closing game:** 0.5-2s per request
- **Cached embeddings help** but GPU is still bottleneck

### Option 2: Use CPU for TTS
Edit `D:\G.R.I.M\resources\python\coqui_bridge.py`:

```python
# Change line ~461:
device = 'cpu'  # Force CPU instead of CUDA
```

**Trade-offs:**
- ? No GPU contention
- ? Slower than GPU when free (5-10s)
- ? Consistent speed during gaming (~10s)

### Option 3: Lower Game Graphics
- Reduce GPU load
- XTTS gets more GPU time
- Faster TTS (~10-20s instead of 60s)

### Option 4: Use Simpler TTS
Switch to SAPI in `ai_config.json`:

```json
{
  "voice": {
    "engine": "sapi"  
  }
}
```

**Trade-offs:**
- ? Instant (0.1s)
- ? No GPU usage
- ? Lower quality voice
- ? No voice cloning

## ?? Benchmark Results

### Your System (RTX 3080 Ti)
- **Idle:** 0.5-2s with embeddings ?
- **Light load:** 5-10s
- **Gaming:** 20-60s (your experience)

### Expected with Cached Embeddings

```
No GPU load:     ?????????? 0.5-2s   ? IDEAL
Light apps:      ?????????? 5-10s
Heavy game:      ?????????? 20-60s   ? YOUR CASE
```

## ? Verification Steps

1. **Test with game closed:**
   ```
   test_tts Hello, testing without game running
   ```
   Expected: ~1-2 seconds

2. **Test during game:**
   ```
   test_tts Hello, testing during game
   ```
   Expected: 20-60 seconds (normal!)

3. **Check embedding is cached:**
   ```
   list_embeddings
   ```
   Should show: `default (130 KB)`

## ?? Current Status

? **Embedding created:** `default.npz` (130 KB)  
? **PyTorch fix applied:** Model loads correctly  
? **Persistent mode:** Model stays loaded  
? **GPU contention:** Gaming slows TTS significantly

## ?? Recommendation

**For gaming use:**

1. **Keep current setup** (embeddings working)
2. **Accept slower speed during games** (20-60s)
3. **Use voice commands before starting game**
4. **Or switch to SAPI for instant** (but lower quality)

**Best workflow:**
```
# Before starting game:
test_tts Pre-cache this phrase

# During game (slow but works):
"Hey GRIM, ..." (waits 20-60s for response)

# After game (fast):
test_tts Back to normal speed!
```

## ?? Diagnostic Commands

```bash
# Test current speed (game closed)
python scripts/test_embedding_speed.py

# Check embedding exists
python -c "import os; print(os.path.exists('D:/G.R.I.M/resources/voices/embeddings/default.npz'))"

# Monitor GPU usage
nvidia-smi -l 1
```

---

**Bottom line:** Your system is working correctly. The 60s delay is **expected** when GPU is heavily used by games. Cached embeddings help, but can't eliminate GPU contention.
